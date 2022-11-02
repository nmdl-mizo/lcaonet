from __future__ import annotations  # type: ignore

from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.typing import Adj
from torch_scatter import scatter

from pyg_material.data import DataKeys
from pyg_material.model.base import BaseGNN
from pyg_material.nn import AtomicNum2Node, Dense, Swish
from pyg_material.utils import activation_resolver


class EGNNConv(MessagePassing):
    """The block to calculate massage passing and update node embeddings. It is
    implemented in the manner of PyTorch Geometric.

    Args:
        x_dim (int or Tuple[int, int]]): number of node dimension. if set to tuple object,
            the first one is input dim, and second one is output dim.
        edge_dim (int): number of edge dimension.
        activation (Callable, optional): activation function. Defaults to `Swish()`.
        edge_attr_dim (int or `None`, optional): number of another edge attribute dimension. Defaults to `None`.
        node_hidden (int, optional): dimension of node hidden layers. Defaults to `256`.
        edge_hidden (int, optional): dimension of edge hidden layers. Defaults to `256`.
        cutoff_net (nn.Module, optional): cutoff network. Defaults to `None`.
        batch_norm (bool, optional): if set to `False`, no batch normalization is used. Defaults to `False`.
        aggr ("add" or "mean", optional): aggregation method. Defaults to `"add"`.
        weight_init (Callable, optional): weight initialization function. Defaults to `glorot_orthogonal`.
    """

    def __init__(
        self,
        x_dim: int | tuple[int, int],
        edge_dim: int,
        activation: Callable[[Tensor], Tensor] = Swish(beta=1.0),
        edge_attr_dim: int | None = None,
        node_hidden: int = 256,
        edge_hidden: int = 256,
        cutoff_net: nn.Module | None = None,
        batch_norm: bool = False,
        aggr: str = "add",
        weight_init: Callable[[Tensor], Tensor] = glorot_orthogonal,
        **kwargs,
    ):
        assert aggr == "add" or aggr == "mean"
        # TODO: pass kwargs to superclass
        super().__init__(aggr=aggr)

        # name node_dim is already used in super class
        self.x_dim = x_dim
        self.edge_dim = edge_dim
        self.edge_attr_dim = edge_attr_dim
        self.node_hidden = node_hidden
        self.edge_hidden = edge_hidden
        if cutoff_net is None:
            self.cutoff_net = None
        else:
            self.cutoff_net = cutoff_net
        self.batch_norm = batch_norm

        if isinstance(x_dim, int):
            x_dim = (x_dim, x_dim)
        assert x_dim[0] == x_dim[1]
        if edge_attr_dim is None:
            edge_attr_dim = 0

        # updata function
        self.edge_func = nn.Sequential(
            Dense(
                x_dim[0] * 2 + 1 + edge_attr_dim,
                edge_hidden,
                bias=True,
                weight_init=weight_init,
                **kwargs,
            ),
            activation,
            Dense(edge_hidden, edge_dim, bias=True, weight_init=weight_init, **kwargs),
            activation,
        )
        self.node_func = nn.Sequential(
            Dense(
                x_dim[0] + edge_dim,
                node_hidden,
                bias=True,
                weight_init=weight_init,
                **kwargs,
            ),
            activation,
            Dense(node_hidden, x_dim[1], bias=True, weight_init=weight_init, **kwargs),
        )
        # attention the edge
        self.atten = nn.Sequential(
            # using xavier uniform initialization
            Dense(
                edge_dim,
                1,
                bias=True,
                weight_init=nn.init.xavier_uniform_,
                gain=1.0,
            ),
            nn.Sigmoid(),
        )
        # batch normalization
        if batch_norm:
            self.bn = nn.BatchNorm1d(x_dim[1])
        else:
            self.bn = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        if self.batch_norm:
            self.bn.reset_parameters()

    def forward(
        self,
        x: Tensor,
        dist: Tensor,
        edge_index: Adj,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        # propagate_type:
        # (x: Tensor, dist: Tensor, edge_attr: Optional[Tensor])
        edge = self.propagate(edge_index, x=x, dist=dist, edge_attr=edge_attr, size=None)
        out = torch.cat([x, edge], dim=-1)
        for nf in self.node_func:
            out = nf(out)
        out = self.bn(out)
        out = out + x
        return out

    def message(
        self,
        x_i: Tensor,
        x_j: Tensor,
        dist: Tensor,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        # update edge
        if edge_attr is None:
            edge_new = torch.cat([x_i, x_j, torch.pow(dist, 2).unsqueeze(-1)], dim=-1)
        else:
            assert edge_attr.size[-1] == self.edge_attr_dim
            edge_new = torch.cat(
                [x_i, x_j, torch.pow(dist, 2).unsqueeze(-1), edge_attr],
                dim=-1,
            )
        edge_new = self.edge_func(edge_new)

        # cutoff net
        if self.cutoff_net is not None:
            edge_new = self.cutoff_net(dist).unsqueeze(-1) * edge_new
        # get attention weight
        edge_new = self.atten(edge_new) * edge_new

        return edge_new


class EGNNOutBlock(nn.Module):
    """The block to compute the global graph proptery from node embeddings. In
    this block, after aggregation, two more Dense layers are calculated.

    Args:
        node_dim (int): number of input dimension.
        hidden_dim (int, optional): number of hidden layers dimension. Defaults to `128`.
        out_dim (int, optional): number of output dimension. Defaults to `1`.
        activation: (Callable, optional): activation function. Defaults to `Swish(beta=1.0)`.
        aggr (`"add"` or `"mean"`): aggregation method. Defaults to `"add"`.
        weight_init (Callable, optional): weight initialization function. Defaults to `torch_geometric.nn.inits.glorot_orthogonal`.
    """  # NOQA: E501

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 1,
        activation: Callable[[Tensor], Tensor] = Swish(beta=1.0),
        aggr: str = "add",
        weight_init: Callable[[Tensor], Tensor] = glorot_orthogonal,
        **kwargs,
    ):
        super().__init__()

        assert aggr == "add" or aggr == "mean"
        self.aggr = aggr
        self.node_transform = nn.Sequential(
            Dense(node_dim, hidden_dim, bias=True, weight_init=weight_init, **kwargs),
            activation,
            Dense(hidden_dim, hidden_dim, bias=True, weight_init=weight_init, **kwargs),
        )
        self.output = nn.Sequential(
            Dense(hidden_dim, hidden_dim, bias=True, weight_init=weight_init, **kwargs),
            activation,
            Dense(hidden_dim, out_dim, bias=False, weight_init=weight_init, **kwargs),
        )

    def forward(self, x: Tensor, batch: Tensor | None = None) -> Tensor:
        """Compute global property from node embeddings.

        Args:
            x (Tensor): node embeddings shape of (n_node x in_dim).
            batch (Tensor, optional): batch index shape of (n_node). Defaults to `None`.

        Returns:
            Tensor: shape of (n_batch x out_dim).
        """
        out = self.node_transform(x)
        out = scatter(out, index=batch, dim=0, reduce=self.aggr)
        return self.output(out)


class EGNN(BaseGNN):
    """EGNN implemeted by using PyTorch Geometric. From atomic structure,
    predict global property such as energy.

    Args:
        node_dim (int): number of node embedding dimension.
        edge_dim (int): number of edge embedding dimension.
        n_conv_layer (int): number of convolutinal layers.
        cutoff_radi (float): cutoff radious. Defaults to `None`.
        out_dim (int, optional): number of output property dimension.
        activation (str, optional): activation function name. Defaults to `"swish"`.
        cutoff_radi (float): cutoff radious. Defaults to `None`.
        cutoff_net (nn.Module, optional): cutoff network. Defaults to `None`.
        hidden_dim (int, optional): number of hidden layers. Defaults to `256`.
        aggr (`"add"` or `"mean"`, optional): aggregation method. Defaults to `"add"`.
        batch_norm (bool, optional): if `False`, no batch normalization in convolution layers. Defaults to `False`.
        edge_attr_dim (int, optional): number of another edge attrbute dimension. Defaults to `None`.
        max_z (int, optional): max number of atomic number. Defaults to `100`.

    Notes:
        PyTorch Geometric:
        https://pytorch-geometric.readthedocs.io/en/latest/

        EGNN:
        [1] V. G. Satorras et al., arXiv (2021), (available at http://arxiv.org/abs/2102.09844).
        [2] https://docs.e3nn.org/en/stable/index.html
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        n_conv_layer: int,
        out_dim: int,
        activation: str = "swish",
        cutoff_net: nn.Module | None = None,
        cutoff_radi: float | None = None,
        hidden_dim: int = 256,
        aggr: str = "add",
        batch_norm: bool = False,
        edge_attr_dim: int | None = None,
        max_z: int | None = 100,
        **kwargs,
    ):
        super().__init__()
        act = activation_resolver(activation)

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.n_conv_layer = n_conv_layer
        self.cutoff_radi = cutoff_radi
        self.out_dim = out_dim
        self.edge_attr = edge_attr_dim
        # layers
        self.node_embed = AtomicNum2Node(node_dim, max_z=max_z)
        if cutoff_net is None:
            self.cutoff_net = None
        else:
            assert cutoff_radi is not None
            self.cutoff_net = cutoff_net(cutoff_radi)

        self.convs = nn.ModuleList(
            [
                EGNNConv(
                    x_dim=node_dim,
                    edge_dim=edge_dim,
                    activation=act,
                    edge_attr_dim=edge_attr_dim,
                    node_hidden=hidden_dim,
                    edge_hidden=hidden_dim,
                    cutoff_net=self.cutoff_net,
                    aggr=aggr,
                    batch_norm=batch_norm,
                    **kwargs,
                )
                for _ in range(n_conv_layer)
            ]
        )

        self.output = EGNNOutBlock(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            activation=act,
            aggr=aggr,
            **kwargs,
        )

        self.reset_parameters()

    def reset_parameters(self):
        if self.cutoff_net is not None:
            if hasattr(self.cutoff_net, "reset_parameters"):
                self.cutoff_net.reset_parameters()

    def forward(self, data_batch) -> Tensor:
        if self.edge_attr is not None:
            data_dict = self.get_data(
                data_batch,
                batch_index=True,
                atom_numbers=True,
                edge_index=True,
                edge_attr=True,
            )
            batch = data_dict[DataKeys.Batch]
            atom_numbers = data_dict[DataKeys.Atom_numbers]
            edge_index = data_dict[DataKeys.Edge_index]
            edge_attr = data_dict[DataKeys.Edge_attr]
        else:
            data_dict = self.get_data(data_batch, batch_index=True, atom_numbers=True, edge_index=True)
            batch = data_dict[DataKeys.Batch]
            atom_numbers = data_dict[DataKeys.Atom_numbers]
            edge_index = data_dict[DataKeys.Edge_index]
            edge_attr = None
        # calc atomic distances
        distances = self.calc_atomic_distances(data_batch)
        # initial embedding
        x = self.node_embed(atom_numbers)

        # convolution
        for conv in self.convs:
            x = conv(x, distances, edge_index, edge_attr)
        # read out property
        x = self.output(x, batch)
        return x

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"node_dim={self.node_dim}, "
            f"edge_dim={self.edge_dim}, "
            f"cutoff_radi={self.cutoff_radi}, "
            f"out_dim={self.out_dim}, "
            f"convolution_layers: {self.convs[0].__class__.__name__} * {self.n_conv_layer})"
        )
