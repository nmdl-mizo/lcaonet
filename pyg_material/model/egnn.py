from __future__ import annotations  # type: ignore

from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot_orthogonal
from torch_scatter import scatter

from pyg_material.data import DataKeys
from pyg_material.model.base import BaseGNN
from pyg_material.nn import AtomicNum2Node, Dense, Swish
from pyg_material.utils import activation_resolver, init_resolver


class EGNNConv(MessagePassing):
    """The block to calculate massage passing and update node embeddings. It is
    implemented in the manner of PyTorch Geometric.

    Args:
        x_dim (int or Tuple[int, int]]): number of node dimension.
            If set to tuple object, the first one is input dim, and second one is output dim.
        edge_dim (int): number of edge dimension.
        activation (nn.Module, optional): activation function. Defaults to `Swish(beta=1.0)`.
        edge_attr_dim (int or `None`, optional): number of another edge attribute dimension. Defaults to `None`.
        cutoff_net (nn.Module, optional): cutoff network. Defaults to `None`.
        batch_norm (bool, optional): if set to `False`, no batch normalization is used. Defaults to `False`.
        aggr ("add" or "mean", optional): aggregation method. Defaults to `"add"`.
        weight_init (Callable, optional): weight initialization function. Defaults to `glorot_orthogonal`.
    """

    def __init__(
        self,
        x_dim: int | tuple[int, int],
        edge_dim: int,
        activation: nn.Module = Swish(beta=1.0),
        edge_attr_dim: int | None = None,
        cutoff_net: nn.Module | None = None,
        batch_norm: bool = False,
        aggr: str = "add",
        weight_init: Callable[[Tensor], Tensor] = glorot_orthogonal,
        **kwargs,
    ):
        assert aggr == "add" or aggr == "mean"
        super().__init__(aggr=aggr, **kwargs)

        self.x_dim = x_dim  # name node_dim is already used in super class
        self.edge_dim = edge_dim
        self.edge_attr_dim = edge_attr_dim
        self.cutoff_net = cutoff_net
        self.batch_norm = batch_norm

        if isinstance(x_dim, int):
            x_dim = (x_dim, x_dim)
        assert x_dim[0] == x_dim[1]
        if edge_attr_dim is None:
            edge_attr_dim = 0

        # updata function
        self.edge_lin = nn.Sequential(
            Dense(x_dim[0] * 2 + 1 + edge_attr_dim, edge_dim, True, weight_init=weight_init),
            activation,
            Dense(edge_dim, edge_dim, True, weight_init=weight_init),
            activation,
        )
        self.node_lin = nn.Sequential(
            Dense(x_dim[0] + edge_dim, x_dim[1], True, weight_init=weight_init),
            activation,
            Dense(x_dim[1], x_dim[1], True, weight_init=weight_init),
        )
        # attention the edge
        self.atten = nn.Sequential(
            # using xavier uniform initialization
            Dense(edge_dim, 1, True, weight_init=nn.init.xavier_uniform_),
            nn.Sigmoid(),
        )
        # batch normalization
        self.bn: nn.Module
        if batch_norm:
            self.bn = nn.BatchNorm1d(x_dim[1])
        else:
            self.bn = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        if self.batch_norm:
            self.bn.reset_parameters()

    def forward(self, x: Tensor, dist: Tensor, edge_index: Tensor, edge_attr: Tensor | None = None) -> Tensor:
        """Forward calculation of egnn convolution block.

        Args:
            x (Tensor): node embeddings of (n_node, x_dim) shape.
            dist (Tensor): inter atomic distances of (n_edge) shape.
            edge_index (Tensor): edge index of (2, n_edge) shape.
            edge_attr (Tensor, optional): edge attributes of (n_edge, edge_attr_dim) shape. Defaults to `None`.
        """
        # propagate_type:
        # (x: Tensor, dist: Tensor, edge_attr: Optional[Tensor])
        edge = self.propagate(edge_index, x=x, dist=dist, edge_attr=edge_attr, size=None)
        out = torch.cat([x, edge], dim=-1)
        for nl in self.node_lin:
            out = nl(out)
        out = self.bn(out)
        # residual connection
        return out + x

    def message(self, x_i: Tensor, x_j: Tensor, dist: Tensor, edge_attr: Tensor | None = None) -> Tensor:
        # update edge
        if edge_attr is None:
            edge_new = torch.cat([x_i, x_j, torch.pow(dist, 2).unsqueeze(-1)], dim=-1)
        else:
            assert edge_attr.size(-1) == self.edge_attr_dim
            edge_new = torch.cat(
                [x_i, x_j, torch.pow(dist, 2).unsqueeze(-1), edge_attr],
                dim=-1,
            )
        edge_new = self.edge_lin(edge_new)

        # cutoff net
        if self.cutoff_net is not None:
            edge_new = self.cutoff_net(dist).unsqueeze(-1) * edge_new
        # get attention weight
        edge_new = self.atten(edge_new) * edge_new

        return edge_new


class EGNNOutBlock(nn.Module):
    """The block to compute the graph-wise proptery from node embeddings. In
    this block, after aggregation, two more Dense layers are added.

    Args:
        node_dim (int): number of input dimension.
        out_dim (int, optional): number of output dimension. Defaults to `1`.
        activation: (nn.Module, optional): activation function. Defaults to `Swish(beta=1.0)`.
        aggr (`"add"` or `"mean"`): aggregation method. Defaults to `"add"`.
        weight_init (Callable, optional): weight initialization function.
            Defaults to `torch_geometric.nn.inits.glorot_orthogonal`.
    """

    def __init__(
        self,
        node_dim: int,
        out_dim: int = 1,
        activation: nn.Module = Swish(beta=1.0),
        aggr: str = "add",
        weight_init: Callable[[Tensor], Tensor] = glorot_orthogonal,
    ):
        super().__init__()

        assert aggr == "add" or aggr == "mean"
        self.aggr = aggr
        self.node_lin = nn.Sequential(
            Dense(node_dim, node_dim, True, weight_init=weight_init),
            activation,
            Dense(node_dim, node_dim, True, weight_init=weight_init),
        )
        self.out_lin = nn.Sequential(
            Dense(node_dim, node_dim // 2, True, weight_init=weight_init),
            activation,
            Dense(node_dim // 2, out_dim, False, weight_init=weight_init),
        )

    def forward(self, x: Tensor, batch_idx: Tensor | None = None) -> Tensor:
        """Compute graph-wise property from node embeddings.

        Args:
            x (Tensor): node embeddings of (n_node, node_dim) shape.
            batch_idx (Tensor, optional): batch index of (n_node) shape. Defaults to `None`.

        Returns:
            Tensor: output values of (n_batch, out_dim) shape.
        """
        out = self.node_lin(x)
        out = out.sum(dim=0) if batch_idx is None else scatter(out, batch_idx, dim=0, reduce=self.aggr)
        return self.out_lin(out)


class EGNN(BaseGNN):
    """EGNN implemeted by using PyTorch Geometric. From atomic structure,
    predict graph-wise property such as formation energy.

    Args:
        node_dim (int): number of node embedding dimension.
        edge_dim (int): number of edge embedding dimension.
        n_conv_layer (int): number of convolutional layers.
        out_dim (int, optional): number of output property dimension.
        activation (str, optional): activation function name. Defaults to `"swish"`.
        cutoff (float): cutoff radious. Defaults to `None`.
        cutoff_net (nn.Module, optional): cutoff network. Defaults to `None`.
        aggr (`"add"` or `"mean"`, optional): aggregation method. Defaults to `"add"`.
        weight_init (str, optional): name of weight initialization function. Defaults to `"glorot_orthogonal"`.
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
        aggr: str = "add",
        weight_init: str = "glorot_orthogonal",
        batch_norm: bool = False,
        edge_attr_dim: int | None = None,
        max_z: int = 100,
        **kwargs,
    ):
        super().__init__()
        act: nn.Module = activation_resolver(activation)
        wi: Callable[[Tensor], Tensor] = init_resolver(weight_init)

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.n_conv_layer = n_conv_layer
        self.out_dim = out_dim
        self.edge_attr = edge_attr_dim

        # layers
        self.node_embed = AtomicNum2Node(node_dim, max_z=max_z)
        self.cutoff_net = cutoff_net

        self.convs = nn.ModuleList(
            [
                EGNNConv(
                    x_dim=node_dim,
                    edge_dim=edge_dim,
                    activation=act,
                    edge_attr_dim=edge_attr_dim,
                    cutoff_net=self.cutoff_net,
                    aggr=aggr,
                    batch_norm=batch_norm,
                    weight_init=wi,
                    **kwargs,
                )
                for _ in range(n_conv_layer)
            ]
        )

        self.output = EGNNOutBlock(
            node_dim=node_dim,
            out_dim=out_dim,
            activation=act,
            weight_init=wi,
            aggr=aggr,
        )

    def forward(self, batch: Batch) -> Tensor:
        batch_idx = batch.get(DataKeys.Batch_idx)
        atom_numbers = batch[DataKeys.Atom_numbers]
        edge_idx = batch[DataKeys.Edge_idx]
        edge_attr = batch.get(DataKeys.Edge_attr)

        # calc atomic distances
        distances = self.calc_atomic_distances(batch)

        # initial embedding
        x = self.node_embed(atom_numbers)

        # convolution
        for conv in self.convs:
            x = conv(x, distances, edge_idx, edge_attr)

        # read out property
        return self.output(x, batch_idx)
