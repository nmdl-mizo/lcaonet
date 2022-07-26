from typing import Callable, Tuple, Union, Optional, Literal, Any

import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.nn.inits import glorot_orthogonal

from pyggnn.nn.activation import Swish
from pyggnn.nn.base import Dense


__all__ = ["EGNNConv"]


class EGNNConv(MessagePassing):
    """
    The block to calculate massage passing and update node embeddings.
    It is implemented in the manner of PyTorch Geometric.
    """

    def __init__(
        self,
        x_dim: Union[int, Tuple[int, int]],
        edge_dim: int,
        activation: Callable[[Tensor], Tensor] = Swish(beta=1.0),
        edge_attr_dim: Optional[int] = None,
        node_hidden: int = 256,
        edge_hidden: int = 256,
        cutoff_net: Optional[nn.Module] = None,
        batch_norm: bool = False,
        aggr: Literal["add", "mean"] = "add",
        weight_init: Callable[[Tensor], Any] = glorot_orthogonal,
        **kwargs,
    ):
        """
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
                weight_init=torch.nn.init.xavier_uniform_,
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
        edge_attr: Optional[Tensor] = None,
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
        edge_attr: Optional[Tensor],
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
