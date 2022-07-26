from typing import Tuple, Union, Optional, Any

import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj

from pyggnn.nn.base import Dense
from pyggnn.utils.resolve import activation_resolver


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
        activation: Union[Any, str] = "swish",
        edge_attr_dim: Optional[int] = None,
        node_hidden: int = 256,
        edge_hidden: int = 256,
        cutoff_net: Optional[nn.Module] = None,
        cutoff_radi: Optional[float] = None,
        batch_norm: bool = False,
        aggr: str = "add",
        **kwargs,
    ):
        """
        Args:
            x_dim (int or Tuple[int, int]]): number of node dimnsion. if set to tuple
                object, the first one is input dim, and second one is output dim.
            edge_dim (int): number of edge dim.
            activation (str or nn.Module): activation function. Defaults to `swish`.
            edge_attr_dim (int or `None`, optional): number of another edge
                attribute dim. Defaults to `None`.
            node_hidden (int, optional): dimension of node hidden layers.
                Defaults to `256`.
            edge_hidden (int, optional): dimension of edge hidden layers.
                Defaults to `256`.
            cutoff_net (nn.Module, optional): cutoff network. Defaults to `None`.
            cutoff_radi (float, optional): cutoff radious. Defaults to `None`.
            batch_norm (bool, optional): if set to `False`, no batch normalization is
                used. Defaults to `False`.
            aggr (str, optional): aggregation method. Defaults to `"add"`.
        """
        # TODO: pass kwargs to superclass
        super().__init__(aggr=aggr)
        act = activation_resolver(activation, **kwargs)

        # name node_dim is already used in super class
        self.x_dim = x_dim
        self.edge_dim = edge_dim
        self.edge_attr_dim = edge_attr_dim
        self.node_hidden = node_hidden
        self.edge_hidden = edge_hidden
        if cutoff_net is None:
            self.cutoff_net = None
        else:
            assert (
                cutoff_radi is not None
            ), "cutoff_radi must be set if cutoff_net is set"
            self.cutoff_net = cutoff_net(cutoff_radi)
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
                activation_name=activation,
                **kwargs,
            ),
            act,
            Dense(
                edge_hidden,
                edge_dim,
                bias=True,
                activation_name=activation,
                **kwargs,
            ),
            act,
        )
        self.node_func = nn.Sequential(
            Dense(
                x_dim[0] + edge_dim,
                node_hidden,
                bias=True,
                activation_name=activation,
                **kwargs,
            ),
            act,
            Dense(node_hidden, x_dim[1], bias=True),
        )
        # attention the edge
        self.atten = nn.Sequential(
            Dense(edge_dim, 1, bias=True, activation_name="sigmoid", **kwargs),
            nn.Sigmoid(),
        )
        # batch normalization
        if batch_norm:
            self.bn = nn.BatchNorm1d(x_dim[1])
        else:
            self.bn = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        for ef in self.edge_func:
            if hasattr(ef, "reset_parameters"):
                ef.reset_parameters()
        for nf in self.node_func:
            if hasattr(nf, "reset_parameters"):
                nf.reset_parameters()
        for at in self.atten:
            if hasattr(at, "reset_parameters"):
                at.reset_parameters()
        if self.cutoff_net is not None:
            self.cutoff_net.reset_parameters()
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
        edge = self.propagate(
            edge_index,
            x=x,
            dist=dist,
            edge_attr=edge_attr,
            size=None,
        )
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
                [x_i, x_j, torch.pow(dist, 2).unsqueeze(-1), edge_attr], dim=-1
            )
        edge_new = self.edge_func(edge_new)

        # cutoff net
        if self.cutoff_net is not None:
            edge_new = self.cutoff_net(dist).unsqueeze(-1) * edge_new
        # get attention weight
        edge_new = self.atten(edge_new) * edge_new

        return edge_new
