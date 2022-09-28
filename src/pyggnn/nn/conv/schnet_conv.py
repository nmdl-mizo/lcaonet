from __future__ import annotations

from collections.abc import Callable

from torch import Tensor
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj

from pyggnn.nn.activation import ShiftedSoftplus
from pyggnn.nn.base import Dense


__all__ = ["SchNetConv"]


class SchNetConv(MessagePassing):
    def __init__(
        self,
        x_dim: int | tuple[int, int],
        edge_filter_dim: int,
        n_gaussian: int,
        activation: Callable[[Tensor], Tensor] = ShiftedSoftplus(),
        node_hidden: int = 256,
        cutoff_net: nn.Module | None = None,
        aggr: str = "add",
        weight_init: Callable[[Tensor], Tensor] = nn.init.xavier_uniform_,
        **kwargs,
    ):
        assert aggr == "add" or aggr == "mean"
        # TODO: pass kwargs to superclass
        super().__init__(aggr=aggr)

        # name node_dim is already used in super class
        self.x_dim = x_dim
        self.edge_filter_dim = edge_filter_dim
        self.node_hidden = node_hidden
        self.n_gaussian = n_gaussian
        if cutoff_net is not None:
            self.cutoff_net = cutoff_net
        else:
            self.cutoff_net = None

        # update functions
        # filter generator
        self.edge_filter_func = nn.Sequential(
            Dense(n_gaussian, edge_filter_dim, bias=True, weight_init=weight_init, **kwargs),
            activation,
            Dense(
                edge_filter_dim,
                edge_filter_dim,
                bias=True,
                weight_init=weight_init,
                **kwargs,
            ),
            activation,
        )
        # node fucntions
        self.node_lin1 = Dense(x_dim, edge_filter_dim, bias=True, weight_init=weight_init, **kwargs)
        self.node_lin2 = nn.Sequential(
            Dense(edge_filter_dim, x_dim, bias=True, weight_init=weight_init, **kwargs),
            activation,
            Dense(x_dim, x_dim, bias=True, weight_init=weight_init, **kwargs),
        )

    def forward(
        self,
        x: Tensor,
        dist: Tensor,
        edge_basis: Tensor,
        edge_index: Adj,
    ) -> Tensor:
        # calc edge filter and cutoff
        W = self.edge_filter_func(edge_basis)
        if self.cutoff_net is not None:
            C = self.cutoff_net(dist)
            W = W * C.view(-1, 1)
        # propagate_type:
        # (x: Tensor, W: Tensor)
        out = self.node_lin1(x)
        out = self.propagate(edge_index, x=out, W=W, size=None)
        out = self.node_lin2(out)
        # residual network
        out = out + x
        return out

    def message(self, x_j: Tensor, W: Tensor) -> Tensor:
        return x_j * W
