from typing import Tuple, Union, Optional, Any

from torch import Tensor
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj

from pyggnn.nn.base import Dense
from pyggnn.utils.resolve import activation_resolver


__all__ = ["SchNetConv"]


class SchNetConv(MessagePassing):
    def __init__(
        self,
        x_dim: Union[int, Tuple[int, int]],
        edge_dim: int,
        n_gaussian: int,
        activation: Union[Any, str] = "shiftedsoftplus",
        node_hidden: int = 256,
        cutoff_net: Optional[nn.Module] = None,
        cutoff_radi: float = 4.0,
        residual: bool = True,
        aggr: str = "add",
        **kwargs,
    ):
        # TODO: pass kwargs to superclass
        super().__init__(aggr=aggr)
        act = activation_resolver(activation, **kwargs)

        self.x_dim = x_dim
        self.edge_dim = edge_dim
        self.node_hidden = node_hidden
        self.n_gaussian = n_gaussian
        if cutoff_net is not None:
            self.cutoff_net = cutoff_net(cutoff_radi)
        else:
            self.cutoff_net = None
        self.cutoff_radi = cutoff_radi
        self.residual = residual

        # update functions
        # filter generator
        self.edge_filter_func = nn.Sequential(
            Dense(n_gaussian, edge_dim, bias=True, activation_name=activation, **kwargs),
            act,
            Dense(edge_dim, edge_dim, bias=True, activation_name=activation, **kwargs),
            act,
        )
        # node fucntions
        self.node_lin1 = Dense(x_dim, edge_dim, bias=True)
        self.node_lin2 = nn.Sequential(
            Dense(edge_dim, x_dim, bias=True, activation_name=activation, **kwargs),
            act,
            Dense(x_dim, x_dim, bias=True),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for ef in self.edge_filter_func:
            if hasattr(ef, "reset_parameters"):
                ef.reset_parameters()
        self.node_lin1.reset_parameters()
        for nf in self.node_lin2:
            if hasattr(nf, "reset_parameters"):
                nf.reset_parameters()

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
        out = self.propagate(
            edge_index,
            x=out,
            W=W,
            size=None,
        )
        out = self.node_lin2(out)
        out = out + x if self.residual else out
        return out

    def message(
        self,
        x_j: Tensor,
        W: Tensor,
    ) -> Tensor:
        return x_j * W
