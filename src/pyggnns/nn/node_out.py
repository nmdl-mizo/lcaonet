from __future__ import annotations  # type: ignore

from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.inits import glorot_orthogonal
from torch_scatter import scatter

from pyggnns.nn.activation import ShiftedSoftplus, Swish
from pyggnns.nn.base import Dense

__all__ = ["Node2Prop1", "Node2Prop2"]


class Node2Prop1(nn.Module):
    """The block to compute the global graph proptery from node embeddings. In
    this block, after aggregation, two more Dense layers are calculated. This
    block is used in EGNN.

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


class Node2Prop2(nn.Module):
    """The block to compute the global graph proptery from node embeddings.
    This block contains two Dense layers and aggregation block. If set
    `scaler`, scaling process before aggregation. This block is used in SchNet.

    Args:
        node_dim (int): number of input dim.
        hidden_dim (int, optional): number of hidden layers dim. Defaults to `128`.
        out_dim (int, optional): number of output dim. Defaults to `1`.
        activation: (Callable, optional): activation function. Defaults to `ShiftedSoftplus()`.
        aggr (`"add"` or `"mean"`): aggregation method. Defaults to `"add"`.
        scaler: (nn.Module, optional): scaler layer. Defaults to `None`.
        mean: (Tensor, optional): mean of the input tensor. Defaults to `None`.
        stddev: (Tensor, optional): stddev of the input tensor. Defaults to `None`.
        weight_init (Callable, optional): weight initialization function. Defaults to `torch.nn.init.xavier_uniform_`.
    """

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 1,
        activation: Callable[[Tensor], Tensor] = ShiftedSoftplus(),
        aggr: str = "add",
        scaler: nn.Module | None = None,
        mean: Tensor | None = None,
        stddev: Tensor | None = None,
        weight_init: Callable[[Tensor], Tensor] = torch.nn.init.xavier_uniform_,
        **kwargs,
    ):
        super().__init__()

        assert aggr == "add" or aggr == "mean"
        self.aggr = aggr
        self.output = nn.Sequential(
            Dense(node_dim, hidden_dim, bias=True, weight_init=weight_init, **kwargs),
            activation,
            Dense(hidden_dim, out_dim, bias=False, weight_init=weight_init, **kwargs),
        )
        if scaler is None:
            self.scaler = None
        else:
            if mean is None:
                mean = torch.FloatTensor([0.0])
            if stddev is None:
                stddev = torch.FloatTensor([1.0])
            self.scaler = scaler(mean, stddev)

    def forward(self, x: Tensor, batch: Tensor | None = None) -> Tensor:
        """Compute global property from node embeddings.

        Args:
            x (Tensor): node embeddings shape of (n_node x in_dim).
            batch (Tensor, optional): batch index shape of (n_node). Defaults to `None`.

        Returns:
            Tensor: shape of (n_batch x out_dim).
        """
        out = self.output(x)
        if self.scaler is not None:
            out = self.scaler(out)
        return scatter(out, index=batch, dim=0, reduce=self.aggr)
