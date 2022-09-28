from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.nn.inits import glorot_orthogonal

from pyggnn.nn.activation import Swish
from pyggnn.nn.base import Dense


__all__ = ["EdgeEmbed"]


class EdgeEmbed(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        n_radial: int,
        activation: Callable[[Tensor], Tensor] = Swish(beta=1.0),
        weight_init: Callable[[Tensor], Tensor] = glorot_orthogonal,
        **kwargs,
    ):
        super().__init__()

        self.rbf_lin = Dense(
            n_radial,
            edge_dim,
            bias=False,
            weight_init=weight_init,
            **kwargs,
        )
        self.edge_embed = nn.Sequential(
            Dense(
                2 * node_dim + edge_dim,
                edge_dim,
                bias=True,
                weight_init=weight_init,
                **kwargs,
            ),
            activation,
        )

    def forward(
        self,
        x: Tensor,
        rbf: Tensor,
        idx_i: torch.LongTensor,
        idx_j: torch.LongTensor,
    ) -> Tensor:
        """
        Computed the initial edge embedding.

        Args:
            x (Tensor): node embedding vector shape of (n_node x node_dim).
            rbf (Tensor): radial basis function shape of (n_edge x n_radial).
            idx_i (LongTensor): index of the first node of the edge shape of (n_edge).
            idx_j (LongTensor): index of the second node of the edge shape of (n_edge).

        Returns:
            Tensor: embedding edge message shape of (n_edge x edge_dim).
        """
        rbf = self.rbf_lin(rbf)
        return self.edge_embed(torch.cat([x[idx_j], x[idx_i], rbf], dim=-1))
