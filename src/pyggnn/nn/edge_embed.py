from typing import Optional, Union, Any

import torch
from torch import Tensor
import torch.nn as nn

from pyggnn.nn.base import Dense
from pyggnn.nn.node_embed import AtomicNum2NodeEmbed
from pyggnn.utils.resolve import activation_resolver

__all__ = ["EdgeEmbed"]


class EdgeEmbed(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        n_radial: int,
        activation: Union[Any, str] = "swish",
        max_z: Optional[int] = 100,
        **kwargs,
    ):
        super().__init__()
        act = activation_resolver(activation, **kwargs)

        self.node_embed = AtomicNum2NodeEmbed(node_dim, max_num=max_z)
        self.rbf_lin = Dense(in_dim=n_radial, out_dim=edge_dim, bias=False)
        self.edge_embed = nn.Sequential(
            Dense(
                in_dim=2 * node_dim + edge_dim,
                out_dim=edge_dim,
                bias=True,
                activation_name=activation,
                **kwargs,
            ),
            act,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.node_embed.reset_parameters()
        self.rbf_lin.reset_parameters()
        for ee in self.edge_embed:
            if hasattr(ee, "reset_parameters"):
                ee.reset_parameters()

    def forward(
        self,
        z: Tensor,
        rbf: Tensor,
        idx_i: torch.LongTensor,
        idx_j: torch.LongTensor,
    ) -> Tensor:
        """
        Computed the initial edge embedding.

        Args:
            z (Tensor): atomic numbers shape of (n_node).
            rbf (Tensor): radial basis function shape of (n_edge x n_radial).
            idx_i (LongTensor): index of the first node of the edge shape of (n_edge).
            idx_j (LongTensor): index of the second node of the edge shape of (n_edge).

        Returns:
            Tensor: embedding edge message shape of (n_edge x edge_dim).
        """
        z = self.node_embed(z)
        rbf = self.rbf_lin(rbf)
        return self.edge_embed(torch.cat([z[idx_j], z[idx_i], rbf], dim=-1))
