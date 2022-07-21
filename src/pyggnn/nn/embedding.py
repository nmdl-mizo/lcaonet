from typing import Optional, Union, Any
import math

import torch
from torch import Tensor
import torch.nn as nn

from pyggnn.nn.base import Dense
from pyggnn.utils.resolve import activation_resolver

__all__ = ["AtomicNum2NodeEmbed", "EdgeEmbed"]


class AtomicNum2NodeEmbed(nn.Embedding):
    """
    The block to calculate initial node embeddings.
    Convert atomic numbers to a vector of arbitrary dimension.
    """

    def __init__(
        self,
        embed_node_dim: int,
        max_num: Optional[int] = None,
    ):
        """
        Args:
            embed_node_dim (int): number of embedding dim.
            max_num (int, optional): number of max value of atomic number.
                if set to`None`, `max_num=100`. Defaults to `None`.
        """
        if max_num is None:
            max_num = 100
        super().__init__(num_embeddings=max_num, embedding_dim=embed_node_dim)
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        # ref:
        # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/dimenet.html#DimeNetPlusPlus
        self.weight.data.uniform_(-math.sqrt(3), math.sqrt(3))

    def forward(self, x: Tensor) -> Tensor:
        """
        Computed the initial node embedding.

        Args:
            x (Tensor): atomic numbers shape of (num_node).

        Returns:
            Tensor: embedding nodes of (num_node x embed_node_dim) shape.
        """
        x = super().forward(x)
        return x


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

        self.node_embed = AtomicNum2NodeEmbed(embed_dim=node_dim, max_num=max_z)
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
        self.rbf_lin.reset_parameters()
        for ee in self.edge_embed:
            if hasattr(ee, "reset_parameters"):
                ee.reset_parameters()

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
            x (Tensor): atomic numbers shape of (num_node).
            rbf (Tensor): radial basis function shape of (num_edge x n_radial).
            idx_i (LongTensor): index of the first node of the edge shape of (num_edge).
            idx_j (LongTensor): index of the second node of the edge shape of (num_edge).

        Returns:
            Tensor: embedding edge message of (num_edge x edge_dim) shape.
        """
        x = self.node_embed(x)
        rbf = self.rbf_lin(rbf)
        return self.edge_embed(torch.cat([x[idx_j], x[idx_i], rbf], dim=-1))
