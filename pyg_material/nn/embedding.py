from __future__ import annotations  # type: ignore

import math

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.inits import glorot_orthogonal


class AtomicNum2Node(nn.Embedding):
    """The block to calculate initial node embeddings. Convert atomic numbers
    to a vector of arbitrary dimension.

    Args:
        node_dim (int): embedding node dimension.
        max_z (int, optional): number of max value of atomic number. if set to`None`, `max_z=100`. Defaults to `None`.
    """  # NOQA: E501

    def __init__(
        self,
        node_dim: int,
        max_z: int | None = None,
    ):
        if max_z is None:
            max_z = 100
        super().__init__(num_embeddings=max_z, embedding_dim=node_dim)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        # ref: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/dimenet.html # NOQA: E501
        self.weight.data.uniform_(-math.sqrt(3), math.sqrt(3))

    def forward(self, z: Tensor) -> Tensor:
        """Computed the initial node embedding.

        Args:
            z (Tensor): atomic numbers shape of (n_node).

        Returns:
            Tensor: embedding nodes shape of (n_node x node_dim).
        """
        return super().forward(z)


SPOOKYNET_DICT = torch.tensor(
    [
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
        [3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [4, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
        [5, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0],
        [6, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0],
        [7, 2, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0],
        [8, 2, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 0, 0],
        [9, 2, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 5, 0, 0],
        [10, 2, 2, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 6, 0, 0],
        [11, 2, 2, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [12, 2, 2, 6, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
        [13, 2, 2, 6, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0],
        [14, 2, 2, 6, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0],
        [15, 2, 2, 6, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0],
        [16, 2, 2, 6, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 0, 0],
        [17, 2, 2, 6, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 5, 0, 0],
        [18, 2, 2, 6, 2, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 6, 0, 0],
        [19, 2, 2, 6, 2, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [20, 2, 2, 6, 2, 6, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
        [21, 2, 2, 6, 2, 6, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0],
        [22, 2, 2, 6, 2, 6, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0],
        [23, 2, 2, 6, 2, 6, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 3, 0],
        [24, 2, 2, 6, 2, 6, 1, 5, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 5, 0],
        [25, 2, 2, 6, 2, 6, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 5, 0],
        [26, 2, 2, 6, 2, 6, 2, 6, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 6, 0],
        [27, 2, 2, 6, 2, 6, 2, 7, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 7, 0],
        [28, 2, 2, 6, 2, 6, 2, 8, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 8, 0],
        [29, 2, 2, 6, 2, 6, 1, 10, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 10, 0],
        [30, 2, 2, 6, 2, 6, 2, 10, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 10, 0],
        [31, 2, 2, 6, 2, 6, 2, 10, 1, 0, 0, 0, 0, 0, 0, 0, 2, 1, 10, 0],
        [32, 2, 2, 6, 2, 6, 2, 10, 2, 0, 0, 0, 0, 0, 0, 0, 2, 2, 10, 0],
        [33, 2, 2, 6, 2, 6, 2, 10, 3, 0, 0, 0, 0, 0, 0, 0, 2, 3, 10, 0],
        [34, 2, 2, 6, 2, 6, 2, 10, 4, 0, 0, 0, 0, 0, 0, 0, 2, 4, 10, 0],
        [35, 2, 2, 6, 2, 6, 2, 10, 5, 0, 0, 0, 0, 0, 0, 0, 2, 5, 10, 0],
        [36, 2, 2, 6, 2, 6, 2, 10, 6, 0, 0, 0, 0, 0, 0, 0, 2, 6, 10, 0],
        [37, 2, 2, 6, 2, 6, 2, 10, 6, 1, 0, 0, 0, 0, 0, 0, 1, 6, 10, 0],
        [38, 2, 2, 6, 2, 6, 2, 10, 6, 2, 0, 0, 0, 0, 0, 0, 2, 6, 10, 0],
        [39, 2, 2, 6, 2, 6, 2, 10, 6, 2, 1, 0, 0, 0, 0, 0, 2, 6, 1, 0],
        [40, 2, 2, 6, 2, 6, 2, 10, 6, 2, 2, 0, 0, 0, 0, 0, 2, 6, 2, 0],
        [41, 2, 2, 6, 2, 6, 2, 10, 6, 1, 4, 0, 0, 0, 0, 0, 1, 6, 4, 0],
        [42, 2, 2, 6, 2, 6, 2, 10, 6, 1, 5, 0, 0, 0, 0, 0, 1, 6, 5, 0],
        [43, 2, 2, 6, 2, 6, 2, 10, 6, 2, 5, 0, 0, 0, 0, 0, 2, 6, 5, 0],
        [44, 2, 2, 6, 2, 6, 2, 10, 6, 1, 7, 0, 0, 0, 0, 0, 1, 6, 7, 0],
        [45, 2, 2, 6, 2, 6, 2, 10, 6, 1, 8, 0, 0, 0, 0, 0, 1, 6, 8, 0],
        [46, 2, 2, 6, 2, 6, 2, 10, 6, 0, 10, 0, 0, 0, 0, 0, 0, 6, 10, 0],
        [47, 2, 2, 6, 2, 6, 2, 10, 6, 1, 10, 0, 0, 0, 0, 0, 1, 6, 10, 0],
        [48, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 0, 0, 0, 0, 0, 2, 6, 10, 0],
        [49, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 1, 0, 0, 0, 0, 2, 1, 10, 0],
        [50, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 2, 0, 0, 0, 0, 2, 2, 10, 0],
        [51, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 3, 0, 0, 0, 0, 2, 3, 10, 0],
        [52, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 4, 0, 0, 0, 0, 2, 4, 10, 0],
        [53, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 5, 0, 0, 0, 0, 2, 5, 10, 0],
        [54, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0, 0, 0, 0, 2, 6, 10, 0],
        [55, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 1, 0, 0, 0, 1, 6, 10, 0],
        [56, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 0, 0, 0, 2, 6, 10, 0],
        [86, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 6, 10, 14],
    ]
)


class AtomicDict2Node(nn.Module):
    """The block to calculate initial node embeddings. Convert atomic numbers
    to a vector of arbitrary dimension.

    Args:
        node_dim (int): embedding node dimension.
        max_z (int, optional): number of max value of atomic number. if set to`None`, `max_z=56`. Defaults to `None`.
    """  # NOQA: E501

    def __init__(
        self,
        node_dim: int,
        max_z: int | None = None,
    ):
        if max_z is None:
            max_z = 56
        else:
            assert max_z <= 56
        super().__init__()
        self.embed = nn.Embedding(max_z, node_dim)
        self.M = nn.Parameter(torch.Tensor(node_dim, 20))

        self.reset_parameters()

    def reset_parameters(self):
        self.embed.weight.data.uniform_(-math.sqrt(3), math.sqrt(3))
        glorot_orthogonal(self.M, scale=2.0)

    def forward(self, z: Tensor) -> Tensor:
        """Computed the initial node embedding.

        Args:
            z (Tensor): atomic numbers shape of (n_node).

        Returns:
            Tensor: embedding nodes shape of (n_node x node_dim).
        """
        device = z.device
        z = torch.einsum("fd, bd->bf", self.M, (SPOOKYNET_DICT[z] / SPOOKYNET_DICT[-1]).to(device)) + self.embed(z)
        return z
