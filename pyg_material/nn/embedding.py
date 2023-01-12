from __future__ import annotations  # type: ignore

import math

import torch.nn as nn
from torch import Tensor


class AtomicNum2Node(nn.Embedding):
    """The block to calculate initial node embeddings.

    This layer converts atomic numbers to a node embedding vector of arbitrary dimension.

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
            z (Tensor): atomic numbers of (n_node) shape.

        Returns:
            Tensor: embedding nodes of (n_node, node_dim) shape.
        """
        return super().forward(z)
