from typing import Optional
import math

from torch import Tensor
import torch.nn as nn

__all__ = ["AtomicNum2NodeEmbed"]


class AtomicNum2NodeEmbed(nn.Embedding):
    """
    The block to calculate initial node embeddings.
    Convert atomic numbers to a vector of arbitrary dimension.
    """

    def __init__(
        self,
        node_dim: int,
        max_num: Optional[int] = None,
    ):
        """
        Args:
            node_dim (int): embedding node dimension.
            max_num (int, optional): number of max value of atomic number.
                if set to`None`, `max_num=100`. Defaults to `None`.
        """
        if max_num is None:
            max_num = 100
        super().__init__(num_embeddings=max_num, embedding_dim=node_dim)
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        # ref:
        # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/dimenet.html
        self.weight.data.uniform_(-math.sqrt(3), math.sqrt(3))

    def forward(self, z: Tensor) -> Tensor:
        """
        Computed the initial node embedding.

        Args:
            z (Tensor): atomic numbers shape of (n_node).

        Returns:
            Tensor: embedding nodes shape of (n_node x node_dim).
        """
        z = super().forward(z)
        return z
