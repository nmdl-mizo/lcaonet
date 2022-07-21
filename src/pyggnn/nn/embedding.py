from typing import Optional
import math

from torch import Tensor
import torch.nn as nn

__all__ = ["AtomicNum2Node"]


class AtomicNum2Node(nn.Embedding):
    """
    The block to calculate initial node embeddings.
    Convert atomic numbers to a vector of arbitrary dimension.
    """

    def __init__(
        self,
        embed_dim: int,
        max_num: Optional[int] = None,
    ):
        """
        Args:
            embed_dim (int): number of embedding dim.
            max_num (int, optional): number of max value of atomic number.
                if set to`None`, `max_num=100`. Defaults to `None`.
        """
        if max_num is None:
            max_num = 100
        super().__init__(num_embeddings=max_num, embedding_dim=embed_dim)
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
            Tensor: embedding nodes of (num_node x embed_dim) shape.
        """
        x = super().forward(x)
        return x
