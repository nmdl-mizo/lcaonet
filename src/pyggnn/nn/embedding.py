from typing import Optional

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

    def forward(self, x: Tensor) -> Tensor:
        """
        Computed the initial node embedding.

        Args:
            x (Tensor): atomic numbers shape of (num_nodes).

        Returns:
            Tensor: embedding nodes of (num_nodes x embed_dim) shape.
        """
        x = super().forward(x)
        return x
