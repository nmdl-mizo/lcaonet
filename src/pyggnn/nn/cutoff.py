import numpy as np
import torch
from torch import Tensor
import torch.nn as nn


class BaseCutoff(nn.Module):
    """BaseCutoff Network"""

    def __init__(self, cutoff_radi: float):
        """
        Args:
            cutoff_radi (float): cutoff radious.
        """
        super().__init__()
        self.cutoff_radi = cutoff_radi

    def forward(self, dist: Tensor) -> Tensor:
        """
        forward calculation of CosineNetwork.

        Args:
            dist (Tensor): inter atomic distances shape of (num_edge)

        Returns:
            Tensor: Cutoff values shape of (num_edge)
        """
        pass


class CosineCutoff(BaseCutoff):
    """CosineCutoff Network"""

    def __init__(self, cutoff_radi: float):
        """
        Args:
            cutoff_radi (float): cutoff radious.
        """
        super().__init__(cutoff_radi)

    def forward(self, dist: Tensor) -> Tensor:
        # Compute values of cutoff function
        cutoffs = 0.5 * (torch.cos(dist * np.pi / self.cutoff_radi) + 1.0)
        # Remove contributions beyond the cutoff radius
        cutoffs *= (dist < self.cutoff_radi).float()
        return cutoffs
