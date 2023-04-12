from __future__ import annotations

from math import pi

import torch
import torch.nn as nn
from torch import Tensor


class BaseCutoff(nn.Module):
    """Layer to calculate radial cutoff function values."""

    def __init__(self, cutoff: float):
        """
        Args:
            cutoff (float): the cutoff radius.
        """
        super().__init__()
        self.cutoff = cutoff

    def extra_repr(self) -> str:
        return f"cutoff={self.cutoff}"

    def forward(self, r: Tensor) -> Tensor:
        raise NotImplementedError


class PolynomialCutoff(BaseCutoff):
    def __init__(self, cutoff: float):
        super().__init__(cutoff)

    def forward(self, r: Tensor) -> Tensor:
        ratio = r / self.cutoff
        # Remove contributions beyond the cutoff radius
        return torch.where(r <= self.cutoff, 1 - 6 * ratio**5 + 15 * ratio**4 - 10 * ratio**3, 0.0)


class CosineCutoff(BaseCutoff):
    def __init__(self, cutoff: float):
        super().__init__(cutoff)

    def forward(self, r: Tensor) -> Tensor:
        cutoffs = 0.5 * (torch.cos(r * pi / self.cutoff) + 1.0)
        # Remove contributions beyond the cutoff radius
        return torch.where(r <= self.cutoff, cutoffs, 0.0)
