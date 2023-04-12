from __future__ import annotations

from math import pi as PI

import torch
import torch.nn as nn
from torch import Tensor


def _polynomial_cutoff(r: Tensor, cutoff: float) -> Tensor:
    """Polynomial cutoff function.

    Args:
        r (torch.Tensor): radius distance tensor.
        cutoff (float): cutoff distance.

    Returns:
        torch.Tensor: polynomial cutoff values.
    """
    ratio = r / cutoff
    # Remove contributions beyond the cutoff radius
    return torch.where(r <= cutoff, 1 - 6 * ratio**5 + 15 * ratio**4 - 10 * ratio**3, 0.0)


def _cosine_cutoff(r: Tensor, cutoff: float) -> Tensor:
    """Cosine cutoff function.

    Args:
        r (torch.Tensor): radius distance tensor.
        cutoff (float): cutoff distance.

    Returns:
        torch.Tensor: cosine cutoff values.
    """
    cutoffs = 0.5 * (torch.cos(r * PI / cutoff) + 1.0)
    # Remove contributions beyond the cutoff radius
    return torch.where(r <= cutoff, cutoffs, 0.0)


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
        return _polynomial_cutoff(r, self.cutoff)


class CosineCutoff(BaseCutoff):
    def __init__(self, cutoff: float):
        super().__init__(cutoff)

    def forward(self, r: Tensor) -> Tensor:
        return _cosine_cutoff(r, self.cutoff)
