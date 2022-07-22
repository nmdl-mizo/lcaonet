import torch
from torch import Tensor
import torch.nn as nn


__all__ = ["ScaleShift", "Standarize"]


class ScaleShift(nn.Module):
    r"""
    The block to scale and shift input tensor.
    .. math::
       out = x \times \sigma + \mu

    Args:
        mean (torch.Tensor): mean value :math:`\mu`.
        stddev (torch.Tensor): standard deviation value :math:`\sigma`.

    Notes:
        reference:
        [1] K. T. Schütt et al., J. Chem. Theory Comput. 15, 448-455 (2019).
    """

    def __init__(self, mean: Tensor, stddev: Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)

    def forward(self, x: Tensor) -> Tensor:
        """Compute layer output.
        Args:
            x (torch.Tensor): input data.

        Returns:
            torch.Tensor: layer output.
        """
        out = x * self.stddev + self.mean
        return out


class Standarize(nn.Module):
    r"""
    The block to standardize input tensor.
    .. math::
       out = \frac{x - \mu}{\sigma}

    Args:
        mean (torch.Tensor): mean value :math:`\mu`.
        stddev (torch.Tensor): standard deviation value :math:`\sigma`.
        eps (float): epsilon.

    Notes:
        reference:
        [1] K. T. Schütt et al., J. Chem. Theory Comput. 15, 448–455 (2019).
    """

    def __init__(self, mean: Tensor, stddev: Tensor, eps: float = 1e-6):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)
        self.register_buffer("eps", torch.ones_like(stddev) * eps)

    def forward(self, x: Tensor) -> Tensor:
        """Compute layer output.
        Args:
            x (torch.Tensor): input data.

        Returns:
            torch.Tensor: layer output.
        """
        out = (x - self.mean) / (self.stddev + self.eps)
        return out
