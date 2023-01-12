import torch
import torch.nn as nn
from torch import Tensor


class BaseScaler(nn.Module):
    def __init__(self, mean: Tensor, stddev: Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)


class ShiftScaler(BaseScaler):
    r"""
    The block to scale and shift input tensor.

    .. math::
       out = x \times \sigma + \mu

    Args:
        mean (torch.Tensor): mean value :math:`\mu`.
        stddev (torch.Tensor): standard deviation value :math:`\sigma`.

    Notes:
        ref:
            [1] K. T. Schütt et al., J. Chem. Theory Comput. 15, 448-455 (2019).
    """

    def __init__(self, mean: Tensor, stddev: Tensor):
        super().__init__(mean=mean, stddev=stddev)

    def forward(self, x: Tensor) -> Tensor:
        """Compute layer output.

        Args:
            x (torch.Tensor): input data.

        Returns:
            torch.Tensor: layer output.
        """
        out = x * self.stddev + self.mean
        return out


class StandarizeScaler(BaseScaler):
    r"""
    The block to standardize input tensor.

    .. math::
       out = \frac{x - \mu}{\sigma}

    Args:
        mean (torch.Tensor): mean value :math:`\mu`.
        stddev (torch.Tensor): standard deviation value :math:`\sigma`.
        eps (float, optional): small offset value to avoid zero division. Defaults to `1e-8`.

    Notes:
        ref:
            [1] K. T. Schütt et al., J. Chem. Theory Comput. 15, 448-455 (2019).
    """

    def __init__(self, mean: Tensor, stddev: Tensor, eps: float = 1e-8):
        super().__init__(mean=mean, stddev=stddev)
        self.register_buffer("eps", torch.ones_like(stddev) * eps)

    def forward(self, x: Tensor) -> Tensor:
        """Compute layer output.
        Args:
            x (torch.Tensor): input data.

        Returns:
            torch.Tensor: layer output.
        """
        out = (x - self.mean) / (self.stddev + self.eps)  # type: ignore
        return out
