import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor


__all__ = ["Swish", "swish", "ShiftedSoftplus", "shifted_softplus"]


def swish(x: Tensor, beta: Tensor = torch.tensor(1.0)) -> Tensor:
    """
    Compute Swish activation function.

    Args:
        x (torch.Tensor): inpu tensor.
        beta (torch.Tensor, optional): beta coefficient of swish activation.
            Defaults to torch.tensor(1.0).

    Returns:
        Tensor: output tensor.
    """
    return x * torch.sigmoid(beta * x)


class Swish(nn.Module):
    """
    Activation function of Swish which reported on ref[1].

    Args:
        beta (float, optional): Coefficent of beta value. Defaults to `1.0`.
        train_beta (bool, optional): if set to `False`, beta is not learnable. Defaults to `True`.

    Notes:
        reference:
        [1] P. Ramachandran et al., arXiv [cs.NE] (2017),
            (available at http://arxiv.org/abs/1710.05941).
        [2] V. G. Satorras et al., arXiv [cs.LG] (2021),
            (available at http://arxiv.org/abs/2102.09844).
    """

    def __init__(self, beta: float = 1.0, train_beta: bool = True):
        super().__init__()
        if train_beta:
            self.beta_coeff = nn.Parameter(torch.tensor(float(beta)))
        else:
            self.register_buffer("beta_coeff", torch.tensor(float(beta)))

    def forward(self, x: Tensor) -> Tensor:
        return swish(x, self.beta_coeff)


def shifted_softplus(x: Tensor) -> Tensor:
    r"""Compute shifted soft-plus activation function.

    .. math::
       y = \ln\left(1 + e^{-x}\right) - \ln(2)

    Args:
        x (torch.Tensor): input tensor.

    Returns:
        torch.Tensor: shifted soft-plus of input.

    Notes:
        reference:
        [1] K. T. Schütt et al., J. Chem. Theory Comput. 15, 448–455 (2019).
    """
    return F.softplus(x) - np.log(2.0)


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return shifted_softplus(x)
