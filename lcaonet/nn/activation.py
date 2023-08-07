import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class Swish(nn.Module):
    """Activation function of Swish which reported on ref[1].

    Args:
        beta (float, optional): Coefficent of beta value. Defaults to `1.0`.
        train_beta (bool, optional): if set to `False`, beta is not learnable. Defaults to `True`.

    Notes:
        ref:
            [1] P. Ramachandran et al., arXiv [cs.NE] (2017), (available at http://arxiv.org/abs/1710.05941).
            [2] V. G. Satorras et al., arXiv [cs.LG] (2021), (available at http://arxiv.org/abs/2102.09844).
    """

    def __init__(self, beta: float = 1.0, train_beta: bool = True):
        super().__init__()
        self.train_beta = train_beta
        if train_beta:
            self.beta = nn.Parameter(torch.tensor(float(beta)))
        else:
            self.register_buffer("beta", torch.tensor(float(beta)))

    def extra_repr(self) -> str:
        return f"beta={self.beta.item():.2f}, trainable_beta:{self.train_beta}"

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(self.beta * x)


class ShiftedSoftplus(nn.Module):
    r"""Compute shifted soft-plus activation function.

    .. math::
       y = \ln\left(1 + e^{-x}\right) - \ln(2)

    Args:
        x (torch.Tensor): input tensor.

    Returns:
        torch.Tensor: shifted soft-plus of input.

    Notes:
        ref:
            [1] K. T. Schütt et al., J. Chem. Theory Comput. 15, 448–455 (2019).
    """

    def __init__(self, shift: float = float(np.log(2.0))):
        super().__init__()
        self.shift = shift

    def extra_repr(self) -> str:
        return f"shift={self.shift:.2f}"

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.softplus(x) - self.shift
