from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


__all__ = ["Swish", "swish"]


def swish(x: Tensor, beta: Tensor = torch.tensor(1.0)) -> Tensor:
    return x * torch.sigmoid(beta * x)


class Swish(nn.Module):
    """
    Activation function of Swish which reported on ref[1].

    Notes:
        reference:
        [1] P. Ramachandran et al., arXiv [cs.NE] (2017),
            (available at http://arxiv.org/abs/1710.05941).
        [2] V. G. Satorras et al., arXiv [cs.LG] (2021),
            (available at http://arxiv.org/abs/2102.09844).
    """

    def __init__(self, beta: Optional[float] = 1.0):
        """
        Activation function of Swish which reported on ref[1].

        Args:
            beta (float, optional): Coefficent of beta value. Defaults to `None`.
        """
        super().__init__()
        self.beta = beta
        if beta is not None:
            self.beta_coeff = nn.Parameter(torch.tensor(beta))
        else:
            self.register_buffer("beta_coeff", torch.tensor(1.0))
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, x: Tensor) -> Tensor:
        return swish(x, self.beta_coeff)
