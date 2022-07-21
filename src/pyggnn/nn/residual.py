from typing import Union, Any

from torch import Tensor
import torch.nn as nn

from pyggnn.utils.resolve import activation_resolver
from pyggnn.nn.base import Dense

__all__ = ["ResidualBlock"]


class ResidualBlock(nn.Module):
    """ResidualBlock"""

    def __init__(
        self,
        hidden_dim: int,
        activation: Union[Any, str],
        n_layers: int = 2,
        last_activation: bool = True,
        **kwargs,
    ):
        """
        Blocks that combine feed-forward and residential networks.

        Args:
            hidden_dim (int): hidden dimension of the feed-forward network.
            activation (str): activation function of the feed-forward network.
            n_layers (int, optional): the number of feed-forward layers. Defaults to `2`.
            last_activation (bool, optional): Defaults to `True`.
        """
        super().__init__()
        act = activation_resolver(activation, **kwargs)

        lins = []
        for _ in range(n_layers - 1):
            lins.append(
                Dense(
                    hidden_dim,
                    hidden_dim,
                    bias=True,
                    activation_name=activation,
                    **kwargs,
                )
            )
            lins.append(act)
        if last_activation:
            lins.append(
                Dense(
                    hidden_dim,
                    hidden_dim,
                    bias=True,
                    activation_name=activation,
                    **kwargs,
                )
            )
            lins.append(act)
        else:
            lins.append(Dense(hidden_dim, hidden_dim, bias=True))
        self.lins = nn.Sequential(*lins)

        self.reset_parameters

    def reset_parameters(self):
        for ll in self.lins:
            if hasattr(ll, "reset_parameters"):
                ll.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward caclulation of the residual block.

        Args:
            x (Tensor): input tensor shape of (* x hidden_dim).

        Returns:
            Tensor: output tensor shape of (* x hidden_dim).
        """
        return x + self.lins(x)
