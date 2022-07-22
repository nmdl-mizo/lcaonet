from typing import Callable, Optional, Union, Any

import torch
from torch import Tensor
import torch.nn as nn

from pyggnn.utils.resolve import activation_gain_resolver, activation_resolver


__all__ = ["Dense", "ResidualBlock"]


class Dense(nn.Linear):
    """
    Applies a linear transformation to the incoming data,
    And using weight initialize method.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        activation_name: Optional[Union[Any, str]] = None,
        weight_init: Callable[[Tensor], Tensor] = nn.init.xavier_normal_,
        bias_init: Callable[[Tensor], Tensor] = nn.init.zeros_,
        **kwargs,
    ):
        """
        Args:
            in_dim (int): input dimension of tensor.
            out_dim (int): output dimension of tensor.
            bias (bool, optional): if `False`, the layer will not return an
                additive bias. Defaults to `True`.
            activation_name (str or nn.Module or `None`, optional): activation fucntion
                class or activation fucntion name. Defaults to `None`.
            weight_init (Callable, optional): Defaults to `nn.init.xavier_normal_`.
            bias_init (Callable, optional): Defaults to `nn.init.zeros_`.
        """
        self.activation_name = (
            "linear"
            if activation_name is None
            else activation_gain_resolver(activation_name)
        )
        if self.activation_name == "leaky_relu":
            # if not set `negative_slople`, set default values of torch.nn.LeakyReLU
            self.negative_slope = kwargs.get("negative_slope", 0.01)
        self.weight_init = weight_init
        if bias:
            assert bias_init is not None, "bias_init must not be None if set to bias"
        self.bias_init = bias_init
        super().__init__(in_dim, out_dim, bias)
        self.reset_parameters()

    def reset_parameters(self):
        if self.activation_name == "leaky_relu":
            gain = torch.nn.init.calculate_gain(
                self.activation_name, self.negative_slope
            )
        else:
            gain = torch.nn.init.calculate_gain(self.activation_name)
        self.weight_init(self.weight, gain=gain)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        forward calculation of Dense layer.

        Args:
            x (torch.Tensor): input tensor shape of (* x in_dim).

        Returns:
            torch.Tensor: output tensor shape of (* x out_dim).
        """
        # compute linear layer y = xW^T + b
        return super().forward(x)


class ResidualBlock(nn.Module):
    """
    The Blocks combining multiple Dense layers and ResNet.
    """

    def __init__(
        self,
        hidden_dim: int,
        activation: Union[Any, str],
        n_layers: int = 2,
        last_act: bool = True,
        **kwargs,
    ):
        """
        Args:
            hidden_dim (int): hidden dimension of the Dense layers.
            activation (str): activation function of the Dense layers.
            n_layers (int, optional): the number of Dense layers. Defaults to `2`.
            last_act (bool, optional): Defaults to `True`.
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
        if last_act:
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
