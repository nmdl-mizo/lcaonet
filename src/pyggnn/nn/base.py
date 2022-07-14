from typing import Callable, Optional, Union, Any

import torch
from torch import Tensor
import torch.nn as nn

from pyggnn.utils.resolve import activation_name_resolver


__all__ = ["Dense"]

GAIN_DICT = {
    "sigmoid": "sigmoid",
    "tanh": "tanh",
    "relu": "relu",
    "selu": "selu",
    "leakyrelu": "leaky_relu",
    "swish": "sigmoid",
}


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
            activation_name (str or `None`, optional): activation fucntion class
                or activation fucntion name. Defaults to `None`.
            weight_init (Callable, optional: Defaults to `nn.init.xavier_normal_`.
            bias_init (Callable, optional): Defaults to `nn.init.zeros_`.
        """
        self.activation_name = (
            "linear"
            if activation_name is None
            else GAIN_DICT[activation_name_resolver(activation_name, **kwargs)]
        )
        if self.activation_name == "leaky_relu":
            self.negative_slope = kwargs["negative_slope"]
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
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: Calculated tensor.
        """
        # compute linear layer y = xW^T + b
        return super().forward(x)
