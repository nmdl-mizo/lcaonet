from typing import Callable, Optional

import torch
from torch import Tensor
import torch.nn as nn


__all__ = ["Dense"]


class Dense(nn.Linear):
    """
    Applies a linear transformation to the incoming data, and if activation is not None,
    apply activation function after linear transformation. And using weight initialize
    method.
    """

    def __init__(
        self,
        in_d: int,
        out_d: int,
        bias: bool = True,
        activation: Optional[Callable[[Tensor], Tensor]] = None,
        activation_name: Optional[str] = None,
        weight_init: Callable[[Tensor], Tensor] = nn.init.xavier_normal_,
        bias_init: Callable[[Tensor], Tensor] = nn.init.zeros_,
    ):
        """
        Args:
            in_d (int): input dimension of tensor.
            out_d (int): output dimension of tensor.
            bias (bool, optional): if `False`, the layer will not return an
                additive bias. Defaults to `True`.
            activation (Callable or None, optional): activation fucnction after
                calculating linear layer. Defaults to `None`.
            activation_name (str or `None`, optional): lower case of activation fucntion
                name. Defaults to `None`.
            weight_init (Callable, optional: Defaults to `nn.init.xavier_normal_`.
            bias_init (Callable, optional): Defaults to `nn.init.zeros_`.
        """
        self.activation = activation
        if self.activation is not None:
            assert activation_name is not None, "Please set 'activation_name'."
        if activation_name is None:
            self.activation_name = "linear"
        else:
            self.activation_name = activation_name
        self.weight_init = weight_init
        self.bias_init = bias_init
        super().__init__(in_d, out_d, bias)

    def reset_parameters(self):
        self.weight_init(
            self.weight, gain=torch.nn.init.calculate_gain(self.activation_name)
        )
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
        y = super().forward(x)
        # add activation function
        if self.activation:
            y = self.activation(y)
        return y
