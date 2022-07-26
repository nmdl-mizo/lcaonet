from typing import Callable, Any

from torch import Tensor
import torch.nn as nn
from torch_geometric.nn.inits import glorot_orthogonal

from pyggnn.utils.resolve import init_param_resolver


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
        weight_init: Callable[[Tensor], Any] = nn.init.xavier_uniform_,
        bias_init: Callable[[Tensor], Tensor] = nn.init.zeros_,
        **kwargs,
    ):
        """
        Args:
            in_dim (int): input dimension of tensor.
            out_dim (int): output dimension of tensor.
            bias (bool, optional): if `False`, the layer will not return an additive bias. Defaults to `True`.
            weight_init (Callable, optional): weight initialize methods. Defaults to `nn.init.xavier_uniform_`.
            bias_init (Callable, optional): bias initialize methods. Defaults to `nn.init.zeros_`.
        """
        if bias:
            assert bias_init is not None, "bias_init must not be None if set bias"
        self.bias_init = bias_init
        self.weight_init = weight_init
        # gain and scale paramer is set to default values
        params = init_param_resolver(weight_init)
        for p in params:
            if p not in kwargs:
                if p == "gain":
                    kwargs[p] = 1.0
                elif p == "scale":
                    kwargs[p] = 2.0
        self.kwargs = kwargs

        super().__init__(in_dim, out_dim, bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.weight_init(self.weight, **self.kwargs)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward calculation of the Dense layer.

        Args:
            x (torch.Tensor): input tensor shape of (* x in_dim).

        Returns:
            torch.Tensor: output tensor shape of (* x out_dim).
        """
        # compute linear layer y = xW^T + b
        return super().forward(x)


class ResidualBlock(nn.Module):
    """
    The Blocks combining the multiple Dense layers and ResNet.
    """

    def __init__(
        self,
        hidden_dim: int,
        activation: Callable[[Tensor], Tensor] = nn.ReLU(),
        n_layers: int = 2,
        weight_init: Callable[[Tensor], Any] = glorot_orthogonal,
        **kwargs,
    ):
        """
        Args:
            hidden_dim (int): hidden dimension of the Dense layers.
            activation (str): activation function of the Dense layers.
            n_layers (int, optional): the number of Dense layers. Defaults to `2`.
            weight_init (Callable, optional): weight initialize methods.
                Defaults to `torch_geometric.nn.inits.glorot_orthogonal`.
        """
        super().__init__()

        denses = []
        for _ in range(n_layers):
            denses.append(
                Dense(
                    hidden_dim,
                    hidden_dim,
                    bias=True,
                    weight_init=weight_init,
                    **kwargs,
                )
            )
            denses.append(activation)
        self.denses = nn.Sequential(*denses)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward caclulation of the residual block.

        Args:
            x (Tensor): input tensor shape of (* x hidden_dim).

        Returns:
            Tensor: output tensor shape of (* x hidden_dim).
        """
        return x + self.denses(x)
