from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.inits import glorot, glorot_orthogonal

from lcaonet.nn.base import Dense

param_Dense = [
    # bias test
    (10, 10, True, None, nn.init.zeros_),
    (10, 10, True, None, nn.init.ones_),
    (10, 10, False, None, nn.init.zeros_),
    (10, 10, True, None, None),
    (10, 10, False, None, None),
    # weight init test
    (10, 10, True, glorot, nn.init.zeros_),
    (10, 10, True, glorot_orthogonal, nn.init.zeros_),
    (10, 10, True, nn.init.xavier_uniform_, nn.init.zeros_),
    # dimension test
    (100, 10, True, None, nn.init.zeros_),
    (10, 100, True, None, nn.init.zeros_),
    (10, 100, False, None, nn.init.zeros_),
]


@pytest.mark.parametrize("in_dim, out_dim, bias, weight_init, bias_init", param_Dense)
def test_Dense(
    in_dim: int,
    out_dim: int,
    bias: bool,
    weight_init: Callable[[Tensor], Tensor] | None,
    bias_init: Callable[[Tensor], Tensor] | None,
):
    if bias and bias_init is None:
        with pytest.raises(ValueError) as e:
            _ = Dense(in_dim, out_dim, bias, weight_init, bias_init)
        assert str(e.value) == "bias_init must not be None if set bias"
    else:
        dense = Dense(in_dim, out_dim, bias, weight_init, bias_init)

        assert isinstance(dense, nn.Linear)
        assert dense.in_features == in_dim
        assert dense.out_features == out_dim
        assert dense.bias is not None if bias else dense.bias is None
        assert dense.weight_init == weight_init
        assert dense.bias_init == bias_init

        input = torch.randn((1, in_dim))
        with torch.no_grad():
            out = dense(input)
            assert out.size() == (1, out_dim)

            jit = torch.jit.export(dense)
            assert torch.allclose(jit(input), out, rtol=1e-5, atol=1e-7)
