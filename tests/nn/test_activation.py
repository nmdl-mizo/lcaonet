from __future__ import annotations

import numpy as np
import pytest
import torch

from lcaonet.nn.activation import ShiftedSoftplus, Swish

param_Swish = [
    (1.0, True),
    (1.0, False),
    (0.1, True),
    (0.1, False),
    (0.01, True),
    (0.01, False),
    (10.0, True),
    (10.0, False),
    (0.0, True),
    (0.0, False),
]


@pytest.mark.parametrize("beta, train_beta", param_Swish)
def test_Swish(
    beta: float,
    train_beta: bool,
):
    input = torch.linspace(-10, 10, 100)
    act = Swish(beta, train_beta)
    out = act(input)

    assert out.size() == input.size()
    assert torch.allclose(act.beta, torch.tensor(beta))
    if train_beta:
        assert act.beta.requires_grad

    assert (out[input < 0] <= torch.zeros_like(out[input < 0])).all()
    assert (out[input == 0] == torch.zeros_like(out[input == 0])).all()
    assert (out[input >= 0] >= torch.zeros_like(out[input >= 0])).all()

    assert torch.allclose(out, input * torch.sigmoid(beta * input))


def test_ShiftedSoftPlus():
    input = torch.linspace(-10, 10, 100)
    act = ShiftedSoftplus()
    out = act(input)

    assert out.size() == input.size()

    assert torch.allclose(out, torch.nn.functional.softplus(input) - np.log(2))
