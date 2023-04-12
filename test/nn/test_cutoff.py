from __future__ import annotations

import pytest
import torch

from lcaonet.nn.cutoff import CosineCutoff, PolynomialCutoff

param_PolynomialCutoff = [
    (1.0),
    (2.0),
    (3.0),
]


@pytest.mark.parametrize("cutoff", param_PolynomialCutoff)
def test_PolynomialCutoff(cutoff: float):
    r = torch.linspace(0.0, cutoff + 2, 100)
    cn = PolynomialCutoff(cutoff)
    cutoff_vals = cn(r)

    assert cutoff_vals.size() == r.size()
    assert cutoff_vals[r > cutoff].sum() == 0.0


param_CosineCutoff = [
    (1.0),
    (2.0),
    (3.0),
]


@pytest.mark.parametrize("cutoff", param_CosineCutoff)
def test_CosineCutoff(cutoff: float):
    r = torch.linspace(0.0, cutoff + 2, 100)
    cn = CosineCutoff(cutoff)
    cutoff_vals = cn(r)

    assert cutoff_vals.size() == r.size()
    assert cutoff_vals[r > cutoff].sum() == 0.0
