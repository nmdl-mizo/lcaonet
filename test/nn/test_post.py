from __future__ import annotations

import pytest
import torch
from torch_scatter import scatter

from lcaonet.nn.post import PostProcess

param_PostProcess = [
    (1, True, None, None),  # default
    (1, False, None, None),  # no extensive
    (1, True, None, torch.ones(1)),  # add mean
    (1, False, None, torch.ones(1)),  # add mean and no extensive
    (1, True, torch.ones((100, 1)), None),  # add atomref
    (1, False, torch.ones((100, 1)), None),  # add atomref and no extensive
    (1, True, torch.ones((100, 1)), torch.ones(1)),  # add atomref and mean
    (1, False, torch.ones((100, 1)), torch.ones(1)),  # add atomref and mean and no extensive
    (10, True, torch.ones((100, 1)), torch.ones(1)),  # add atomref and mean
    (10, False, torch.ones((100, 1)), torch.ones(1)),  # add atomref and mean and no extensive
]


@pytest.mark.parametrize("out_dim, is_extensive, atomref, mean", param_PostProcess)
def test_PostProcess(
    out_dim: int,
    is_extensive: bool,
    atomref: torch.Tensor | None,
    mean: torch.Tensor | None,
):
    n_batch, n_node = 10, 100
    out = torch.rand((n_batch, out_dim))
    zs = torch.randint(0, 100, (n_node,))
    batch_idx = torch.randint(0, n_batch, (n_node,))
    pp_layer = PostProcess(out_dim, is_extensive, atomref, mean)

    out_pp = pp_layer(out, zs, batch_idx)

    assert out_pp.size() == (n_batch, out_dim)

    expected = out
    reduce = "sum" if is_extensive else "mean"
    if atomref is not None:
        expected += scatter(atomref[zs], batch_idx, dim=0, dim_size=n_batch, reduce=reduce)
    if mean is not None:
        mean = mean.unsqueeze(0).expand(n_node, -1)
        mean = scatter(mean, batch_idx, dim=0, dim_size=n_batch, reduce=reduce)
        expected += mean
    assert torch.allclose(out_pp, expected)
