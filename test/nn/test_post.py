from __future__ import annotations

import pytest
import torch
from torch_scatter import scatter

from lcaonet.nn.post import PostProcess

param_PostProcess = [
    (1, None, False, None, True),  # default
    (1, None, False, None, False),  # no extensive
    (1, None, True, None, True),  # add mean
    (1, None, True, torch.ones(1), True),  # add mean (1)
    (1, None, True, torch.ones(1), False),  # add mean (1) and no extensive
    (1, torch.ones((100, 1)), False, None, True),  # add atomref
    (1, torch.ones((100, 1)), False, None, False),  # add atomref and no extensive
    (1, torch.ones((100, 1)), True, None, True),  # add atomref and mean
    (1, torch.ones((100, 1)), True, torch.ones(1), True),  # add atomref and mean (1)
    (1, torch.ones((100, 1)), True, torch.ones(1), False),  # add atomref and mean (1) and no extensive
    (10, torch.ones((100, 1)), True, torch.ones(1), True),  # add atomref and mean (1)
    (10, torch.ones((100, 1)), True, torch.ones(1), False),  # add atomref and mean (1) and no extensive
]


@pytest.mark.parametrize("out_dim, atomref, mean, is_extensive", param_PostProcess)
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
