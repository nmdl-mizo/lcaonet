from __future__ import annotations

import math

import numpy as np
import pytest
import scipy
import torch

from lcaonet.atomistic.info import ElecInfo
from lcaonet.nn.shbf import SphericalHarmonicsBasis

param_SphericalHarmonicsBasis = [
    (12, None, 1),
    (12, None, 2),
    (12, "3s", 1),
    (12, "3s", 2),
    (36, None, 1),
    (36, None, 2),
    (36, "6s", 1),
    (36, "6s", 1),
    (84, None, 1),
    (84, "6d", 1),
]


@pytest.mark.parametrize("max_z, max_orb, n_per_orb", param_SphericalHarmonicsBasis)
def test_SphericalHarmonicsBasis(
    max_z: int,
    max_orb: str | None,
    n_per_orb: int,
):
    n_triplet = 200
    angle = torch.linspace(0, 2 * math.pi, n_triplet)
    ei = ElecInfo(max_z, max_orb, None, n_per_orb)

    shbf = SphericalHarmonicsBasis(ei)
    shb = shbf(angle)

    assert shb.size() == (n_triplet, ei.n_orb)

    # check with scipy function
    angle_numpy = angle.numpy()
    for i in range(ei.n_orb):
        lq = ei.nl_list[i][1].item()
        shb_scipy = scipy.special.sph_harm(0, lq, 0, angle_numpy).astype(np.float64)
        assert torch.allclose(shb[:, i], torch.tensor(shb_scipy, dtype=torch.float32), rtol=1e-5, atol=1e-7)
