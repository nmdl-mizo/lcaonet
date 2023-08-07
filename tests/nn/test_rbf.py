from __future__ import annotations

import numpy as np
import pytest
import torch

from lcaonet.atomistic.info import ElecInfo
from lcaonet.nn.cutoff import EnvelopeCutoff
from lcaonet.nn.rbf import HydrogenRadialBasis

param_HydrogenRadialBasis = [
    (1.0, 12, None, 1),
    (3.0, 12, None, 1),
    (1.0, 12, None, 2),
    (3.0, 12, None, 2),
    (1.0, 12, "3s", 1),
    (3.0, 12, "3s", 1),
    (1.0, 12, "3s", 2),
    (3.0, 12, "3s", 2),
    (1.0, 36, None, 1),
    (3.0, 36, None, 1),
    (1.0, 36, None, 2),
    (3.0, 36, None, 2),
    (1.0, 36, "6s", 1),
    (3.0, 36, "6s", 1),
    (1.0, 36, "6s", 1),
    (3.0, 36, "6s", 1),
    (1.0, 84, None, 1),
    (3.0, 84, None, 1),
    (1.0, 84, "6d", 1),
    (3.0, 84, "6d", 1),
    (1.0, 84, "6d", 2),
    (3.0, 84, "6d", 2),
]

# fmt: off
rbfs = {
    (1, 0): lambda r, a0: (1 / a0)**(3 / 2) * np.exp(-r / a0) * 2,  # 1s
    (2, 0): lambda r, a0: (1 / a0)**(3 / 2) * (2 - r / a0) * np.exp(-r / 2 / a0) / 2 / np.sqrt(2),  # 2s
    (2, 1): lambda r, a0: (1 / a0)**(3 / 2) * r / a0 * np.exp(-r / 2 / a0) / 2 / np.sqrt(6),  # 2p
    (3, 0): lambda r, a0: (1 / a0)**(3 / 2) * (27 - 18 * r / a0 + 2 * r**2 / a0**2) * np.exp(-r / 3 / a0) * 2 / 81 / np.sqrt(3),  # 3s # noqa: E501
    (3, 1): lambda r, a0: (1 / a0)**(3 / 2) * (6 - r / a0) * r / a0 * np.exp(-r / 3 / a0) * 4 / 81 / np.sqrt(6),  # 3p
    (3, 2): lambda r, a0: (1 / a0)**(3 / 2) * (r / a0)**2 * np.exp(-r / 3 / a0) * 4 / 81 / np.sqrt(30),  # 3d
    (4, 0): lambda r, a0: (1 / a0)**(3 / 2) * (192 - 144 * r / a0 + 24 * r**2 / a0**2 - r**3 / a0**3) * np.exp(-r / 4 / a0) * 1 / 768,  # 4s # noqa: E501
    (4, 1): lambda r, a0: (1 / a0)**(3 / 2) * (80 - 20 * r / a0 + r**2 / a0**2) * r / a0 * np.exp(-r / 4 / a0) * 1 / 256 / np.sqrt(15),  # 4p # noqa: E501
    (4, 2): lambda r, a0: (1 / a0)**(3 / 2) * (12 - r / a0) * (r / a0)**2 * np.exp(-r / 4 / a0) * 1 / 768 / np.sqrt(5),  # 4d # noqa: E501
    (4, 3): lambda r, a0: (1 / a0)**(3 / 2) * (r / a0)**3 * np.exp(-r / 4 / a0) * 1 / 768 / np.sqrt(35),  # 4f
}
# fmt: on


@pytest.mark.parametrize("cutoff, max_z, max_orb, n_per_orb", param_HydrogenRadialBasis)
def test_HydrogenRadialBasis(
    cutoff: float,
    max_z: int,
    max_orb: str | None,
    n_per_orb: int,
):
    n_edge = 200
    r = torch.linspace(0, 10, n_edge)
    ei = ElecInfo(max_z, max_orb, None, n_per_orb)
    cn = EnvelopeCutoff(cutoff)

    rbf = HydrogenRadialBasis(cutoff, ei, cn)
    rb = rbf(r)

    assert rb.size() == (n_edge, ei.n_orb)

    # check function
    r_numpy = r.numpy()
    cw = cn(r).detach().numpy()
    for i, nl in enumerate(ei.nl_list):
        nl_tuple = (nl[0].item(), nl[1].item())
        func = rbfs.get(nl_tuple, None)
        if func is None:
            continue
        rbf_numpy = func(r_numpy, rbf.bohr_radius) * cw
        assert torch.allclose(rb[:, i], torch.tensor(rbf_numpy), rtol=1e-5, atol=1e-7)
