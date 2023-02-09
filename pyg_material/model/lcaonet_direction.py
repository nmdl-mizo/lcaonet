from __future__ import annotations  # type: ignore

import math
from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch
from torch_scatter import scatter

from pyg_material.data import DataKeys
from pyg_material.model.base import BaseGNN
from pyg_material.nn import Dense
from pyg_material.utils import activation_resolver, init_resolver

# 1s, 2s, 2p, 3s, 3p, 4s, 3d, 4p, 5s, 4d, 5p, 6s, 4f, 5d, 6p, 7s, 5f, 6d
ELEC_DICT = torch.tensor(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # dummy
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # H
        [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # He
        [2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Li
        [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Be
        [2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # B
        [2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # C
        [2, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # N
        [2, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # O
        [2, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # F
        [2, 2, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Ne
        [2, 2, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Na
        [2, 2, 6, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Mg
        [2, 2, 6, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Al
        [2, 2, 6, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Si
        [2, 2, 6, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # P
        [2, 2, 6, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # S
        [2, 2, 6, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Cl
        [2, 2, 6, 2, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Ar (18)
        [2, 2, 6, 2, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # K
        [2, 2, 6, 2, 6, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Ca
        [2, 2, 6, 2, 6, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Sc
        [2, 2, 6, 2, 6, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Ti
        [2, 2, 6, 2, 6, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # V
        [2, 2, 6, 2, 6, 1, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Cr
        [2, 2, 6, 2, 6, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Mn
        [2, 2, 6, 2, 6, 2, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Fe
        [2, 2, 6, 2, 6, 2, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Co
        [2, 2, 6, 2, 6, 2, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Ni
        [2, 2, 6, 2, 6, 1, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Cu
        [2, 2, 6, 2, 6, 2, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Zn
        [2, 2, 6, 2, 6, 2, 10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Ga
        [2, 2, 6, 2, 6, 2, 10, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Ge
        [2, 2, 6, 2, 6, 2, 10, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # As
        [2, 2, 6, 2, 6, 2, 10, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Se
        [2, 2, 6, 2, 6, 2, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Br
        [2, 2, 6, 2, 6, 2, 10, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Kr (36)
        [2, 2, 6, 2, 6, 2, 10, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Rb
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Sr
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Y
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],  # Zr
        [2, 2, 6, 2, 6, 2, 10, 6, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0],  # Nb
        [2, 2, 6, 2, 6, 2, 10, 6, 1, 5, 0, 0, 0, 0, 0, 0, 0, 0],  # Mo
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0],  # Tc
        [2, 2, 6, 2, 6, 2, 10, 6, 1, 7, 0, 0, 0, 0, 0, 0, 0, 0],  # Ru
        [2, 2, 6, 2, 6, 2, 10, 6, 1, 8, 0, 0, 0, 0, 0, 0, 0, 0],  # Rh
        [2, 2, 6, 2, 6, 2, 10, 6, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0],  # Pd
        [2, 2, 6, 2, 6, 2, 10, 6, 1, 10, 0, 0, 0, 0, 0, 0, 0, 0],  # Ag
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 0, 0, 0, 0, 0, 0, 0, 0],  # Cd
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 1, 0, 0, 0, 0, 0, 0, 0],  # In
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 2, 0, 0, 0, 0, 0, 0, 0],  # Sn
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 3, 0, 0, 0, 0, 0, 0, 0],  # Sb
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 4, 0, 0, 0, 0, 0, 0, 0],  # Te
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 5, 0, 0, 0, 0, 0, 0, 0],  # I
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0, 0, 0, 0, 0, 0, 0],  # Xe (54)
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 1, 0, 0, 0, 0, 0, 0],  # Cs
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 0, 0, 0, 0, 0, 0],  # Ba
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 0, 1, 0, 0, 0, 0],  # La
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 1, 1, 0, 0, 0, 0],  # Ce
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 3, 0, 0, 0, 0, 0],  # Pr
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 4, 0, 0, 0, 0, 0],  # Nd
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 5, 0, 0, 0, 0, 0],  # Pm
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 6, 0, 0, 0, 0, 0],  # Sm
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 7, 0, 0, 0, 0, 0],  # Eu
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 7, 1, 0, 0, 0, 0],  # Gd
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 9, 0, 0, 0, 0, 0],  # Tb
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 10, 0, 0, 0, 0, 0],  # Dy
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 11, 0, 0, 0, 0, 0],  # Ho
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 12, 0, 0, 0, 0, 0],  # Er
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 13, 0, 0, 0, 0, 0],  # Tm
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 0, 0, 0, 0, 0],  # Yb
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 1, 0, 0, 0, 0],  # Lu (71)
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 2, 0, 0, 0, 0],  # Hf
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 3, 0, 0, 0, 0],  # Ta
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 4, 0, 0, 0, 0],  # W
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 5, 0, 0, 0, 0],  # Re
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 6, 0, 0, 0, 0],  # Os
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 7, 0, 0, 0, 0],  # Ir
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 1, 14, 9, 0, 0, 0, 0],  # Pt
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 1, 14, 10, 0, 0, 0, 0],  # Au
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 0, 0, 0, 0],  # Hg
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 1, 0, 0, 0],  # Tl
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 2, 0, 0, 0],  # Pb
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 3, 0, 0, 0],  # Bi
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 4, 0, 0, 0],  # Po
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 5, 0, 0, 0],  # At
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 0, 0, 0],  # Rn (86)
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 1, 0, 0],  # Fr
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 0, 0],  # Ra
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 0, 1],  # Ac
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 0, 2],  # Th
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 2, 1],  # Pa
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 3, 1],  # U
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 4, 1],  # Np
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 6, 0],  # Pu (94)
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 7, 0],  # Am
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 7, 1],  # Cm (96)
    ]
)
MAX_IND = [3, 3, 7, 3, 7, 3, 11, 7, 3, 11, 7, 3, 15, 11, 7, 3, 15, 11]


def modify_elec_dict(elec: Tensor = ELEC_DICT, max_ind: list[int] = MAX_IND) -> tuple[Tensor, Tensor]:
    # 1s,
    # 2s,
    # 2p, 2p, 2p,
    # 3s,
    # 3p, 3p, 3p,
    # 4s,
    # 3d, 3d, 3d, 3d, 3d, 3d,
    # 4p, 4p, 4p,
    # 5s,
    # 4d, 4d, 4d, 4d, 4d, 4d,
    # 5p, 5p, 5p,
    # 6s,
    # 4f, 4f, 4f, 4f, 4f, 4f, 4f, 4f, 4f, 4f
    # 5d, 5d, 5d, 5d, 5d, 5d,
    # 6p, 6p, 6p,
    # 7s
    # 5f, 5f, 5f, 5f, 5f, 5f, 5f, 5f, 5f, 5f,
    # 6d, 6d, 6d, 6d, 6d, 6d,
    OUT_ORB = 66

    def modify_one(one_elec: Tensor) -> Tensor:
        out_elec = torch.zeros((OUT_ORB,), dtype=torch.long)
        j = 0
        for i in range(one_elec.size(0)):
            # s
            if np.isin([0, 1, 3, 5, 8, 11, 15], i).any():
                out_elec[j] = one_elec[i]
                j += 1
            # p
            if np.isin([2, 4, 7, 10, 14], i).any():
                for _ in range(3):
                    out_elec[j] = one_elec[i]
                    j += 1
            # d
            if np.isin([6, 9, 13, 17], i).any():
                for _ in range(6):
                    out_elec[j] = one_elec[i]
                    j += 1
            # f
            if np.isin([12, 16], i).any():
                for _ in range(10):
                    out_elec[j] = one_elec[i]
                    j += 1
        return out_elec

    def modify_max_ind() -> Tensor:
        out_max_ind = torch.zeros((OUT_ORB,), dtype=torch.long)
        j = 0
        for i in range(len(max_ind)):
            # s
            if np.isin([0, 1, 3, 5, 8, 11, 15], i).any():
                out_max_ind[j] = max_ind[i]
                j += 1
            # p
            if np.isin([2, 4, 7, 10, 14], i).any():
                for _ in range(3):
                    out_max_ind[j] = max_ind[i]
                    j += 1
            # d
            if np.isin([6, 9, 13, 17], i).any():
                for _ in range(6):
                    out_max_ind[j] = max_ind[i]
                    j += 1
            # f
            if np.isin([12, 16], i).any():
                for _ in range(10):
                    out_max_ind[j] = max_ind[i]
                    j += 1
        return out_max_ind

    out_elec = torch.zeros((len(elec), OUT_ORB), dtype=torch.long)
    out_max_ind = modify_max_ind()
    for i in range(len(elec)):
        out_elec[i] = modify_one(elec[i])

    return out_elec, out_max_ind


ELEC_DICT_BIG, MAX_IND_BIG = modify_elec_dict()


def factorial(n: int):
    return torch.tensor([math.factorial(n)])


def assoc_laguerre(x: Tensor, n: int, k: int) -> Tensor:
    device = x.device
    ranges_1 = torch.tensor([m + k for m in range(n - k + 1)]).to(device)
    m = torch.pow(-1.0, ranges_1)[None, ...]
    del ranges_1
    denomi = (
        torch.tensor([math.factorial(m) for m in range(n - k + 1)])
        * torch.tensor([math.factorial(m + k) for m in range(n - k + 1)])
        * torch.tensor([math.factorial(n - m - k) for m in range(n - k + 1)])
    )
    denomi = denomi[None, ...].to(device)
    ranges_2 = torch.tensor([m for m in range(n - k + 1)]).to(device)
    x = torch.pow(x[..., None], ranges_2)
    del ranges_2
    nume = factorial(n).to(device)
    return torch.sum(m / denomi * x, dim=-1) * torch.pow(nume, 2)


def R_nl_assoc_lag(nq: int, lq: int) -> Callable[[Tensor, Tensor], Tensor]:
    """RBF with the associated Laguerre polynomial.

    Args:
        nq (int): principal quantum number.
        lq (int): azimuthal quantum number.

    Returns:
        Callable[[Tensor, Tensor], Tensor]: RBF function.
    """

    def r_nl(r: Tensor, a0: Tensor) -> Tensor:
        zeta = 2.0 / nq / a0 * r
        # # Standardization in all spaces
        # device = r.device
        # f_coeff = -(
        #     ((2.0 / nq / a0) ** 3 * factorial(nq - lq - 1).to(device) / 2.0 / nq / (factorial(nq + lq).to(device) ** 3)) # NOQA: E501
        #     ** (1.0 / 2.0)
        # )

        # no standardization
        f_coeff = -2.0 / nq / a0

        al = assoc_laguerre(zeta, nq + lq, 2 * lq + 1)
        o = f_coeff * al * torch.pow(zeta, lq) * torch.exp(-zeta / 2.0)

        return o

    return r_nl


def R_nl_learnable(nq: int) -> Callable[[Tensor, Tensor, Tensor], Tensor]:
    """RBF with the learnable coefficient and the exponential decay.

    Args:
        nq (int): principal quantum number.

    Returns:
        Callable[[Tensor, Tensor, Tensor], Tensor]: RBF function.
    """

    def r_nl(r: Tensor, coeff: Tensor, zeta: Tensor) -> Tensor:
        o = coeff * torch.pow(r, nq - 1) * torch.exp(-zeta * r)
        return o

    return r_nl


class SlaterDirectionOrbBasis(nn.Module):
    n_radial: int = ELEC_DICT_BIG.size(-1)

    def __init__(self, cutoff: float | None = None, assoc_lag: bool = False, **kwargs):
        super().__init__()
        self.cutoff = cutoff
        self.assoc_lag = assoc_lag
        if self.assoc_lag:
            # if R_nl_assoc_lag func, a0 is always learnable
            self.radius = 0.529
            self.a0 = nn.Parameter(torch.ones((1,)) * self.radius)
        else:
            self.nl_embed = EmbedNL(**kwargs)
        # fmt: off
        nl_list = [
            (1, 0),  # 1s
            (2, 0),  # 2s
            (2, 1), (2, 1), (2, 1),  # 2p
            (3, 0),  # 3s
            (3, 1), (3, 1), (3, 1),  # 3p
            (4, 0),  # 4s
            (3, 2), (3, 2), (3, 2), (3, 2), (3, 2), (3, 2),  # 3d
            (4, 1), (4, 1), (4, 1),  # 4p
            (5, 0),  # 5s
            (4, 2), (4, 2), (4, 2), (4, 2), (4, 2), (4, 2),  # 4d
            (5, 1), (5, 1), (5, 1),  # 5p
            (6, 0),  # 6s
            (4, 3), (4, 3), (4, 3), (4, 3), (4, 3), (4, 3), (4, 3), (4, 3), (4, 3), (4, 3),  # 4f
            (5, 2), (5, 2), (5, 2), (5, 2), (5, 2), (5, 2),  # 5d
            (6, 1), (6, 1), (6, 1),  # 6p
            (7, 0),  # 7s
            (5, 3), (5, 3), (5, 3), (5, 3), (5, 3), (5, 3), (5, 3), (5, 3), (5, 3), (5, 3),  # 5f
            (6, 2), (6, 2), (6, 2), (6, 2), (6, 2), (6, 2),  # 6d
        ]
        # fmt: on

        def make_directional_func() -> list[Callable]:
            l: list[Callable] = []
            for i in range(len(MAX_IND)):
                # s
                if np.isin([0, 1, 3, 5, 8, 11, 15], i).any():
                    l.append(lambda _: 1.0)
                # p
                if np.isin([2, 4, 7, 10, 14], i).any():
                    for k in range(3):
                        l.append(lambda x: x[:, k])
                # d
                if np.isin([6, 9, 13, 17], i).any():
                    l.append(lambda x: (x[:, 0] * x[:, 0]))
                    l.append(lambda x: (x[:, 0] * x[:, 1]))
                    l.append(lambda x: (x[:, 0] * x[:, 2]))
                    l.append(lambda x: (x[:, 1] * x[:, 1]))
                    l.append(lambda x: (x[:, 1] * x[:, 2]))
                    l.append(lambda x: (x[:, 2] * x[:, 2]))
                # f
                if np.isin([12, 16], i).any():
                    l.append(lambda x: (x[:, 0] * x[:, 0] * x[:, 0]))
                    l.append(lambda x: (x[:, 0] * x[:, 0] * x[:, 1]))
                    l.append(lambda x: (x[:, 0] * x[:, 0] * x[:, 2]))
                    l.append(lambda x: (x[:, 1] * x[:, 1] * x[:, 1]))
                    l.append(lambda x: (x[:, 1] * x[:, 1] * x[:, 0]))
                    l.append(lambda x: (x[:, 1] * x[:, 1] * x[:, 2]))
                    l.append(lambda x: (x[:, 2] * x[:, 2] * x[:, 2]))
                    l.append(lambda x: (x[:, 2] * x[:, 2] * x[:, 0]))
                    l.append(lambda x: (x[:, 2] * x[:, 2] * x[:, 1]))
                    l.append(lambda x: (x[:, 0] * x[:, 1] * x[:, 2]))

            return l

        self.directional_func = make_directional_func()

        self.basis_func_assoc = []
        self.basis_func_learnable = []
        for n, l in nl_list:
            if self.assoc_lag:
                r_nl_assoc = R_nl_assoc_lag(n, l)
                self.basis_func_assoc.append(r_nl_assoc)
            else:
                r_nl_learnable = R_nl_learnable(n)
                self.basis_func_learnable.append(r_nl_learnable)

    def forward(self, dist: Tensor, vec: Tensor) -> Tensor:
        """Forward calculation of SlaterDirectionOrbBasis.

        Args:
            dist (Tensor): atomic distances with (n_edge) shape.
            vec (Tensor): inter atomic vector with (n_edge, 3) shape.

        Returns:
            rbf (Tensor): rbf with (n_edge, n_orb) shape.
        """
        if self.assoc_lag:
            rbf = torch.stack(
                [f(dist, self.a0) * fd(vec) for f, fd in zip(self.basis_func_assoc, self.directional_func)], dim=1
            )
        else:
            coeff, zeta = self.nl_embed()
            rbf = torch.stack(
                [
                    f(dist, coeff[i], zeta[i]) * fd(vec)
                    for i, f, fd in zip(
                        range(SlaterDirectionOrbBasis.n_radial), self.basis_func_learnable, self.directional_func
                    )
                ],
                dim=1,
            )
        return rbf


class EmbedNL(nn.Module):
    def __init__(
        self,
        nl_hidden_dim: int = 32,
        device: str = "cpu",
        activation: nn.Module = nn.SiLU(),
        weight_init: Callable[[Tensor], Tensor] | None = None,
    ):
        super().__init__()
        self.nl_hidden_dim = nl_hidden_dim
        self.n_orb = SlaterDirectionOrbBasis.n_radial
        self.n_embed = nn.Embedding(8, nl_hidden_dim)
        self.l_embed = nn.Embedding(4, nl_hidden_dim)
        self.coeffs_lin = nn.Sequential(
            activation,
            Dense(2 * nl_hidden_dim, nl_hidden_dim, weight_init=weight_init),
            activation,
            Dense(nl_hidden_dim, nl_hidden_dim, False, weight_init=weight_init),
        )
        # fmt: off
        self.n_list = torch.tensor(
            [
                1,  # 1s
                2,  # 2s
                2, 2, 2,  # 2p
                3,  # 3s
                3, 3, 3,  # 3p
                4,  # 4s
                3, 3, 3, 3, 3, 3,  # 3d
                4, 4, 4,  # 4p
                5,  # 5s
                4, 4, 4, 4, 4, 4,  # 4d
                5, 5, 5,  # 5p
                6,  # 6s
                4, 4, 4, 4, 4, 4, 4, 4, 4, 4,  # 4f
                5, 5, 5, 5, 5, 5,  # 5d
                6, 6, 6,  # 6p
                7,  # 7s
                5, 5, 5, 5, 5, 5, 5, 5, 5, 5,  # 5f
                6, 6, 6, 6, 6, 6,  # 6d
            ]
        ).to(device)
        self.l_list = torch.tensor(
            [
                0,  # 1s
                0,  # 2s
                1, 1, 1,  # 2p
                0,  # 3s
                1, 1, 1,  # 3p
                0,  # 4s
                2, 2, 2, 2, 2, 2,  # 3d
                1, 1, 1,  # 4p
                0,  # 5s
                2, 2, 2, 2, 2, 2,  # 4d
                1, 1, 1,  # 5p
                0,  # 6s
                3, 3, 3, 3, 3, 3, 3, 3, 3, 3,  # 4f
                2, 2, 2, 2, 2, 2,  # 5d
                1, 1, 1,  # 6p
                0,  # 7s
                3, 3, 3, 3, 3, 3, 3, 3, 3, 3,  # 5f
                2, 2, 2, 2, 2, 2,  # 6d
            ]
        ).to(device)
        # fmt: on

        self.reset_parameters()

    def reset_parameters(self):
        self.n_embed.weight.data.uniform_(-math.sqrt(3), math.sqrt(3))
        self.l_embed.weight.data.uniform_(-math.sqrt(3), math.sqrt(3))

    def forward(self) -> tuple[Tensor, Tensor]:
        """
        Returns:
            coeffs (torch.Tensor): coeffs of (n_orb) shape.
            zeta (torch.Tensor): zeta vectors of (n_orb) shape.
        """
        # (n_orb, hidden_dim)
        nq = self.n_embed(self.n_list)

        # (n_orb, hidden_dim)
        lq = self.l_embed(self.l_list)

        # concat and linear transformation
        # (n_orb, 2 * hidden_dim)
        c = torch.cat([nq, lq], dim=-1)

        # (n_orb, 2)
        c = self.coeffs_lin(c)
        coeff, zeta = torch.chunk(c, 2, dim=-1)

        return coeff.flatten(), zeta.flatten()


class EmbedZ(nn.Module):
    def __init__(self, embed_dim: int, max_z: int = 37):
        super().__init__()
        self.z_embed = nn.Embedding(max_z, embed_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.z_embed.weight.data.uniform_(-math.sqrt(3), math.sqrt(3))

    def forward(self, z: Tensor) -> Tensor:
        """
        Args:
            z (torch.Tensor): atomic numbers of (n_node) shape.

        Returns:
            z_embed (torch.Tensor): coefficent vectors of (n_node, embed_dim) shape.
        """
        # (n_node) -> (n_node, embed_dim)
        z_embed = self.z_embed(z)

        return z_embed


class EmbedElec(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        device: str,
        n_orb: int = SlaterDirectionOrbBasis.n_radial,
    ):
        super().__init__()
        self.elec = ELEC_DICT_BIG.to(device)
        self.n_orb = n_orb
        self.embed_dim = embed_dim
        self.elec_embeds = nn.ModuleList([nn.Embedding(m, embed_dim, padding_idx=0) for m in MAX_IND_BIG])
        # !!Caution!! padding_idx parameter stops working when reset_parameter() is called

    def forward(self, z: Tensor, z_embed: Tensor) -> Tensor:
        """
        Args:
            z (torch.Tensor): atomic numbers of (n_node) shape.
            z_embed (torch.Tensor): embedding of atomic number with (n_node, embed_dim) shape.

        Returns:
            elec_embed (torch.Tensor): embedding vectors of (n_node, n_orb, embed_dim) shape.
        """
        # (n_node) -> (n_node, n_orb)
        elec = self.elec[z]
        # (n_orb, n_node)
        elec = torch.transpose(elec, 0, 1)
        # (n_orb, n_node, embed_dim)
        elec_embed = torch.stack([ce(elec[i]) for i, ce in enumerate(self.elec_embeds)], dim=0)
        # (n_node, n_orb, embed_dim)
        elec_embed = torch.transpose(elec_embed, 0, 1)

        # (n_node) -> (n_node, 1, embed_dim)
        z_embed = z_embed.unsqueeze(1)
        # inject atomic information to nl_embed vectors
        elec_embed = elec_embed + elec_embed * z_embed

        return elec_embed


class EmbedNode(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        z_dim: int,
        elec_dim: int,
        activation: nn.Module = nn.SiLU(),
        weight_init: Callable[[Tensor], Tensor] | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.elec_dim = elec_dim

        self.elec_z_lin = nn.Sequential(
            activation,
            Dense(z_dim + elec_dim, hidden_dim, True, weight_init),
            activation,
            Dense(hidden_dim, hidden_dim, False, weight_init),
        )

    def forward(self, z_embed: Tensor, elec_embed: Tensor) -> Tensor:
        """
        Args:
            z_embed (torch.Tensor): embedding of atomic number with (n_node, z_dim) shape.
            elec_embed (torch.Tensor): embedding of nl values with (n_node, n_orb, elec_dim) shape.

        Returns:
            torch.Tensor: node embedding vector of (n_node, hidden_dim) shape.
        """
        elec_z_embed = torch.cat([z_embed, elec_embed.sum(1)], dim=-1)
        return self.elec_z_lin(elec_z_embed)


class LCAOConv(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        down_dim: int,
        coeffs_dim: int,
        activation: nn.Module = nn.SiLU(),
        weight_init: Callable[[Tensor], Tensor] | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.down_dim = down_dim
        self.coeffs_dim = coeffs_dim

        self.node_lin = nn.Sequential(
            activation,
            Dense(hidden_dim, hidden_dim, True, weight_init),
            activation,
            Dense(hidden_dim, down_dim, True, weight_init),
        )
        # No bias is used to keep coefficient values at 0
        self.coeffs_lin = nn.Sequential(
            activation,
            Dense(coeffs_dim, hidden_dim, False, weight_init),
            activation,
            Dense(hidden_dim, down_dim, False, weight_init),
        )
        self.up_lin = nn.Sequential(Dense(down_dim, hidden_dim, False, weight_init))

    def forward(self, x: Tensor, edge_index: Tensor, rbfs: Tensor, coeffs: Tensor) -> Tensor:
        edge_src, edge_dst = edge_index[0], edge_index[1]

        x = self.node_lin(x)

        coeffs = self.coeffs_lin(coeffs)
        coeffs = coeffs[edge_dst] * coeffs[edge_src] + coeffs[edge_dst]
        coeffs = F.normalize(coeffs, dim=-1)

        # get rbf values
        rbfs = torch.einsum("ed,edh->eh", rbfs, coeffs)
        rbfs = F.normalize(rbfs, dim=-1)

        return self.up_lin(scatter(x[edge_dst] * rbfs, edge_src, dim=0))


class LCAOOut(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        activation: nn.Module = nn.SiLU(),
        weight_init: Callable[[Tensor], Tensor] | None = None,
        aggr: str = "sum",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.aggr = aggr

        self.out_lin = nn.Sequential(
            activation,
            Dense(hidden_dim, hidden_dim, True, weight_init),
            activation,
            Dense(hidden_dim, hidden_dim // 2, True, weight_init),
            activation,
            Dense(hidden_dim // 2, out_dim, False, weight_init),
        )

    def forward(self, x: Tensor, batch_idx: Tensor | None) -> Tensor:
        out = self.out_lin(x)
        return out.sum(dim=0, keepdim=True) if batch_idx is None else scatter(out, batch_idx, dim=0, reduce=self.aggr)


class LCAODirectionNet(BaseGNN):
    def __init__(
        self,
        hidden_dim: int = 128,
        down_dim: int = 64,
        coeffs_dim: int = 16,
        out_dim: int = 1,
        n_conv_layer: int = 3,
        cutoff: float | None = 3.5,
        activation: str = "SiLU",
        weight_init: str | None = "glorotorthogonal",
        assoc_lag: bool = False,
        max_z: int = 100,
        aggr: str = "sum",
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__()
        wi: Callable[[Tensor], Tensor] | None = init_resolver(weight_init) if weight_init is not None else None
        act: nn.Module = activation_resolver(activation)

        self.n_conv_layer = n_conv_layer
        self.hidden_dim = hidden_dim
        self.coeffs_dim = coeffs_dim
        self.device = device

        # RBF
        if assoc_lag:
            self.wfrbf = SlaterDirectionOrbBasis(cutoff, assoc_lag=assoc_lag)
        else:
            kwargs["weight_init"] = wi
            kwargs["activation"] = act
            kwargs["device"] = device
            kwargs["nl_hidden_dim"] = 32 if kwargs.get("nl_hidden_dim") is None else kwargs.get("nl_hidden_dim")
            self.wfrbf = SlaterDirectionOrbBasis(cutoff, assoc_lag=assoc_lag, **kwargs)

        # z and elec embedding layers
        self.node_z_embed_dim = 32  # fix
        self.node_elec_embed_dim = 32  # fix
        self.elec_embed_dim = self.coeffs_dim + self.node_elec_embed_dim
        self.z_embed_dim = self.elec_embed_dim + self.node_z_embed_dim
        self.z_embed = EmbedZ(embed_dim=self.z_embed_dim, max_z=max_z)
        self.elec_embed = EmbedElec(embed_dim=self.elec_embed_dim, device=device)
        self.node_embed = EmbedNode(
            hidden_dim, z_dim=self.node_z_embed_dim, elec_dim=self.node_elec_embed_dim, activation=act, weight_init=wi
        )

        self.conv_layers = nn.ModuleList(
            [LCAOConv(hidden_dim, down_dim, coeffs_dim, act, wi) for _ in range(n_conv_layer)]
        )

        self.out_layer = LCAOOut(hidden_dim, out_dim, act, wi, aggr=aggr)

    def forward(self, batch: Batch) -> Tensor:
        batch_idx: Tensor | None = batch.get(DataKeys.Batch_idx)
        z = batch[DataKeys.Atom_numbers]
        edge_idx = batch[DataKeys.Edge_idx]

        # calc atomic distances
        distances, vec = self.calc_atomic_distances(batch, return_vec=True)

        # calc rbf values
        rbfs = self.wfrbf(distances, vec)

        # calc node and coefficent vectors
        z_embed = self.z_embed(z)
        z_embed1, z_embed2 = torch.split(z_embed, [self.elec_embed_dim, self.node_z_embed_dim], dim=-1)

        elec_embed = self.elec_embed(z, z_embed1)
        elec_embed1, elec_embed2 = torch.split(elec_embed, [self.coeffs_dim, self.node_elec_embed_dim], dim=-1)
        x = self.node_embed(z_embed2, elec_embed2)

        # conv layers
        for conv in self.conv_layers:
            x = x + conv(x, edge_idx, rbfs, elec_embed1)

        # out layer
        out = self.out_layer(x, batch_idx)

        return out
