from __future__ import annotations  # type: ignore

import math
from collections.abc import Callable
from math import pi

import numpy as np
import sympy as sym
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.integrate import quad
from torch import Tensor
from torch_geometric.data import Batch
from torch_scatter import scatter

from pyg_material.data import DataKeys
from pyg_material.model.base import BaseGNN
from pyg_material.nn import Dense
from pyg_material.utils import activation_resolver, init_resolver

# 1s, 2s, 2p, 3s, 3p, 4s, 3d, 4p, 5s, 4d, 5p, 6s, 4f, 5d, 6p, 7s, 5f, 6d
ELEC_TABLE = torch.tensor(
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
MAX_IDX = [3, 3, 7, 3, 7, 3, 11, 7, 3, 11, 7, 3, 15, 11, 7, 3, 15, 11]
ORB_SPH = 37


def modify_elec_table(elec: Tensor = ELEC_TABLE, max_idx: list[int] = MAX_IDX) -> tuple[Tensor, Tensor]:
    # 1s,
    # 2s,
    # 2p, 2p,
    # 3s,
    # 3p, 3p,
    # 4s,
    # 3d, 3d, 3d,
    # 4p, 4p,
    # 5s,
    # 4d, 4d, 4d,
    # 5p, 5p,
    # 6s,
    # 4f, 4f, 4f, 4f,
    # 5d, 5d, 5d,
    # 6p, 6p,
    # 7s
    # 5f, 5f, 5f, 5f,
    # 6d, 6d, 6d,

    def modify_one(one_elec: Tensor) -> Tensor:
        out_elec = torch.zeros((ORB_SPH,), dtype=torch.long)
        j = 0
        for i in range(one_elec.size(0)):
            # s
            if np.isin([0, 1, 3, 5, 8, 11, 15], i).any():
                out_elec[j] = one_elec[i]
                j += 1
            # p
            if np.isin([2, 4, 7, 10, 14], i).any():
                for _ in range(2):
                    out_elec[j] = one_elec[i]
                    j += 1
            # d
            if np.isin([6, 9, 13, 17], i).any():
                for _ in range(3):
                    out_elec[j] = one_elec[i]
                    j += 1
            # f
            if np.isin([12, 16], i).any():
                for _ in range(4):
                    out_elec[j] = one_elec[i]
                    j += 1
        return out_elec

    def modify_max_ind() -> Tensor:
        out_max_ind = torch.zeros((ORB_SPH,), dtype=torch.long)
        j = 0
        for i in range(len(max_idx)):
            # s
            if np.isin([0, 1, 3, 5, 8, 11, 15], i).any():
                out_max_ind[j] = max_idx[i]
                j += 1
            # p
            if np.isin([2, 4, 7, 10, 14], i).any():
                for _ in range(2):
                    out_max_ind[j] = max_idx[i]
                    j += 1
            # d
            if np.isin([6, 9, 13, 17], i).any():
                for _ in range(3):
                    out_max_ind[j] = max_idx[i]
                    j += 1
            # f
            if np.isin([12, 16], i).any():
                for _ in range(4):
                    out_max_ind[j] = max_idx[i]
                    j += 1
        return out_max_ind

    out_elec = torch.zeros((len(elec), ORB_SPH), dtype=torch.long)
    out_max_ind = modify_max_ind()
    for i in range(len(elec)):
        out_elec[i] = modify_one(elec[i])

    return out_elec, out_max_ind


ELEC_TABLE_BIG, MAX_IDX_BIG = modify_elec_table()


class RadialOrbitalBasis(nn.Module):
    n_orb: int = ORB_SPH
    # fmt: off
    nl_list: list[tuple[int, int]] = [
        (1, 0),                             # 1s
        (2, 0),                             # 2s
        (2, 1), (2, 1),                     # 2p
        (3, 0),                             # 3s
        (3, 1), (3, 1),                     # 3p
        (4, 0),                             # 4s
        (3, 2), (3, 2), (3, 2),             # 3d
        (4, 1), (4, 1),                     # 4p
        (5, 0),                             # 5s
        (4, 2), (4, 2), (4, 2),             # 4d
        (5, 1), (5, 1),                     # 5p
        (6, 0),                             # 6s
        (4, 3), (4, 3), (4, 3), (4, 3),     # 4f
        (5, 2), (5, 2), (5, 2),             # 5d
        (6, 1), (6, 1),                     # 6p
        (7, 0),                             # 7s
        (5, 3), (5, 3), (5, 3), (5, 3),     # 5f
        (6, 2), (6, 2), (6, 2),             # 6d
    ]
    # fmt: on

    def __init__(self, cutoff: float | None = None, stand_in_cutoff: bool = True, radius: float = 0.529):
        super().__init__()
        self.cutoff = cutoff
        self.stand_in_cutoff = stand_in_cutoff
        if self.stand_in_cutoff and self.cutoff is None:
            raise ValueError("Cutoff must be specified if standardize is True.")
        self.radius = radius
        # self.a0 = nn.Parameter(torch.ones((1.0,)) * self.radius)

        self.basis_func = []
        self.stand_coeff = []
        for n, l in RadialOrbitalBasis.nl_list:
            r_nl = self._R_nl(n, l, self.radius)
            self.basis_func.append(r_nl)
            if self.stand_in_cutoff:
                self.stand_coeff.append(self._standardized_coeff(r_nl).requires_grad_(True))

    def _R_nl(self, nq: int, lq: int, a0: float = 0.529) -> Callable[[Tensor | float], Tensor | float]:
        """RBF with the associated Laguerre polynomial.

        Args:
            nq (int): principal quantum number.
            lq (int): azimuthal quantum number.
            a0 (float): bohr radius.

        Returns:
            r_nl (Callable[[Tensor | float], Tensor | float]): Orbital Basis function.
        """
        x = sym.Symbol("x", real=True)
        # modify in n, l parameter
        # ref: https://zenn.dev/shittoku_xxx/articles/13afd6fdfac44e
        assoc_lag_coeff = sym.lambdify(
            [x],
            sym.simplify(
                sym.assoc_laguerre(nq - lq - 1, 2 * lq + 1, x) * sym.factorial(nq + lq) * (-1) ** (2 * lq + 1)
            ),
        )
        if self.stand_in_cutoff:
            stand_coeff = -2.0 / nq / a0
        else:
            # standardize in all space
            stand_coeff = -math.sqrt(
                (2.0 / nq / a0) ** 3 * math.factorial(nq - lq - 1) / 2.0 / nq / math.factorial(nq + lq) ** 3
            )

        def r_nl(r: Tensor | float) -> Tensor | float:
            zeta = 2.0 / nq / a0 * r

            if isinstance(r, float):
                return stand_coeff * assoc_lag_coeff(zeta) * zeta**lq * math.exp(-zeta / 2.0)

            return stand_coeff * assoc_lag_coeff(zeta) * torch.pow(zeta, lq) * torch.exp(-zeta / 2.0)  # type: ignore

        return r_nl

    def _standardized_coeff(self, func: Callable[[Tensor | float], Tensor | float]) -> Tensor:
        cutoff = self.cutoff if self.cutoff is not None else 10.0
        with torch.no_grad():

            def interad_func(r):
                return (r * func(r)) ** 2

            inte = quad(interad_func, 0.0, cutoff)
            return 1 / (torch.sqrt(torch.tensor([inte[0]])) + 1e-12)

    def forward(self, dist: Tensor) -> Tensor:
        """Forward calculation of SlaterOrbBasis.

        Args:
            dist (Tensor): atomic distances with (n_edge) shape.

        Returns:
            rbf (Tensor): rbf with (n_edge, n_orb) shape.
        """
        if self.stand_in_cutoff:
            device = dist.device
            rbf = torch.stack([f(dist) * st.to(device) for f, st in zip(self.basis_func, self.stand_coeff)], dim=1)
        else:
            rbf = torch.stack([f(dist) for f in self.basis_func], dim=1)  # type: ignore
        return rbf


def _y00(theta: Tensor, phi: Tensor) -> Tensor:
    r"""
    Spherical Harmonics with `l=m=0`.
    ..math::
        Y_0^0 = \frac{1}{2} \sqrt{\frac{1}{\pi}}

    Args:
        theta: the azimuthal angle.
        phi: the polar angle.

    Returns:
        `Y_0^0`: the spherical harmonics with `l=m=0`.
    """
    dtype = theta.dtype
    return (0.5 * torch.ones_like(theta) * math.sqrt(1.0 / pi)).to(dtype)


class SphericalHarmonicsBasis(nn.Module):
    """Layer that expand inter atomic distances and angles in spherical
    harmonics function.

    Args:
    """

    def __init__(self, cutoff: float | None = None, stand_in_cutoff: bool = True, radius: float = 0.529):
        super().__init__()
        self.radial_basis = RadialOrbitalBasis(cutoff=cutoff, stand_in_cutoff=stand_in_cutoff, radius=radius)

        # make spherical basis functions
        self.sph_funcs = self._calculate_symbolic_sh_funcs()

    def _calculate_symbolic_sh_funcs(self) -> list:
        funcs = []
        theta, phi = sym.symbols("theta phi")
        modules = {"sin": torch.sin, "cos": torch.cos, "conjugate": torch.conj, "sqrt": torch.sqrt, "exp": torch.exp}
        for nl in RadialOrbitalBasis.nl_list:
            # !! only zero m is used
            m_list = [0]
            for m in m_list:
                if nl[1] == 0:
                    funcs.append(_y00)
                else:
                    func = sym.functions.special.spherical_harmonics.Znm(nl[1], m, theta, phi).expand(func=True)
                    func = sym.simplify(func).evalf()
                    funcs.append(sym.lambdify([theta, phi], func, modules))
        self.orig_funcs = funcs

        return funcs

    def forward(self, dist: Tensor, angle: Tensor, edge_idx_kj: torch.LongTensor) -> Tensor:
        """Compute expanded distances and angles with Bessel spherical and
        radial basis.

        Args:
            dist (Tensor): interatomic distances of (n_edge) shape.
            angle (Tensor): angles of triplets of (n_triplets) shape.
            edge_idx_kj (torch.LongTensor): edge index from atom k to j of (n_triplets) shape.

        Returns:
            Tensor: expanded distances and angles of (n_triplets, n_orb) shape.
        """
        # (n_edge, n_orb)
        rob = self.radial_basis(dist)
        # (n_triplets, n_orb)
        shb = torch.stack([f(angle, None) for f in self.sph_funcs], dim=1)

        # (n_triplets, n_orb)
        return rob[edge_idx_kj] * shb


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
    ):
        super().__init__()
        self.elec = ELEC_TABLE_BIG.to(device)
        self.n_orb = RadialOrbitalBasis.n_orb
        self.embed_dim = embed_dim
        self.elec_embeds = nn.ModuleList([nn.Embedding(m, embed_dim, padding_idx=0) for m in MAX_IDX_BIG])
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

    def forward(
        self,
        x: Tensor,
        robs: Tensor,
        shbs: Tensor,
        coeffs: Tensor,
        edge_index: Tensor,
        idx_j: Tensor,
        idx_k: Tensor,
        edge_idx_kj: torch.LongTensor,
        edge_idx_ji: torch.LongTensor,
    ) -> Tensor:
        edge_src, edge_dst = edge_index[0], edge_index[1]

        x = self.node_lin(x)

        coeffs = self.coeffs_lin(coeffs)

        # triple conv
        coeffs_kj = coeffs[idx_k] * coeffs[idx_j] + coeffs[idx_k]
        coeffs_kj = F.normalize(coeffs_kj, dim=-1)
        triple_weight = (shbs * robs[edge_idx_kj]).unsqueeze(-1) * coeffs_kj
        triple_weight = scatter(triple_weight, edge_idx_ji, dim=0, dim_size=robs.size(0))

        coeffs_ji = coeffs[edge_dst] * coeffs[edge_src] + coeffs[edge_dst]
        coeffs_ji = coeffs_ji * triple_weight
        coeffs_ji = F.normalize(coeffs_ji, dim=-1)
        # LCAO conv: summation of all orbitals multiplied by coefficient vectors
        robs = torch.einsum("ed,edh->eh", robs, coeffs_ji)
        robs = F.normalize(robs, dim=-1)

        return self.up_lin(scatter(x[edge_dst] * robs, edge_src, dim=0))


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


class LCAOSpheNet(BaseGNN):
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
        standardize_in_cutoff: bool = True,
        bohr_radius: float = 0.529,
        max_z: int = 100,
        aggr: str = "sum",
        device: str = "cpu",
    ):
        super().__init__()
        wi: Callable[[Tensor], Tensor] | None = init_resolver(weight_init) if weight_init is not None else None
        act: nn.Module = activation_resolver(activation)

        self.hidden_dim = hidden_dim
        self.down_dim = down_dim
        self.coeffs_dim = coeffs_dim
        self.out_dim = out_dim
        self.n_conv_layer = n_conv_layer
        self.device = device

        # Radial basis
        self.rob = RadialOrbitalBasis(cutoff, radius=bohr_radius, stand_in_cutoff=standardize_in_cutoff)
        # Sphereical harmonics basis
        self.shb = SphericalHarmonicsBasis(cutoff, radius=bohr_radius, stand_in_cutoff=standardize_in_cutoff)

        # z and elec embedding layers
        self.node_z_embed_dim = 32  # fix
        self.node_elec_embed_dim = 32  # fix
        self.elec_embed_dim = self.coeffs_dim + self.node_elec_embed_dim
        self.z_embed_dim = self.elec_embed_dim + self.node_z_embed_dim
        self.z_embed = EmbedZ(embed_dim=self.z_embed_dim, max_z=max_z)
        self.elec_embed = EmbedElec(embed_dim=self.elec_embed_dim, device=device)
        self.node_embed = EmbedNode(hidden_dim, self.node_z_embed_dim, self.node_elec_embed_dim, act, wi)

        self.conv_layers = nn.ModuleList(
            [LCAOConv(hidden_dim, down_dim, coeffs_dim, act, wi) for _ in range(n_conv_layer)]
        )

        self.out_layer = LCAOOut(hidden_dim, out_dim, act, wi, aggr=aggr)

    def forward(self, batch: Batch) -> Tensor:
        batch_idx: Tensor | None = batch.get(DataKeys.Batch_idx)
        pos = batch[DataKeys.Position]
        z = batch[DataKeys.Atom_numbers]
        edge_idx = batch[DataKeys.Edge_idx]

        # calc atomic distances
        distances = self.calc_atomic_distances(batch)

        # get index and triplets
        (
            _,
            _,
            tri_idx_i,
            tri_idx_j,
            tri_idx_k,
            edge_idx_kj,
            edge_idx_ji,
        ) = self.get_triplets(batch)

        # calc angle each triplets
        # arctan is more stable than arccos
        pos_i = pos[tri_idx_i]
        pos_ji, pos_ki = pos[tri_idx_j] - pos_i, pos[tri_idx_k] - pos_i
        inner = (pos_ji * pos_ki).sum(dim=-1)
        outter = torch.cross(pos_ji, pos_ki).norm(dim=-1)
        angle = torch.atan2(outter, inner)

        # calc rbf values
        robs = self.rob(distances)
        shbs = self.shb(distances, angle, edge_idx_kj)

        # calc node and coefficent vectors
        z_embed = self.z_embed(z)
        z_embed1, z_embed2 = torch.split(z_embed, [self.elec_embed_dim, self.node_z_embed_dim], dim=-1)

        elec_embed = self.elec_embed(z, z_embed1)
        elec_embed1, elec_embed2 = torch.split(elec_embed, [self.coeffs_dim, self.node_elec_embed_dim], dim=-1)
        x = self.node_embed(z_embed2, elec_embed2)

        # conv layers
        for conv in self.conv_layers:
            x = x + conv(x, robs, shbs, elec_embed1, edge_idx, tri_idx_j, tri_idx_k, edge_idx_kj, edge_idx_ji)

        # out layer
        out = self.out_layer(x, batch_idx)

        return out
