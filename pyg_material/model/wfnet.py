from __future__ import annotations  # type: ignore

import math
from collections.abc import Callable

import torch
import torch.nn as nn
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


def R_nl(nq: int, lq: int) -> Callable[[Tensor, Tensor], Tensor]:
    def r_nl(r: Tensor, a0: Tensor) -> Tensor:
        # device = r.device

        zeta = 2.0 / nq / a0 * r
        # # Standardization in all spaces
        # f_coeff = -(
        #     ((2.0 / nq / a0) ** 3 * factorial(nq - lq - 1).to(device) / 2.0 / nq / (factorial(nq + lq).to(device) ** 3)) # NOQA: E501
        #     ** (1.0 / 2.0)
        # )
        # no standardization
        f_coeff = 2.0 / nq / a0

        al = assoc_laguerre(zeta, nq + lq, 2 * lq + 1)
        f = f_coeff * torch.exp(-zeta / 2.0) * zeta**lq * al

        return f

    return r_nl


class RadialWaveBasis(nn.Module):
    n_radial: int = ELEC_DICT.size(-1)

    def __init__(self, cutoff: float | None = None, radius: float = 0.529, learnable: bool = False):
        super().__init__()
        self.cutoff = cutoff
        self.radius = radius
        self.learnable = learnable
        if self.learnable:
            self.a0 = nn.Parameter(torch.ones((1,)) * self.radius)
        else:
            self.register_buffer("a0", torch.ones((1,)) * self.radius)

        nl_list = [
            (1, 0),  # 1s
            (2, 0),  # 2s
            (2, 1),  # 2p
            (3, 0),  # 3s
            (3, 1),  # 3p
            (4, 0),  # 4s
            (3, 2),  # 3d
            (4, 1),  # 4p
            (5, 0),  # 5s
            (4, 2),  # 4d
            (5, 1),  # 5p
            (6, 0),  # 6s
            (4, 3),  # 4f
            (5, 2),  # 5d
            (6, 1),  # 6p
            (7, 0),  # 7s
            (5, 3),  # 5f
            (6, 2),  # 6d
        ]
        self.basis_func = []
        for n, l in nl_list:
            r_nl = R_nl(n, l)
            self.basis_func.append(r_nl)

    def forward(self, dist: Tensor) -> Tensor:
        rbf = torch.stack([f(dist, self.a0) for f in self.basis_func], dim=1)
        return rbf


class EmbedCoeffs(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        device: str,
        max_z: int = 37,
        n_orb: int = RadialWaveBasis.n_radial,
    ):
        super().__init__()
        self.elec = ELEC_DICT.to(device)
        self.n_orb = n_orb
        self.embed_dim = embed_dim
        self.z_embed = nn.Embedding(max_z, embed_dim, padding_idx=0)
        self.coeff_embeds = nn.ModuleList([nn.Embedding(m, embed_dim, padding_idx=0) for m in MAX_IND])

    def forward(self, z: Tensor) -> Tensor:
        """
        Args:
            z (torch.Tensor): atomic numbers of (n_node) shape.

        Returns:
            coeffs (torch.Tensor): coefficent vectors of (n_node, n_orb, dim) shape.
        """
        # (n_node) -> (n_node, n_orb)
        elec = self.elec[z]
        # (n_orb, n_node)
        elec = torch.transpose(elec, 0, 1)
        # (n_orb, n_node, dim)
        coeffs = torch.stack([ce(elec[i]) for i, ce in enumerate(self.coeff_embeds)], dim=0)
        # (n_node, n_orb, dim)
        coeffs = torch.transpose(coeffs, 0, 1)

        # (n_node) -> (n_node, 1, dim)
        z = self.z_embed(z).unsqueeze(1)
        # inject atomic information to coefficent vectors
        coeffs = coeffs + coeffs * z

        return coeffs


class EmbedZ(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        coeffs_dim: int,
        activation: nn.Module = nn.SiLU(),
        max_z: int = 100,
        weight_init: Callable[[Tensor], Tensor] | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.coeffs_dim = coeffs_dim

        self.z_embed = nn.Embedding(max_z, hidden_dim)
        # No bias is used to keep coefficient values at 0
        self.coeffs_lin = nn.Sequential(
            activation,
            Dense(coeffs_dim, hidden_dim, False, weight_init),
            activation,
            Dense(hidden_dim, hidden_dim, False, weight_init),
        )

    def forward(self, z: Tensor, coeffs: Tensor) -> Tensor:
        return self.z_embed(z) + self.coeffs_lin(coeffs).sum(1)


class WFConv(nn.Module):
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
        coeffs = coeffs / (torch.sqrt(torch.sum(coeffs**2, dim=1, keepdim=True)) + 1e-12)

        # get rbf values
        rbfs = torch.einsum("ed,edh->eh", rbfs, coeffs)
        rbfs = rbfs / (torch.sqrt(torch.sum(rbfs**2, dim=1, keepdim=True)) + 1e-12)

        return self.up_lin(scatter(x[edge_dst] * rbfs, edge_src, dim=0))


class WFOut(nn.Module):
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


class WFNet(BaseGNN):
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
        max_z: int = 100,
        aggr: str = "sum",
        learnable_rbf: bool = False,
        device: str = "cpu",
    ):
        super().__init__()
        wi: Callable[[Tensor], Tensor] | None = init_resolver(weight_init) if weight_init is not None else None
        act: nn.Module = activation_resolver(activation)

        self.n_conv_layer = n_conv_layer
        self.device = device

        self.coeffs_embed = EmbedCoeffs(coeffs_dim, device, max_z=max_z)
        self.wfrbf = RadialWaveBasis(cutoff, learnable=learnable_rbf)

        self.initial_embed = EmbedZ(hidden_dim, coeffs_dim, act, weight_init=wi, max_z=max_z)

        self.conv_layers = nn.ModuleList(
            [WFConv(hidden_dim, down_dim, coeffs_dim, act, wi) for _ in range(n_conv_layer)]
        )
        self.out_layer = WFOut(hidden_dim, out_dim, act, wi, aggr=aggr)

    def forward(self, batch: Batch) -> Tensor:
        batch_idx: Tensor | None = batch.get(DataKeys.Batch_idx)
        atom_numbers = batch[DataKeys.Atom_numbers]
        edge_idx = batch[DataKeys.Edge_idx]

        # calc atomic distances
        distances = self.calc_atomic_distances(batch)

        # calc rbf values
        rbfs = self.wfrbf(distances)

        # calc coefficent vectors
        coeffs = self.coeffs_embed(atom_numbers)

        # embedding
        x = self.initial_embed(atom_numbers, coeffs)

        # conv layers
        for conv in self.conv_layers:
            x = x + conv(x, edge_idx, rbfs, coeffs)

        # out layer
        out = self.out_layer(x, batch_idx)

        return out
