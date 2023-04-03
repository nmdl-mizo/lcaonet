from __future__ import annotations

import math
from collections.abc import Callable
from typing import Union

import sympy as sym
import torch
import torch.nn as nn
from scipy.integrate import quad
from torch import Tensor

from lcaonet.atomistic.info import BaseAtomisticInformation


class BaseRadialBasis(nn.Module):
    limit_n_orb: int = 18

    def __init__(self, cutoff: float | None, atom_info: BaseAtomisticInformation):
        super().__init__()
        self.cutoff = cutoff
        self.atom_info = atom_info
        self.n_orb = atom_info.n_orb
        self.n_l_list = atom_info.get_nl_list

    def extra_repr(self) -> str:
        return "cutoff={}, atom_info={}(max_z={},n_orb={})".format(
            self.cutoff, self.atom_info.__class__.__name__, self.atom_info.max_z, self.n_orb
        )


class HydrogenRadialWaveFunctionBasis(BaseRadialBasis):
    limit_n_orb: int = 18
    basis_type = Callable[[Union[Tensor, float]], Union[Tensor, float]]
    """The layer that expand the interatomic distance with the radial
    wavefunctions of hydrogen."""

    def __init__(
        self,
        cutoff: float | None,
        atom_info: BaseAtomisticInformation,
        bohr_radius: float = 0.529,
    ):
        """
        Args:
            cutoff (float | None): the cutoff radius.
            atom_info (lcaonet.atomistic.info.BaseAtomisticInformation): the atomistic information.
            bohr_radius (float | None, optional): the bohr radius. Defaults to `0.529`.
        """
        super().__init__(cutoff, atom_info)
        self.bohr_radius = bohr_radius
        self.max_z = self.atom_info.max_z

        self.basis_func: list[HydrogenRadialWaveFunctionBasis.basis_type] = []
        self.normalize_coeff: dict[tuple[int, int], Tensor] = {}
        for n, l in self.n_l_list:
            self.normalize_coeff[(n, l)] = torch.ones(1)
            r_nl = self._get_r_nl(n, l, self.bohr_radius)
            self.basis_func.append(r_nl)
            if self.cutoff is not None:
                self.normalize_coeff[(n, l)] = self._get_normalize_coeff(r_nl)
                self.normalize_coeff[(n, l)].requires_grad_(True)

    def _get_r_nl(self, nq: int, lq: int, rb: float = 0.529) -> HydrogenRadialWaveFunctionBasis.basis_type:
        """Get HydrogenRadialWaveFunctionBasis functions with the associated
        Laguerre polynomial.

        Args:
            nq (int): principal quantum number.
            lq (int): azimuthal quantum number.
            rb (float): bohr radius. Defaults to `0.529`.

        Returns:
            r_nl (Callable[[Tensor | float], Tensor | float]): Basis functions based on hydrogen wave fucntions.
        """
        x = sym.Symbol("x", real=True)
        # modify for n, l parameter
        # ref: https://zenn.dev/shittoku_xxx/articles/13afd6fdfac44e
        assoc_lag_coeff = sym.lambdify(
            [x],
            sym.simplify(
                sym.assoc_laguerre(nq - lq - 1, 2 * lq + 1, x) * sym.factorial(nq + lq) * (-1) ** (2 * lq + 1)
            ),
        )
        norm_nl = self.normalize_coeff[(nq, lq)]

        def r_nl(r: Tensor | float) -> Tensor | float:
            # funcs
            sqrt = math.sqrt
            power = pow if isinstance(r, float) else torch.pow
            exp = math.exp if isinstance(r, float) else torch.exp

            r0 = rb

            if self.cutoff is not None:
                # normalize in the cutoff radius
                normal_coeff = -2.0 / nq / r0
                if isinstance(r, float):
                    normal_coeff *= norm_nl.item()
                else:
                    normal_coeff = normal_coeff * norm_nl.to(r.device)  # type: ignore
            else:
                # normalize in all space
                normal_coeff = -sqrt(  # type: ignore
                    (2.0 / nq / r0) ** 3 * math.factorial(nq - lq - 1) / 2.0 / nq / math.factorial(nq + lq) ** 3
                )

            zeta = 2.0 / nq / r0 * r

            return normal_coeff * assoc_lag_coeff(zeta) * power(zeta, lq) * exp(-zeta / 2.0)  # type: ignore

        return r_nl

    def _get_normalize_coeff(self, func: HydrogenRadialWaveFunctionBasis.basis_type) -> Tensor:
        """If a cutoff radius is specified, the normalization coefficient is
        computed by numerical integration.

        Args:
            func (Callable[[Tensor | float], [Tensor | int]], Tensor | float]): the hydrogen radial wave function.

        Raises:
            ValueError: Occurs when cutoff radius is not specified.

        Returns:
            torch.Tensor: Normalize coefficient.
        """
        if self.cutoff is None:
            raise ValueError("cutoff is None")
        cutoff = self.cutoff

        with torch.no_grad():

            def interad_func(r):
                return (r * func(r)) ** 2

            inte = quad(interad_func, 0.0, cutoff)
            return 1 / (torch.sqrt(torch.tensor([inte[0]])) + 1e-12)

    def forward(self, d: Tensor) -> Tensor:
        """Forward calculation of HydrogenRadialWaveFunctionBasis.

        Args:
            d (torch.Tensor): the interatomic distance with (n_edge) shape.

        Returns:
            rbf (torch.Tensor): the expanded distance with (n_edge, n_orb) shape.
        """
        rbf = torch.stack([f(d) for f in self.basis_func], dim=1)  # type: ignore
        return rbf


class SlaterOrbitalBasis(BaseRadialBasis):
    # Exponent table is only valid for n_orb <= 15
    limit_n_orb: int = 15
    basis_type = Callable[[Union[Tensor, float], Union[Tensor, int]], Union[Tensor, float]]
    """The layer that expand the interatomic distance with the slater-type orbital functions."""

    def __init__(
        self,
        cutoff: float | None,
        atom_info: BaseAtomisticInformation,
        learnable_exp: bool = False,
    ):
        super().__init__(cutoff, atom_info)
        self.max_z = self.atom_info.max_z
        self.n_map: dict[int, float] = {1: 1.0, 2: 2.0, 3: 3.0, 4: 3.7, 5: 4.0, 6: 4.2}
        if learnable_exp:
            self.exponent = nn.ParameterDict(
                {f"{k[0]}-{k[1]}": v.requires_grad_(True) for k, v in atom_info.get_exponent_dict.items()}
            )
        else:
            self.exponent = nn.ParameterDict(  # type: ignore
                {f"{k[0]}-{k[1]}": v.requires_grad_(False) for k, v in atom_info.get_exponent_dict.items()}
            )

        self.basis_func: list[SlaterOrbitalBasis.basis_type] = []
        self.normalize_coeff: dict[tuple[int, int], Tensor] = {}
        for n, l in self.n_l_list:
            self.normalize_coeff[(n, l)] = torch.ones(self.max_z + 1)
            r_nl = self._get_r_nl(n, l)
            self.basis_func.append(r_nl)
            if self.cutoff is not None:
                for z in range(1, self.max_z + 1):
                    self.normalize_coeff[(n, l)][z] = self._get_normalize_coeff(r_nl, z)
                self.normalize_coeff[(n, l)].requires_grad_(True)

    def _get_r_nl(self, nq: int, lq: int) -> SlaterOrbitalBasis.basis_type:
        norm_nl = self.normalize_coeff[(nq, lq)]
        exponent = self.exponent[f"{nq}-{lq}"]
        n_eff = self.n_map[nq]

        def r_nl(r: Tensor | float, z_j: Tensor | int) -> Tensor | float:
            # funcs
            power = pow if isinstance(r, float) else torch.pow

            rbf: Tensor = norm_nl[z_j] * power(r, n_eff - 1) * torch.exp(-exponent[z_j] * r)  # type: ignore
            return rbf.detach().numpy() if isinstance(r, float) else rbf

        return r_nl

    def _get_normalize_coeff(self, func: SlaterOrbitalBasis.basis_type, z_j: int) -> Tensor:
        """If a cutoff radius is specified, the normalization coefficient is
        computed by numerical integration.

        Args:
            func (Callable[[Tensor | float, Tensor | float], Tensor | float]): the slater radial basis.
            z_j (int): the atomic number.

        Raises:
            ValueError: Occurs when cutoff radius is not specified.

        Returns:
            torch.Tensor: Normalize coefficient.
        """
        if self.cutoff is None:
            raise ValueError("cutoff is None")
        cutoff = self.cutoff

        with torch.no_grad():

            def interad_func(r):
                return (r * func(r, z_j)) ** 2

            inte = quad(interad_func, 0.0, cutoff)
            return 1 / (torch.sqrt(torch.tensor([inte[0]])) + 1e-12)

    def forward(self, d: Tensor, z_j: Tensor) -> Tensor:
        """Forward calculation of HydrogenRadialWaveFunctionBasis.

        Args:
            d (torch.Tensor): the interatomic distance with (n_edge) shape.
            z_j (torch.Tensor): the atomic number with (n_edge) shape.
        Returns:
            rbf (torch.Tensor): the expanded distance with (n_edge, n_orb) shape.
        """
        rbf = torch.stack([f(d, z_j) for f in self.basis_func], dim=1)  # type: ignore
        return rbf


class GaussianOrbitalBasis(BaseRadialBasis):
    """The layer that expand the interatomic distance with the gaussian-type
    orbital functions."""

    def __init__(self, cutoff: float | None, atom_info: BaseAtomisticInformation):
        super().__init__(cutoff, atom_info)