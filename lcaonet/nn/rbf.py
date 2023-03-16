from __future__ import annotations

import math
from collections.abc import Callable

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


class HydrogenRadialWaveFunctionBasis(BaseRadialBasis):
    limit_n_orb: int = 18
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

        self.basis_func = []
        self.normalize_coeff = []
        for n, l in self.n_l_list:
            r_nl = self._get_r_nl(n, l, self.bohr_radius)
            self.basis_func.append(r_nl)
            if self.cutoff is not None:
                self.normalize_coeff.append(self._get_normalize_coeff(r_nl).requires_grad_(True))

    def _get_r_nl(self, nq: int, lq: int, r0: float = 0.529) -> Callable[[Tensor | float], Tensor | float]:
        """Get HydrogenRadialWaveFunctionBasis functions with the associated
        Laguerre polynomial.

        Args:
            nq (int): principal quantum number.
            lq (int): azimuthal quantum number.
            r0 (float): bohr radius. Defaults to `0.529`.

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
        if self.cutoff is not None:
            normal_coeff = -2.0 / nq / r0
        else:
            # normalize in all space
            normal_coeff = -math.sqrt(
                (2.0 / nq / r0) ** 3 * math.factorial(nq - lq - 1) / 2.0 / nq / math.factorial(nq + lq) ** 3
            )

        def r_nl(r: Tensor | float) -> Tensor | float:
            zeta = 2.0 / nq / r0 * r

            if isinstance(r, float):
                return normal_coeff * assoc_lag_coeff(zeta) * zeta**lq * math.exp(-zeta / 2.0)

            return normal_coeff * assoc_lag_coeff(zeta) * torch.pow(zeta, lq) * torch.exp(-zeta / 2.0)  # type: ignore

        return r_nl

    def _get_normalize_coeff(self, func: Callable[[Tensor | float], Tensor | float]) -> Tensor:
        """If a cutoff radius is specified, the normalization coefficient is
        computed by numerical integration.

        Args:
            func (Callable[[Tensor | float], Tensor | float]): the hydrogen radial wave function.

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
        if self.cutoff is not None:
            device = d.device
            rbf = torch.stack([f(d) * nc.to(device) for f, nc in zip(self.basis_func, self.normalize_coeff)], dim=1)
        else:
            rbf = torch.stack([f(d) for f in self.basis_func], dim=1)  # type: ignore
        return rbf


class SlaterOrbitalBasis(BaseRadialBasis):
    # Exponent table is only valid for n_orb <= 15
    limit_n_orb: int = 15
    """The layer that expand the interatomic distance with the slater-type orbital functions."""

    def __init__(self, cutoff: float | None, atom_info: BaseAtomisticInformation):
        super().__init__(cutoff, atom_info)

    #     self.basis_func = []
    #     self.normalize_coeff = []
    #     for n, l in self.n_l_list:
    #         r_nl = self._get_r_nl(n, l, self.bohr_radius)
    #         self.basis_func.append(r_nl)
    #         if self.cutoff is not None:
    #             self.normalize_coeff.append(self._get_normalize_coeff(r_nl).requires_grad_(True))

    # def _get_normalize_coeff(self, func: Callable[[Tensor | float, Tensor | float], Tensor | float]) -> Tensor:
    #     """If a cutoff radius is specified, the normalization coefficient is
    #     computed by numerical integration.

    #     Args:
    #         func (Callable[[Tensor | float, Tensor | float], Tensor | float]): the slater radial basis.

    #     Raises:
    #         ValueError: Occurs when cutoff radius is not specified.

    #     Returns:
    #         torch.Tensor: Normalize coefficient.
    #     """
    #     if self.cutoff is None:
    #         raise ValueError("cutoff is None")
    #     cutoff = self.cutoff

    #     with torch.no_grad():

    #         def interad_func(r):
    #             return (r * func(r)) ** 2

    #         inte = quad(interad_func, 0.0, cutoff)
    #         return 1 / (torch.sqrt(torch.tensor([inte[0]])) + 1e-12)


class GaussianOrbitalBasis(BaseRadialBasis):
    """The layer that expand the interatomic distance with the gaussian-type
    orbital functions."""

    def __init__(self, cutoff: float | None, atom_info: BaseAtomisticInformation):
        super().__init__(cutoff, atom_info)
