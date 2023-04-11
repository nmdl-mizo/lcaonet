from __future__ import annotations

import math
from collections.abc import Callable

import sympy as sym
import torch
import torch.nn as nn
from scipy.integrate import quad
from torch import Tensor

from lcaonet.atomistic.info import ElecInfo


class BaseRadialBasis(nn.Module):
    def __init__(self, cutoff: float | None, elec_info: ElecInfo):
        super().__init__()
        self.cutoff = cutoff
        self.elec_info = elec_info

    def extra_repr(self) -> str:
        return "cutoff={}, elec_info={}(max_z={}, n_orb={}, n_per_orb={})".format(
            self.cutoff,
            self.elec_info.__class__.__name__,
            self.elec_info.max_z,
            self.elec_info.n_orb,
            self.elec_info.n_per_orb,
        )


class HydrogenRadialBasis(BaseRadialBasis):
    """The layer that expand the interatomic distance with the radial
    wavefunctions of hydrogen."""

    def __init__(
        self,
        cutoff: float | None,
        elec_info: ElecInfo,
        bohr_radius: float = 0.529,
    ):
        """
        Args:
            cutoff (float | None, optional): the cutoff radius.
            elec_info (lcaonet.atomistic.info.ElecInfo): the object that contains the information about the number of electrons.
            bohr_radius (float | None, optional): the bohr radius. Defaults to `0.529`.
        """  # NOQA: E501
        super().__init__(cutoff, elec_info)
        self.cutoff = cutoff
        self.n_orb = elec_info.n_orb
        self.bohr_radius = bohr_radius

        self.basis_func = []
        self.stand_coeff = []
        for nl in elec_info.nl_list:
            r_nl = self._get_r_nl(nl[0].item(), nl[1].item(), self.bohr_radius)
            self.basis_func.append(r_nl)
            if self.cutoff is not None:
                self.stand_coeff.append(self._get_standardized_coeff(r_nl).requires_grad_(True))

    def _get_r_nl(self, nq: int, lq: int, r0: float = 0.529) -> Callable[[Tensor | float], Tensor | float]:
        """Get RadialOrbitalBasis functions with the associated Laguerre
        polynomial.

        Args:
            nq (int): principal quantum number.
            lq (int): azimuthal quantum number.
            r0 (float): bohr radius. Defaults to `0.529`.

        Returns:
            r_nl (Callable[[Tensor | float], Tensor | float]): Orbital Basis function.
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
            stand_coeff = -2.0 / nq / r0
        else:
            # standardize in all space
            stand_coeff = -math.sqrt(
                (2.0 / nq / r0) ** 3 * math.factorial(nq - lq - 1) / 2.0 / nq / math.factorial(nq + lq) ** 3
            )

        def r_nl(r: Tensor | float) -> Tensor | float:
            zeta = 2.0 / nq / r0 * r

            if isinstance(r, float):
                return stand_coeff * assoc_lag_coeff(zeta) * zeta**lq * math.exp(-zeta / 2.0)

            return stand_coeff * assoc_lag_coeff(zeta) * torch.pow(zeta, lq) * torch.exp(-zeta / 2.0)  # type: ignore

        return r_nl

    def _get_standardized_coeff(self, func: Callable[[Tensor | float], Tensor | float]) -> Tensor:
        """If a cutoff radius is specified, the standardization coefficient is
        computed by numerical integration.

        Args:
            func (Callable[[Tensor | float], Tensor | float]): radial wave function.

        Raises:
            ValueError: Occurs when cutoff radius is not specified.

        Returns:
            torch.Tensor: Standardization coefficient such that the probability of existence
                within the cutoff sphere is 1.
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
        """Forward calculation of RadialOrbitalBasis.

        Args:
            d (torch.Tensor): the interatomic distance with (n_edge) shape.

        Returns:
            rb (torch.Tensor): the expanded distance with HydrogenRadialBasis with (n_edge, n_orb) shape.
        """
        if self.cutoff is not None:
            rb = torch.stack([f(d) * sc.to(d.device) for f, sc in zip(self.basis_func, self.stand_coeff)], dim=1)
        else:
            rb = torch.stack([f(d) for f in self.basis_func], dim=1)  # type: ignore
        return rb
