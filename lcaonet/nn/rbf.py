from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import sympy as sym
import torch
import torch.nn as nn
from scipy.integrate import quad
from torch import Tensor

from ..atomistic.info import ElecInfo
from ..nn.cutoff import BaseCutoff


class BaseRadialBasis(nn.Module):
    def __init__(self, cutoff: float, elec_info: ElecInfo, cutoff_net: BaseCutoff):
        super().__init__()
        self.cutoff = cutoff
        self.elec_info = elec_info
        self.cutoff_net = cutoff_net

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
        cutoff: float,
        elec_info: ElecInfo,
        cutoff_net: BaseCutoff,
        bohr_radius: float = 0.529,
        integral_norm: bool = False,
    ):
        """
        Args:
            cutoff (float, optional): the cutoff radius.
            elec_info (lcaonet.atomistic.info.ElecInfo): the object that contains the information about the number of electrons.
            cutoff_net (torch.nn.Module): torch.nn.Moduel of the cutoff function.
            bohr_radius (float | None, optional): the bohr radius. Defaults to `0.529`.
            integral_norm (bool, optional): If True, the standardization coefficient is computed by numerical integration. Defaults to `False`.
        """  # noqa: E501
        super().__init__(cutoff, elec_info, cutoff_net)
        self.n_orb = elec_info.n_orb
        self.bohr_radius = bohr_radius
        self.integral_norm = integral_norm

        self.basis_func = []
        self.stand_coeff = []
        for nl in elec_info.nl_list:
            r_nl = self._get_r_nl(nl[0].item(), nl[1].item(), self.bohr_radius)
            self.basis_func.append(r_nl)
            if self.integral_norm:
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
        if self.integral_norm:
            stand_coeff = -2.0 / nq / r0
        else:
            # standardize in all space
            stand_coeff = -math.sqrt(
                (2.0 / nq / r0) ** 3 * math.factorial(nq - lq - 1) / 2.0 / nq / math.factorial(nq + lq) ** 3
            )

        def r_nl(r: Tensor | float) -> Tensor | float:
            zeta = 2.0 / nq / r0 * r

            if isinstance(r, float):
                cw = float(self.cutoff_net(torch.tensor([r])).item())
                return cw * stand_coeff * assoc_lag_coeff(zeta) * zeta**lq * math.exp(-zeta / 2.0)

            return self.cutoff_net(r) * stand_coeff * assoc_lag_coeff(zeta) * torch.pow(zeta, lq) * torch.exp(-zeta / 2.0)  # type: ignore # Since mypy cannot determine that the type of zeta is tensor # noqa: E501

        return r_nl

    def _get_standardized_coeff(self, func: Callable[[Tensor | float], Tensor | float]) -> Tensor:
        """If integral_norm=True, the standardization coefficient is computed
        by numerical integration.

        Args:
            func (Callable[[Tensor | float], Tensor | float]): radial wave function.

        Raises:
            ValueError: Occurs when cutoff radius is not specified.

        Returns:
            torch.Tensor: Standardization coefficient such that the probability of existence
                within the cutoff sphere is 1.
        """
        with torch.no_grad():

            def interad_func(r):
                return (r * func(r)) ** 2

            inte = quad(interad_func, 0.0, self.cutoff)
            return 1 / (torch.sqrt(torch.tensor([inte[0]])) + 1e-12)

    def forward(self, r: Tensor) -> Tensor:
        """Forward calculation of RadialOrbitalBasis.

        Args:
            r (torch.Tensor): the interatomic distance with (E) shape.

        Returns:
            rb (torch.Tensor): the expanded distance with HydrogenRadialBasis with (E, n_orb) shape.
        """
        if self.integral_norm:
            rb = torch.stack([f(r) * sc.to(r.device) for f, sc in zip(self.basis_func, self.stand_coeff)], dim=1)
        else:
            rb = torch.stack([f(r) for f in self.basis_func], dim=1)  # type: ignore # Since mypy cannot determine that the return type of a function is tensor # noqa: E501
        return rb


class SphericalBesselRadialBasis(BaseRadialBasis):
    """Layer to compute the basis of the spherical Bessel functions that decay
    in the cutoff sphere."""

    def __init__(self, cutoff: float, elec_info: ElecInfo, cutoff_net: BaseCutoff):
        """
        Args:
            cutoff (float): cutoff radius.
            elec_info (lcaonet.atomistic.info.ElecInfo): the object that contains the information about the number of electrons.
            cutoff_net (torch.nn.Module): torch.nn.Moduel of the cutoff function.
        """  # noqa: E501
        super().__init__(cutoff, elec_info, cutoff_net)
        self.n_orb = elec_info.n_orb
        self.basis_func = []
        for nl in elec_info.nl_list:
            r_nl = self._get_r_nl(nl[0].item(), nl[1].item())
            self.basis_func.append(r_nl)

    def _get_r_nl(self, nq: int, lq: int) -> Callable[[Tensor], Tensor]:
        freq = np.pi * nq

        def r_nl(r: Tensor) -> Tensor:
            return self.cutoff_net(r) * (freq * r / self.cutoff).sin() / r

        return r_nl

    def forward(self, r: Tensor) -> Tensor:
        r"""Forward calculation of SphericalBesselBasis.

        Args:
            r (torch.Tensor): inter atomic distance with (E) shape.

        Returns:
            sbb (torch.Tensor): the spherical bessel basis functions with (E, n_orb) shape.
        """
        sbb = torch.stack([f(r) for f in self.basis_func], dim=1)  # type: ignore # Since mypy cannot determine that the return type of a function is tensor # noqa: E501

        return sbb
