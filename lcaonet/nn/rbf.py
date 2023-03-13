from __future__ import annotations

import math
from collections.abc import Callable

import sympy as sym
import torch
import torch.nn as nn
from scipy.integrate import quad
from torch import Tensor

from lcaonet.elec import get_max_nl_index_byorb, get_max_nl_index_byz, get_nl_list


class HydrogenRadialWaveBasis(nn.Module):
    """The layer that expand the interatomic distance with the radial
    wavefunctions of hydrogen."""

    def __init__(
        self,
        cutoff: float | None = None,
        bohr_radius: float = 0.529,
        max_z: int = 36,
        max_orb: str | None = None,
    ):
        """
        Args:
            cutoff (float | None, optional): the cutoff radius. Defaults to `None`.
            bohr_radius (float | None, optional): the bohr radius. Defaults to `0.529`.
            max_z (int, optional): the maximum atomic number. Defaults to `36`.
            max_orb (str | None, optional): the maximum orbital name like "2p". Defaults to `None`.
        """
        super().__init__()
        # get elec table
        if max_orb is None:
            max_idx = get_max_nl_index_byz(max_z)
        else:
            max_idx = max(get_max_nl_index_byorb(max_orb), get_max_nl_index_byz(max_z))
        self.n_orb = max_idx + 1
        self.n_l_list = get_nl_list(max_idx)
        self.cutoff = cutoff
        self.bohr_radius = bohr_radius

        self.basis_func = []
        self.normalize_coeff = []
        for n, l in self.n_l_list:
            r_nl = self._get_r_nl(n, l, self.bohr_radius)
            self.basis_func.append(r_nl)
            if self.cutoff is not None:
                self.normalize_coeff.append(self._get_normalize_coeff(r_nl).requires_grad_(True))

    def _get_r_nl(self, nq: int, lq: int, r0: float = 0.529) -> Callable[[Tensor | float], Tensor | float]:
        """Get HydrogenRadialWaveBasis functions with the associated Laguerre
        polynomial.

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

    def _get_normalize_coeff(self, func: Callable[[Tensor | float], Tensor | float]) -> Tensor:
        """If a cutoff radius is specified, the normalization coefficient is
        computed by numerical integration.

        Args:
            func (Callable[[Tensor | float], Tensor | float]): hydrogen radial wave function.

        Raises:
            ValueError: Occurs when cutoff radius is not specified.

        Returns:
            torch.Tensor: Normalize coefficient such that the probability of existence
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

    def forward(self, dist: Tensor) -> Tensor:
        """Forward calculation of HydrogenRadialWaveBasis.

        Args:
            dist (torch.Tensor): the interatomic distance with (n_edge) shape.

        Returns:
            rbf (torch.Tensor): the expanded distance with (n_edge, n_orb) shape.
        """
        if self.cutoff is not None:
            device = dist.device
            rbf = torch.stack([f(dist) * nc.to(device) for f, nc in zip(self.basis_func, self.normalize_coeff)], dim=1)
        else:
            rbf = torch.stack([f(dist) for f in self.basis_func], dim=1)  # type: ignore
        return rbf


class SlaterOrbitalBasis(nn.Module):
    """The layer that expand the interatomic distance with the slater orbital
    functions."""

    def __init__(
        self,
        cutoff: float | None = None,
        max_z: int = 36,
        max_orb: str | None = None,
    ):
        """
        Args:
            cutoff (float | None, optional): the cutoff radius. Defaults to `None`.
            max_z (int, optional): the maximum atomic number. Defaults to `36`.
            max_orb (str | None, optional): the maximum orbital name like "2p". Defaults to `None`.
        """
        super().__init__()
        # get elec table
        if max_orb is None:
            max_idx = get_max_nl_index_byz(max_z)
        else:
            max_idx = max(get_max_nl_index_byorb(max_orb), get_max_nl_index_byz(max_z))
        self.n_orb = max_idx + 1
        self.n_l_list = get_nl_list(max_idx)
        self.cutoff = cutoff

        self.basis_func = []
        self.normalize_coeff = []
        for n, l in self.n_l_list:
            r_nl = self._get_r_nl(n, l)
            self.basis_func.append(r_nl)
            if self.cutoff is not None:
                self.normalize_coeff.append(self._get_normalize_coeff(r_nl).requires_grad_(True))

    def _get_r_nl(self, nq: int, lq: int, r0: float = 0.592) -> Callable[[Tensor | float], Tensor | float]:
        """Get SlaterOrbitalBasis functions with the associated Laguerre
        polynomial.

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

    def _get_normalize_coeff(self, func: Callable[[Tensor | float], Tensor | float]) -> Tensor:
        """If a cutoff radius is specified, the normalization coefficient is
        computed by numerical integration.

        Args:
            func (Callable[[Tensor | float], Tensor | float]): hydrogen radial wave function.

        Raises:
            ValueError: Occurs when cutoff radius is not specified.

        Returns:
            torch.Tensor: Normalize coefficient such that the probability of existence
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

    def forward(self, dist: Tensor) -> Tensor:
        """Forward calculation of SlaterOrbitalBasis.

        Args:
            dist (torch.Tensor): the interatomic distance with (n_edge) shape.

        Returns:
            rbf (torch.Tensor): the expanded distance with (n_edge, n_orb) shape.
        """
        if self.cutoff is not None:
            device = dist.device
            rbf = torch.stack([f(dist) * nc.to(device) for f, nc in zip(self.basis_func, self.normalize_coeff)], dim=1)
        else:
            rbf = torch.stack([f(dist) for f in self.basis_func], dim=1)  # type: ignore
        return rbf


class GaussianOrbitalBasis(nn.Module):
    """The layer that expand the interatomic distance with the gaussian orbital
    functions."""

    def __init__(
        self,
        cutoff: float | None = None,
        max_z: int = 36,
        max_orb: str | None = None,
    ):
        """
        Args:
            cutoff (float | None, optional): the cutoff radius. Defaults to `None`.
            max_z (int, optional): the maximum atomic number. Defaults to `36`.
            max_orb (str | None, optional): the maximum orbital name like "2p". Defaults to `None`.
        """
        super().__init__()
        # get elec table
        if max_orb is None:
            max_idx = get_max_nl_index_byz(max_z)
        else:
            max_idx = max(get_max_nl_index_byorb(max_orb), get_max_nl_index_byz(max_z))
        self.n_orb = max_idx + 1
        self.n_l_list = get_nl_list(max_idx)
        self.cutoff = cutoff

        self.basis_func = []
        self.normalize_coeff = []
        for n, l in self.n_l_list:
            r_nl = self._get_r_nl(n, l)
            self.basis_func.append(r_nl)
            if self.cutoff is not None:
                self.normalize_coeff.append(self._get_normalize_coeff(r_nl).requires_grad_(True))

    def _get_r_nl(self, nq: int, lq: int, r0: float = 0.529) -> Callable[[Tensor | float], Tensor | float]:
        """Get GaussianOrbitalBasis functions with the associated Laguerre
        polynomial.

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

    def _get_normalize_coeff(self, func: Callable[[Tensor | float], Tensor | float]) -> Tensor:
        """If a cutoff radius is specified, the normalization coefficient is
        computed by numerical integration.

        Args:
            func (Callable[[Tensor | float], Tensor | float]): hydrogen radial wave function.

        Raises:
            ValueError: Occurs when cutoff radius is not specified.

        Returns:
            torch.Tensor: Normalize coefficient such that the probability of existence
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

    def forward(self, dist: Tensor) -> Tensor:
        """Forward calculation of GaussianOrbitalBasis.

        Args:
            dist (torch.Tensor): the interatomic distance with (n_edge) shape.

        Returns:
            rbf (torch.Tensor): the expanded distance with (n_edge, n_orb) shape.
        """
        if self.cutoff is not None:
            device = dist.device
            rbf = torch.stack([f(dist) * nc.to(device) for f, nc in zip(self.basis_func, self.normalize_coeff)], dim=1)
        else:
            rbf = torch.stack([f(dist) for f in self.basis_func], dim=1)  # type: ignore
        return rbf
