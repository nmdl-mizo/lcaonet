from __future__ import annotations

import math
from math import pi

import sympy as sym
import torch
import torch.nn as nn
from torch import Tensor

from lcaonet.atomistic.info import ElecInfo


class SphericalHarmonicsBasis(nn.Module):
    """The layer that expand three body angles with spherical harmonics
    functions."""

    def __init__(self, elec_info: ElecInfo):
        """
        Args:
            elec_info (lcaonet.atomistic.info.ElecInfo): the object that contains the information about the number of electrons.
        """  # NOQA: E501
        super().__init__()
        self.elec_info = elec_info
        self.sph_funcs = self._calculate_symbolic_sh_funcs()

    @staticmethod
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

    def _calculate_symbolic_sh_funcs(self) -> list:
        """Calculate symbolic spherical harmonics functions.

        Returns:
            funcs (list[Callable]): the list of spherical harmonics functions.
        """
        funcs = []
        theta, phi = sym.symbols("theta phi")
        modules = {"sin": torch.sin, "cos": torch.cos, "conjugate": torch.conj, "sqrt": torch.sqrt, "exp": torch.exp}
        for nl in self.elec_info.nl_list:
            # !! only m=zero is used
            m_list = [0]
            for m in m_list:
                if nl[1] == 0:
                    funcs.append(SphericalHarmonicsBasis._y00)
                else:
                    func = sym.expand_func(sym.functions.special.spherical_harmonics.Znm(nl[1].item(), m, theta, phi))
                    func = sym.simplify(func).evalf()
                    funcs.append(sym.lambdify([theta, phi], func, modules))

        return funcs

    def extra_repr(self) -> str:
        return "elec_info={}(max_z={}, n_orb={}, n_per_orb={})".format(
            self.elec_info.__class__.__name__,
            self.elec_info.max_z,
            self.elec_info.n_orb,
            self.elec_info.n_per_orb,
        )

    def forward(self, angle: Tensor) -> Tensor:
        """Forward calculation of SphericalHarmonicsBasis.

        Args:
            angle (torch.Tensor): the angles of triplets with (n_triplets) shape.

        Returns:
            shb (torch.Tensor): the expanded angles with SphericalHarmonicsFunctions with (n_triplets, n_orb) shape.
        """
        # (n_triplets, n_orb)
        shb = torch.stack([f(angle, None) for f in self.sph_funcs], dim=1)

        return shb
