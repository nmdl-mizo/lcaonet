import torch
from torch import Tensor
import torch.nn as nn

from pyggnn.nn.cutoff import EnvelopeCutoff

__all__ = ["BesselSBF"]


class BesselSBF(nn.Module):
    """Bessel Spherical basis functions"""

    def __init__(
        self,
        n_radial: int,
        n_spherical: int,
        cutoff_radi: float = 5.0,
        envelope_exponent: int = 5,
    ):
        """
        Expand inter atomic distances and angles by Bessel spherical and radial basis.

        Args:
            n_radial (int): number of radial basis.
            n_spherical (int): number of spherical basis.
            cutoff_radi (float, optional): cutoff radius. Defaults to `5.0`.
            envelope_exponent (int, optional): exponent of envelope cutoff fucntion.
                Defaults to `5`.
        """
        super().__init__()
        import sympy as sym
        from torch_geometric.nn.models.dimenet_utils import (
            bessel_basis,
            real_sph_harm,
        )

        assert n_radial <= 64, "n_radial must be under 64"
        self.n_radial = n_radial
        self.n_spherical = n_spherical
        self.cutoff_radi = cutoff_radi
        self.envelope = EnvelopeCutoff(cutoff_radi, envelope_exponent)

        bessel_forms = bessel_basis(n_spherical, n_radial)
        sph_harm_forms = real_sph_harm(n_spherical)
        self.sph_funcs = []
        self.bessel_funcs = []

        x, theta = sym.symbols("x theta")
        modules = {"sin": torch.sin, "cos": torch.cos}
        for i in range(n_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta], sph_harm_forms[i][0], modules)(0)
                self.sph_funcs.append(lambda x: torch.zeros_like(x) + sph1)
            else:
                sph = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(sph)
            for j in range(n_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    def forward(
        self,
        dist: Tensor,
        angle: Tensor,
        edge_idx_kj: torch.LongTensor,
    ) -> Tensor:
        """
        Extend distances and angles with Bessel spherical and radial basis.

        Args:
            dist (Tensor): interatomic distance values shape of (n_edge).
            angle (Tensor): angles of triplets shape of (n_triplets).
            edge_idx_kj (torch.LongTensor): edge index from atom k to j
                shape of (n_triplets).

        Returns:
            Tensor: extended distances and angles of
                (n_triplets x (n_spherical x n_radial)) shape.
        """
        dist = dist / self.cutoff_radi
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        # apply cutoff
        rbf = self.envelope(dist).unsqueeze(-1) * rbf

        cbf = torch.stack([f(angle) for f in self.sph_funcs], dim=1)

        n, k = self.n_spherical, self.n_radial
        return (rbf[edge_idx_kj].view(-1, n, k) * cbf.view(-1, n, 1)).view(-1, n * k)
