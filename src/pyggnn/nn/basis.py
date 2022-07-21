from math import pi as PI

import torch
from torch import Tensor
import torch.nn as nn

from pyggnn.nn.cutoff import EnvelopeCutoff

__all__ = ["GaussianRB", "BesselRB", "BesselSB"]


def gaussian_rbf(
    distances: Tensor,
    offsets: Tensor,
    widths: Tensor,
    centered: bool = False,
) -> Tensor:
    """
    Filtered interatomic distance values using Gaussian basis.

    Notes:
        reference:
        [1] https://github.com/atomistic-machine-learning/schnetpack
    """
    if centered:
        # if Gaussian functions are centered, use offsets to compute widths
        eta = 0.5 / torch.pow(offsets, 2)
        # if Gaussian functions are centered, no offset is subtracted
        diff = distances[:, None]

    else:
        # compute width of Gaussian functions (using an overlap of 1 STDDEV)
        eta = 0.5 / torch.pow(widths, 2)
        # Use advanced indexing to compute the individual components
        diff = distances[:, None] - offsets[None, :]

    # compute smear distance values
    filtered_distances = torch.exp(-eta * torch.pow(diff, 2))
    return filtered_distances


class GaussianRB(nn.Module):
    """Gassian radial basis function"""

    def __init__(
        self,
        start: float = 0.0,
        stop: float = 6.0,
        n_gaussian: int = 20,
        centered: bool = False,
        trainable: bool = True,
    ):
        """
        Expand interatomic distance values using Gaussian radial basis.

        Args:
            start (float, optional): start value of shift of Gaussian. Defaults to `0.0`.
            stop (float, optional): end value of shift of Gaussian. Defaults to `6.0`.
            n_gaussian (int, optional): number of gaussian radial basis.
                Defaults to `20`.
            centered (bool, optional): if set to `True`, using 0 centered
                gaussian function. Defaults to `False`.
            trainable (bool, optional): whether gaussian params trainable.
                Defaults to `True`.
        """
        super().__init__()
        offset = torch.linspace(start=start, end=stop, steps=n_gaussian)
        width = torch.tensor((offset[1] - offset[0]) * torch.ones_like(offset))
        self.centered = centered
        if trainable:
            self.width = nn.Parameter(width)
            self.offset = nn.Parameter(offset)
        else:
            self.register_buffer("width", width)
            self.register_buffer("offset", offset)

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, dist: Tensor) -> Tensor:
        """
        Compute filtered distances with Gaussian basis.

        Args:
            dist (Tensor): interatomic distance values of (num_edge) shape.

        Returns:
            Tensor: filtered distances of (num_edge x n_dim) shape.
        """
        return gaussian_rbf(
            dist, offsets=self.offset, widths=self.width, centered=self.centered
        )


class BesselRB(torch.nn.Module):
    """Bessel radial basis functions"""

    def __init__(
        self,
        n_radial: int,
        cutoff_radi: float = 5.0,
        envelope_exponent: int = 5,
    ):
        """
        Expand inter atomic distances by Bessel radial basis.

        Args:
            n_radial (int): number of radial basis.
            cutoff_radi (float, optional): cutoff radius. Defaults to `5.0`.
            envelope_exponent (int, optional): exponent of cutoff envelope function.
                Defaults to `5`.
        """
        super().__init__()
        self.cutoff_radi = cutoff_radi
        self.envelope = EnvelopeCutoff(
            cutoff_radi=cutoff_radi, envelope_exponent=envelope_exponent
        )
        self.freq = torch.nn.Parameter(torch.Tensor(n_radial))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            torch.arange(1, self.freq.numel() + 1, out=self.freq).mul_(
                PI / self.cutoff_radi
            )
        self.freq.requires_grad_()

    def forward(self, dist: Tensor) -> Tensor:
        """
        Extend distances with Bessel basis.

        Args:
            dist (Tensor): interatomic distance values of (num_edge) shape.

        Returns:
            Tensor: extend distances of (num_edge x n_radial) shape.
        """
        return self.envelope(dist).unsqueeze(-1) * (self.freq * dist.unsqueeze(-1)).sin()


class BesselSB(torch.nn.Module):
    """Bessel Sphericla basis functions"""

    def __init__(
        self,
        n_spherical: int,
        n_radial: int,
        cutoff_radi: float = 5.0,
        envelope_exponent: int = 5,
    ):
        """
        Expand inter atomic distances and angles by Bessel spherical and radial basis.

        Args:
            n_spherical (int): number of spherical basis.
            n_radial (int): number of radial basis.
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
        self.n_spherical = n_spherical
        self.n_radial = n_radial
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
            dist (Tensor): interatomic distance values of (num_edge) shape.
            angle (Tensor): angles of triplets of (n_triplets) shape.
            edge_idx_kj (torch.LongTensor): edge index from atom k to j
                shape of (n_triplets).

        Returns:
            Tensor: extend distances and angles of
                (n_triplets x (n_pherical x n_radial)) shape.
        """
        rbf = torch.stack([f(dist / self.cutoff_radi) for f in self.bessel_funcs], dim=1)
        # apply cutoff
        rbf = self.envelope(dist).unsqueeze(-1) * rbf

        cbf = torch.stack([f(angle) for f in self.sph_funcs], dim=1)

        n, k = self.n_spherical, self.n_radial
        return (rbf[edge_idx_kj].view(-1, n, k) * cbf.view(-1, n, 1)).view(-1, n * k)
