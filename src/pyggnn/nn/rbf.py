from math import pi as PI

import torch
from torch import Tensor
import torch.nn as nn

from pyggnn.nn.cutoff import EnvelopeCutoff

__all__ = ["GaussianRBF", "BesselRBF"]


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


class GaussianRBF(nn.Module):
    """Gassian radial basis function.
    Expand interatomic distance values using Gaussian radial basis.

    Args:
        start (float, optional): start value of shift of Gaussian. Defaults to `0.0`.
        stop (float, optional): end value of shift of Gaussian. Defaults to `6.0`.
        n_gaussian (int, optional): number of gaussian radial basis. Defaults to `20`.
        centered (bool, optional): if set to `True`, using 0 centered gaussian function. Defaults to `False`.
        trainable (bool, optional): whether gaussian params trainable. Defaults to `True`.
    """

    def __init__(
        self,
        start: float = 0.0,
        stop: float = 6.0,
        n_gaussian: int = 20,
        centered: bool = False,
        trainable: bool = True,
    ):
        super().__init__()
        offset = torch.linspace(start=start, end=stop, steps=n_gaussian)
        width = torch.ones_like(offset, dtype=offset.dtype) * (offset[1] - offset[0])
        self.centered = centered
        if trainable:
            self.width = nn.Parameter(width)
            self.offset = nn.Parameter(offset)
        else:
            self.register_buffer("width", width)
            self.register_buffer("offset", offset)

    def forward(self, dist: Tensor) -> Tensor:
        """
        Compute extended distances with Gaussian basis.

        Args:
            dist (Tensor): interatomic distance values of (n_edge) shape.

        Returns:
            Tensor: extended distances of (n_edge x n_gaussian) shape.
        """
        return gaussian_rbf(dist, offsets=self.offset, widths=self.width, centered=self.centered)


class BesselRBF(torch.nn.Module):
    """
    Bessel radial basis functions.
    Expand inter atomic distances by Bessel radial basis.

    Args:
        n_radial (int): number of radial basis.
        cutoff_radi (float, optional): cutoff radius. Defaults to `5.0`.
        envelope_exponent (int, optional): exponent of cutoff envelope function. Defaults to `5`.
    """

    def __init__(
        self,
        n_radial: int,
        cutoff_radi: float = 5.0,
        envelope_exponent: int = 5,
    ):
        super().__init__()
        self.cutoff_radi = cutoff_radi
        # TODO: separate the cutoff function
        self.cutoff = EnvelopeCutoff(cutoff_radi, envelope_exponent)
        self.freq = torch.nn.Parameter(torch.Tensor(n_radial))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            torch.arange(1, self.freq.numel() + 1, out=self.freq).mul_(PI)
        self.freq.requires_grad_()

    def forward(self, dist: Tensor) -> Tensor:
        """
        Compute extended distances with Bessel basis.

        Args:
            dist (Tensor): interatomic distance values of (n_edge) shape.

        Returns:
            Tensor: extended distances of (n_edge x n_radial) shape.
        """
        dist = dist / self.cutoff_radi
        return self.cutoff(dist).unsqueeze(-1) * (self.freq * dist.unsqueeze(-1)).sin()
