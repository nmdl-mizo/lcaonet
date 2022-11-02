from __future__ import annotations  # type: ignore

from collections.abc import Callable
from math import pi as PI
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_scatter import scatter

from pyg_material.data import DataKeys
from pyg_material.model.base import BaseGNN
from pyg_material.nn import AtomicNum2Node, Dense, ScaleShift, ShiftedSoftplus
from pyg_material.utils import activation_resolver, init_resolver


class CosineCutoff(nn.Module):
    """CosineCutoff Network.

    Args:
        cutoff_radi (float): cutoff radious
    """

    def __init__(self, cutoff_radi: float):
        super().__init__()
        self.cutof_radi = cutoff_radi

    def forward(self, dist: Tensor) -> Tensor:
        """forward calculation of CosineNetwork.

        Args:
            dist (Tensor): inter atomic distances shape of (n_edge)

        Returns:
            Tensor: Cutoff values shape of (n_edge)
        """
        # Compute values of cutoff function
        cutoffs = 0.5 * (torch.cos(dist * PI / self.cutoff_radi) + 1.0)
        # Remove contributions beyond the cutoff radius
        return cutoffs * (dist < self.cutoff_radi).to(dist.dtype)


def _gaussian_rbf(
    distances: Tensor,
    offsets: Tensor,
    widths: Tensor,
    centered: bool = False,
) -> Tensor:
    """Filtered interatomic distance values using Gaussian basis.

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
    """Gassian radial basis function. Expand interatomic distance values using
    Gaussian radial basis.

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
        """Compute extended distances with Gaussian basis.

        Args:
            dist (Tensor): interatomic distance values of (n_edge) shape.

        Returns:
            Tensor: extended distances of (n_edge x n_gaussian) shape.
        """
        return _gaussian_rbf(dist, offsets=self.offset, widths=self.width, centered=self.centered)


class SchNetConv(MessagePassing):
    def __init__(
        self,
        x_dim: int,
        edge_filter_dim: int,
        n_gaussian: int,
        activation: Callable[[Tensor], Tensor] = ShiftedSoftplus(),
        node_hidden: int = 256,
        cutoff_net: nn.Module | None = None,
        aggr: str = "add",
        weight_init: Callable[[Tensor], Tensor] = nn.init.xavier_uniform_,
        **kwargs,
    ):
        assert aggr == "add" or aggr == "mean"
        # TODO: pass kwargs to superclass
        super().__init__(aggr=aggr)

        # name node_dim is already used in super class
        self.x_dim = x_dim
        self.edge_filter_dim = edge_filter_dim
        self.node_hidden = node_hidden
        self.n_gaussian = n_gaussian
        if cutoff_net is not None:
            self.cutoff_net = cutoff_net
        else:
            self.cutoff_net = None

        # update functions
        # filter generator
        self.edge_filter_func = nn.Sequential(
            Dense(
                n_gaussian,
                edge_filter_dim,
                bias=True,
                weight_init=weight_init,
                **kwargs,
            ),
            activation,
            Dense(
                edge_filter_dim,
                edge_filter_dim,
                bias=True,
                weight_init=weight_init,
                **kwargs,
            ),
            activation,
        )
        # node fucntions
        self.node_lin1 = Dense(x_dim, edge_filter_dim, bias=True, weight_init=weight_init, **kwargs)
        self.node_lin2 = nn.Sequential(
            Dense(edge_filter_dim, x_dim, bias=True, weight_init=weight_init, **kwargs),
            activation,
            Dense(x_dim, x_dim, bias=True, weight_init=weight_init, **kwargs),
        )

    def forward(
        self,
        x: Tensor,
        dist: Tensor,
        edge_basis: Tensor,
        edge_index: Adj,
    ) -> Tensor:
        # calc edge filter and cutoff
        W = self.edge_filter_func(edge_basis)
        if self.cutoff_net is not None:
            C = self.cutoff_net(dist)
            W = W * C.view(-1, 1)
        # propagate_type:
        # (x: Tensor, W: Tensor)
        out = self.node_lin1(x)
        out = self.propagate(edge_index, x=out, W=W, size=None)
        out = self.node_lin2(out)
        # residual network
        out = out + x
        return out

    def message(self, x_j: Tensor, W: Tensor) -> Tensor:
        return x_j * W


class SchNetOutBlock(nn.Module):
    """The block to compute the global graph proptery from node embeddings.
    This block contains two Dense layers and aggregation block. If set
    `scaler`, scaling process before aggregation. This block is used in SchNet.

    Args:
        node_dim (int): number of input dim.
        hidden_dim (int, optional): number of hidden layers dim. Defaults to `128`.
        out_dim (int, optional): number of output dim. Defaults to `1`.
        activation: (Callable, optional): activation function. Defaults to `ShiftedSoftplus()`.
        aggr (`"add"` or `"mean"`): aggregation method. Defaults to `"add"`.
        scaler: (nn.Module, optional): scaler layer. Defaults to `None`.
        mean: (Tensor, optional): mean of the input tensor. Defaults to `None`.
        stddev: (Tensor, optional): stddev of the input tensor. Defaults to `None`.
        weight_init (Callable, optional): weight initialization function. Defaults to `torch.nn.init.xavier_uniform_`.
    """

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 1,
        activation: Callable[[Tensor], Tensor] = ShiftedSoftplus(),
        aggr: str = "add",
        scaler: nn.Module | None = None,
        mean: Tensor | None = None,
        stddev: Tensor | None = None,
        weight_init: Callable[[Tensor], Tensor] = torch.nn.init.xavier_uniform_,
        **kwargs,
    ):
        super().__init__()

        assert aggr == "add" or aggr == "mean"
        self.aggr = aggr
        self.output = nn.Sequential(
            Dense(node_dim, hidden_dim, bias=True, weight_init=weight_init, **kwargs),
            activation,
            Dense(hidden_dim, out_dim, bias=False, weight_init=weight_init, **kwargs),
        )
        if scaler is None:
            self.scaler = None
        else:
            if mean is None:
                mean = torch.FloatTensor([0.0])
            if stddev is None:
                stddev = torch.FloatTensor([1.0])
            self.scaler = scaler(mean, stddev)

    def forward(self, x: Tensor, batch: Tensor | None = None) -> Tensor:
        """Compute global property from node embeddings.

        Args:
            x (Tensor): node embeddings shape of (n_node x in_dim).
            batch (Tensor, optional): batch index shape of (n_node). Defaults to `None`.

        Returns:
            Tensor: shape of (n_batch x out_dim).
        """
        out = self.output(x)
        if self.scaler is not None:
            out = self.scaler(out)
        return scatter(out, index=batch, dim=0, reduce=self.aggr)


class SchNet(BaseGNN):
    """SchNet implemeted by using PyTorch Geometric. From atomic structure,
    predict global property such as energy.

    Args:
        node_dim (int): node embedding dimension.
        edge_filter_dim (int): edge filter embedding dimension.
        n_conv_layer (int): number of convolution layers.
        out_dim (int): output dimension.
        n_gaussian (int): number of gaussian radial basis.
        activation (str, optional): activation function or function name. Defaults to `"shifted_softplus"`.
        cutoff_net (nn.Module, optional): cutoff networck. Defaults to `pyg_material.nn.CosineCutoff`.
        cutoff_radi (float, optional): cutoff radius. Defaults to `4.0`.
        hidden_dim (int, optional): hidden dimension in convolution layers. Defaults to `256`.
        aggr ("add" or "mean", optional): aggregation method. Defaults to `"add"`.
        scaler (nn.Module, optional): scaler network. Defaults to `pyg_material.nn.ScaleShift`.
        mean (float, optional): mean of node property. Defaults to `None`.
        stddev (float, optional): standard deviation of node property. Defaults to `None`.
        weight_init (str, optional): weight initialization function. Defaults to `"xavier_uniform_"`.
        share_weight (bool, optional): share weight parameter all convolution. Defaults to `False`.
        max_z (int, optional): max atomic number. Defaults to `100`.

    Notes:
        PyTorch Geometric:
        https://pytorch-geometric.readthedocs.io/en/latest/

        SchNet:
        [1] K. T. Schütt et al., J. Chem. Phys. 148, 241722 (2018).
        [2] https://github.com/atomistic-machine-learning/schnetpack
    """

    def __init__(
        self,
        node_dim: int,
        edge_filter_dim: int,
        n_conv_layer: int,
        out_dim: int,
        n_gaussian: int,
        activation: str = "shifted_softplus",
        cutoff_net: nn.Module = CosineCutoff,
        cutoff_radi: float = 4.0,
        hidden_dim: int = 256,
        aggr: str = "add",
        scaler: nn.Module | None = ScaleShift,
        mean: float | None = None,
        stddev: float | None = None,
        weight_init: str = "xavier_uniform_",
        share_weight: bool = False,
        max_z: int | None = 100,
        **kwargs,
    ):
        super().__init__()
        act = activation_resolver(activation)
        wi: Callable[[Any], Any] = init_resolver(weight_init)

        self.node_dim = node_dim
        self.edge_filter_dim = edge_filter_dim
        self.n_conv_layer = n_conv_layer
        self.n_gaussian = n_gaussian
        self.cutoff_radi = cutoff_radi
        self.out_dim = out_dim
        self.scaler = scaler
        # layers
        self.node_embed = AtomicNum2Node(node_dim, max_z=max_z)
        self.rbf = GaussianRBF(start=0.0, stop=cutoff_radi - 0.5, n_gaussian=n_gaussian)
        if cutoff_net is None:
            self.cutoff_net = None
        else:
            assert cutoff_radi is not None
            self.cutoff_net = cutoff_net(cutoff_radi)

        if share_weight:
            self.convs = nn.ModuleList(
                [
                    SchNetConv(
                        x_dim=node_dim,
                        edge_filter_dim=edge_filter_dim,
                        n_gaussian=n_gaussian,
                        activation=act,
                        node_hidden=hidden_dim,
                        cutoff_net=self.cutoff_net,
                        weight_init=wi,
                        aggr=aggr,
                        **kwargs,
                    )
                ]
                * n_conv_layer
            )
        else:
            self.convs = nn.ModuleList(
                [
                    SchNetConv(
                        x_dim=node_dim,
                        edge_filter_dim=edge_filter_dim,
                        n_gaussian=n_gaussian,
                        activation=act,
                        node_hidden=hidden_dim,
                        cutoff_net=self.cutoff_net,
                        weight_init=wi,
                        aggr=aggr,
                        **kwargs,
                    )
                    for _ in range(n_conv_layer)
                ]
            )

        self.output = SchNetOutBlock(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            activation=act,
            aggr=aggr,
            scaler=scaler,
            mean=mean,
            stddev=stddev,
            weight_init=wi,
            **kwargs,
        )

        self.reset_parameters()

    def reset_parameters(self):
        if self.cutoff_net is not None:
            if hasattr(self.cutoff_net, "reset_parameters"):
                self.cutoff_net.reset_parameters()

    def forward(self, data_batch) -> Tensor:
        data_dict = self.get_data(data_batch, batch_index=True, atom_numbers=True, edge_index=True)
        batch = data_dict[DataKeys.Batch]
        atom_numbers = data_dict[DataKeys.Atom_numbers]
        edge_index = data_dict[DataKeys.Edge_index]
        # calc atomic distances
        distances = self.calc_atomic_distances(data_batch)
        # expand with Gaussian radial basis
        edge_basis = self.rbf(distances)
        # initial embedding
        x = self.node_embed(atom_numbers)

        # convolution
        for conv in self.convs:
            x = conv(x, distances, edge_basis, edge_index)
        # read out property
        x = self.output(x, batch)
        return x

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"node_dim={self.node_dim}, "
            f"edge_filter_dim={self.edge_filter_dim}, "
            f"n_gaussian={self.n_gaussian}, "
            f"cutoff_radi={self.cutoff_radi}, "
            f"out_dim={self.out_dim}, "
            f"convolution_layers: {self.convs[0].__class__.__name__} * {self.n_conv_layer})"
        )
