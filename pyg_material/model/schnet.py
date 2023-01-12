from __future__ import annotations  # type: ignore

from collections.abc import Callable
from math import pi as PI

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter

from pyg_material.data import DataKeys
from pyg_material.model.base import BaseGNN
from pyg_material.nn import (
    AtomicNum2Node,
    BaseScaler,
    Dense,
    ShiftedSoftplus,
    ShiftScaler,
)
from pyg_material.utils import activation_resolver, init_resolver


class CosineCutoff(nn.Module):
    """Cosine cutoff network used in SchNet.

    Args:
        cutoff (float): cutoff radious.
    """

    def __init__(self, cutoff: float):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, dist: Tensor) -> Tensor:
        """Forward calculation of cosine cutoff layer.

        Args:
            dist (Tensor): inter atomic distances of (n_edge) shape.

        Returns:
            Tensor: cutoff values of (n_edge) shape
        """
        # Compute values of cutoff function
        cutoffs = 0.5 * (torch.cos(dist * PI / self.cutoff) + 1.0)
        # Remove contributions beyond the cutoff radius
        return cutoffs * (dist < self.cutoff).to(dist.dtype)


def _gaussian_rbf(distances: Tensor, offsets: Tensor, widths: Tensor, centered: bool = False) -> Tensor:
    """Expand interatomic distances using Gaussian basis.

    Notes:
        ref:
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
    """Layer that expand inter atomic distances in the Gassian radial basis
    functions.

    Args:
        start (float, optional): start value of shift of Gaussian. Defaults to `0.0`.
        stop (float, optional): end value of shift of Gaussian. Defaults to `6.0`.
        n_rad (int, optional): number of gaussian radial basis. Defaults to `20`.
        centered (bool, optional): if set to `True`, using 0 centered gaussian function. Defaults to `False`.
        trainable (bool, optional): whether gaussian params trainable. Defaults to `True`.
    """

    def __init__(
        self,
        start: float = 0.0,
        stop: float = 6.0,
        n_rad: int = 20,
        centered: bool = False,
        trainable: bool = True,
    ):
        super().__init__()
        self.n_rad = n_rad
        offset = torch.linspace(start=start, end=stop, steps=n_rad)
        width = torch.ones_like(offset, dtype=offset.dtype) * (offset[1] - offset[0])
        self.centered = centered
        if trainable:
            self.width = nn.Parameter(width)
            self.offset = nn.Parameter(offset)
        else:
            self.register_buffer("width", width)
            self.register_buffer("offset", offset)

    def forward(self, dist: Tensor) -> Tensor:
        """Compute expanded distances with Gaussian basis functions.

        Args:
            dist (Tensor): interatomic distances of (n_edge) shape.

        Returns:
            Tensor: expanded distances of (n_edge, n_gaussian) shape.
        """
        return _gaussian_rbf(dist, offsets=self.offset, widths=self.width, centered=self.centered)


class SchNetConv(MessagePassing):
    def __init__(
        self,
        x_dim: int,
        edge_filter_dim: int,
        n_rad: int,
        activation: nn.Module = ShiftedSoftplus(),
        cutoff_net: nn.Module | None = None,
        aggr: str = "add",
        weight_init: Callable[[Tensor], Tensor] = nn.init.xavier_uniform_,
        **kwargs,
    ):
        assert aggr == "add" or aggr == "mean"
        super().__init__(aggr=aggr, **kwargs)

        self.x_dim = x_dim  # name node_dim is already used in super class
        self.edge_filter_dim = edge_filter_dim
        self.n_rad = n_rad
        self.cutoff_net = cutoff_net

        # update functions
        # filter generate funcion
        self.edge_filter_lin = nn.Sequential(
            Dense(n_rad, edge_filter_dim, True, weight_init=weight_init),
            activation,
            Dense(edge_filter_dim, edge_filter_dim, True, weight_init=weight_init),
            activation,
        )
        # node fucntions
        self.node_lin1 = Dense(x_dim, edge_filter_dim, True, weight_init=weight_init)
        self.node_lin2 = nn.Sequential(
            Dense(edge_filter_dim, x_dim, True, weight_init=weight_init),
            activation,
            Dense(x_dim, x_dim, True, weight_init=weight_init),
        )

    def forward(self, x: Tensor, dist: Tensor, edge_basis: Tensor, edge_index: Tensor) -> Tensor:
        # calc edge filter and cutoff
        W = self.edge_filter_lin(edge_basis)
        if self.cutoff_net is not None:
            C = self.cutoff_net(dist)
            W = W * C.view(-1, 1)
        # propagate_type:
        # (x: Tensor, W: Tensor)
        out = self.node_lin1(x)
        out = self.propagate(edge_index, x=out, W=W, size=None)
        out = self.node_lin2(out)
        # residual network
        return out + x

    def message(self, x_j: Tensor, W: Tensor) -> Tensor:
        return x_j * W


class SchNetOutBlock(nn.Module):
    """The block to compute the graph-wise proptery from node embeddings. This
    block contains two Dense layers and aggregation block. If set `scaler`,
    scaling process before aggregation. This block is used in SchNet.

    Args:
        node_dim (int): number of input dimension.
        hidden_dim (int, optional): number of hidden layers dimension. Defaults to `128`.
        out_dim (int, optional): number of output dimension. Defaults to `1`.
        activation: (torch.nn.Module, optional): activation function. Defaults to `ShiftedSoftplus()`.
        aggr (`"add"` or `"mean"`): aggregation method. Defaults to `"add"`.
        scaler: (type[pyg_material.nn.BaseScaler], optional): scaler layer. Defaults to `ShiftScaler`.
        mean: (float, optional): mean of the input tensor. Defaults to `None`.
        stddev: (float, optional): stddev of the input tensor. Defaults to `None`.
        weight_init (Callable, optional): weight initialization function. Defaults to `torch.nn.init.xavier_uniform_`.
    """

    def __init__(
        self,
        node_dim: int,
        out_dim: int = 1,
        activation: nn.Module = ShiftedSoftplus(),
        aggr: str = "add",
        scaler: type[BaseScaler] | None = ShiftScaler,
        mean: float | None = None,
        stddev: float | None = None,
        weight_init: Callable[[Tensor], Tensor] = torch.nn.init.xavier_uniform_,
    ):
        super().__init__()

        assert aggr == "add" or aggr == "mean"
        self.aggr = aggr
        self.output_lin = nn.Sequential(
            Dense(node_dim, node_dim // 2, True, weight_init=weight_init),
            activation,
            Dense(node_dim // 2, out_dim, False, weight_init=weight_init),
        )
        if scaler is None:
            self.scaler = None
        else:
            if mean is None:
                mean = 0.0
            if stddev is None:
                stddev = 1.0
            self.scaler = scaler(torch.FloatTensor([mean]), torch.FloatTensor([stddev]))

    def forward(self, x: Tensor, batch_idx: Tensor | None = None) -> Tensor:
        """Compute global property from node embeddings.

        Args:
            x (Tensor): node embeddings of (n_node, node_dim) shape.
            batch_idx (Tensor, optional): batch index of (n_node) shape. Defaults to `None`.

        Returns:
            Tensor: graph-wise property of (n_batch, out_dim) shape.
        """
        out = self.output_lin(x)
        # aggregation
        out = out.sum(dim=0) if batch_idx is None else scatter(out, batch_idx, dim=0, reduce=self.aggr)
        # scaler
        if self.scaler is not None:
            out = self.scaler(out)
        return out


class SchNet(BaseGNN):
    """SchNet implemeted by using PyTorch Geometric. From atomic structure,
    predict graph-wise property such as formation energy.

    Args:
        node_dim (int): node embedding dimension.
        edge_filter_dim (int): edge filter embedding dimension.
        n_conv_layer (int): number of convolution layers.
        out_dim (int): output dimension.
        n_rad (int): number of gaussian radial basis.
        activation (str, optional): name of activation function. Defaults to `"shifted_softplus"`.
        cutoff (float, optional): cutoff radius. Defaults to `4.0`.
        cutoff_net (type[CosineCutoff], optional): cutoff networck. Defaults to `CosineCutoff`.
        aggr ("add" or "mean", optional): aggregation method. Defaults to `"add"`.
        scaler (type[pyg_material.nn.BaseScaler], optional): output scaler network. Defaults to `pyg_material.nn.ShiftScaler`.
        mean (float, optional): mean of node property. Defaults to `None`.
        stddev (float, optional): standard deviation of node property. Defaults to `None`.
        weight_init (str, optional): name of weight initialization function. Defaults to `"xavier_uniform_"`.
        share_weight (bool, optional): share weight parameter in all convolution. Defaults to `False`.
        max_z (int, optional): max atomic number. Defaults to `100`.

    Notes:
        PyTorch Geometric:
            https://pytorch-geometric.readthedocs.io/en/latest/

        SchNet:
            [1] K. T. Schütt et al., J. Chem. Phys. 148, 241722 (2018).
            [2] https://github.com/atomistic-machine-learning/schnetpack
    """  # NOQA: E501

    def __init__(
        self,
        node_dim: int,
        edge_filter_dim: int,
        n_conv_layer: int,
        out_dim: int,
        n_rad: int,
        activation: str = "shifted_softplus",
        cutoff: float = 4.0,
        cutoff_net: type[CosineCutoff] | None = CosineCutoff,
        aggr: str = "add",
        scaler: type[BaseScaler] | None = ShiftScaler,
        mean: float | None = None,
        stddev: float | None = None,
        weight_init: str = "xavier_uniform_",
        share_weight: bool = False,
        max_z: int | None = 100,
        **kwargs,
    ):
        super().__init__()
        act: nn.Module = activation_resolver(activation)
        wi: Callable[[Tensor], Tensor] = init_resolver(weight_init)

        self.node_dim = node_dim
        self.edge_filter_dim = edge_filter_dim
        self.n_conv_layer = n_conv_layer
        self.n_rad = n_rad
        self.cutoff = cutoff
        self.out_dim = out_dim

        # layers
        self.node_embed = AtomicNum2Node(node_dim, max_z=max_z)
        self.rbf = GaussianRBF(start=0.0, stop=cutoff - 0.5, n_rad=n_rad)
        self.cutoff_net: nn.Module | None
        if cutoff_net is None:
            self.cutoff_net = None
        else:
            assert cutoff is not None, "cutoff must be specified if cutoff_net is not None."
            self.cutoff_net = cutoff_net(cutoff)

        if share_weight:
            self.convs = nn.ModuleList(
                [
                    SchNetConv(
                        x_dim=node_dim,
                        edge_filter_dim=edge_filter_dim,
                        n_rad=n_rad,
                        activation=act,
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
                        n_rad=n_rad,
                        activation=act,
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
            out_dim=out_dim,
            activation=act,
            aggr=aggr,
            scaler=scaler,
            mean=mean,
            stddev=stddev,
            weight_init=wi,
        )

    def forward(self, batch: Batch) -> Tensor:
        batch_idx = batch.get(DataKeys.Batch_idx)
        atom_numbers = batch[DataKeys.Atom_numbers]
        edge_idx = batch[DataKeys.Edge_idx]

        # calc atomic distances
        distances = self.calc_atomic_distances(batch)

        # expand with Gaussian radial basis
        edge_basis = self.rbf(distances)
        # initial embedding
        x = self.node_embed(atom_numbers)

        # convolution
        for conv in self.convs:
            x = conv(x, distances, edge_basis, edge_idx)

        # read out property
        return self.output(x, batch_idx)
