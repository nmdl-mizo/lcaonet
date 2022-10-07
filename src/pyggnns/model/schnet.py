from __future__ import annotations  # type: ignore

from collections.abc import Callable
from typing import Any

import torch.nn as nn
from torch import Tensor

from pyggnns.model.base import BaseGNN
from pyggnns.nn.conv.schnet_conv import SchNetConv
from pyggnns.nn.cutoff import CosineCutoff
from pyggnns.nn.node_embed import AtomicNum2Node
from pyggnns.nn.node_out import Node2Prop2
from pyggnns.nn.rbf import GaussianRBF
from pyggnns.utils.resolve import activation_resolver, init_resolver

__all__ = ["SchNet"]


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
        cutoff_net (nn.Module, optional): cutoff networck. Defaults to `CosineCutoff`.
        cutoff_radi (float, optional): cutoff radius. Defaults to `4.0`.
        hidden_dim (int, optional): hidden dimension in convolution layers. Defaults to `256`.
        aggr ("add" or "mean", optional): aggregation method. Defaults to `"add"`.
        scaler (nn.Module, optional): scaler network. Defaults to `None`.
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
        scaler: nn.Module | None = None,
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
        self.node_embed = AtomicNum2Node(node_dim, max_num=max_z)
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

        self.output = Node2Prop2(
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
        batch, edge_index, atom_numbers = self.get_data(
            data_batch, batch_index=True, edge_index=True, atom_numbers=True
        )
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
            f"cutoff={self.cutoff_radi}, "
            f"out_dim={self.out_dim}, "
            f"convolution_layers: {self.convs[0].__class__.__name__} * {self.n_conv_layer})"
        )
