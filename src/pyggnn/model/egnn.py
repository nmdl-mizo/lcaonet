from __future__ import annotations

from typing import Any

import torch.nn as nn
from torch import Tensor

from pyggnn.model.base import BaseGNN
from pyggnn.nn.node_embed import AtomicNum2Node
from pyggnn.nn.conv.egnn_conv import EGNNConv
from pyggnn.nn.node_out import Node2Prop1
from pyggnn.utils.resolve import activation_resolver

__all__ = ["EGNN"]


class EGNN(BaseGNN):
    """
    EGNN implemeted by using PyTorch Geometric.
    From atomic structure, predict global property such as energy.

    Args:
        node_dim (int): number of node embedding dimension.
        edge_dim (int): number of edge embedding dimension.
        n_conv_layer (int): number of convolutinal layers.
        cutoff_radi (float): cutoff radious. Defaults to `None`.
        out_dim (int, optional): number of output property dimension.
        activation (str or nn.Module, optional): activation function or function name.
        cutoff_radi (float): cutoff radious. Defaults to `None`.
        cutoff_net (nn.Module, optional): cutoff network. Defaults to `None`.
        hidden_dim (int, optional): number of hidden layers. Defaults to `256`.
        aggr (`"add"` or `"mean"`, optional): aggregation method. Defaults to `"add"`.
        batch_norm (bool, optional): if `False`, no batch normalization in convolution layers. Defaults to `False`.
        edge_attr_dim (int, optional): number of another edge attrbute dimension. Defaults to `None`.
        share_weight (bool, optional): if `True`, all convolution layers share the parameters. Defaults to `False`.
        max_z (int, optional): max number of atomic number. Defaults to `100`.

    Notes:
        PyTorch Geometric:
        https://pytorch-geometric.readthedocs.io/en/latest/

        EGNN:
        [1] V. G. Satorras et al., arXiv (2021), (available at http://arxiv.org/abs/2102.09844).
        [2] https://docs.e3nn.org/en/stable/index.html
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        n_conv_layer: int,
        out_dim: int,
        activation: Any | str = "swish",
        cutoff_net: nn.Module | None = None,
        cutoff_radi: float | None = None,
        hidden_dim: int = 256,
        aggr: str = "add",
        batch_norm: bool = False,
        edge_attr_dim: int | None = None,
        share_weight: bool = False,
        max_z: int | None = 100,
        **kwargs,
    ):
        super().__init__()
        act = activation_resolver(activation)

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.n_conv_layer = n_conv_layer
        self.cutoff_radi = cutoff_radi
        self.out_dim = out_dim
        self.edge_attr = edge_attr_dim
        # layers
        self.node_embed = AtomicNum2Node(node_dim, max_num=max_z)
        if cutoff_net is None:
            self.cutoff_net = None
        else:
            assert cutoff_radi is not None
            self.cutoff_net = cutoff_net(cutoff_radi)

        if share_weight:
            self.convs = nn.ModuleList(
                [
                    EGNNConv(
                        x_dim=node_dim,
                        edge_dim=edge_dim,
                        activation=act,
                        edge_attr_dim=edge_attr_dim,
                        node_hidden=hidden_dim,
                        edge_hidden=hidden_dim,
                        cutoff_net=self.cutoff_net,
                        aggr=aggr,
                        batch_norm=batch_norm,
                        **kwargs,
                    )
                ]
                * n_conv_layer
            )
        else:
            self.convs = nn.ModuleList(
                [
                    EGNNConv(
                        x_dim=node_dim,
                        edge_dim=edge_dim,
                        activation=act,
                        edge_attr_dim=edge_attr_dim,
                        node_hidden=hidden_dim,
                        edge_hidden=hidden_dim,
                        cutoff_net=self.cutoff_net,
                        aggr=aggr,
                        batch_norm=batch_norm,
                        **kwargs,
                    )
                    for _ in range(n_conv_layer)
                ]
            )

        self.output = Node2Prop1(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            activation=act,
            aggr=aggr,
            **kwargs,
        )

        self.reset_parameters()

    def reset_parameters(self):
        if self.cutoff_net is not None:
            if hasattr(self.cutoff_net, "reset_parameters"):
                self.cutoff_net.reset_parameters()

    def forward(self, data_batch) -> Tensor:
        if self.edge_attr is not None:
            batch, edge_index, atom_numbers, edge_attr = self.get_data(
                data_batch,
                batch_index=True,
                edge_index=True,
                atom_numbers=True,
                edge_attr=True,
            )
        else:
            batch, edge_index, atom_numbers = self.get_data(
                data_batch,
                batch_index=True,
                edge_index=True,
                atom_numbers=True,
            )
            edge_attr = None
        # calc atomic distances
        distances = self.calc_atomic_distances(data_batch)
        # initial embedding
        x = self.node_embed(atom_numbers)
        # convolution
        for conv in self.convs:
            x = conv(x, distances, edge_index, edge_attr)
        # read out property
        x = self.output(x, batch)
        return x

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"node_dim={self.node_dim}, "
            f"edge_dim={self.edge_dim}, "
            f"cutoff={self.cutoff_radi}, "
            f"out_dim={self.out_dim}, "
            f"convolution_layers: {self.convs[0].__class__.__name__} * {self.n_conv_layer})"
        )
