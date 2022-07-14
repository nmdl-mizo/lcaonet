from typing import Literal, Optional

import torch.nn as nn
from torch import Tensor

from pyggnn.data.datakeys import DataKeys
from pyggnn.nn.embedding import AtomicNum2Node
from pyggnn.nn.conv import EGNNConv
from pyggnn.nn.out import Node2Property
from pyggnn.model.base import BaseGNN

__all__ = ["EGNN"]


class EGNN(BaseGNN):
    """
    EGNN implemeted by using PyTorch Geometric.
    From atomic structure, predict global property such as total energy.

    Notes:
        PyTorch Geometric:
        https://pytorch-geometric.readthedocs.io/en/latest/

        EGNN:
        [1] V. G. Satorras et al., arXiv (2021),
            (available at http://arxiv.org/abs/2102.09844).
        [2] https://docs.e3nn.org/en/stable/index.html
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        n_conv_layer: int,
        out_dim: int = 1,
        cutoff_net: Optional[nn.Module] = None,
        cutoff_radi: Optional[float] = None,
        hidden_dim: int = 256,
        aggr: Literal["add", "mean"] = "add",
        residual: bool = True,
        batch_norm: bool = False,
        edge_attr_dim: Optional[int] = None,
        share_weight: bool = False,
        swish_beta: Optional[float] = 1.0,
        max_z: Optional[int] = 100,
    ):
        """
        Args:
            node_dim (int): number of node embedding dim.
            edge_dim (int): number of edge embedding dim.
            n_conv_layer (int): number of convolutinal layers.
            cutoff_radi (float): cutoff radious.
            out_dim (int, optional): number of output property dimension.
                Defaults to `1`.
            hidden_dim (int, optional): number of hidden layers.
                Defaults to `256`.
            aggr (`"add"` or `"mean"`, optional): if set to `"add"`, sumaggregation
                is done along node dimension. Defaults to `"add"`.
            residual (bool, optional): if `False`, no residual network.
                Defaults to `True`.
            batch_norm (bool, optional): if `False`, no batch normalization in
                convolution layers. Defaults to `False`.
            edge_attr_dim (int, optional): number of edge attrbute dim.
                Defaults to `None`.
            share_weight (bool, optional): if `True`, all convolution layers
                share the parameters. Defaults to `False`.
            swish_beta (float, optional): beta coefficient of Swish activation func.
                Defaults to `1.0`.
            max_z (int, optional): max number of atomic number. Defaults to `100`.
        """
        super().__init__()
        self.node_initialize = AtomicNum2Node(embedding_dim=node_dim, max_num=max_z)

        if share_weight:
            self.convs = nn.ModuleList(
                [
                    EGNNConv(
                        x_dim=node_dim,
                        edge_dim=edge_dim,
                        edge_attr_dim=edge_attr_dim,
                        node_hidden=hidden_dim,
                        edge_hidden=hidden_dim,
                        cutoff_net=cutoff_net,
                        cutoff_radi=cutoff_radi,
                        beta=swish_beta,
                        aggr=aggr,
                        residual=residual,
                        batch_norm=batch_norm,
                    )
                    * n_conv_layer
                ]
            )
        else:
            self.convs = nn.ModuleList(
                [
                    EGNNConv(
                        x_dim=node_dim,
                        edge_dim=edge_dim,
                        edge_attr_dim=edge_attr_dim,
                        node_hidden=hidden_dim,
                        edge_hidden=hidden_dim,
                        cutoff_net=cutoff_net,
                        cutoff_radi=cutoff_radi,
                        beta=swish_beta,
                        aggr=aggr,
                        residual=residual,
                        batch_norm=batch_norm,
                    )
                    for _ in range(n_conv_layer)
                ]
            )

        self.output = Node2Property(
            in_dim=node_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            beta=swish_beta,
            aggr=aggr,
        )

    def forward(self, data_batch) -> Tensor:
        batch = data_batch[DataKeys.Batch]
        atomic_numbers = data_batch[DataKeys.Atomic_num]
        edge_index = data_batch[DataKeys.Edge_index]
        edge_attr = data_batch.get(DataKeys.Edge_attr, None)
        # calc atomic distances
        distances = self.calc_atomic_distances(data_batch)
        # initial embedding
        x = self.node_initialize(atomic_numbers)
        # convolution
        for conv in self.convs:
            x = conv(x, distances, edge_index, edge_attr)
        # read out property
        x = self.output(x, batch)
        return x
