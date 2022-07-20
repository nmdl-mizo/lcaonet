from typing import Union, Optional, Literal, Any

from torch import Tensor
import torch.nn as nn

from pyggnn.data.datakeys import DataKeys
from pyggnn.model.base import BaseGNN
from pyggnn.nn.cutoff import CosineCutoff
from pyggnn.nn.embedding import AtomicNum2Node
from pyggnn.nn.basis import GaussianRBF
from pyggnn.nn.conv.schnet_conv import SchNetConv
from pyggnn.nn.out import Node2Property2

__all__ = ["SchNet"]


class SchNet(BaseGNN):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        n_conv_layer: int,
        out_dim: int,
        n_gaussian: int,
        activation: Union[Any, str] = "shifted_softplus",
        cutoff_net: Optional[nn.Module] = CosineCutoff,
        cutoff_radi: float = 4.0,
        hidden_dim: int = 256,
        aggr: Literal["add", "mean"] = "add",
        scaler: Optional[nn.Module] = None,
        mean: Optional[float] = None,
        stddev: Optional[float] = None,
        residual: bool = True,
        share_weight: bool = False,
        max_z: Optional[int] = 100,
        **kwargs,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.n_conv_layer = n_conv_layer
        self.cutoff_radi = cutoff_radi
        self.out_dim = out_dim
        self.scaler = scaler
        # layers
        self.node_initialize = AtomicNum2Node(embed_dim=node_dim, max_num=max_z)
        self.edge_smearing = GaussianRBF(start=0.0, stop=cutoff_radi, n_dim=n_gaussian)

        if share_weight:
            self.convs = nn.ModuleList(
                [
                    SchNetConv(
                        x_dim=node_dim,
                        edge_dim=edge_dim,
                        n_gaussian=n_gaussian,
                        activation=activation,
                        node_hidden=hidden_dim,
                        cutoff_net=cutoff_net,
                        cutoff_radi=cutoff_radi,
                        aggr=aggr,
                        residual=residual,
                        **kwargs,
                    )
                    * n_conv_layer
                ]
            )
        else:
            self.convs = nn.ModuleList(
                [
                    SchNetConv(
                        x_dim=node_dim,
                        edge_dim=edge_dim,
                        n_gaussian=n_gaussian,
                        activation=activation,
                        node_hidden=hidden_dim,
                        cutoff_net=cutoff_net,
                        cutoff_radi=cutoff_radi,
                        aggr=aggr,
                        residual=residual,
                        **kwargs,
                    )
                    for _ in range(n_conv_layer)
                ]
            )

        self.output = Node2Property2(
            in_dim=node_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            activation=activation,
            aggr=aggr,
            scaler=scaler,
            mean=mean,
            stddev=stddev,
            **kwargs,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.node_initialize.reset_parameters()
        self.edge_smearing.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.output.reset_parameters()

    def forward(self, data_batch) -> Tensor:
        batch = data_batch[DataKeys.Batch]
        atomic_numbers = data_batch[DataKeys.Atomic_num]
        edge_index = data_batch[DataKeys.Edge_index]
        # calc atomic distances
        distances = self.calc_atomic_distances(data_batch)
        # smearing with Gaussian basis
        edge_basis = self.edge_smearing(distances)
        # initial embedding
        x = self.node_initialize(atomic_numbers)
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
            f"edge_dim={self.edge_dim}, "
            f"n_conv_layer={self.n_conv_layer}, "
            f"cutoff={self.cutoff_radi}, "
            f"out_dim={self.out_dim})"
        )
