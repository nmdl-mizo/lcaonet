from typing import Callable, Literal, Optional, Any

import torch
from torch import Tensor
import torch.nn as nn
from torch_scatter import scatter
from torch_geometric.nn.inits import glorot_orthogonal

from pyggnn.nn.activation import Swish
from pyggnn.nn.base import Dense


__all__ = ["Edge2NodeProp"]


class Edge2NodeProp(nn.Module):
    """
    The block to compute the node-wise proptery from edge embeddings.
    This block contains some Dense layers and aggregation block of all neighbors.
    This block is used in Dimenet.
    """

    def __init__(
        self,
        edge_dim: int,
        n_radial: int,
        out_dim: int = 1,
        n_layers: int = 3,
        activation: Callable[[Tensor], Tensor] = Swish(beta=1.0),
        aggr: Literal["add", "mean"] = "add",
        weight_init: Callable[[Tensor], Any] = glorot_orthogonal,
        **kwargs,
    ):
        super().__init__()

        assert aggr == "add" or aggr == "mean"
        self.aggr = aggr
        # linear layer for radial basis
        self.rbf_dense = Dense(
            n_radial,
            edge_dim,
            bias=False,
            weight_init=weight_init,
            **kwargs,
        )
        # linear layer for edge embedding
        denses = []
        for _ in range(n_layers):
            denses.append(
                Dense(
                    edge_dim,
                    edge_dim,
                    bias=True,
                    weight_init=weight_init,
                    **kwargs,
                )
            )
            denses.append(activation)
        denses.append(
            Dense(
                edge_dim,
                out_dim,
                bias=False,
                weight_init=weight_init,
                **kwargs,
            )
        )
        self.denses = nn.Sequential(*denses)

    def forward(
        self,
        x: Tensor,
        rbf: Tensor,
        idx_i: torch.LongTensor,
        num_nodes: Optional[int] = None,
    ) -> Tensor:
        """
        Compute node-wise property from edge embeddings.

        Args:
            x (Tensor): edge embedding shape of (n_edge x edge_dim).
            rbf (Tensor): radial basis function shape of (n_node x n_radial).
            idx_i (torch.LongTensor): node index center atom i shape of (n_edge).
            num_nodes (Optional[int], optional): number of edge. Defaults to `None`.

        Returns:
            Tensor: node-wise properties shape of (n_node x out_dim).
        """
        x = self.rbf_dense(rbf) * x
        # add all neighbor atoms
        x = scatter(x, idx_i, dim=0, dim_size=num_nodes, reduce=self.aggr)
        return self.denses(x)
