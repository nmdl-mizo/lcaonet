from typing import Literal, Optional, Union, Any

import torch
from torch import Tensor
import torch.nn as nn
from torch_scatter import scatter

from pyggnn.nn.base import Dense
from pyggnn.utils.resolve import activation_resolver


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
        activation: Union[Any, str] = "swish",
        aggr: Literal["add", "mean"] = "add",
        **kwargs,
    ):
        super().__init__()
        act = activation_resolver(activation, **kwargs)

        assert aggr == "add" or aggr == "mean"
        self.aggr = aggr
        self.rbf_dense = Dense(n_radial, edge_dim, bias=False)
        denses = []
        for _ in range(n_layers):
            denses.append(
                Dense(
                    edge_dim,
                    edge_dim,
                    bias=True,
                    activation_name=activation,
                    **kwargs,
                )
            )
            denses.append(act)
        denses.append(Dense(edge_dim, out_dim, bias=False))
        self.denses = nn.Sequential(*denses)

        self.reset_parameters()

    def reset_parameters(self):
        self.rbf_dense.reset_parameters()
        for ll in self.denses:
            if hasattr(ll, "reset_parameters"):
                ll.reset_parameters()

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
