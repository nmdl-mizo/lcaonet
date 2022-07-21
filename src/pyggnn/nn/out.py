from typing import Literal, Optional, Union, Any

import torch
from torch import Tensor
import torch.nn as nn
from torch_scatter import scatter

from pyggnn.nn.base import Dense
from pyggnn.utils.resolve import activation_resolver


__all__ = ["Node2Property1", "Node2Property2", "Edge2NodeProperty"]


class Node2Property1(nn.Module):
    """
    The block to compute the global graph proptery from node embeddings.
    In this block, after aggregation, two more FNNs are used to deform embeddings.
    This block is used in EGNN.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 1,
        activation: Union[Any, str] = "swish",
        aggr: Literal["add", "mean"] = "add",
        **kwargs,
    ):
        """
        Args:
            in_dim (int): number of input dim.
            hidden_dim (int, optional): number of hidden layers dim. Defaults to `128`.
            out_dim (int, optional): number of output dim. Defaults to `1`.
            activation: (str or nn.Module, optional): activation function class or name.
                Defaults to `Swish`.
            aggr (`"add"` or `"mean"`): aggregation method. Defaults to `"add"`.
        """
        super().__init__()

        act = activation_resolver(activation, **kwargs)

        assert aggr == "add" or aggr == "mean"
        self.aggr = aggr
        self.node_transform = nn.Sequential(
            Dense(in_dim, hidden_dim, bias=True, activation_name=activation, **kwargs),
            act,
            Dense(hidden_dim, hidden_dim, bias=True),
        )
        self.predict = nn.Sequential(
            Dense(
                hidden_dim,
                hidden_dim,
                bias=True,
                activation_name=activation,
                **kwargs,
            ),
            act,
            Dense(hidden_dim, out_dim, bias=False),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.node_transform:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        for layer in self.predict:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(self, x: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        """
        Compute global property from node embeddings.

        Args:
            x (Tensor): node embeddings shape of (num_node x in_dim).
            batch (Tensor, optional): batch index. Defaults to `None`.

        Returns:
            Tensor: shape of (num_batch x out_dim).
        """
        out = self.node_transform(x)
        out = scatter(out, index=batch, dim=0, reduce=self.aggr)
        return self.predict(out)


class Node2Property2(nn.Module):
    """
    The block to compute the global graph proptery from node embeddings.
    This block contains two FNN layers and aggregation block.
    If set `scler`, scaling process before aggregation.
    This block is used in SchNet.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 1,
        activation: Union[Any, str] = "shifted_softplus",
        aggr: Literal["add", "mean"] = "add",
        scaler: Optional[nn.Module] = None,
        mean: Optional[Tensor] = None,
        stddev: Optional[Tensor] = None,
        **kwargs,
    ):
        """
        Args:
            in_dim (int): number of input dim.
            hidden_dim (int, optional): number of hidden layers dim. Defaults to `128`.
            out_dim (int, optional): number of output dim. Defaults to `1`.
            activation: (str or nn.Module, optional): activation function class or name.
                Defaults to `shifted_softplus`.
            aggr (`"add"` or `"mean"`): aggregation method. Defaults to `"add"`.
            scaler: (nn.Module, optional): scaler layer. Defaults to `None`.
            mean: (Tensor, optional): mean of the input tensor. Defaults to `None`.
            stddev: (Tensor, optional): stddev of the input tensor. Defaults to `None`.
        """
        super().__init__()
        act = activation_resolver(activation, **kwargs)

        assert aggr == "add" or aggr == "mean"
        self.aggr = aggr
        self.predict = nn.Sequential(
            Dense(in_dim, hidden_dim, bias=True, activation_name=activation, **kwargs),
            act,
            Dense(hidden_dim, out_dim, bias=False),
        )
        if scaler is None:
            self.scaler = None
        else:
            if mean is None:
                mean = torch.FloatTensor([0.0])
            if stddev is None:
                stddev = torch.FloatTensor([1.0])
            self.scaler = scaler(mean, stddev)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.predict:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(self, x: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        """
        Compute global property from node embeddings.

        Args:
            x (Tensor): node embeddings shape of (num_node x in_dim).
            batch (Tensor, optional): batch index. Defaults to `None`.

        Returns:
            Tensor: shape of (num_batch x out_dim).
        """
        out = self.predict(x)
        if self.scaler is not None:
            out = self.scaler(out)
        return scatter(out, index=batch, dim=0, reduce=self.aggr)


class Edge2NodeProperty(nn.Module):
    """
    The block to compute the node-wise proptery from edge embeddings.
    This block contains FNN layers and aggregation block of all neighbor.
    This block is used in Dimenet.
    """

    def __init__(
        self,
        hidden_dim: int,
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
        self.rbf_lin = Dense(n_radial, hidden_dim, bias=False)
        lins = []
        for _ in range(n_layers):
            lins.append(
                Dense(
                    hidden_dim,
                    hidden_dim,
                    bias=True,
                    activation_name=activation,
                    **kwargs,
                )
            )
            lins.append(act)
        lins.append(Dense(hidden_dim, out_dim, bias=False))
        self.lins = nn.Sequential(*lins)

        self.reset_parameters()

    def reset_parameters(self):
        for ll in self.lins:
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
            x (Tensor): edge embedding shape of (num_edge x hidden_dim).
            rbf (Tensor): radial basis function shape of (num_node x n_radial).
            idx_i (torch.LongTensor): node index center atom i shape of (num_edge).
            num_nodes (Optional[int], optional): number of edge. Defaults to `None`.

        Returns:
            Tensor: node-wise properties shape of (num_node x out_dim).
        """
        x = self.rbf_lin(rbf) * x
        # add all neighbor atoms
        x = scatter(x, idx_i, dim=0, dim_size=num_nodes, reduce=self.aggr)
        return self.lins(x)
