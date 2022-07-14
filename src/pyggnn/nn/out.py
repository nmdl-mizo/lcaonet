from typing import Literal, Optional

from torch import Tensor
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool

from pyggnn.nn.activation import Swish
from pyggnn.nn.base import Dense


__all__ = ["Node2Property"]


class Node2Property(nn.Module):
    """
    The block to compute the global graph proptery from node embeddings.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 1,
        beta: Optional[float] = None,
        aggr: Literal["add", "mean"] = "add",
    ):
        """
        Args:
            in_dim (int): number of input dim.
            hidden_dim (int, optional): number of hidden layers dim. Defaults to `128`.
            out_dim (int, optional): number of output dim. Defaults to `1`.
            beta (float, optional): if set to `None`, beta is not learnable parameters.
                Defaults to `None`.
            aggr (`"add"` or `"mean"`): aggregation method. Defaults to `"add"`.
        """
        super().__init__()
        aggregation = {"add": global_add_pool, "mean": global_mean_pool}
        assert aggr == "add" or aggr == "mean"
        self.aggr = aggr
        self.node_transform = nn.Sequential(
            Dense(in_dim, hidden_dim, bias=True),
            Swish(beta),
            Dense(hidden_dim, hidden_dim, bias=True),
        )
        self.aggregate = aggregation[aggr]
        self.predict = nn.Sequential(
            Dense(hidden_dim, hidden_dim, bias=True),
            Swish(beta),
            Dense(hidden_dim, out_dim, bias=False),
        )

    def forward(self, x: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        out = self.node_transform(x)
        out = self.aggregate(out, batch=batch)
        return self.predict(out)
