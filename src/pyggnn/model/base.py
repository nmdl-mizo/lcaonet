import torch
from torch import Tensor
import torch.nn as nn

from pyggnn.data import DataKeys


__all__ = ["BaseGNN"]


class BaseGNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def calc_atomic_distances(self, data) -> Tensor:
        if data.get(DataKeys.Batch) is not None:
            batch = data[DataKeys.Batch]
        else:
            batch = data[DataKeys.Position].new_zeros(
                data[DataKeys.Position].shape[0], dtype=torch.long
            )

        edge_src, edge_dst = data[DataKeys.Edge_index][0], data[DataKeys.Edge_index][1]
        edge_batch = batch[edge_src]
        edge_vec = (
            data[DataKeys.Position][edge_dst]
            - data[DataKeys.Position][edge_src]
            + torch.einsum(
                "ni,nij->nj",
                data[DataKeys.Edge_shift],
                data[DataKeys.Lattice][edge_batch],
            )
        )
        return torch.norm(edge_vec, dim=1)
