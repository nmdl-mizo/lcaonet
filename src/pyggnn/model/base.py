import torch
from torch import Tensor
import torch.nn as nn

from pyggnn.data.datakeys import DataKeys


__all__ = ["BaseGNN"]


class BaseGNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def reset_parameters(self):
        # must be implemented in child class
        return NotImplementedError

    def calc_atomic_distances(self, data_batch) -> Tensor:
        if data_batch.get(DataKeys.Batch) is not None:
            batch = data_batch[DataKeys.Batch]
        else:
            batch = data_batch[DataKeys.Position].new_zeros(
                data_batch[DataKeys.Position].shape[0], dtype=torch.long
            )

        edge_src, edge_dst = (
            data_batch[DataKeys.Edge_index][0],
            data_batch[DataKeys.Edge_index][1],
        )
        edge_batch = batch[edge_src]
        edge_vec = (
            data_batch[DataKeys.Position][edge_dst]
            - data_batch[DataKeys.Position][edge_src]
            + torch.einsum(
                "ni,nij->nj",
                data_batch[DataKeys.Edge_shift],
                data_batch[DataKeys.Lattice][edge_batch],
            )
        )
        return torch.norm(edge_vec, dim=1)
