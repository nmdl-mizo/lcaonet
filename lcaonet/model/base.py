from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch
from torch_sparse import SparseTensor

from lcaonet.data.keys import GraphKeys


class BaseMPNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def calc_atomic_distances(self, batch: Batch, return_vec: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        """calculate atomic distances for periodic boundary conditions.

        Args:
            batch (torch_geometric.data.Batch): material graph batch.
            return_vec (bool, optional): return distance vector. Defaults to `False`.

        Returns:
            distance (torch.Tensor): inter atomic distances of (num_edge) shape.
        """
        if batch.get(GraphKeys.Batch_idx) is not None:
            batch_ind = batch[GraphKeys.Batch_idx]
        else:
            batch_ind = batch[GraphKeys.Pos].new_zeros(batch[GraphKeys.Pos].shape[0], dtype=torch.long)

        # order is "source_to_traget" i.e. [index_j, index_i]
        edge_dst, edge_src = batch[GraphKeys.Edge_idx]
        edge_batch = batch_ind[edge_src]
        edge_vec = (
            batch[GraphKeys.Pos][edge_dst]
            - batch[GraphKeys.Pos][edge_src]
            + torch.einsum("ni,nij->nj", batch[GraphKeys.Edge_shift], batch[GraphKeys.Lattice][edge_batch]).contiguous()
        )
        if return_vec:
            return torch.norm(edge_vec, dim=1), edge_vec
        return torch.norm(edge_vec, dim=1)

    def get_triplets(self, batch: Batch) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Convert edge_index to triplets.

        Args:
            batch (torch_geometirc.data.Batch): material graph batch.

        Returns:
            idx_i (Tensor): index of center atom i of (num_edge) shape.
            idx_j (Tensor): index of neighbor atom j of (num_edge) shape.
            tri_idx_i (Tensor): index of atom i of (num_triplets) shape.
            tri_idx_j (Tensor): index of atom j of (num_triplets) shape.
            tri_idx_k (Tensor): index of atom k of (num_triplets) shape.
            edge_idx_kj (Tensor): edge index of center k to j of (num_triplets) shape.
            edge_idx_ji (Tensor): edge index of center j to i of (num_triplets) shape.

        Notes:
            ref:
                https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/dimenet.html
        """
        # order is "source_to_traget" i.e. [index_j, index_i]
        idx_j, idx_i = batch[GraphKeys.Edge_idx]

        value = torch.arange(idx_j.size(0), device=idx_j.device)
        num_nodes = batch[GraphKeys.Z].size(0)
        adj_t = SparseTensor(row=idx_i, col=idx_j, value=value, sparse_sizes=(num_nodes, num_nodes))
        adj_t_row = adj_t[idx_j]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        # Node indices (i, j, k) for triplets.
        tri_idx_i = idx_i.repeat_interleave(num_triplets)
        tri_idx_j = idx_j.repeat_interleave(num_triplets)
        tri_idx_k = adj_t_row.storage.col()
        # Remove i == k triplets.
        mask = tri_idx_i != tri_idx_k
        tri_idx_i, tri_idx_j, tri_idx_k = (tri_idx_i[mask], tri_idx_j[mask], tri_idx_k[mask])

        # Edge indices (k -> j) and (j -> i) for triplets.
        edge_idx_kj = adj_t_row.storage.value()[mask]
        edge_idx_ji = adj_t_row.storage.row()[mask]

        return (idx_i, idx_j, tri_idx_i, tri_idx_j, tri_idx_k, edge_idx_kj, edge_idx_ji)

    @property
    def n_param(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
