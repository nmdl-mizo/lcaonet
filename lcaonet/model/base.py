from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_sparse import SparseTensor

from lcaonet.data.keys import GraphKeys


class BaseMPNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def calc_atomic_distances(self, graph: Batch, return_vec: bool = False) -> Batch:
        """calculate atomic distances for periodic boundary conditions.

        Args:
            graph (torch_geometric.data.Batch): material graph batch.
            return_vec (bool, optional): return distance vector from i to j atom. Defaults to `False`.

        Returns:
            graph (torch_geometric.data.Batch): material graph batch with edge information:
                distance (torch.Tensor): inter atomic distances of (E) shape.
                pair_vec_ij (torch.Tensor): inter atomic distance vector from i to j atom of (E, 3) shape.
        """
        if graph.get(GraphKeys.Batch_idx) is not None:
            batch_ind = graph[GraphKeys.Batch_idx]
        else:
            batch_ind = graph[GraphKeys.Pos].new_zeros(graph[GraphKeys.Pos].shape[0], dtype=torch.long)

        # order is "source_to_traget" i.e. [index_j, index_i]
        edge_dst, edge_src = graph[GraphKeys.Edge_idx]
        edge_batch = batch_ind[edge_src]
        edge_vec = (
            graph[GraphKeys.Pos][edge_dst]
            - graph[GraphKeys.Pos][edge_src]
            + torch.einsum("ni,nij->nj", graph[GraphKeys.Edge_shift], graph[GraphKeys.Lattice][edge_batch]).contiguous()
        )
        graph[GraphKeys.Edge_dist] = torch.norm(edge_vec, dim=1)
        if return_vec:
            graph[GraphKeys.Edge_vec] = edge_vec
        return graph

    def calc_3body_angles(self, graph: Batch) -> Batch:
        """calculate 3body angles for periodic boundary conditions.

        Args:
            graph (torch_geometric.data.Batch): material graph batch.

        Returns:
            graph (torch_geometric.data.Batch): material graph batch with 3body angles:
                angles (torch.Tensor): angle of ijk with (n_triplets) shape.
        """
        pair_vec_ij = graph.get(GraphKeys.Edge_vec)
        if pair_vec_ij is None:
            raise ValueError("edge_vec is not calculated. Please run calc_atomic_distances(return_vec=True) first.")
        edge_idx_ji, edge_idx_kj = graph[GraphKeys.Edge_idx_ji_3b], graph[GraphKeys.Edge_idx_kj_3b]

        vec_ij, vec_jk = pair_vec_ij[edge_idx_ji], pair_vec_ij[edge_idx_kj]
        inner = (vec_ij * vec_jk).sum(dim=-1)
        outter = torch.cross(vec_ij, vec_jk).norm(dim=-1)
        # arctan is more stable than arccos
        angles = torch.atan2(outter, inner)

        graph[GraphKeys.Angles_3b] = angles
        return graph

    def get_triplets(self, graph: Batch) -> Batch:
        """Convert edge_index to triplets.

        Args:
            graph (torch_geometirc.data.Batch): material graph batch.

        Returns:
            graph (torch_geometric.data.Batch): material graph batch with 3body index:
                tri_idx_i (Tensor): index of atom i of (n_triplets) shape.
                tri_idx_j (Tensor): index of atom j of (n_triplets) shape.
                tri_idx_k (Tensor): index of atom k of (n_triplets) shape.
                edge_idx_kj (Tensor): edge index of center k to j of (n_triplets) shape.
                edge_idx_ji (Tensor): edge index of center j to i of (n_triplets) shape.

        Notes:
            ref:
                https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/dimenet.html
        """
        # order is "source_to_traget" i.e. [index_j, index_i]
        idx_j, idx_i = graph[GraphKeys.Edge_idx]

        value = torch.arange(idx_j.size(0), device=idx_j.device)
        num_nodes = graph[GraphKeys.Z].size(0)
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
        # The position of j in the pair edge_index becomes k and the position of i becomes j
        edge_idx_kj = adj_t_row.storage.value()[mask]
        edge_idx_ji = adj_t_row.storage.row()[mask]

        graph[GraphKeys.Idx_i_3b] = tri_idx_i
        graph[GraphKeys.Idx_j_3b] = tri_idx_j
        graph[GraphKeys.Idx_k_3b] = tri_idx_k
        graph[GraphKeys.Edge_idx_kj_3b] = edge_idx_kj
        graph[GraphKeys.Edge_idx_ji_3b] = edge_idx_ji
        return graph

    @property
    def n_param(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
