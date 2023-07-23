from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Batch

from lcaonet.data.keys import GraphKeys


class BaseMPNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def calc_atomic_distances(self, graph: Batch, return_vec: bool = False) -> Batch:
        """calculate atomic distances for periodic boundary conditions.

        Args:
            graph (torch_geometric.data.Batch): material graph batch.
            return_vec (bool, optional): return normalized distance vector from i to j atom. Defaults to `False`.

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
            graph[GraphKeys.Edge_vec] = edge_vec / graph[GraphKeys.Edge_dist].unsqueeze(-1)
        return graph

    @property
    def n_param(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
