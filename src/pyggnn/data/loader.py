from typing import List

import torch
import torch_geometric
from pyggnn.data.datakeys import DataKeys

from pyggnn.data.dataset import BaseGraphDataset

__all__ = ["GraphLoader"]


class Collater:
    def __init__(self, device: torch.device):
        self.device = device

    def __call__(self, batch: List[BaseGraphDataset]):
        n_batch = 0
        edge_ind_max = 0
        batch_ind = torch.empty(0).to(torch.long)
        datas = torch_geometric.data.Data()
        for b in batch:
            n_node = b[DataKeys.Atomic_num].size(0)
            # add batch index
            batch_ind = torch.cat(
                [batch_ind, torch.full((n_node,), n_batch, dtype=torch.long)], dim=0
            )
            # shift the index of the edge
            edge_ind = b[DataKeys.Edge_index] + edge_ind_max
            # update all keys
            for k, v in b:
                if k == DataKeys.Edge_index:
                    if datas.get(k) is None:
                        datas[k] = edge_ind
                    else:
                        datas[k] = torch.cat([datas[k], edge_ind], dim=1)
                if datas.get(k) is None:
                    datas[k] = v
                else:
                    datas[k] = torch.cat([datas[k], v], dim=0)
            n_batch += 1
            edge_ind_max += n_node
        # add batch index
        datas[DataKeys.Batch] = batch_ind

        return datas


class GraphLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset: BaseGraphDataset,
        batch_size: int = 1,
        shuffle: bool = False,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ):
        if kwargs.get("collate_fn") is None:
            del kwargs["collate_fn"]

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=Collater(device),
            **kwargs,
        )
