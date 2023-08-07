from __future__ import annotations

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
from torch_geometric.data import Data


def full_linked_graph(n_nodes: int) -> tuple[ndarray, ndarray]:
    # get all pair permutations of atom indices
    r = np.arange(n_nodes)
    x, y = np.meshgrid(r, r)
    ind = np.stack([y.ravel(), x.ravel()], axis=1)
    # remove self edge
    ind = ind[ind[:, 0] != ind[:, 1]]

    shift = np.zeros((ind.shape[0], 3), np.float32)

    return ind.T, shift


def _set_data(
    data: Data,
    k: str,
    v: int | float | ndarray | Tensor,
    add_dim: bool,
    add_batch: bool,
    dtype: torch.dtype,
):
    if add_dim:
        val = torch.tensor([v], dtype=dtype)
    else:
        val = torch.tensor(v, dtype=dtype)
    data[k] = val.unsqueeze(0) if add_batch else val


def set_properties(
    data: Data,
    k: str,
    v: int | float | str | ndarray | Tensor,
    add_batch: bool = True,
):
    if isinstance(v, int):
        _set_data(data, k, v, add_dim=True, add_batch=add_batch, dtype=torch.long)
    elif isinstance(v, float):
        _set_data(data, k, v, add_dim=True, add_batch=add_batch, dtype=torch.float32)
    elif isinstance(v, str):
        data[k] = v
    elif len(v.shape) == 0:
        # for 0-dim array
        if isinstance(v, ndarray):
            dtype = torch.long if v.dtype == int else torch.float32
        elif isinstance(v, Tensor):
            dtype = v.dtype
        else:
            raise ValueError(f"Unknown type of {v}")
        _set_data(data, k, v, add_dim=True, add_batch=add_batch, dtype=dtype)
    else:
        # for array-like
        if isinstance(v, ndarray):
            dtype = torch.long if v.dtype == int else torch.float32
        elif isinstance(v, Tensor):
            dtype = v.dtype
        else:
            raise ValueError(f"Unknown type of {v}")
        _set_data(data, k, v, add_dim=False, add_batch=add_batch, dtype=dtype)
