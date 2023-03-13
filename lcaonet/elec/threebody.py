from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from .table import ELEC_TABLE, MAX_ELEC_IDX, VALENCE_TABLE

# 1s,
# 2s,
# 2p, 2p,
# 3s,
# 3p, 3p,
# 4s,
# 3d, 3d, 3d,
# 4p, 4p,
# 5s,
# 4d, 4d, 4d,
# 5p, 5p,
# 6s,
# 4f, 4f, 4f, 4f,
# 5d, 5d, 5d,
# 6p, 6p,
# 7s
# 5f, 5f, 5f, 5f,
# 6d, 6d, 6d,
N_ORB_THREE_BODY = 37


def _modify_elec_table(elec: Tensor = ELEC_TABLE) -> Tensor:
    def modify_one(one_elec: Tensor) -> Tensor:
        out_elec = torch.zeros((N_ORB_THREE_BODY,), dtype=torch.long)
        j = 0
        for i in range(one_elec.size(0)):
            # s
            if np.isin([0, 1, 3, 5, 8, 11, 15], i).any():
                out_elec[j] = one_elec[i]
                j += 1
            # p
            if np.isin([2, 4, 7, 10, 14], i).any():
                for _ in range(2):
                    out_elec[j] = one_elec[i]
                    j += 1
            # d
            if np.isin([6, 9, 13, 17], i).any():
                for _ in range(3):
                    out_elec[j] = one_elec[i]
                    j += 1
            # f
            if np.isin([12, 16], i).any():
                for _ in range(4):
                    out_elec[j] = one_elec[i]
                    j += 1
        return out_elec

    out_elec = torch.zeros((len(elec), N_ORB_THREE_BODY), dtype=torch.long)
    for i in range(len(elec)):
        out_elec[i] = modify_one(elec[i])

    return out_elec


def _modify_max_idx(max_idx: list[int] = MAX_ELEC_IDX) -> Tensor:
    out_max_ind = torch.zeros((N_ORB_THREE_BODY,), dtype=torch.long)
    j = 0
    for i in range(len(max_idx)):
        # s
        if np.isin([0, 1, 3, 5, 8, 11, 15], i).any():
            out_max_ind[j] = max_idx[i]
            j += 1
        # p
        if np.isin([2, 4, 7, 10, 14], i).any():
            for _ in range(2):
                out_max_ind[j] = max_idx[i]
                j += 1
        # d
        if np.isin([6, 9, 13, 17], i).any():
            for _ in range(3):
                out_max_ind[j] = max_idx[i]
                j += 1
        # f
        if np.isin([12, 16], i).any():
            for _ in range(4):
                out_max_ind[j] = max_idx[i]
                j += 1
    return out_max_ind


ELEC_TABLE_THREE_BODY = _modify_elec_table()
VALENCE_TABLE_THREE_BODY = _modify_elec_table(VALENCE_TABLE)
MAX_ELEC_IDX_THREE_BODY = _modify_max_idx()

# fmt: off
NL_LIST_THREE_BODY: list[tuple[int, int]] = [
    (1, 0),                             # 1s
    (2, 0),                             # 2s
    (2, 1), (2, 1),                     # 2p
    (3, 0),                             # 3s
    (3, 1), (3, 1),                     # 3p
    (4, 0),                             # 4s
    (3, 2), (3, 2), (3, 2),             # 3d
    (4, 1), (4, 1),                     # 4p
    (5, 0),                             # 5s
    (4, 2), (4, 2), (4, 2),             # 4d
    (5, 1), (5, 1),                     # 5p
    (6, 0),                             # 6s
    (4, 3), (4, 3), (4, 3), (4, 3),     # 4f
    (5, 2), (5, 2), (5, 2),             # 5d
    (6, 1), (6, 1),                     # 6p
    (7, 0),                             # 7s
    (5, 3), (5, 3), (5, 3), (5, 3),     # 5f
    (6, 2), (6, 2), (6, 2),             # 6d
]
# fmt: on


def get_max_nl_index_byz(max_z: int) -> int:
    if max_z <= 2:
        return 0
    if max_z <= 4:
        return 1
    if max_z <= 10:
        return 3
    if max_z <= 12:
        return 4
    if max_z <= 18:
        return 6
    if max_z <= 20:
        return 7
    if max_z <= 30:
        return 10
    if max_z <= 36:
        return 12
    if max_z <= 38:
        return 13
    if max_z <= 48:
        return 16
    if max_z <= 54:
        return 18
    if max_z <= 56:
        return 19
    if max_z <= 80:
        return 26
    if max_z <= 86:
        return 28
    if max_z <= 88:
        return 29
    if max_z <= 96:
        return 36
    raise ValueError(f"max_z={max_z} is too large")


def get_max_nl_index_byorb(max_orb: str) -> int:
    if max_orb == "1s":
        return 0
    if max_orb == "2s":
        return 1
    if max_orb == "2p":
        return 3
    if max_orb == "3s":
        return 4
    if max_orb == "3p":
        return 6
    if max_orb == "4s":
        return 7
    if max_orb == "3d":
        return 10
    if max_orb == "4p":
        return 12
    if max_orb == "5s":
        return 13
    if max_orb == "4d":
        return 16
    if max_orb == "5p":
        return 18
    if max_orb == "6s":
        return 19
    if max_orb == "4f":
        return 23
    if max_orb == "5d":
        return 26
    if max_orb == "6p":
        return 28
    if max_orb == "7s":
        return 29
    if max_orb == "5f":
        return 33
    if max_orb == "6d":
        return 36
    raise ValueError(f"max_orb={max_orb} is not supported")


def get_elec_table(max_z: int, max_idx: int) -> Tensor:
    return ELEC_TABLE_THREE_BODY[: max_z + 1, : max_idx + 1]


def get_valence_table(max_z: int, max_idx: int) -> Tensor:
    return VALENCE_TABLE_THREE_BODY[: max_z + 1, : max_idx + 1]


def get_max_elec_idx(max_z: int) -> Tensor:
    return MAX_ELEC_IDX_THREE_BODY[: max_z + 1]


def get_nl_list(max_idx: int) -> list[tuple[int, int]]:
    return NL_LIST_THREE_BODY[: max_idx + 1]
