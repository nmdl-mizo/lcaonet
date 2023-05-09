from __future__ import annotations

import torch
from torch import Tensor

from .elec import ELEC_TABLE, MAX_ELEC_IDX, NL_LIST, VALENCE_TABLE


class ElecInfo:
    """Objects that manage electronic structure information, valence electron
    information, and quantum numbers for each element."""

    def __init__(self, max_z: int, max_orb: str | None, n_per_orb: int = 1):
        """
        Args:
            max_z (int): the maximum atomic number.
            max_orb (str | None): the maximum orbital name like `"3s"`. When using this parameter, use it when you want to include orbitals beyond the valence orbitals.
                For example, when `max_z=6`, the largest orbital is `"2p"`, but by setting `max_orb="3s"`, the message passing can take into account the basis up to 3s orbitals.
            n_per_orb (int): Number of bases used per orbit. Default is `1`.
        """  # noqa: E501
        self.max_z = max_z
        self.max_orb = max_orb
        self.n_per_orb = n_per_orb

        self._max_orb_idx = self._get_max_orb_idx_byz(max_z)
        if max_orb is not None:
            self._max_orb_idx = max(self._max_orb_idx, self._get_max_orb_idx_byorb(max_orb))
        # Number of orbits used for MP
        self._n_orb = (self._max_orb_idx + 1) * self.n_per_orb

        # original values
        self._elec_table = ELEC_TABLE
        self._valence_table = VALENCE_TABLE
        self._max_elec_idx = MAX_ELEC_IDX
        self._nl_list = NL_LIST

    def _get_max_orb_idx_byz(self, max_z: int) -> int:
        if max_z <= 0:
            raise ValueError(f"max_z={max_z} is too small.")
        if max_z <= 2:
            return 0
        elif max_z <= 4:
            return 1
        elif max_z <= 10:
            return 2
        elif max_z <= 12:
            return 3
        elif max_z <= 18:
            return 4
        elif max_z <= 20:
            return 5
        elif max_z <= 30:
            return 6
        elif max_z <= 36:
            return 7
        elif max_z <= 38:
            return 8
        elif max_z <= 48:
            return 9
        elif max_z <= 54:
            return 10
        elif max_z <= 56:
            return 11
        elif max_z <= 80:
            return 13
        elif max_z <= 86:
            return 14
        elif max_z <= 88:
            return 15
        elif max_z <= 96:
            return 17
        else:
            raise ValueError(f"max_z={max_z} is too large.")

    def _get_max_orb_idx_byorb(self, max_orb: str) -> int:
        orb_idx_dict = {
            "1s": 0,
            "2s": 1,
            "2p": 2,
            "3s": 3,
            "3p": 4,
            "4s": 5,
            "3d": 6,
            "4p": 7,
            "5s": 8,
            "4d": 9,
            "5p": 10,
            "6s": 11,
            "4f": 12,
            "5d": 13,
            "6p": 14,
            "7s": 15,
            "5f": 16,
            "6d": 17,
        }
        idx = orb_idx_dict.get(max_orb, None)
        if idx is None:
            raise ValueError(f"max_orb={max_orb} is not supported.")
        return idx

    @property
    def n_orb(self) -> int:
        return self._n_orb

    @property
    def elec_table(self) -> Tensor:
        return self._elec_table[: self.max_z + 1, : self._max_orb_idx + 1].repeat_interleave(self.n_per_orb, dim=1)

    @property
    def elec_mask(self) -> Tensor:
        return torch.where(self.elec_table > 0, 1, 0)

    @property
    def valence_table(self) -> Tensor:
        return self._valence_table[: self.max_z + 1, : self._max_orb_idx + 1].repeat_interleave(self.n_per_orb, dim=1)

    @property
    def max_elec_idx(self) -> Tensor:
        return self._max_elec_idx[: self._max_orb_idx + 1].repeat_interleave(self.n_per_orb, dim=0)

    @property
    def nl_list(self) -> Tensor:
        return self._nl_list[: self._max_orb_idx + 1].repeat_interleave(self.n_per_orb, dim=0)
