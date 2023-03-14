from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from .elec import ELEC_TABLE, MAX_ELEC_IDX, NL_LIST, VALENCE_TABLE
from .exponent import EXPONENT_TABLE


class BaseAtomisticInformation:
    def __init__(self, max_z: int, max_orb: str | None, limit_n_orb: int | None = None):
        self.max_z = max_z
        self.max_orb = max_orb
        self.limit_n_orb = limit_n_orb

        # original values
        self._elec_table = ELEC_TABLE
        self._valence_table = VALENCE_TABLE
        self._exponent_table = EXPONENT_TABLE

        self._max_elec_idx = MAX_ELEC_IDX

        self._nl_list = NL_LIST

    @property
    def get_elec_table(self) -> Tensor:
        raise NotImplementedError

    @property
    def get_valence_table(self) -> Tensor:
        raise NotImplementedError

    @property
    def get_exponent_table(self) -> Tensor:
        raise NotImplementedError

    @property
    def get_max_elec_idx(self) -> Tensor:
        raise NotImplementedError

    @property
    def get_nl_list(self) -> list[tuple[int, int]]:
        raise NotImplementedError


class ThreeBodyAtomisticInformation(BaseAtomisticInformation):
    def __init__(self, max_z: int, max_orb: str | None, limit_n_orb: int | None = None):
        super().__init__(max_z, max_orb, limit_n_orb)

        # get threebody values
        self.threeb_n_orb = 37 if self.limit_n_orb is None else self._calc_three_body_n_orb(self.limit_n_orb)
        self.max_orb_idx = self._get_max_orb_idx_byz(self.threeb_n_orb)
        if self.max_orb_idx is not None:
            self.max_orb_idx = max(self.max_orb_idx, self._get_max_orb_idx_byorb(self.threeb_n_orb))

        self._elec_table_threeb = self._mod_table_threeb(self._elec_table, self.threeb_n_orb)
        self._valence_table_threeb = self._mod_table_threeb(self._valence_table, self.threeb_n_orb)
        self._exponent_table_threeb = self._mod_table_threeb(self._exponent_table, self.threeb_n_orb)

        self._max_elec_idx_threeb = self._mod_max_idx_threeb(self._max_elec_idx, self.threeb_n_orb)

        self._nl_list_threeb = self._mod_nl_list_threeb(self._nl_list)

    def _calc_three_body_n_orb(self, limit_n_orb: int) -> int:
        three_body_n_orb = 0
        for i in range(limit_n_orb):
            # s
            if np.isin([0, 1, 3, 5, 8, 11, 15], i).any():
                three_body_n_orb += 1
            # p
            elif np.isin([2, 4, 7, 10, 14], i).any():
                three_body_n_orb += 2
            # d
            elif np.isin([6, 9, 13, 17], i).any():
                three_body_n_orb += 3
            # f
            elif np.isin([12, 16], i).any():
                three_body_n_orb += 4
        return three_body_n_orb

    def _get_max_orb_idx_byz(self, limit_threeb_n_orb: int) -> int:
        max_z = self.max_z
        idx = 0
        if max_z <= 2:
            idx = 0
        elif max_z <= 4:
            idx = 1
        elif max_z <= 10:
            idx = 3
        elif max_z <= 12:
            idx = 4
        elif max_z <= 18:
            idx = 6
        elif max_z <= 20:
            idx = 7
        elif max_z <= 30:
            idx = 10
        elif max_z <= 36:
            idx = 12
        elif max_z <= 38:
            idx = 13
        elif max_z <= 48:
            idx = 16
        elif max_z <= 54:
            idx = 18
        elif max_z <= 56:
            idx = 19
        elif max_z <= 80:
            idx = 26
        elif max_z <= 86:
            idx = 28
        elif max_z <= 88:
            idx = 29
        elif max_z <= 96:
            idx = 36
        else:
            raise ValueError(f"max_z={max_z} is too large")
        if idx >= limit_threeb_n_orb:
            raise ValueError(f"max_z={max_z} is too large in limit_threeb_n_orb={limit_threeb_n_orb}")
        return idx

    def _get_max_orb_idx_byorb(self, limit_threeb_n_orb: int) -> int:
        max_orb = self.max_orb
        idx = 0
        if max_orb == "1s":
            idx = 0
        elif max_orb == "2s":
            idx = 1
        elif max_orb == "2p":
            idx = 3
        elif max_orb == "3s":
            idx = 4
        elif max_orb == "3p":
            idx = 6
        elif max_orb == "4s":
            idx = 7
        elif max_orb == "3d":
            idx = 10
        elif max_orb == "4p":
            idx = 12
        elif max_orb == "5s":
            idx = 13
        elif max_orb == "4d":
            idx = 16
        elif max_orb == "5p":
            idx = 18
        elif max_orb == "6s":
            idx = 19
        elif max_orb == "4f":
            idx = 23
        elif max_orb == "5d":
            idx = 26
        elif max_orb == "6p":
            idx = 28
        elif max_orb == "7s":
            idx = 29
        elif max_orb == "5f":
            idx = 33
        elif max_orb == "6d":
            idx = 36
        else:
            raise ValueError(f"max_orb={max_orb} is not supported")
        if idx >= limit_threeb_n_orb:
            raise ValueError(f"max_orb={max_orb} is too large in limit_threeb_n_orb={limit_threeb_n_orb}")
        return idx

    def _mod_table_threeb(self, tbl: Tensor, three_body_n_orb: int) -> Tensor:
        def mod_one(one_elec: Tensor) -> Tensor:
            out_elec = torch.zeros((three_body_n_orb,), dtype=torch.long)
            j = 0
            # iteration on original table
            for i in range(one_elec.size(0)):
                if self.limit_n_orb is not None and i >= self.limit_n_orb:
                    break
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

        out_elec = torch.zeros((len(tbl), three_body_n_orb), dtype=torch.long)
        for i in range(len(tbl)):
            out_elec[i] = mod_one(tbl[i])

        return out_elec

    def _mod_max_idx_threeb(self, max_idx: list[int], three_body_n_orb: int) -> Tensor:
        out_max_ind = torch.zeros((three_body_n_orb,), dtype=torch.long)
        j = 0
        # iteration on original table
        for i in range(len(max_idx)):
            if self.limit_n_orb is not None and i >= self.limit_n_orb:
                break
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

    def _mod_nl_list_threeb(self, nl_list: list[tuple[int, int]]) -> list[tuple[int, int]]:
        out_nl_list = []
        # iteration on original list
        for i, nl in enumerate(nl_list):
            if self.limit_n_orb is not None and i >= self.limit_n_orb:
                break
            # s
            if nl[1] == 0:
                out_nl_list.append(nl)
            # p
            elif nl[1] == 1:
                for _ in range(2):
                    out_nl_list.append(nl)
            # d
            elif nl[1] == 2:
                for _ in range(3):
                    out_nl_list.append(nl)
            # f
            elif nl[1] == 3:
                for _ in range(4):
                    out_nl_list.append(nl)
        return out_nl_list

    @property
    def get_elec_table(self) -> Tensor:
        return self._elec_table_threeb[: self.max_z + 1, : self.max_orb_idx + 1]

    @property
    def get_valence_table(self) -> Tensor:
        return self._valence_table_threeb[: self.max_z + 1, : self.max_orb_idx + 1]

    @property
    def get_exponent_table(self) -> Tensor:
        return self._exponent_table_threeb[: self.max_z + 1, : self.max_orb_idx + 1]

    @property
    def get_max_elec_idx(self) -> Tensor:
        return self._max_elec_idx_threeb[: self.max_orb_idx + 1]

    @property
    def get_nl_list(self) -> list[tuple[int, int]]:
        return self._nl_list_threeb[: self.max_orb_idx + 1]
