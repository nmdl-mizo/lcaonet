from __future__ import annotations  # type: ignore

import logging

import pytorch_lightning as pl
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from pyggnns.data.dataset import Db2GraphDataset, Hdf2GraphDataset

log = logging.getLogger(__name__)


__all__ = ["GraphDataModule", "GraphDataModuleSplit"]


class GraphDataModule(pl.LightningDataModule):
    def __init__(
        self,
        file_pth: str,
        file_type: str,
        cutoff_radi: float,
        property_names: list[str],
        pbc: bool | tuple[bool, ...],
        num_train: int,
        num_val: int,
        num_test: int | None = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = False,
    ):
        super().__init__()
        if file_type == "hdf5":
            self.dataset = Hdf2GraphDataset(file_pth, cutoff_radi, property_names, pbc)
        elif file_type == "db":
            self.dataset = Db2GraphDataset(file_pth, cutoff_radi, property_names, pbc)
        else:
            log.error(f"file_type {file_type} is not supported. Please use hdf5 or db.")
            raise ValueError("file_type is not supported. Please use hdf5 or db.")
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: str | None = None):
        # TODO: split
        self.train_idx = list(range(self.num_train))
        self.val_idx = list(range(self.num_train, self.num_train + self.num_val))
        self.test_idx = []
        if self.num_test is not None:
            self.test_idx = list(range(self.num_train + self.num_val, self.num_train + self.num_val + self.num_test))
        self._train_dataset = Subset(self.dataset, self.train_idx)
        self._val_dataset = Subset(self.dataset, self.val_idx)
        self._test_dataset = Subset(self.dataset, self.test_idx)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader | None:
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader | None:
        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class GraphDataModuleSplit(pl.LightningDataModule):
    pass
