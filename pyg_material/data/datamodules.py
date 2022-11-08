from __future__ import annotations  # type: ignore

import logging

import pytorch_lightning as pl
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from pyg_material.data.dataset import (
    BaseGraphDataset,
    Db2GraphDataset,
    Hdf2GraphDataset,
)

logger = logging.getLogger(__name__)


class File2GraphDataModule(pl.LightningDataModule):
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
            logger.error(f"file_type={file_type} is not supported. Please use hdf5 or db.")
            raise ValueError(f"file_type={file_type} is not supported. Please use hdf5 or db.")
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
            self.test_idx = list(
                range(
                    self.num_train + self.num_val,
                    self.num_train + self.num_val + self.num_test,
                )
            )
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

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class Dataset2GraphDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset: BaseGraphDataset,
        val_dataset: BaseGraphDataset,
        test_dataset: BaseGraphDataset | None = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        logger.info(f"Number of train_dataset: {len(self.train_dataset)}")
        self.val_dataset = val_dataset
        logger.info(f"Number of val_dataset: {len(self.val_dataset)}")
        if test_dataset is None:
            self.test_dataset = Subset(self.val_dataset, [])
            logger.info("test_dataset is None.")
        else:
            self.test_dataset = test_dataset
            logger.info(f"Number of test_dataset: {len(self.test_dataset)}")
        self.batch_size = batch_size
        logger.info(f"batch_size: {self.batch_size}")
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
