from __future__ import annotations

import logging

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

log = logging.getLogger(__name__)

__all__ = ["get_data"]


def get_data(config: DictConfig) -> pl.LightningDataModule:
    """Get data from config.

    Args:
        config (DictConfig): Config.

    Returns:
        pl.LightningDataModule: Data module object.
    """
    if config.module._target_ == "pyggnns.data.GraphDataModule":
        datamodule: pl.LightningDataModule = hydra.utils.instantiate(config)
    elif config.module._target_ == "pyggnns.data.GraphDataModuleSplit":
        log.info(f"Setting up training data: {config.train_dataset._target_}")
        train_dataset = None
        for pth in config.files.hdf5_path:
            if train_dataset is None:
                train_dataset = hydra.utils.instantiate(config.train_dataset, hdf5_path=pth)
            else:
                train_dataset += hydra.utils.instantiate(config.train_dataset, hdf5_path=pth)
        log.info(f"Setting up validation data: {config.val_dataset._target_}")
        val_dataset = None
        for pth in config.files.hdf5_path:
            if val_dataset is None:
                val_dataset = hydra.utils.instantiate(config.val_dataset, hdf5_path=pth)
            else:
                val_dataset += hydra.utils.instantiate(config.val_dataset, hdf5_path=pth)
        test_dataset = None
        if config.test_dataset is not None:
            log.info(f"Setting up test data: {config.test_dataset._target_}")
            for pth in config.files.hdf5_path:
                if test_dataset is None:
                    test_dataset = hydra.utils.instantiate(config.test_dataset, hdf5_path=pth)
                else:
                    test_dataset += hydra.utils.instantiate(config.test_dataset, hdf5_path=pth)
        datamodule = hydra.utils.instantiate(
            config.module,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
        )
    else:
        log.error(f"Unknown data module: {config._target_}")
        raise NotImplementedError(f"Unknown data module: {config._target_}")
    return datamodule
