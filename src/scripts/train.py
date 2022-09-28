from __future__ import annotations

from collections import Callable
import logging

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

log = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="train", version_base=None)
def train(config: DictConfig):
    # set seed
    seed_everything(config.seed, workers=True)
    log.info(f"Setting seed: {config.training.seed}")

    # setup data
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(config.data)
    log.info(f"Setting up data: {config.data._target_}")

    # setup model
    model: torch.nn.Module = hydra.utils.instantiate(config.model)
    log.info(f"Setting up model: {config.model._target_}")

    # setup optimizer
    optimizer: torch.optim.Optimizer = hydra.utils.instantiate(config.optimizer, params=model.parameters())
    log.info(f"Setting up optimizer: {config.optimizer._target_}")

    # setup scheduler
    if config.scheduler is not None:
        scheduler: torch.optim.lr_scheduler._LRScheduler = hydra.utils.instantiate(config.scheduler, optimizer=optimizer)
        log.info(f"Setting up scheduler: {config.scheduler._target_}")
    else:
        scheduler = None
        log.info("No scheduler is set")

    # setup loss function
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = hydra.utils.instantiate(config.loss_fn)
    log.info(f"Setting up loss function: {config.loss_fn._target_}")

    # setup lightning module
    plmodule: pl.LightningModule = hydra.utils.instantiate(
        config.plmodule,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
    )
    log.info(f"Setting up lightning module: {config.plmodule._target_}")

    # setup callbacks
    callbacks: list[pl.Callback] = []
    if config.callbacks is not None:
        for _, conf in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(conf))
            log.info(f"Setting up callback: {conf._target_}")

    # setup logger
    logger: list[pl.loggers.LightningLoggerBase] = []
    if config.logger is not None:
        for _, conf in config.logger.items():
            logger.append(hydra.utils.instantiate(conf))
            log.info(f"Setting up logger: {conf._target_}")

    # setup trainer
    trainer: pl.Trainer = hydra.utils.instantiate(
        config.trainer, logger=logger, callbacks=callbacks, _convert_="partial"
    )
    log.info(f"Setting up trainer: {config.trainer._target_}")

    # train
    log.info("Starting training...")
    trainer.fit(plmodule, datamodule)

    # test
    log.info("Starting testing...")
    trainer.test(plmodule, datamodule, ckpt_path="best")

    # Store best model
    best_path = trainer.checkpoint_callback.best_model_path
    log.info(f"Best checkpoint path:\n{best_path}")

    log.info("Done.")
