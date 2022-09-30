from __future__ import annotations  # type: ignore

import logging
from collections.abc import Callable

import hydra
import pyrootutils
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

log = logging.getLogger(__name__)


root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)


@hydra.main(config_path=root / "configs", config_name="train", version_base=None)
def training(config: DictConfig):
    # set seed
    log.info(f"Setting seed: {config.configs.seed}")
    seed_everything(config.training.seed, workers=True)

    # setup data
    log.info(f"Setting up data: {config.data._target_}")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(config.data)

    # setup model
    log.info(f"Setting up model: {config.model._target_}")
    model: torch.nn.Module = hydra.utils.instantiate(config.model)

    # setup optimizer
    log.info(f"Setting up optimizer: {config.optimizer._target_}")
    optimizer: torch.optim.Optimizer = hydra.utils.instantiate(config.optimizer, params=model.parameters())

    # setup scheduler
    if config.scheduler is not None:
        log.info(f"Setting up scheduler: {config.scheduler._target_}")
        scheduler: torch.optim.lr_scheduler._LRScheduler = hydra.utils.instantiate(
            config.scheduler, optimizer=optimizer
        )
    else:
        log.info("No scheduler is set")
        scheduler = None

    # setup loss function
    log.info(f"Setting up loss function: {config.loss_fn._target_}")
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = hydra.utils.instantiate(config.loss_fn)

    # setup lightning module
    log.info(f"Setting up lightning module: {config.plmodule._target_}")
    plmodule: pl.LightningModule = hydra.utils.instantiate(
        config.plmodule,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
    )

    # setup callbacks
    callbacks: list[pl.Callback] = []
    if config.callbacks is not None:
        for _, conf in config.callbacks.items():
            log.info(f"Setting up callback: {conf._target_}")
            callbacks.append(hydra.utils.instantiate(conf))

    # setup logger
    logger: list[pl.loggers.LightningLoggerBase] = []
    if config.logger is not None:
        for _, conf in config.logger.items():
            log.info(f"Setting up logger: {conf._target_}")
            logger.append(hydra.utils.instantiate(conf))

    # setup trainer
    log.info(f"Setting up trainer: {config.trainer._target_}")
    trainer: pl.Trainer = hydra.utils.instantiate(
        config.trainer, logger=logger, callbacks=callbacks, _convert_="partial"
    )

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


if __name__ == "__main__":
    training()
