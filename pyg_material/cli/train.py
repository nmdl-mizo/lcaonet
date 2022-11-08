from __future__ import annotations  # type: ignore

import logging
from collections.abc import Callable

import hydra
import pyrootutils
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pyggnns.cli.utils import get_data
from pyggnns.train.loss import BaseLoss
from pytorch_lightning import seed_everything

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "src"],
    pythonpath=True,
    dotenv=True,
)


@hydra.main(config_path=root / "configs", config_name="train", version_base=None)
def training(config: DictConfig):
    # check device
    if config.training.device == "gpu" and not torch.cuda.is_available():
        logger.warning("CUDA not available, setting device to CPU")
        config.training.device = "cpu"
    if config.training.device == "cpu":
        logger.info("Running on CPU")
    else:
        logger.info("Running on GPU")

    # set seed
    logger.info(f"Setting seed: {config.seed}")
    seed_everything(config.seed, workers=True)

    # setup data
    logger.info(f"Setting up data: {config.datamodule.module._target_}")
    datamodule: pl.LightningDataModule = get_data(config.datamodule)

    # setup model
    logger.info(f"Setting up model: {config.model._target_}")
    model: torch.nn.Module = hydra.utils.instantiate(config.model)

    # setup optimizer
    logger.info(f"Setting up optimizer: {config.optimizer._target_}")
    optimizer: torch.optim.Optimizer = hydra.utils.instantiate(config.optimizer, params=model.parameters())

    # setup scheduler
    if config.scheduler is not None:
        logger.info(f"Setting up scheduler: {config.scheduler._target_}")
        scheduler: torch.optim.lr_scheduler._LRScheduler = hydra.utils.instantiate(
            config.scheduler, optimizer=optimizer
        )
    else:
        logger.info("No scheduler is set")
        scheduler = None

    # setup loss function
    logger.info(f"Setting up loss function: {config.loss_fn._target_}")
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = hydra.utils.instantiate(config.loss_fn)

    # setup metrics
    metrics: list[BaseLoss] = []
    if config.get("metrics") is not None:
        for _, m in config.metrics.items():
            logger.info(f"Setting up metric: {m._target_}")
            metrics.append(hydra.utils.instantiate(m))
    else:
        logger.info("No metrics are set")

    # setup lightning module
    logger.info(f"Setting up lightning module: {config.plmodule._target_}")
    plmodule: pl.LightningModule = hydra.utils.instantiate(
        config.plmodule,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        metrics=metrics,
    )

    # setup callbacks
    callbacks: list[pl.Callback] = []
    if config.get("callbacks") is not None:
        for _, conf in config.callbacks.items():
            logger.info(f"Setting up callback: {conf._target_}")
            callbacks.append(hydra.utils.instantiate(conf))
    else:
        logger.info("No callbacks are set")

    # setup logger
    loggers_list: list[pl.loggers.LightningLoggerBase] = []
    if config.get("logger") is not None:
        for _, conf in config.logger.items():
            logger.info(f"Setting up logger: {conf._target_}")
            loggers_list.append(hydra.utils.instantiate(conf))
    else:
        logger.info("No logger is set")

    # setup trainer
    logger.info(f"Setting up trainer: {config.trainer._target_}")
    trainer: pl.Trainer = hydra.utils.instantiate(
        config.trainer, logger=loggers_list, callbacks=callbacks, _convert_="partial"
    )

    # train
    logger.info("Starting training...")
    trainer.fit(plmodule, datamodule)

    # test
    logger.info("Starting testing...")
    trainer.test(plmodule, datamodule, ckpt_path="best")

    # Store best model
    best_path = trainer.checkpoint_callback.best_model_path
    logger.info(f"Best checkpoint path:\n{best_path}")

    logger.info("Done.")


if __name__ == "__main__":
    training()
