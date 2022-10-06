from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch_geometric

from pyggnns.train.loss import BaseLoss

__all__ = ["GNNModule"]


class GNNModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: BaseLoss,
        metrics: list[BaseLoss] = [],
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        predict_key: str = "formation_energy",
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.scheduler = scheduler
        self.predict_key = predict_key

    def forward(self, x) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: torch_geometric.data.Data, batch_idx) -> torch.Tensor:
        self.log("train/batch_size", len(batch), on_step=False, on_epoch=True, logger=True)
        x = batch
        y = batch[self.predict_key]
        y_hat = self(x)
        loss = self.loss_fn.calc(y_hat, y)
        self.log(f"train/{self.loss_fn.name()}", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        for m in self.metrics:
            self.log(f"train/{m.name()}", m.calc(y_hat, y), on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch: torch_geometric.data.Data, batch_idx) -> torch.Tensor:
        self.log("val/batch_size", len(batch), on_step=False, on_epoch=True, logger=True)
        x = batch
        y = batch[self.predict_key]
        y_hat = self(x)
        loss = self.loss_fn.calc(y_hat, y)
        self.log(f"val/{self.loss_fn.name()}", loss, on_step=True, on_epoch=True, logger=True)
        for m in self.metrics:
            self.log(f"val/{m.name()}", m.calc(y_hat, y), on_step=True, on_epoch=True, logger=True)
        return loss

    def test_step(self, batch: torch_geometric.data.Data, batch_idx):
        pass

    def configure_optimizers(
        self,
    ) -> torch.optim.Optimizer | dict:
        if self.scheduler is not None:
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    "scheduler": self.scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return self.optimizer
