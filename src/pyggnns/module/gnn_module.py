from __future__ import annotations

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch_geometric

from pyggnns.train.loss import BaseLoss

__all__ = ["GNNModule"]


class GNNModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: BaseLoss,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        predict_key: str = "formation_energy",
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.predict_key = predict_key

    def forward(self, x) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: torch_geometric.data.Data, batch_idx) -> torch.Tensor:
        x = batch
        y = batch[self.predict_key]
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(
        self,
    ) -> torch.optim.Optimizer | tuple[list[torch.optim.Optimizer], list[torch.optim.lr_scheduler._LRScheduler]]:
        if self.scheduler is not None:
            return [self.optimizer], [self.scheduler]
        return self.optimizer