import abc

import torch
import torch.nn.functional as F


__all__ = ["BaseLoss", "RMSELoss", "MSELoss"]


class BaseLoss(abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass


class RMSELoss(BaseLoss):
    def __call__(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(F.mse_loss(y_hat, y))


class MSELoss(BaseLoss):
    def __call__(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(y_hat, y)
