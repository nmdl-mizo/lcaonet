import abc

import torch
import torch.nn.functional as F

__all__ = ["BaseLoss", "RMSELoss", "MSELoss"]


class BaseLoss(abc.ABC):
    @abc.abstractmethod
    def calc(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass


class RMSELoss(BaseLoss):
    def calc(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(F.mse_loss(y_hat, y))


class MSELoss(BaseLoss):
    def calc(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(y_hat, y)
