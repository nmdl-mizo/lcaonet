import abc

import torch
import torch.nn.functional as F

__all__ = ["BaseLoss", "RMSELoss", "MSELoss", "MAELoss"]


class BaseLoss(abc.ABC):
    @abc.abstractmethod
    def calc(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    def name(self) -> str:
        return self.__class__.__name__


class RMSELoss(BaseLoss):
    def calc(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(F.mse_loss(y_hat, y))


class MSELoss(BaseLoss):
    def calc(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(y_hat, y)


class MAELoss(BaseLoss):
    def calc(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(y_hat, y)
