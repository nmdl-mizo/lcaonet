from turtle import forward
import torch
from torch import Tensor
import torch.nn as nn

from pyggnn.model.base import BaseGNN

__all__ = ["SchNet"]


class SchNet(BaseGNN):
    def __init__(self):
        super().__init__()

    def reset_parameters(self):
        pass

    def forward(sefl, data_batch) -> Tensor:
        pass
