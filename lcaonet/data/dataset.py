from __future__ import annotations

import os
import pathlib

import ase
import torch
from torch_geometric.data import Data, Dataset

from .convert import graphdata2atoms


class GraphDataset(Dataset):
    def __init__(self, save_dir: str | pathlib.Path, inmemory: bool = False):
        super().__init__()

        if isinstance(save_dir, str):
            self.save_dir = pathlib.Path(save_dir)
        else:
            self.save_dir = save_dir
        if not self.save_dir.exists():
            raise FileNotFoundError(f"{self.save_dir} does not exist. Please convert the dataset first.")

        self.inmemory = inmemory
        if inmemory:
            self._data_list = [None for _ in range(self.len())]

    def len(self) -> int:
        length = len(os.listdir(self.save_dir))
        if length == 0:
            raise ValueError("The dataset is empty.")
        return length

    def get(self, idx: int) -> Data:
        if not self.inmemory:
            return torch.load(f"{self.save_dir}/{idx}.pt")
        if idx >= self.len():
            raise IndexError("index out of range")
        if self._data_list[idx] is None:
            try:
                self._data_list[idx] = torch.load(f"{self.save_dir}/{idx}.pt")
            except FileNotFoundError:
                raise IndexError("Inproper index")
        return self._data_list[idx]

    def get_atoms(self, idx: int) -> ase.Atoms:
        return graphdata2atoms(self.get(idx))
