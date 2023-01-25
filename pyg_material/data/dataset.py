from __future__ import annotations  # type: ignore

import ase
import ase.neighborlist
import numpy as np
import torch
from numpy import ndarray
from pymatgen.core import Structure
from torch import Tensor
from torch_geometric.data import Data, Dataset

from pyg_material.data.datakeys import DataKeys


class BaseGraphDataset(Dataset):
    def __init__(
        self,
        cutoff: float,
        max_neighbors: int = 32,
        self_interaction: bool = False,
        pbc: bool | tuple[bool, ...] = True,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.self_interaction = self_interaction
        self.pbc = pbc

    def len(self) -> int:
        raise NotImplementedError

    def get(self, idx: int) -> Data:
        raise NotImplementedError

    def _structure2atoms(self, s: Structure) -> ase.Atoms:
        """Helper function to convert one `Structure` object to `ase.Atoms`.

        Args:
            s (pymatgen.core.Structure): one structure object.

        Returns:
            atoms (ase.Atoms): one atoms object.
        """
        atom_num = np.array(s.atomic_numbers, dtype=int)
        ce = s.lattice.matrix
        pos = s.cart_coords
        atoms = ase.Atoms(numbers=atom_num, positions=pos, pbc=self.pbc, cell=ce)

        return atoms

    def _atoms2geometricdata(self, atoms: ase.Atoms) -> Data:
        """Helper function to convert one `Atoms` object to
        `torch_geometric.data.Data`.

        Args:
            atoms (ase.Atoms): one atoms object.

        Returns:
            data (torch_geometric.data.Data): one Data object with edge information include pbc.
        """
        # for edge_shift
        edge_src, edge_dst, dist, edge_shift = ase.neighborlist.neighbor_list(
            "ijdS",
            a=atoms,
            cutoff=self.cutoff,
            self_interaction=self.self_interaction,
        )

        # only max neighbor
        if self.max_neighbors is not None:
            # only max_neighbors
            idx_i = np.zeros(1, dtype=int) - 100
            idx_j = np.zeros(1, dtype=int) - 100
            s = np.zeros((1, 3)) - 100
            unique = np.unique(edge_src)
            for i in unique:
                ind = np.argsort(dist[edge_src == i])
                idx_i = np.concatenate([idx_i, edge_src[edge_src == i][ind[: self.max_neighbors]]], axis=0)
                idx_j = np.concatenate([idx_j, edge_dst[edge_src == i][ind[: self.max_neighbors]]], axis=0)
                s = np.concatenate([s, edge_shift[edge_src == i][ind[: self.max_neighbors]]], axis=0)
            edge_src = idx_i[idx_i > -100]
            edge_dst = idx_j[idx_j > -100]
            edge_shift = s[1:]

        data = Data(edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0))
        data[DataKeys.Position] = torch.tensor(atoms.get_positions(), dtype=torch.float32)
        data[DataKeys.Atom_numbers] = torch.tensor(atoms.numbers, dtype=torch.long)
        # add batch dimension
        data[DataKeys.Lattice] = torch.tensor(atoms.cell.array, dtype=torch.float32).unsqueeze(0)
        data[DataKeys.Edge_shift] = torch.tensor(edge_shift, dtype=torch.float32)
        return data

    def _geometricdata2structure(self, data: Data) -> Structure:
        """Helper function to convert one `torch_geometric.data.Data` object to
        `pymatgen.core.Structure`.

        Args:
            data (torch_geometric.data.Data): one Data object with edge information include pbc.

        Returns:
            s (pymatgen.core.Structure): one structure object.
        """
        pos = data[DataKeys.Position].numpy()
        atom_num = data[DataKeys.Atom_numbers].numpy()
        ce = data[DataKeys.Lattice].numpy()[0]  # remove batch dimension
        s = Structure(lattice=ce, species=atom_num, coords=pos, coords_are_cartesian=True)
        return s

    def _set_data(
        self,
        data: Data,
        k: str,
        v: int | float | ndarray | Tensor,
        add_dim: bool,
        add_batch: bool,
        dtype: torch.dtype,
    ):
        if add_dim:
            val = torch.tensor([v], dtype=dtype)
        else:
            val = torch.tensor(v, dtype=dtype)
        data[k] = val.unsqueeze(0) if add_batch else val

    def _set_properties(self, data: Data, k: str, v: int | float | str | ndarray | Tensor, add_batch: bool = True):
        if isinstance(v, int):
            self._set_data(data, k, v, add_dim=True, add_batch=add_batch, dtype=torch.long)
        elif isinstance(v, float):
            self._set_data(data, k, v, add_dim=True, add_batch=add_batch, dtype=torch.float32)
        elif isinstance(v, str):
            data[k] = v
        elif len(v.shape) == 0:
            # for 0-dim array
            if isinstance(v, ndarray):
                dtype = torch.long if v.dtype == int else torch.float32
            elif isinstance(v, Tensor):
                dtype = v.dtype
            else:
                raise ValueError(f"Unknown type of {v}")
            self._set_data(data, k, v, add_dim=True, add_batch=add_batch, dtype=dtype)
        else:
            # for array-like
            if isinstance(v, ndarray):
                dtype = torch.long if v.dtype == int else torch.float32
            elif isinstance(v, Tensor):
                dtype = v.dtype
            else:
                raise ValueError(f"Unknown type of {v}")
            self._set_data(data, k, v, add_dim=False, add_batch=add_batch, dtype=dtype)


class List2GraphDataset(BaseGraphDataset):
    def __init__(
        self,
        structures: list[Structure | ase.Atoms],
        y_values: dict[str, list[int | float | str | ndarray | Tensor] | ndarray | Tensor],
        cutoff: float,
        max_neighbors: int = 32,
        self_interaction: bool = False,
        pbc: bool | tuple[bool, ...] = True,
        remove_batch_key: list[str] | None = None,
    ):
        super().__init__(cutoff, max_neighbors, self_interaction, pbc)
        self.graph_data_list: list[Data] = []
        self.remove_batch_key = remove_batch_key
        self._preprocess(structures, y_values)
        del structures
        del y_values

    def save(self, save_pth: str):
        import pickle

        with open(save_pth, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, load_pth: str):
        import pickle

        with open(load_pth, "rb") as f:
            return pickle.load(f)

    def _preprocess(
        self,
        structures: list[Structure | ase.Atoms],
        y_values: dict[str, list[int | float | str | ndarray | Tensor] | ndarray | Tensor],
    ):
        for i, s in enumerate(structures):
            if isinstance(s, Structure):
                s = self._structure2atoms(s)
            data = self._atoms2geometricdata(s)
            for k, v in y_values.items():
                add_batch = True
                if self.remove_batch_key is not None and k in self.remove_batch_key:
                    add_batch = False
                self._set_properties(data, k, v[i], add_batch)
            self.graph_data_list.append(data)

    def to_structure(self, idx: int) -> Structure:
        return self._geometricdata2structure(self.graph_data_list[idx])

    def len(self) -> int:
        if len(self.graph_data_list) == 0:
            raise ValueError("The dataset is empty.")
        return len(self.graph_data_list)

    def get(self, idx: int) -> Data:
        return self.graph_data_list[idx]
