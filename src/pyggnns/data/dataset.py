from __future__ import annotations  # type: ignore

import logging
import pathlib

import ase
import ase.neighborlist
import h5py
import numpy as np
import torch
from ase.db import connect
from numpy import ndarray
from torch_geometric.data import Data, Dataset

from pyggnns.data.datakeys import DataKeys

log = logging.getLogger(__name__)


__all__ = [
    "BaseGraphDataset",
    "Db2GraphDataset",
    "Hdf2GraphDataset",
    "Hdf2PartialGraphDataset",
    "List2GraphDataset",
]


class BaseGraphDataset(Dataset):
    def __init__(
        self,
        cutoff_radi: float,
        property_names: list[str] | None,
        pbc: bool | tuple[bool, ...] = True,
    ):
        super().__init__()
        self.cutoff_radi = cutoff_radi
        self.property_names = property_names
        self.pbc = pbc

    def len(self) -> int:
        raise NotImplementedError

    def getitem(self, idx) -> Data:
        raise NotImplementedError

    def _atoms2geometricdata(self, atoms: ase.Atoms) -> Data:
        """Helper function to convet one Atoms object to
        torch_geometric.data.Data.

        Args:
            atoms (ase.Atoms): one atoms object.

        Returns:
            data: torch_geometric.data.Data
        """
        # for edge_shift
        edge_src, edge_dst, edge_shift = ase.neighborlist.neighbor_list(
            "ijS",
            a=atoms,
            cutoff=self.cutoff_radi,
            self_interaction=False,
        )
        data = Data(
            edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0),
        )
        data[DataKeys.Position] = torch.tensor(atoms.get_positions(), dtype=torch.float32)
        data[DataKeys.Atom_numbers] = torch.tensor(atoms.numbers, dtype=torch.long)
        # add batch dimension
        data[DataKeys.Lattice] = torch.tensor(atoms.cell.array, dtype=torch.float32).unsqueeze(0)
        data[DataKeys.Edge_shift] = torch.tensor(edge_shift, dtype=torch.float32)
        return data

    def _set_properties(self, data: torch, k: str, v: int | float | ndarray | torch.Tensor):
        # add a dimension for batching
        if isinstance(v, int) or isinstance(v, float):
            # for value
            data[k] = torch.tensor([v], dtype=torch.float32).unsqueeze(0)
        elif len(v.shape) == 0:
            # for 0-dim array
            data[k] = torch.tensor([float(v)], dtype=torch.float32).unsqueeze(0)
        else:
            # for array-like
            data[k] = torch.tensor(v, dtype=torch.float32).unsqueeze(0)


class Hdf2GraphDataset(BaseGraphDataset):
    """Dataset for graph data such as crystal or molecule by using atoms object
    from hdf5 file. This dataset corresponds to periodic boundary conditions.

    Args:
        hdf5_path (str or pathlib.Path): path to the database.
        cutoff_radi (float): cutoff radius.
        property_names (List[str], optional): properties to add to the dataset. Defaults to `None`.
        pbc (bool, optional): whether to use periodic boundary conditions. Defaults to `True`.
    """

    def __init__(
        self,
        hdf5_path: str | pathlib.Path,
        cutoff_radi: float,
        property_names: list[str] | None = None,
        pbc: bool | tuple[bool, ...] = True,
    ):
        super().__init__(cutoff_radi, property_names, pbc)
        if isinstance(hdf5_path, str):
            hdf5_path = pathlib.Path(hdf5_path)
        if not hdf5_path.exists():
            log.error(f"{hdf5_path} is not found.")
            raise FileNotFoundError(f"{hdf5_path} does not exist.")
        self.hdf5_path = str(hdf5_path)
        # open file
        self.hdf5_file = h5py.File(self.hdf5_path, "r", locking=False)

    def len(self) -> int:
        return len(self.hdf5_file)

    def get(self, idx) -> Data:
        # each system is a group in the hdf5 file
        system_group = self.hdf5_file[list(self.hdf5_file.keys())[idx]]
        atoms: ase.Atoms = self._make_atoms(system_group)
        geometric_data = self._atoms2geometricdata(atoms)
        # add properties
        if self.property_names is not None:
            for k in self.property_names:
                if system_group.attrs.get(k) is not None:
                    v = system_group.attrs.get(k)
                elif system_group.get(k) is not None:
                    v = system_group.get(k)[...]
                else:
                    raise KeyError(f"{k} is not found in the {self.hdf5_path}.")
                self._set_properties(geometric_data, k, v)
        return geometric_data

    # !!Rewrite here if data structure is different
    def _make_atoms(self, system_group: h5py.Group) -> ase.Atoms:
        # get lattice
        lattice = system_group[DataKeys.Lattice][...]
        # get positions
        positions = system_group[DataKeys.Position][...]
        # get atomic numbers
        atomic_num = system_group[DataKeys.Atom_numbers][...]
        # make atoms object
        atoms = ase.Atoms(
            positions=positions,
            numbers=atomic_num,
            cell=lattice,
            pbc=self.pbc,
        )
        return atoms

    def __del__(self):
        self.hdf5_file.close()


class Hdf2PartialGraphDataset(BaseGraphDataset):
    """Dataset for graph data such as crystal or molecule by using atoms object
    from hdf5 file. This dataset corresponds to periodic boundary conditions.

    Args:
        hdf5_path (str or pathlib.Path): path to the database.
        cutoff_radi (float): cutoff radius.
        property_names (List[str], optional): properties to add to the dataset. Defaults to `None`.
        pbc (bool, optional): whether to use periodic boundary conditions. Defaults to `True`.
        atom_numbers (List[int], optional): Atomic numbers to be used. Defaults to `None`.
        specific_atom_numbers (List[int], optional): Include systems in the data set that contain only the atomic numbers specified in this parameter. Defaults to `None`.
        threshold (float, optional): threshold of property. Defaults to `None`.
    """  # NOQA: E501

    def __init__(
        self,
        hdf5_path: str | pathlib.Path,
        cutoff_radi: float,
        property_names: list[str] | None = None,
        pbc: bool | tuple[bool, ...] = True,
        atom_numbers: list[int] | None = None,
        specific_atom_numbers: list[int] | None = None,
        threshold: float | None = None,
    ):
        super().__init__(cutoff_radi, property_names, pbc)
        if isinstance(hdf5_path, str):
            hdf5_path = pathlib.Path(hdf5_path)
        if not hdf5_path.exists():
            log.error(f"{hdf5_path} does not exist.")
            raise FileNotFoundError(f"{hdf5_path} does not exist.")
        self.hdf5_path = str(hdf5_path)
        # open file
        self.hdf5_file = h5py.File(self.hdf5_path, "r", locking=False)
        if atom_numbers is None:
            self.atom_numbers = np.array(range(1, 120))
        else:
            self.atom_numbers = np.array(atom_numbers)
        if specific_atom_numbers is None:
            self.specific_atom_numbers = np.array([self.atom_numbers[0]])
        else:
            self.specific_atom_numbers = np.array(specific_atom_numbers)
        self.threshold = threshold
        # load data from hdf5 file which contains one or more atomic numbers in atom_numbers
        self.pyg_data_list: list[Data] = []
        self._load_atoms()
        self.hdf5_file.close()

    def _load_atoms(self):
        """Helper Function to read only systems containing one or more atomic
        numbers in `self.atom_numbers`, convert them to
        `torch_geometric.data.Data`, and add them to the `list` object of
        `self.pyg_data_list`."""
        for key in self.hdf5_file.keys():
            system_group = self.hdf5_file[key]
            try:
                atom_numbers = system_group[DataKeys.Atom_numbers][...]
            except KeyError:
                log.warning(f"{DataKeys.Atom_numbers} is not found in {key}.")
                continue
            if (
                np.isin(atom_numbers, self.atom_numbers).any()
                or np.isin(atom_numbers, self.specific_atom_numbers).all()
            ):
                atoms: ase.Atoms | None = self._make_atoms(system_group)
                if atoms is None:
                    continue
                geometric_data: Data = self._atoms2geometricdata(atoms)
                flag: bool = True
                # add properties
                if self.property_names is not None:
                    for k in self.property_names:
                        if system_group.attrs.get(k) is not None:
                            v = system_group.attrs.get(k)
                            # exclude data if property is leargeer than threshold
                            if v > self.threshold:
                                flag = False
                                break
                        elif system_group.get(k) is not None:
                            v = system_group.get(k)[...]
                        else:
                            log.warning(f"{k} is not found in the {system_group.name}.")
                            flag = False
                            break
                        self._set_properties(geometric_data, k, v)
                if flag:
                    self.pyg_data_list.append(geometric_data)

    def len(self) -> int:
        return len(self.pyg_data_list)

    def get(self, idx) -> Data:
        return self.pyg_data_list[idx]

    # !!Rewrite here if data structure is different
    def _make_atoms(self, system_group: h5py.Group) -> ase.Atoms | None:
        # get lattice
        try:
            lattice = system_group[DataKeys.Lattice][...]
        except KeyError:
            log.warning(f"{DataKeys.Lattice} is not found in {system_group.name}.")
            return None
        # get positions
        positions = system_group[DataKeys.Position][...]
        # get atomic numbers
        atomic_num = system_group[DataKeys.Atom_numbers][...]
        # make atoms object
        atoms = ase.Atoms(
            positions=positions,
            numbers=atomic_num,
            cell=lattice,
            pbc=self.pbc,
        )
        return atoms


class Db2GraphDataset(BaseGraphDataset):
    """Dataset for graph data such as crystal or molecule by using atoms object
    from db file. This dataset corresponds to periodic boundary conditions.

    Args:
        db_path (str or pathlib.Path): path to the database.
        cutoff_radi (float): cutoff radius.
        property_names (list[str], optional): property names to add to the dataset. Defaults to `None`.
        pbc (bool or tuple[bool], optional): whether to use periodic boundary conditions. Defaults to `True`
    """

    def __init__(
        self,
        db_path: str | pathlib.Path,
        cutoff_radi: float,
        property_names: list[str] | None = None,
        pbc: bool | tuple[bool, ...] = True,
    ):
        super().__init__(cutoff_radi, property_names, pbc)
        if isinstance(db_path, str):
            db_path = pathlib.Path(db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"{db_path} does not exist.")
        self.db_path = str(db_path)

    def len(self) -> int:
        with connect(self.db_path) as db:
            return db.count()

    def get(self, idx) -> Data:
        with connect(self.db_path) as db:
            row = db.get(idx + 1)
        atoms: ase.Atoms = row.toatoms()
        atoms.pbc = self.pbc
        geometric_data = self._atoms2geometricdata(atoms)
        # add properties
        if self.property_names is not None:
            for k in self.property_names:
                if row.get(k) is not None:
                    v = row.get(k)
                elif row.data.get(k) is not None:
                    v = row.data.get(k)
                else:
                    raise KeyError(f"{k} is not found in the {self.db_path}.")
                self._set_properties(geometric_data, k, v)
        return geometric_data


class List2GraphDataset(BaseGraphDataset):
    """Dataset for graph data such as crystal or molecule from list of Atoms
    object. This dataset corresponds to periodic boundary conditions.

    Args:
        atoms_list (list[ase.Atoms]): list of ase.Atoms object.
        cutoff_radi (float): cutoff radius.
        property_names (list[str], optional): properties to add to the dataset. Defaults to `None`.
        pbc (bool or tuple[bool], optional): whether to use periodic boundary conditions. Defaults to `True`.
    """

    def __init__(
        self,
        atoms_list: list[ase.Atoms],
        cutoff_radi: float,
        property_names: list[str] | None = None,
        pbc: bool | tuple[bool, ...] = True,
    ):
        super().__init__(cutoff_radi, property_names, pbc)
        self.atoms_list = atoms_list

    def len(self) -> int:
        return len(self.atoms_list)

    def get(self, idx) -> Data:
        atoms = self.atoms_list[idx]
        geometric_data = self._atoms2geometricdata(atoms)
        # add properties
        if self.property_names is not None:
            for k in self.property_names:
                if atoms.info.get(k) is not None:
                    v = atoms.info.get(k)
                else:
                    raise KeyError(f"{k} is not found in the Atoms info.")
                self._set_properties(geometric_data, k, v)
        return geometric_data
