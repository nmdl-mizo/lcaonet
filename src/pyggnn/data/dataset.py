from os import PathLike
import pathlib
from typing import List, Optional, Tuple, Union

import h5py
import ase
import ase.neighborlist
from ase.db import connect
from numpy import ndarray
import torch
from torch_geometric.data import Data

from pyggnn.data.datakeys import DataKeys


__all__ = [
    "BaseGraphDataset",
    "Db2GraphDataset",
    "Hdf2GraphDataset",
    "List2GraphDataset",
]
# TODO: load missing data values from hdf5 or db file


class BaseGraphDataset(torch.utils.data.Dataset):
    def __init__(self, cutoff_radi: float, property_names: List[str], pbc: bool = True):
        self.cutoff_radi = cutoff_radi
        self.property_names = property_names
        self.pbc = pbc

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def _atoms2geometricdata(self, atoms: ase.Atoms):
        """
        Helper function to convet one Atoms object to torch_geometric.Data.

        Args:
            atoms (ase.Atoms): one atoms object.
            cutoff_radi (float): cutoff radious.

        Returns:
            data: torch_geometric.Data
        """
        edge_src, edge_dst, edge_shift = ase.neighborlist.neighbor_list(
            "ijS",
            a=atoms,
            cutoff=self.cutoff_radi,
            self_interaction=False,
        )
        data = Data(
            edge_index=torch.stack(
                [torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0
            ),
        )
        data[DataKeys.Position] = torch.tensor(atoms.get_positions())
        data[DataKeys.Atomic_num] = torch.tensor(atoms.numbers)
        # add batch dimension
        data[DataKeys.Lattice] = torch.tensor(atoms.cell.array).unsqueeze(0)
        data[DataKeys.Edge_shift] = torch.tensor(
            edge_shift, dtype=torch.get_default_dtype()
        )
        return data

    def _set_properties(self, data, k: str, v: Union[int, float, ndarray, torch.Tensor]):
        # add a dimension for batching
        if isinstance(v, int) or isinstance(v, float):
            # for value
            data[k] = torch.tensor([v]).unsqueeze(0)
        elif len(v.shape) == 0:
            # for 0-dim array
            data[k] = torch.tensor([float(v)]).unsqueeze(0)
        else:
            # for array-like
            data[k] = torch.tensor(v).unsqueeze(0)


class Db2GraphDataset(BaseGraphDataset):
    """
    Dataset for graph data such as crystal or molecule by using atoms object
    from db file. This dataset corresponds to periodic boundary conditions.
    """

    def __init__(
        self,
        db_path: PathLike,
        cutoff_radi: float,
        property_names: Optional[List[str]] = None,
        pbc: Union[bool, Tuple[bool]] = True,
    ):
        """
        Args:
            db_path (Pathlike): path to the database.
            cutoff_radi (float): cutoff radius.
            property_names (List[str], optional): property names to add to the
                dataset. Defaults to `None`.
            pbc (bool, optional): whether to use periodic boundary conditions.
                Defaults to `True`.
        """
        if isinstance(db_path, str):
            db_path = pathlib.Path(db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"{db_path} does not exist.")
        self.db_path = str(db_path)
        super().__init__(cutoff_radi, property_names, pbc)

    def __len__(self):
        with connect(self.db_path) as db:
            return db.count()

    def __getitem__(self, idx):
        with connect(self.db_path) as db:
            row = db.get(idx + 1)
        atoms = row.toatoms()
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


class Hdf2GraphDataset(BaseGraphDataset):
    """
    Dataset for graph data such as crystal or molecule by using atoms object
    from hdf5 file. This dataset corresponds to periodic boundary conditions.
    """

    def __init__(
        self,
        hdf5_path: PathLike,
        cutoff_radi: float,
        property_names: Optional[List[str]] = None,
        pbc: Union[bool, Tuple[bool]] = True,
    ):
        """
        Args:
            hdf5_path (Pathlike): path to the database.
            cutoff_radi (float): cutoff radius.
            property_names (List[str], optional): properties to add to the dataset.
                Defaults to `None`.
            pbc (bool, optional): whether to use periodic boundary conditions.
                Defaults to `True`.
        """
        if isinstance(hdf5_path, str):
            hdf5_path = pathlib.Path(hdf5_path)
        if not hdf5_path.exists():
            raise FileNotFoundError(f"{hdf5_path} does not exist.")
        self.hdf5_path = str(hdf5_path)
        # open file
        self.hdf5_file = h5py.File(self.hdf5_path, "r")
        super().__init__(cutoff_radi, property_names, pbc)

    def __len__(self):
        return len(self.hdf5_file)

    def __getitem__(self, idx):
        # each system is a group in the hdf5 file
        system_group = self.hdf5_file[list(self.hdf5_file.keys())[idx]]
        atoms = self._make_atoms(system_group)
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
        atomic_num = system_group[DataKeys.Atomic_num][...]
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


class List2GraphDataset(BaseGraphDataset):
    """
    Dataset for graph data such as crystal or molecule from list of Atoms object.
    This dataset corresponds to periodic boundary conditions.
    """

    def __init__(
        self,
        atoms_list: List[ase.Atoms],
        cutoff_radi: float,
        property_names: Optional[List[str]] = None,
        pbc: Union[bool, Tuple[bool]] = True,
    ):
        """
        Args:
            atoms_list (list of ase.Atoms): list of ase.Atoms object.
            cutoff_radi (float): cutoff radius.
            property_names (List[str], optional): properties to add to the dataset.
                Defaults to `None`.
            pbc (bool, optional): whether to use periodic boundary conditions.
                Defaults to `True`.
        """
        self.atoms_list = atoms_list
        super().__init__(cutoff_radi, property_names, pbc)

    def __len__(self):
        return len(self.atoms_list)

    def __getitem__(self, idx):
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
