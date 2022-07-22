from os import PathLike
import pathlib
from typing import List, Optional, Tuple, Union

import h5py
import ase
from ase.db import connect
import torch
from torch_geometric.data import Data

from pyggnn.data.datakeys import DataKeys


__all__ = ["Db2GraphDataset", "Hdf2GraphDataset", "List2GraphDataset"]


def _geometric_data(atoms: ase.Atoms, cutoff_radi: float):
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
        cutoff=cutoff_radi,
        self_interaction=False,
    )
    # TODO: set data key names
    data = Data(
        pos=torch.tensor(atoms.get_positions()),
        lattice=torch.tensor(atoms.cell.array).unsqueeze(0),
        atom_numbers=torch.tensor(atoms.numbers),
        edge_index=torch.stack(
            [torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0
        ),
        edge_shift=torch.tensor(edge_shift),
    )
    return data


class Db2GraphDataset(torch.utils.data.Dataset):
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
        self.cutoff_radi = cutoff_radi
        self.property_names = property_names
        self.pbc = pbc

    def __len__(self):
        with connect(self.db_path) as db:
            return db.count()

    def __getitem__(self, idx):
        with connect(self.db_path) as db:
            row = db.get(idx + 1)
        atoms = row.toatoms()
        atoms.pbc = self.pbc
        geometric_data = _geometric_data(atoms, self.cutoff_radi)
        # add properties
        if self.property_names is not None:
            for k in self.property_names:
                if row.get(k) is not None:
                    v = row.get(k)
                elif row.data.get(k) is not None:
                    v = row.data.get(k)
                else:
                    raise KeyError(f"{k} is not found in the {self.db_path}.")
                if isinstance(v, int) or isinstance(v, float):
                    # add a dimension for batching
                    geometric_data[k] = torch.tensor([v]).unsqueeze(0)
                elif len(v.shape) == 0:
                    # 0-dim array
                    geometric_data[k] = torch.tensor([float(v)]).unsqueeze(0)
                else:
                    # add a dimension for batching
                    geometric_data[k] = torch.tensor(v).unsqueeze(0)
        return geometric_data


class Hdf2GraphDataset(torch.utils.data.Dataset):
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
        self.cutoff_radi = cutoff_radi
        self.property_names = property_names
        self.pbc = pbc
        # open file
        self.hdf5_file = h5py.File(self.hdf5_path, "r")

    def __len__(self):
        return len(self.hdf5_file)

    def __getitem__(self, idx):
        # each system is a group in the hdf5 file
        system_group = self.hdf5_file[list(self.hdf5_file.keys())[idx]]
        atoms = self._make_atoms(system_group)
        geometric_data = _geometric_data(atoms, self.cutoff_radi)
        # add properties
        if self.property_names is not None:
            for k in self.property_names:
                if system_group.attrs.get(k) is not None:
                    v = system_group.attrs.get(k)
                elif system_group.get(k) is not None:
                    v = system_group.get(k)[...]
                else:
                    raise KeyError(f"{k} is not found in the {self.hdf5_path}.")
                if isinstance(v, int) or isinstance(v, float):
                    # add a dimension for batching
                    geometric_data[k] = torch.tensor([v]).unsqueeze(0)
                elif len(v.shape) == 0:
                    # 0-dim array
                    print("aa")
                    geometric_data[k] = torch.tensor([float(v)]).unsqueeze(0)
                else:
                    # add a dimension for batching
                    print("bb")
                    geometric_data[k] = torch.tensor(v).unsqueeze(0)
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


class List2GraphDataset(torch.utils.data.Dataset):
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
        self.cutoff_radi = cutoff_radi
        self.property_names = property_names
        self.pbc = pbc

    def __len__(self):
        return len(self.atoms_list)

    def __getitem__(self, idx):
        atoms = self.atoms_list[idx]
        geometric_data = _geometric_data(atoms, self.cutoff_radi)
        # add properties
        if self.property_names is not None:
            for k in self.property_names:
                if atoms.info.get(k) is not None:
                    v = atoms.info.get(k)
                else:
                    raise KeyError(f"{k} is not found in the Atoms info.")
                if isinstance(v, int) or isinstance(v, float):
                    # add a dimension for batching
                    geometric_data[k] = torch.tensor([v]).unsqueeze(0)
                elif len(v.shape) == 0:
                    # 0 dim array
                    geometric_data[k] = torch.tensor([float(v)]).unsqueeze(0)
                else:
                    # add a dimension for batching
                    geometric_data[k] = torch.tensor(v).unsqueeze(0)
        return geometric_data
