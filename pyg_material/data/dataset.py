from __future__ import annotations  # type: ignore

import ase
import ase.neighborlist
import numpy as np
import torch
from numpy import ndarray
from pymatgen.core import Structure
from scipy.interpolate import RegularGridInterpolator
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


class List2ChgFiedlDataset(List2GraphDataset):
    def __init__(
        self,
        structures: list[Structure | ase.Atoms],
        y_values: dict[str, list[int | float | str | ndarray | Tensor] | ndarray | Tensor],
        chgcar: list[np.ndarray],
        cutoff: float,
        out_field_radi: float,
        in_field_radi: float,
        field_grid_interval: float,
        max_neighbors: int = 32,
        self_interaction: bool = False,
        pbc: bool | tuple[bool, ...] = True,
        remove_batch_key: list[str] | None = None,
    ):
        super().__init__(structures, y_values, cutoff, max_neighbors, self_interaction, pbc, remove_batch_key)
        self.out_field_radi = out_field_radi
        self.in_field_radi = in_field_radi
        self.field_grid_interval = field_grid_interval
        self._preprocess_chg(chgcar)
        del chgcar

    def _preprocess_chg(self, chgcar):
        """Preprocess the graph information list to make the graph Data with
        node field information."""
        sphere = self._create_sphere(self.out_field_radi, self.in_field_radi, self.field_grid_interval)
        for i, g in enumerate(self.graph_data_list):
            pos = g[DataKeys.Position]
            ce = g[DataKeys.Lattice]
            chg_data = self._preprocess_chgcar(chgcar[i], ce)

            # get field data
            ffc = self._create_field(sphere, pos, ce)
            self._set_chg_interpolator(chg_data)
            densities = self._get_chg_densities(ffc)

            # add chg info
            g["field_dens"] = torch.tensor(densities)
            g["sphere_coords"] = torch.tensor(sphere).unsqueeze(0)

    def _create_sphere(self, out_radius: float, in_radious: float, grid_interval: float) -> np.ndarray:
        xyz = np.arange(-out_radius, out_radius + 1e-3, grid_interval)
        sphere = [
            [x, y, z]
            for x in xyz
            for y in xyz
            for z in xyz
            if (x**2 + y**2 + z**2 <= out_radius**2)
            and [x, y, z] != [0, 0, 0]
            and (x**2 + y**2 + z**2 > in_radious**2)
        ]
        return np.array(sphere)

    def _create_field(self, sphere: np.ndarray, coords: np.ndarray, lat_mat: np.ndarray) -> np.ndarray:
        """Create the grid field of a material.

        Args:
            sphere (np.ndarray): Sphere to be placed on each atom of a material.
            coords (np.ndarray): Cartesian coordinates of atoms of a material.
            lat_mat (np.ndarray): Lattice matrix of a material.

        Returns:
            ffc (np.ndarray): Fractional coordinates of the grid field shape of (n_node, n_field, 3).

        Notes:
            ref: https://github.com/masashitsubaki/QuantumDeepField_molecule
        """
        fcc_list = [sphere + c for c in coords]
        fcc = np.array(fcc_list)
        # fractional coords
        ffc: np.ndarray = np.array([np.dot(f, np.linalg.inv(lat_mat)) for f in fcc])
        # move negative to positive, over 1 to less than 1
        ffc = np.where(ffc < 0, ffc + 1, ffc)
        ffc = np.where(ffc > 1, ffc - 1, ffc)
        return ffc

    def _preprocess_chgcar(self, chgcar: np.ndarray, lat_matrix: np.ndarray) -> np.ndarray:
        """Preprocess the charge density data.

        Args:
            chgcar (np.ndarray): Charge density data shape of (nx, ny, nz).
            lat_matrix (np.ndarray): Lattice matrix shape of (3, 3, 3).

        Returns:
            chgcar (np.ndarray): Preprocessed charge density data shape of (nx, ny, nz).
        """
        volume = float(abs(np.dot(np.cross(lat_matrix[0], lat_matrix[1]), lat_matrix[2])))
        return chgcar / volume

    def _set_chg_interpolator(self, chg_data: np.ndarray):
        """Set the interpolator for the charge density.

        Args:
            chg_data (np.ndarray): Charge density data shape of (nx, ny, nz).

        Notes:
            ref: https://github.com/materialsproject/pymatgen
        """
        dim = chg_data.shape
        xpoints = np.linspace(0.0, 1.0, num=dim[0])
        ypoints = np.linspace(0.0, 1.0, num=dim[1])
        zpoints = np.linspace(0.0, 1.0, num=dim[2])
        self.chg_interpolator = RegularGridInterpolator((xpoints, ypoints, zpoints), chg_data, bounds_error=True)

    def _get_chg_densities(self, ffc: np.ndarray) -> np.ndarray:
        """Get the charge density at a fractional point (x, y, z).

        Args:
            ffc (np.ndarray): Fractional coordinates of field shape of (n_node, n_field, 3)

        Returns:
            d (np.ndarray): Charge densities shape of (n_node, n_field)

        Notes:
            ref: https://github.com/materialsproject/pymatgen
        """
        try:
            d = self.chg_interpolator(ffc)
        except AttributeError:
            raise AttributeError("The interpolator is not set. Please call `self._set_chg_interpolator` first.")
        return d
