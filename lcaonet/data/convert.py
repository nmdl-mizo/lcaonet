from __future__ import annotations

import ase
import numpy as np
import torch
from ase.data import atomic_masses
from ase.neighborlist import neighbor_list
from numpy import ndarray
from torch import Tensor
from torch_geometric.data import Data

from .keys import KEYS, GraphKeys


def atoms2graphdata(
    atoms: ase.Atoms,
    subtract_center_of_mass: bool,
    cutoff: float,
    basis_cutoff: float,
) -> Data:
    """Convert one `ase.Atoms` object to `torch_geometric.data.Data` with edge
    index information include pbc.

    Args:
        atoms (ase.Atoms): one atoms object

    Returns:
        data (torch_geometric.data.Data): one Data object with edge information include pbc and the rotation matrix.
    """
    if subtract_center_of_mass:
        masses = np.array(atomic_masses[atoms.numbers])
        pos = atoms.positions
        atoms.positions -= (masses[:, None] * pos).sum(0) / masses.sum()

    # edge information including pbc
    edge_src, edge_dst, dist, edge_vec, edge_shift = neighbor_list(
        "ijdDS",
        a=atoms,
        cutoff=basis_cutoff,
        self_interaction=False,
    )

    idx_s = []
    idx_t = []
    shift = []

    unique = np.unique(edge_src)
    for i in unique:
        center_mask = edge_src == i
        dist_i = dist[center_mask]
        sorted_ind = np.argsort(dist_i)
        dist_mask = (dist_i <= cutoff)[sorted_ind]
        # center_mask to retrieve information on central atom i
        # reorder by soreted_ind in order of distance
        # extract only the information within the cutoff radius with dist_mask
        idx_s_i = edge_src[center_mask][sorted_ind][dist_mask]
        idx_s.append(idx_s_i)
        idx_t.append(edge_dst[center_mask][sorted_ind][dist_mask])
        shift.append(edge_shift[center_mask][sorted_ind][dist_mask])

    edge_src = np.concatenate(idx_s, axis=0)
    edge_dst = np.concatenate(idx_t, axis=0)
    edge_shift = np.concatenate(shift, axis=0)

    # edge_index order is "source_to_target"
    data = Data(edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0))
    # node info
    data[GraphKeys.Pos] = torch.tensor(atoms.get_positions(), dtype=torch.float32)
    data[GraphKeys.Z] = torch.tensor(atoms.numbers, dtype=torch.long)
    # edge info
    data[GraphKeys.Edge_shift] = torch.tensor(edge_shift, dtype=torch.float32)

    # graph info
    data[GraphKeys.Lattice] = torch.tensor(atoms.cell.array, dtype=torch.float32).unsqueeze(0)
    data[GraphKeys.PBC] = torch.tensor(atoms.pbc, dtype=torch.long).unsqueeze(0)
    data[GraphKeys.Neighbors] = torch.tensor([edge_dst.shape[0]])

    return data


def _set_data(
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


def set_properties(
    data: Data,
    k: str,
    v: int | float | str | ndarray | Tensor,
    add_batch: bool = True,
):
    if isinstance(v, int):
        _set_data(data, k, v, add_dim=True, add_batch=add_batch, dtype=torch.long)
    elif isinstance(v, float):
        _set_data(data, k, v, add_dim=True, add_batch=add_batch, dtype=torch.float32)
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
        _set_data(data, k, v, add_dim=True, add_batch=add_batch, dtype=dtype)
    else:
        # for array-like
        if isinstance(v, ndarray):
            dtype = torch.long if v.dtype == int else torch.float32
        elif isinstance(v, Tensor):
            dtype = v.dtype
        else:
            raise ValueError(f"Unknown type of {v}")
        _set_data(data, k, v, add_dim=False, add_batch=add_batch, dtype=dtype)


def graphdata2atoms(data: Data) -> ase.Atoms:
    """Convert one `torch_geometric.data.Data` object to `ase.Atoms`.

    Args:
        data (torch_geometric.data.Data): one graph data object with edge information include pbc
    Returns:
        atoms (ase.Atoms): one Atoms object
    """
    pos = data[GraphKeys.Pos].numpy()
    atom_num = data[GraphKeys.Z].numpy()
    ce = data[GraphKeys.Lattice].numpy()[0]  # remove batch dimension
    pbc = data[GraphKeys.PBC].numpy()[0]  # remove batch dimension
    info = {}
    for k, v in data.items():
        if k not in KEYS:
            info[k] = v
    atoms = ase.Atoms(numbers=atom_num, positions=pos, pbc=pbc, cell=ce, info=info)
    return atoms


# class List2ChgFiedlDataset(List2GraphDataset):
#     def __init__(
#         self,
#         structures: list[ase.Atoms],
#         y_values: dict[str, list[int | float | str | ndarray | Tensor] | ndarray | Tensor],
#         chgcar: list[np.ndarray],
#         cutoff: float,
#         out_field_radi: float,
#         in_field_radi: float,
#         field_grid_interval: float,
#         max_neighbors: int = 32,
#         self_interaction: bool = False,
#         remove_batch_key: list[str] | None = None,
#     ):
#         super().__init__(
#             structures, y_values, cutoff, max_neighbors, self_interaction, remove_batch_key=remove_batch_key
#         )
#         self.out_field_radi = out_field_radi
#         self.in_field_radi = in_field_radi
#         self.field_grid_interval = field_grid_interval
#         self._preprocess_chg(chgcar)
#         del chgcar

#     def _preprocess_chg(self, chgcar):
#         """Preprocess the graph information list to make the graph Data with
#         node field information."""
#         sphere = self._create_sphere(self.out_field_radi, self.in_field_radi, self.field_grid_interval)
#         for i, g in enumerate(self.graph_data_list):
#             pos = np.array(g[GraphKeys.Pos])
#             ce = np.array(g[GraphKeys.Lattice][0])
#             chg_data = self._preprocess_chgcar(chgcar[i], ce)

#             # get field data
#             ffc = self._create_field(sphere, pos, ce)
#             self._set_chg_interpolator(chg_data)
#             densities = self._get_chg_densities(ffc)

#             # add chg info
#             g["field_dens"] = torch.tensor(densities)
#             g["sphere_coords"] = torch.tensor(sphere).unsqueeze(0)

#     def _create_sphere(self, out_radius: float, in_radious: float, grid_interval: float) -> np.ndarray:
#         xyz = np.arange(-out_radius, out_radius + 1e-3, grid_interval)
#         sphere = [
#             [x, y, z]
#             for x in xyz
#             for y in xyz
#             for z in xyz
#             if (x**2 + y**2 + z**2 <= out_radius**2)
#             and [x, y, z] != [0, 0, 0]
#             and (x**2 + y**2 + z**2 > in_radious**2)
#         ]
#         return np.array(sphere)

#     def _create_field(self, sphere: np.ndarray, coords: np.ndarray, lat_mat: np.ndarray) -> np.ndarray:
#         """Create the grid field of a material.

#         Args:
#             sphere (np.ndarray): Sphere to be placed on each atom of a material.
#             coords (np.ndarray): Cartesian coordinates of atoms of a material.
#             lat_mat (np.ndarray): Lattice matrix of a material.

#         Returns:
#             ffc (np.ndarray): Fractional coordinates of the grid field shape of (n_node, n_field, 3).

#         Notes:
#             ref: https://github.com/masashitsubaki/QuantumDeepField_molecule
#         """
#         fcc_list = [sphere + c for c in coords]
#         fcc = np.array(fcc_list)
#         # fractional coords
#         ffc: np.ndarray = np.array([np.dot(f, np.linalg.inv(lat_mat)) for f in fcc])
#         # move negative to positive, over 1 to less than 1
#         ffc = np.where(ffc < 0, ffc + 1, ffc)
#         ffc = np.where(ffc > 1, ffc - 1, ffc)
#         return ffc

#     def _preprocess_chgcar(self, chgcar: np.ndarray, lat_matrix: np.ndarray) -> np.ndarray:
#         """Preprocess the charge density data.

#         Args:
#             chgcar (np.ndarray): Charge density data shape of (nx, ny, nz).
#             lat_matrix (np.ndarray): Lattice matrix shape of (3, 3, 3).

#         Returns:
#             chgcar (np.ndarray): Preprocessed charge density data shape of (nx, ny, nz).
#         """
#         volume = float(abs(np.dot(np.cross(lat_matrix[0], lat_matrix[1]), lat_matrix[2])))
#         return chgcar / volume

#     def _set_chg_interpolator(self, chg_data: np.ndarray):
#         """Set the interpolator for the charge density.

#         Args:
#             chg_data (np.ndarray): Charge density data shape of (nx, ny, nz).

#         Notes:
#             ref: https://github.com/materialsproject/pymatgen
#         """
#         dim = chg_data.shape
#         xpoints = np.linspace(0.0, 1.0, num=dim[0])
#         ypoints = np.linspace(0.0, 1.0, num=dim[1])
#         zpoints = np.linspace(0.0, 1.0, num=dim[2])
#         self.chg_interpolator = RegularGridInterpolator((xpoints, ypoints, zpoints), chg_data, bounds_error=True)

#     def _get_chg_densities(self, ffc: np.ndarray) -> np.ndarray:
#         """Get the charge density at a fractional point (x, y, z).

#         Args:
#             ffc (np.ndarray): Fractional coordinates of field shape of (n_node, n_field, 3)

#         Returns:
#             d (np.ndarray): Charge densities shape of (n_node, n_field)

#         Notes:
#             ref: https://github.com/materialsproject/pymatgen
#         """
#         try:
#             d = self.chg_interpolator(ffc)
#         except AttributeError:
#             raise AttributeError("The interpolator is not set. Please call `self._set_chg_interpolator` first.")
#         return d
