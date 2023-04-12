import pytest
import torch
from torch_geometric.data import Data


@pytest.fixture(scope="module")
def one_graph_data():
    # 3 dimensional cubic lattice
    lattice = torch.tensor([[[15.0, 0.0, 0.0], [0.0, 15.0, 0.0], [0.0, 0.0, 15.0]]], dtype=torch.float32)
    # 3 nodes
    pos = torch.tensor([[5.187, 7.50, 7.50], [9.812, 7.50, 7.50], [5.187, 6.56, 7.50]], dtype=torch.float32)
    z = torch.tensor([1, 3, 5], dtype=torch.long)
    # 8 edges
    edge_shift = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, -1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    edge_index = torch.tensor(
        [
            [0, 0, 1, 1, 2, 2, 2, 2],
            [1, 2, 0, 2, 0, 1, 1, 1],
        ],
        dtype=torch.long,
    )

    return Data(edge_index=edge_index, atom_numbers=z, pos=pos, edge_shift=edge_shift, lattice=lattice)
