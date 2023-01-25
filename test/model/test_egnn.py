from __future__ import annotations  # type: ignore

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from pyg_material.data import DataKeys
from pyg_material.model import EGNN
from pyg_material.model.schnet import CosineCutoff

testdata = [
    (16, 16, 1, True, None, None),
    (16, 16, 1, False, None, None),
    (16, 16, 2, False, None, None),
    (100, 12, 1, True, None, None),
    (16, 16, 1, False, None, CosineCutoff(2.0)),
    (16, 16, 1, False, 2, None),
    (16, 16, 1, True, 3, CosineCutoff(2.0)),
    (16, 16, 2, True, 2, CosineCutoff(2.0)),
]


@pytest.mark.parametrize("node_dim, edge_dim, out_dim, batch_norm, edge_attr_dim, cutoff_net", testdata)
def test_egnn(
    one_graph_data: Data,
    node_dim: int,
    edge_dim: int,
    out_dim: int,
    batch_norm: bool,
    edge_attr_dim: int | None,
    cutoff_net: torch.nn.Module | None,
):
    if edge_attr_dim is not None:
        one_graph_data[DataKeys.Edge_attr] = torch.randn(
            (one_graph_data[DataKeys.Edge_idx].size(1), edge_attr_dim),
        )
    max_z = one_graph_data[DataKeys.Atom_numbers].max().item() + 1
    model = EGNN(
        node_dim=node_dim,
        edge_dim=edge_dim,
        n_conv_layer=2,
        out_dim=out_dim,
        activation="swish",
        cutoff=2.0,
        cutoff_net=cutoff_net,
        aggr="add",
        weight_init="glorot_orthogonal",
        batch_norm=batch_norm,
        edge_attr_dim=edge_attr_dim,
        max_z=max_z,
    )

    with torch.no_grad():
        out = model(one_graph_data)
        # no batch
        assert out.size() == (1, out_dim)

        jit = torch.jit.export(model)
        assert torch.allclose(jit(one_graph_data), out)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    min_loss = float("inf")
    for _ in range(100):
        optimizer.zero_grad()
        out = model(one_graph_data)
        loss = F.l1_loss(out, torch.ones((1, out_dim)))
        loss.backward()
        optimizer.step()
        min_loss = min(float(loss), min_loss)
    assert min_loss < 2
