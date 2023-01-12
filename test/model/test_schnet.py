from __future__ import annotations  # type: ignore

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from pyg_material.data import DataKeys
from pyg_material.model import SchNet
from pyg_material.model.schnet import CosineCutoff
from pyg_material.nn import BaseScaler, ShiftScaler

testdata = [
    (16, 16, 1, None, None),
    (12, 10, 1, None, None),
    (16, 16, 1, CosineCutoff, None),
    (16, 16, 1, CosineCutoff, ShiftScaler),
    (16, 10, 1, CosineCutoff, ShiftScaler),
    (12, 10, 2, None, None),
    (16, 10, 2, None, ShiftScaler),
]


@pytest.mark.parametrize("node_dim, edge_filter_dim, out_dim, cutoff_net, scaler", testdata)
def test_schnet(
    one_graph_data: Data,
    node_dim: int,
    edge_filter_dim: int,
    out_dim: int,
    cutoff_net: type[CosineCutoff] | None,
    scaler: type[BaseScaler],
):
    max_z = one_graph_data[DataKeys.Atom_numbers].max().item() + 1
    model = SchNet(
        node_dim=node_dim,
        edge_filter_dim=edge_filter_dim,
        n_conv_layer=2,
        out_dim=out_dim,
        n_rad=8,
        activation="shiftedsoftplus",
        cutoff=2.0,
        cutoff_net=cutoff_net,
        aggr="add",
        scaler=scaler,
        weight_init="xavier_uniform_",
        share_weight=False,
        max_z=max_z,
    )

    with torch.no_grad():
        out = model(one_graph_data)
        assert out.size() == (out_dim,)

        jit = torch.jit.export(model)
        assert torch.allclose(jit(one_graph_data), out)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    min_loss = float("inf")
    for _ in range(100):
        optimizer.zero_grad()
        out = model(one_graph_data)
        loss = F.l1_loss(out, torch.ones((out_dim,)))
        loss.backward()
        optimizer.step()
        min_loss = min(float(loss), min_loss)
    assert min_loss < 2
