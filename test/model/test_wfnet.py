from __future__ import annotations  # type: ignore

import pytest
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torch_geometric.data import Data

from pyg_material.data import DataKeys
from pyg_material.model import WFNet

testdata = [
    (16, 8, 10, 1, 2.0),
    (16, 32, 10, 1, 2.0),
    (16, 8, 10, 2, 2.0),
    (16, 8, 10, 1, None),
    (16, 10, 10, 2, None),
]


@pytest.mark.parametrize("hidden_dim, down_dim, coeffs_dim, out_dim, cutoff", testdata)
def test_wfnet(
    one_graph_data: Data,
    hidden_dim: int,
    down_dim: int,
    coeffs_dim: int,
    out_dim: int,
    cutoff: float | None,
):
    seed_everything(42)
    max_z = one_graph_data[DataKeys.Atom_numbers].max().item() + 1
    model = WFNet(
        hidden_dim=hidden_dim,
        down_dim=down_dim,
        coeffs_dim=coeffs_dim,
        out_dim=out_dim,
        n_conv_layer=2,
        cutoff=cutoff,
        standarize_basis=True if cutoff is not None else False,
        activation="Silu",
        weight_init="glorot_orthogonal",
        max_z=max_z,
        device="cpu",
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
