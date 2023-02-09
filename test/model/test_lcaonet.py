from __future__ import annotations  # type: ignore

import pytest
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torch_geometric.data import Data

from pyg_material.data import DataKeys
from pyg_material.model.wfnet import ELEC_DICT, EmbedElec, EmbedZ, LCAONet


@pytest.fixture(scope="module")
def set_seed():
    seed_everything(42)


@pytest.mark.parametrize("embed_dim", [2, 5, 10, 32])
def test_embed_elec(
    set_seed,  # NOQA: F811
    embed_dim: int,
):
    zs = torch.tensor([i for i in range(95)])
    ez = EmbedZ(embed_dim, 96)
    z_embed = ez(zs)
    ec = EmbedElec(embed_dim, "cpu")
    coeffs = ec(zs, z_embed)
    assert coeffs.size() == (zs.size(0), ec.n_orb, embed_dim)
    for i, z in enumerate(zs):
        assert coeffs[i, ELEC_DICT[z] == 0, :].sum().item() == 0


param_wfnet = [
    (16, 8, 10, 1, 2.0, False),
    (16, 32, 10, 1, 2.0, False),
    (16, 8, 10, 2, 2.0, False),
    (16, 8, 10, 2, 2.0, True),
    (16, 8, 10, 1, None, False),
    (16, 10, 10, 2, None, False),
    (16, 10, 10, 2, None, True),
]


@pytest.mark.parametrize("hidden_dim, down_dim, coeffs_dim, out_dim, cutoff, assoc_lag", param_wfnet)
def test_LCAONet(
    one_graph_data: Data,
    hidden_dim: int,
    down_dim: int,
    coeffs_dim: int,
    out_dim: int,
    cutoff: float | None,
    assoc_lag: bool,
):
    max_z = one_graph_data[DataKeys.Atom_numbers].max().item() + 1
    model = LCAONet(
        hidden_dim=hidden_dim,
        down_dim=down_dim,
        coeffs_dim=coeffs_dim,
        out_dim=out_dim,
        n_conv_layer=2,
        cutoff=cutoff,
        activation="Silu",
        assoc_lag=assoc_lag,
        weight_init="glorot_orthogonal",
        max_z=max_z,
        device="cpu",
    )

    with torch.no_grad():
        out = model(one_graph_data)
        # no batch
        assert out.size() == (1, out_dim)

        jit = torch.jit.export(model)
        assert torch.allclose(jit(one_graph_data), out)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    min_loss = float("inf")
    for _ in range(100):
        optimizer.zero_grad()
        out = model(one_graph_data)
        loss = F.mse_loss(out, torch.ones((1, out_dim)))
        loss.backward()
        optimizer.step()
        min_loss = min(float(loss), min_loss)
    assert min_loss < 2
