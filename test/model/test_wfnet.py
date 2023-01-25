from __future__ import annotations  # type: ignore

import pytest
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torch_geometric.data import Data

from pyg_material.data import DataKeys
from pyg_material.model.wfnet import ELEC_DICT, EmbedCoeffs, WFNet


@pytest.fixture(scope="module")
def set_seed():
    seed_everything(42)


@pytest.mark.parametrize("embed_dim", [2, 5, 10, 32])
def test_embed_coeffs(
    set_seed,  # NOQA: F811
    embed_dim: int,
):
    zs = torch.tensor([i for i in range(95)])
    ec = EmbedCoeffs(embed_dim, "cpu", 96)
    coeffs = ec(zs)
    assert coeffs.size() == (zs.size(0), ec.n_orb, embed_dim)
    for i, z in enumerate(zs):
        assert coeffs[i, ELEC_DICT[z] == 0, :].sum().item() == 0


param_wfnet = [
    (16, 8, 10, 1, 2.0),
    (16, 32, 10, 1, 2.0),
    (16, 8, 10, 2, 2.0),
    (16, 8, 10, 1, None),
    (16, 10, 10, 2, None),
]


@pytest.mark.parametrize("hidden_dim, down_dim, coeffs_dim, out_dim, cutoff", param_wfnet)
def test_wfnet(
    one_graph_data: Data,
    hidden_dim: int,
    down_dim: int,
    coeffs_dim: int,
    out_dim: int,
    cutoff: float | None,
):
    max_z = one_graph_data[DataKeys.Atom_numbers].max().item() + 1
    model = WFNet(
        hidden_dim=hidden_dim,
        down_dim=down_dim,
        coeffs_dim=coeffs_dim,
        out_dim=out_dim,
        n_conv_layer=2,
        cutoff=cutoff,
        activation="Silu",
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
