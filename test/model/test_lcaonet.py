from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torch_geometric.data import Data
from torch_geometric.nn.inits import glorot_orthogonal

from lcaonet.atomistic.info import ElecInfo
from lcaonet.data.keys import GraphKeys
from lcaonet.model.lcaonet import (
    EmbedCoeffs,
    EmbedElec,
    EmbedNode,
    EmbedZ,
    LCAONet,
    LCAOOut,
    ValenceMask,
)
from lcaonet.nn.cutoff import BaseCutoff


@pytest.fixture(scope="module")
def set_seed():
    seed_everything(42)


param_EmbedZ = [
    (10, 2),
    (10, 36),
    (10, 84),
    (100, 2),
    (100, 36),
    (100, 84),
]


@pytest.mark.parametrize("embed_dim, max_z", param_EmbedZ)
def test_EmbedZ(
    embed_dim: int,
    max_z: int,
):
    n_node = max_z * 3
    zs = torch.randint(0, max_z + 1, (n_node,))

    ez = EmbedZ(embed_dim, max_z)
    z_embed = ez(zs)

    assert z_embed.size() == (n_node, embed_dim)

    for i in range(max_z + 1):
        assert (z_embed[zs == i, :] == ez.z_embed.weight.data[i]).all()


param_EmbedElec = [
    (2, 96, None, 1, False),
    (2, 96, None, 1, True),
    (2, 96, None, 2, True),
    (2, 96, None, 2, True),
    (2, 5, "4s", 1, False),
    (2, 5, "4s", 1, True),
    (2, 5, "4s", 2, False),
    (2, 5, "4s", 2, True),
    (32, 96, None, 1, False),
    (32, 96, None, 1, True),
    (32, 5, "4p", 1, False),
    (32, 5, "4p", 1, True),
]


@pytest.mark.parametrize("embed_dim, max_z, max_orb, n_per_orb, extend_orb", param_EmbedElec)
def test_EmbedElec(
    embed_dim: int,
    max_z: int,
    max_orb: str | None,
    n_per_orb: int,
    extend_orb: bool,
):
    n_node = max_z * 3
    zs = torch.randint(0, max_z + 1, (n_node,))
    ei = ElecInfo(max_z, max_orb, n_per_orb)

    ee = EmbedElec(embed_dim, ei, extend_orb)
    elec_embed = ee(zs)

    assert elec_embed.size() == (zs.size(0), ei.n_orb, embed_dim)

    for i, z in enumerate(zs):
        # check padding_idx
        if not extend_orb:
            assert (elec_embed[i, ee.elec[z] == 0, :] == torch.zeros_like(elec_embed[i, ee.elec[z] == 0, :])).all()  # type: ignore # noqa: E501
        if extend_orb:
            assert (elec_embed[i, ee.elec[z] == 0, :] != torch.zeros_like(elec_embed[i, ee.elec[z] == 0, :])).all()  # type: ignore # noqa: E501


param_ValenceMask = [
    (2, 96, None, 1),
    (2, 96, None, 2),
    (2, 5, "4s", 1),
    (2, 5, "4s", 2),
    (32, 96, None, 1),
    (32, 96, None, 2),
    (32, 5, "4s", 1),
    (32, 5, "4s", 2),
]


@pytest.mark.parametrize("embed_dim, max_z, max_orb, n_per_orb", param_ValenceMask)
def test_ValenceMask(
    embed_dim: int,
    max_z: int,
    max_orb: str | None,
    n_per_orb: int,
):
    n_node, n_edge = max_z * 3, max_z * 5
    zs = torch.randint(0, max_z + 1, (n_node,))
    idx_j = torch.randint(0, zs.size(0), (n_edge,))
    ei = ElecInfo(max_z, max_orb, n_per_orb)

    vm = ValenceMask(embed_dim, ei)
    mask = vm(zs, idx_j)

    assert mask.size() == (n_edge, ei.n_orb, embed_dim)

    for i, z in enumerate(zs[idx_j]):
        # check mask values
        assert (mask[i, vm.valence[z] == 0, :] == torch.zeros_like(mask[i, vm.valence[z] == 0, :])).all()  # type: ignore # noqa: E501
        assert (mask[i, vm.valence[z] == 1, :] == torch.ones_like(mask[i, vm.valence[z] == 1, :])).all()  # type: ignore # noqa: E501


param_EmbedNode = [
    (10, 10, True, 10),
    (10, 10, False, 10),
    (10, 10, True, None),
    (10, 10, False, None),
    (100, 10, True, 10),
    (100, 10, False, 10),
    (100, 10, True, None),
    (100, 10, False, None),
]


@pytest.mark.parametrize("hidden_dim, z_dim, use_elec, e_dim", param_EmbedNode)
def test_EmbedNode(
    hidden_dim: int,
    z_dim: int,
    use_elec: bool,
    e_dim: int | None,
):
    n_node, n_orb = 100, 20
    z_embed = torch.rand((n_node, z_dim))
    if use_elec:
        e_embed = torch.rand((n_node, n_orb, e_dim)) if e_dim else None
    else:
        e_embed = None

    if use_elec and e_dim is None:
        with pytest.raises(AssertionError) as e:
            _ = EmbedNode(hidden_dim, z_dim, use_elec, e_dim, weight_init=glorot_orthogonal)
        assert str(e.value) == ""
    else:
        en = EmbedNode(hidden_dim, z_dim, use_elec, e_dim, weight_init=glorot_orthogonal)
        node_embed = en(z_embed, e_embed)
        assert node_embed.size() == (n_node, hidden_dim)


param_EmbedCoeff = [
    (10, 10, 10),
    (100, 10, 10),
    (10, 100, 10),
    (10, 10, 100),
    (100, 100, 100),
    (10, 20, 30),
]


@pytest.mark.parametrize("hidden_dim, z_dim, e_dim", param_EmbedCoeff)
def test_EmbedCoeffs(
    hidden_dim: int,
    z_dim: int,
    e_dim: int,
):
    n_node, n_orb, n_edge = 100, 20, 300
    z_embed = torch.rand((n_node, z_dim))
    e_embed = torch.rand((n_node, n_orb, e_dim)) if e_dim else None
    idx_i = torch.randint(0, n_node, (n_edge,))
    idx_j = torch.randint(0, n_node, (n_edge,))

    ec = EmbedCoeffs(hidden_dim, z_dim, e_dim, weight_init=glorot_orthogonal)
    coeff_embed = ec(z_embed, e_embed, idx_i, idx_j)

    assert coeff_embed.size() == (n_edge, n_orb, hidden_dim)


param_LCAOOut = [
    (10, 1, True, True),
    (10, 1, True, False),
    (10, 1, False, True),
    (10, 1, False, False),
    (10, 2, True, True),
    (10, 2, True, False),
    (10, 2, False, True),
    (10, 2, False, False),
    (100, 1, True, True),
    (100, 2, True, True),
]


@pytest.mark.parametrize("hidden_dim, out_dim, is_extensive, batch", param_LCAOOut)
def test_LCAOOut(
    hidden_dim: int,
    out_dim: int,
    is_extensive: bool,
    batch: bool,
):
    n_node = 100
    n_batch = 5 if batch else 1
    batch_idx = torch.randint(0, n_batch, (n_node,)) if batch else None
    node_embed = torch.rand((n_node, hidden_dim))

    lcao_out = LCAOOut(hidden_dim, out_dim, is_extensive, weight_init=glorot_orthogonal)
    out = lcao_out(node_embed, batch_idx)

    assert out.size() == (n_batch, out_dim)


param_LCAONet = [
    (16, 16, 10, 1, 1, None, None, None, True, False, False, True),  # default small
    # boolean param test
    (16, 16, 10, 1, 1, None, None, None, True, False, False, False),
    (16, 16, 10, 1, 1, None, None, None, True, False, True, True),
    (16, 16, 10, 1, 1, None, None, None, True, False, True, False),
    (16, 16, 10, 1, 1, None, None, None, True, True, False, True),
    (16, 16, 10, 1, 1, None, None, None, True, True, False, False),
    (16, 16, 10, 1, 1, None, None, None, True, True, True, True),
    (16, 16, 10, 1, 1, None, None, None, True, True, True, False),
    (16, 16, 10, 1, 1, None, None, None, False, False, False, True),
    (16, 16, 10, 1, 1, None, None, None, False, False, False, False),
    (16, 16, 10, 1, 1, None, None, None, False, False, True, True),
    (16, 16, 10, 1, 1, None, None, None, False, False, True, False),
    (16, 16, 10, 1, 1, None, None, None, False, True, False, True),
    (16, 16, 10, 1, 1, None, None, None, False, True, False, False),
    (16, 16, 10, 1, 1, None, None, None, False, True, True, True),
    (16, 16, 10, 1, 1, None, None, None, False, True, True, False),
    # cutoff test
    (16, 16, 10, 1, 1, 5.0, None, None, True, False, False, True),
    (16, 16, 10, 1, 1, 5.0, "polynomial", None, True, False, False, True),
    (16, 16, 10, 1, 1, None, "polynomial", None, True, False, False, True),
    (16, 16, 10, 1, 1, 5.0, "cosine", None, True, False, False, True),
    (16, 16, 10, 1, 1, None, "cosine", None, True, False, False, True),
    (16, 16, 10, 1, 1, 2.0, "polynomial", None, True, True, True, True),
    (16, 16, 10, 1, 1, 2.0, "cosine", None, True, True, True, True),
    # dimension test
    (32, 16, 10, 1, 1, None, None, None, True, False, False, True),
    (16, 32, 10, 1, 1, None, None, None, True, False, False, True),
    (10, 16, 32, 1, 1, None, None, None, True, False, False, True),
    (10, 32, 16, 2, 1, None, None, None, True, False, False, True),
    (10, 32, 16, 2, 1, None, None, None, True, True, True, True),
    # n_per_orb test
    (16, 16, 10, 1, 2, None, None, None, True, False, False, True),
    (16, 16, 10, 1, 2, None, None, None, True, True, True, True),
    (10, 32, 16, 2, 2, None, None, None, True, False, False, True),
    # max_orb test
    (16, 16, 10, 1, 2, None, None, "4p", True, False, False, True),
    (16, 16, 10, 1, 2, None, None, "3s", True, False, False, True),
    (16, 16, 10, 1, 2, None, None, "1s", True, False, False, True),
    (16, 16, 10, 1, 2, None, None, "4p", True, True, True, True),
    (10, 32, 16, 2, 1, None, None, "3s", True, False, False, True),
]


@pytest.mark.model
@pytest.mark.parametrize(
    """hidden_dim, coeffs_dim, conv_dim, out_dim, n_per_orb, cutoff, cutoff_net, max_orb,
    elec_to_node, add_valence, extend_orb, is_extensive""",
    param_LCAONet,
)
def test_LCAONet(
    one_graph_data: Data,
    hidden_dim: int,
    coeffs_dim: int,
    conv_dim: int,
    out_dim: int,
    n_per_orb: int,
    cutoff: float | None,
    cutoff_net: str | type[BaseCutoff] | None,
    max_orb: str | None,
    elec_to_node: bool,
    add_valence: bool,
    extend_orb: bool,
    is_extensive: bool,
):
    max_z = one_graph_data[GraphKeys.Z].max().item()
    if cutoff_net is not None and cutoff is None:
        with pytest.raises(ValueError) as e:
            _ = LCAONet(
                hidden_dim=hidden_dim,
                coeffs_dim=coeffs_dim,
                conv_dim=conv_dim,
                out_dim=out_dim,
                n_interaction=2,
                n_per_orb=n_per_orb,
                cutoff=cutoff,
                rbf_type="hydrogen",
                cutoff_net=cutoff_net,
                max_z=max_z,
                max_orb=max_orb,
                elec_to_node=elec_to_node,
                add_valence=add_valence,
                extend_orb=extend_orb,
                is_extensive=is_extensive,
                activation="SiLU",
                weight_init="glorotorthogonal",
            )
        assert str(e.value) == "cutoff must be specified when cutoff_net is used"
    else:
        model = LCAONet(
            hidden_dim=hidden_dim,
            coeffs_dim=coeffs_dim,
            conv_dim=conv_dim,
            out_dim=out_dim,
            n_interaction=2,
            n_per_orb=n_per_orb,
            cutoff=cutoff,
            rbf_type="hydrogen",
            cutoff_net=cutoff_net,
            max_z=max_z,
            max_orb=max_orb,
            elec_to_node=elec_to_node,
            add_valence=add_valence,
            extend_orb=extend_orb,
            is_extensive=is_extensive,
            activation="SiLU",
            weight_init="glorotorthogonal",
        )

        with torch.no_grad():
            out = model(one_graph_data)
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
