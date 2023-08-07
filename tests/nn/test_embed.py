from __future__ import annotations

import pytest
import torch
from pytorch_lightning import seed_everything
from torch_geometric.nn.inits import glorot_orthogonal

from lcaonet.atomistic.info import ElecInfo
from lcaonet.nn.embed import EmbedCoeffs, EmbedElec, EmbedNode, EmbedZ, ValenceMask


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


@pytest.mark.parametrize("emb_size, max_z", param_EmbedZ)
def test_EmbedZ(
    emb_size: int,
    max_z: int,
):
    n_node = max_z * 3
    zs = torch.randint(1, max_z + 1, (n_node,))

    ez = EmbedZ(emb_size, max_z)
    z_embed = ez(zs)

    assert z_embed.size() == (n_node, emb_size)

    for i in range(1, max_z + 1):
        assert (z_embed[zs == i, :] == ez.z_embed.weight.data[i - 1]).all()


param_EmbedElec = [
    # all element test
    (2, 96, None, None, 1, False),
    (2, 96, None, None, 1, True),
    (2, 96, None, None, 3, False),
    (2, 96, None, None, 3, True),
    # dimension test
    (32, 96, None, None, 1, False),
    (32, 96, None, None, 1, True),
    (32, 96, None, None, 3, False),
    (32, 96, None, None, 3, True),
    # max_z and max_orb test
    (2, 5, "4s", None, 1, False),
    (2, 5, "4s", None, 1, True),
    (32, 5, "4p", None, 1, False),
    (32, 5, "4p", None, 1, True),
    (32, 25, "5s", None, 1, False),
    (32, 25, "5s", None, 1, True),
    (32, 25, "5s", None, 3, False),
    (32, 25, "5s", None, 3, True),
    # small orb test (not valid max_orb parameter)
    (32, 5, "1s", None, 1, False),
    (32, 5, "1s", None, 1, True),
    (32, 40, "2s", None, 1, False),
    (32, 40, "2s", None, 1, True),
    # min_orb test
    (32, 96, None, "3p", 1, False),
    (32, 96, None, "3p", 2, False),
    (32, 96, None, "3p", 3, False),
    (32, 96, None, "3p", 3, True),
    (32, 20, "4s", "4s", 3, False),
    (32, 20, "5p", "4s", 3, False),
]


@pytest.mark.parametrize("emb_size, max_z, max_orb, min_orb, n_per_orb, extend_orb", param_EmbedElec)
def test_EmbedElec(
    emb_size: int,
    max_z: int,
    max_orb: str | None,
    min_orb: str | None,
    n_per_orb: int,
    extend_orb: bool,
):
    n_node = max_z * 3
    zs = torch.randint(0, max_z + 1, (n_node,))
    ei = ElecInfo(max_z, max_orb, min_orb, n_per_orb)

    ee = EmbedElec(emb_size, ei, extend_orb)
    elec_embed = ee(zs)

    assert elec_embed.size() == (zs.size(0), ei.n_orb, emb_size)

    # check padding_idx
    for i, z in enumerate(zs):
        if extend_orb:
            # not 0 padding
            assert (elec_embed[i, ee.elec[z] == 0, :] != torch.zeros_like(elec_embed[i, ee.elec[z] == 0, :])).all()  # type: ignore # noqa: E501
            assert (elec_embed[i, ee.elec[z] != 0, :] != torch.zeros_like(elec_embed[i, ee.elec[z] != 0, :])).all()  # type: ignore # noqa: E501
        else:
            if min_orb is None:
                # 0 padding
                assert (elec_embed[i, ee.elec[z] == 0, :] == torch.zeros_like(elec_embed[i, ee.elec[z] == 0, :])).all()  # type: ignore # noqa: E501
                assert (elec_embed[i, ee.elec[z] != 0, :] != torch.zeros_like(elec_embed[i, ee.elec[z] != 0, :])).all()  # type: ignore # noqa: E501
            else:
                min_idx = ei.min_orb_idx
                for orb in range(ei.n_orb):
                    if orb <= min_idx:  # type: ignore # Since mypy cannot determine min_idx is not None
                        # not 0 padding
                        assert (elec_embed[i, orb] != torch.zeros(emb_size)).all()
                    else:
                        if ee.elec[z, orb].item() == 0:  # type: ignore # Since mypy cannnot determine elec is Tensor
                            # 0 padding for out of min_idx
                            assert (elec_embed[i, orb] == torch.zeros(emb_size)).all()
                        else:
                            assert (elec_embed[i, orb] != torch.zeros(emb_size)).all()


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


@pytest.mark.parametrize("emb_size, max_z, max_orb, n_per_orb", param_ValenceMask)
def test_ValenceMask(
    emb_size: int,
    max_z: int,
    max_orb: str | None,
    n_per_orb: int,
):
    n_node, n_edge = max_z * 3, max_z * 5
    zs = torch.randint(0, max_z + 1, (n_node,))
    idx_j = torch.randint(0, zs.size(0), (n_edge,))
    ei = ElecInfo(max_z, max_orb, None, n_per_orb)

    vm = ValenceMask(emb_size, ei)
    mask = vm(zs, idx_j)

    assert mask.size() == (n_edge, ei.n_orb, emb_size)

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


@pytest.mark.parametrize("emb_size, emb_size_z, use_elec, emb_size_e", param_EmbedNode)
def test_EmbedNode(
    emb_size: int,
    emb_size_z: int,
    use_elec: bool,
    emb_size_e: int | None,
):
    n_node, n_orb = 100, 20
    z_embed = torch.rand((n_node, emb_size_z))
    if use_elec:
        e_embed = torch.rand((n_node, n_orb, emb_size_e)) if emb_size_e else None
    else:
        e_embed = None

    if use_elec and emb_size_e is None:
        with pytest.raises(AssertionError) as e:
            _ = EmbedNode(emb_size, emb_size_z, use_elec, emb_size_e, weight_init=glorot_orthogonal)
        assert str(e.value) == ""
    else:
        en = EmbedNode(emb_size, emb_size_z, use_elec, emb_size_e, weight_init=glorot_orthogonal)
        node_embed = en(z_embed, e_embed)
        assert node_embed.size() == (n_node, emb_size)


param_EmbedCoeff = [
    (10, 10, 10),
    (100, 10, 10),
    (10, 100, 10),
    (10, 10, 100),
    (100, 100, 100),
    (10, 20, 30),
]


@pytest.mark.parametrize("emb_size, emb_size_z, emb_size_e", param_EmbedCoeff)
def test_EmbedCoeffs(
    emb_size: int,
    emb_size_z: int,
    emb_size_e: int,
):
    n_node, n_orb, n_edge = 100, 20, 300
    z_embed = torch.rand((n_node, emb_size_z))
    e_embed = torch.rand((n_node, n_orb, emb_size_e))
    idx_i = torch.randint(0, n_node, (n_edge,))
    idx_j = torch.randint(0, n_node, (n_edge,))

    ec = EmbedCoeffs(emb_size, emb_size_z, emb_size_e, n_orb, weight_init=glorot_orthogonal)
    coeff_embed = ec(z_embed, e_embed, idx_i, idx_j)

    assert coeff_embed.size() == (n_edge, n_orb, emb_size)
