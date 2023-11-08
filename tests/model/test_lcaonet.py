from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torch_geometric.data import Data
from torch_geometric.nn.inits import glorot_orthogonal

from lcaonet.data.keys import GraphKeys
from lcaonet.model.lcaonet import LCAONet, LCAOOut
from lcaonet.nn.cutoff import BaseCutoff


@pytest.fixture(scope="module")
def set_seed():
    seed_everything(42)


# TODO add force test
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


@pytest.mark.parametrize("emb_size, out_size, is_extensive, batch", param_LCAOOut)
def test_LCAOOut(
    emb_size: int,
    out_size: int,
    is_extensive: bool,
    batch: bool,
):
    n_node, n_edge = 50, 100
    n_batch = 5 if batch else 1
    node_emb = torch.rand((n_node, emb_size))
    batch_idx = torch.randint(0, n_batch, (n_node,)) if batch else None
    idx_s, idx_t = torch.randint(0, n_node, (n_edge,)), torch.randint(0, n_node, (n_edge,))
    pos = torch.rand((n_node, 3))
    edge_vec_st = pos[idx_t] - pos[idx_s]

    lcao_out = LCAOOut(emb_size, out_size, is_extensive, weight_init=glorot_orthogonal)
    out = lcao_out(node_emb, batch_idx, idx_s, idx_t, edge_vec_st, pos)

    assert out.size() == (n_batch, out_size)

    if batch:
        after_lin = lcao_out.out_lin(node_emb)
        out_test = torch.zeros((n_batch, out_size))
        for i, idx in enumerate(batch_idx):  # type: ignore # Since mypy cannnot determine that batch_idx is iterable
            out_test[idx] += after_lin[i]
        if not is_extensive:
            out_test /= torch.bincount(batch_idx, minlength=n_batch).unsqueeze(1) + 1e-9  # type: ignore # Since mypy cannot determine that batch_idx is Tensor # noqa: E501
        assert torch.allclose(out, out_test)
    else:
        out_test = lcao_out.out_lin(node_emb).sum(dim=0, keepdim=True)
        if not is_extensive:
            out_test /= n_node
        assert torch.allclose(out, out_test)


# fmt: off
param_LCAONet = [
    # default small
    (16, 16, 10, 1, 1, 6.0, "envelope", None, True, False, False, True, "hydrogen", None, False, False, True),
    # boolean param test
    (16, 16, 10, 1, 1, 6.0, "envelope", None, True, False, False, False, "hydrogen", None, False, False, True),
    (16, 16, 10, 1, 1, 6.0, "envelope", None, True, False, True, True, "hydrogen", None, False, False, True),
    (16, 16, 10, 1, 1, 6.0, "envelope", None, True, False, True, False, "hydrogen", None, False, False, True),
    (16, 16, 10, 1, 1, 6.0, "envelope", None, True, True, False, True, "hydrogen", None, False, False, True),
    (16, 16, 10, 1, 1, 6.0, "envelope", None, True, True, False, False, "hydrogen", None, False, False, True),
    (16, 16, 10, 1, 1, 6.0, "envelope", None, True, True, True, True, "hydrogen", None, False, False, True),
    (16, 16, 10, 1, 1, 6.0, "envelope", None, True, True, True, False, "hydrogen", None, False, False, True),
    (16, 16, 10, 1, 1, 6.0, "envelope", None, False, False, False, True, "hydrogen", None, False, False, True),
    (16, 16, 10, 1, 1, 6.0, "envelope", None, False, False, False, False, "hydrogen", None, False, False, True),
    (16, 16, 10, 1, 1, 6.0, "envelope", None, False, False, True, True, "hydrogen", None, False, False, True),
    (16, 16, 10, 1, 1, 6.0, "envelope", None, False, False, True, False, "hydrogen", None, False, False, True),
    (16, 16, 10, 1, 1, 6.0, "envelope", None, False, True, False, True, "hydrogen", None, False, False, True),
    (16, 16, 10, 1, 1, 6.0, "envelope", None, False, True, False, False, "hydrogen", None, False, False, True),
    (16, 16, 10, 1, 1, 6.0, "envelope", None, False, True, True, True, "hydrogen", None, False, False, True),
    (16, 16, 10, 1, 1, 6.0, "envelope", None, False, True, True, False, "hydrogen", None, False, False, True),
    # cutoff test
    (16, 16, 10, 1, 1, 6.0, "polynomial", None, True, False, False, True, "hydrogen", None, False, False, True),
    (16, 16, 10, 1, 1, 6.0, "cosine", None, True, False, False, True, "hydrogen", None, False, False, True),
    (16, 16, 10, 1, 1, 6.0, "envelope", None, True, False, False, True, "hydrogen", None, False, False, True),
    (16, 16, 10, 1, 1, 2.0, "polynomial", None, True, True, True, True, "hydrogen", None, False, False, True),
    (16, 16, 10, 1, 1, 2.0, "cosine", None, True, True, True, True, "hydrogen", None, False, False, True),
    (16, 16, 10, 1, 1, 2.0, "envelope", None, True, True, True, True, "hydrogen", None, False, False, True),
    # dimension test
    (32, 16, 10, 1, 1, 6.0, "envelope", None, True, False, False, True, "hydrogen", None, False, False, True),
    (16, 32, 10, 1, 1, 6.0, "envelope", None, True, False, False, True, "hydrogen", None, False, False, True),
    (10, 16, 32, 1, 1, 6.0, "envelope", None, True, False, False, True, "hydrogen", None, False, False, True),
    (10, 32, 16, 2, 1, 6.0, "envelope", None, True, False, False, True, "hydrogen", None, False, False, True),
    (10, 32, 16, 2, 1, 6.0, "envelope", None, True, True, True, True, "hydrogen", None, False, False, True),
    # n_per_orb test
    (16, 16, 10, 1, 2, 6.0, "envelope", None, True, False, False, True, "hydrogen", None, False, False, True),
    (16, 16, 10, 1, 2, 6.0, "envelope", None, True, True, True, True, "hydrogen", None, False, False, True),
    (10, 32, 16, 2, 2, 6.0, "envelope", None, True, False, False, True, "hydrogen", None, False, False, True),
    # max_orb test
    (16, 16, 10, 1, 2, 6.0, "envelope", "4p", True, False, False, True, "hydrogen", None, False, False, True),
    (16, 16, 10, 1, 2, 6.0, "envelope", "3s", True, False, False, True, "hydrogen", None, False, False, True),
    (16, 16, 10, 1, 2, 6.0, "envelope", "1s", True, False, False, True, "hydrogen", None, False, False, True),
    (16, 16, 10, 1, 2, 6.0, "envelope", "4p", True, True, True, True, "hydrogen", None, False, False, True),
    (10, 32, 16, 2, 1, 6.0, "envelope", "3s", True, False, False, True, "hydrogen", None, False, False, True),
    # rbf_type test
    (16, 16, 10, 1, 1, 2.0, "envelope", None, True, False, False, True, "sphericalbessel", None, False, False, True),
    (16, 16, 10, 1, 1, 2.0, "envelope", "4p", True, False, False, True, "sphericalbessel", None, False, False, True),
    (16, 16, 10, 1, 2, 2.0, "envelope", None, True, False, False, True, "sphericalbessel", None, False, False, True),
    (16, 16, 10, 1, 2, 2.0, "envelope", None, True, True, True, True, "sphericalbessel", None, False, False, True),
    (16, 16, 10, 1, 2, 2.0, "envelope", "4p", True, True, True, True, "sphericalbessel", None, False, False, True),
    # mean and atomref test
    (16, 16, 10, 1, 1, 2.0, "envelope", None, True, False, False, True, "hydrogen", torch.tensor([1.0]), False, False,True),
    (16, 16, 10, 1, 1, 2.0, "envelope", None, True, False, False, False, "hydrogen", torch.tensor([1.0]), False, False, True),
    (16, 16, 10, 1, 1, 2.0, "envelope", None, True, False, False, True, "hydrogen", None, True, False, True),
    (16, 16, 10, 1, 1, 2.0, "envelope", None, True, False, False, False, "hydrogen", None, True, False, True),
    (16, 16, 10, 1, 1, 2.0, "envelope", None, True, False, False, True, "hydrogen", torch.tensor([1.0]), True, False, True),
    (16, 16, 10, 1, 1, 2.0, "envelope", None, True, False, False, False, "hydrogen", torch.tensor([1.0]), True, False, True),
    # force test
    (16, 16, 10, 1, 1, 6.0, "envelope", None, True, False, False, True, "hydrogen", None, False, True, True),
    (16, 16, 10, 1, 1, 6.0, "envelope", None, True, False, False, True, "hydrogen", None, False, True, False),
]
# fmt: on


# @pytest.mark.model
@pytest.mark.parametrize(
    """emb_size, emb_size_coeff, emb_size_conv, out_size, n_per_orb, cutoff, cutoff_net, max_orb,
    elec_to_node, add_valence, extend_orb, is_extensive, rbf_type, mean, atomref, regress_forces, direct_forces""",
    param_LCAONet,
)
def test_LCAONet(
    one_graph_data: Data,
    emb_size: int,
    emb_size_coeff: int,
    emb_size_conv: int,
    out_size: int,
    n_per_orb: int,
    cutoff: float,
    cutoff_net: str | type[BaseCutoff],
    max_orb: str | None,
    elec_to_node: bool,
    add_valence: bool,
    extend_orb: bool,
    is_extensive: bool,
    rbf_type: str,
    mean: torch.Tensor | None,
    atomref: bool,
    regress_forces: bool,
    direct_forces: bool,
):
    max_z = one_graph_data[GraphKeys.Z].max().item()
    aref = torch.ones((max_z + 1, out_size)) if atomref else None
    model = LCAONet(
        emb_size=emb_size,
        emb_size_coeff=emb_size_coeff,
        emb_size_conv=emb_size_conv,
        out_size=out_size,
        n_interaction=2,
        n_per_orb=n_per_orb,
        cutoff=cutoff,
        rbf_type=rbf_type,
        cutoff_net=cutoff_net,
        max_z=max_z,
        max_orb=max_orb,
        elec_to_node=elec_to_node,
        add_valence=add_valence,
        extend_orb=extend_orb,
        is_extensive=is_extensive,
        activation="SiLU",
        weight_init="glorotorthogonal",
        mean=mean,
        atomref=aref,
        regress_forces=regress_forces,
        direct_forces=direct_forces,
    )

    if not direct_forces and regress_forces:
        out = model(one_graph_data)
        if regress_forces:
            assert len(out) == 2
            assert out[0].size() == (1, out_size)
            assert out[1].size() == (one_graph_data["pos"].size(0), 3)
        else:
            assert out.size() == (1, out_size)

        jit = torch.jit.export(model)
        if regress_forces:
            assert len(jit(one_graph_data)) == 2
            assert torch.allclose(jit(one_graph_data)[0], out[0])
            assert torch.allclose(jit(one_graph_data)[1], out[1])
        else:
            assert torch.allclose(jit(one_graph_data), out)
    else:
        with torch.no_grad():
            out = model(one_graph_data)
            if regress_forces:
                assert len(out) == 2
                assert out[0].size() == (1, out_size)
                assert out[1].size() == (one_graph_data["pos"].size(0), 3)
            else:
                assert out.size() == (1, out_size)

            jit = torch.jit.export(model)
            if regress_forces:
                assert len(jit(one_graph_data)) == 2
                assert torch.allclose(jit(one_graph_data)[0], out[0])
                assert torch.allclose(jit(one_graph_data)[1], out[1])
            else:
                assert torch.allclose(jit(one_graph_data), out)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    min_loss = float("inf")
    if regress_forces:
        for _ in range(100):
            optimizer.zero_grad()
            out = model(one_graph_data)
            e_loss = F.mse_loss(out[0], torch.ones((1, out_size)))
            f_loss = F.mse_loss(out[1], torch.ones((one_graph_data["pos"].size(0), 3)))
            loss = (1 - 0.999) * e_loss + 0.999 * f_loss
            loss.backward()
            loss = torch.nan_to_num(loss)
            optimizer.step()
            min_loss = min(float(loss), min_loss)
        assert min_loss < 2

    else:
        for _ in range(100):
            optimizer.zero_grad()
            out = model(one_graph_data)
            loss = F.mse_loss(out, torch.ones((1, out_size)))
            loss.backward()
            optimizer.step()
            min_loss = min(float(loss), min_loss)
        assert min_loss < 2
