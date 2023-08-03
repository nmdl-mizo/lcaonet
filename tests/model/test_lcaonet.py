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


param_LCAONet = [
    (16, 16, 10, 1, 1, 6.0, "envelope", None, True, False, False, True, "hydrogen"),  # default small
    # boolean param test
    (16, 16, 10, 1, 1, 6.0, "envelope", None, True, False, False, False, "hydrogen"),
    (16, 16, 10, 1, 1, 6.0, "envelope", None, True, False, True, True, "hydrogen"),
    (16, 16, 10, 1, 1, 6.0, "envelope", None, True, False, True, False, "hydrogen"),
    (16, 16, 10, 1, 1, 6.0, "envelope", None, True, True, False, True, "hydrogen"),
    (16, 16, 10, 1, 1, 6.0, "envelope", None, True, True, False, False, "hydrogen"),
    (16, 16, 10, 1, 1, 6.0, "envelope", None, True, True, True, True, "hydrogen"),
    (16, 16, 10, 1, 1, 6.0, "envelope", None, True, True, True, False, "hydrogen"),
    (16, 16, 10, 1, 1, 6.0, "envelope", None, False, False, False, True, "hydrogen"),
    (16, 16, 10, 1, 1, 6.0, "envelope", None, False, False, False, False, "hydrogen"),
    (16, 16, 10, 1, 1, 6.0, "envelope", None, False, False, True, True, "hydrogen"),
    (16, 16, 10, 1, 1, 6.0, "envelope", None, False, False, True, False, "hydrogen"),
    (16, 16, 10, 1, 1, 6.0, "envelope", None, False, True, False, True, "hydrogen"),
    (16, 16, 10, 1, 1, 6.0, "envelope", None, False, True, False, False, "hydrogen"),
    (16, 16, 10, 1, 1, 6.0, "envelope", None, False, True, True, True, "hydrogen"),
    (16, 16, 10, 1, 1, 6.0, "envelope", None, False, True, True, False, "hydrogen"),
    # cutoff test
    (16, 16, 10, 1, 1, 6.0, "polynomial", None, True, False, False, True, "hydrogen"),
    (16, 16, 10, 1, 1, 6.0, "cosine", None, True, False, False, True, "hydrogen"),
    (16, 16, 10, 1, 1, 6.0, "envelope", None, True, False, False, True, "hydrogen"),
    (16, 16, 10, 1, 1, 2.0, "polynomial", None, True, True, True, True, "hydrogen"),
    (16, 16, 10, 1, 1, 2.0, "cosine", None, True, True, True, True, "hydrogen"),
    (16, 16, 10, 1, 1, 2.0, "envelope", None, True, True, True, True, "hydrogen"),
    # dimension test
    (32, 16, 10, 1, 1, 6.0, "envelope", None, True, False, False, True, "hydrogen"),
    (16, 32, 10, 1, 1, 6.0, "envelope", None, True, False, False, True, "hydrogen"),
    (10, 16, 32, 1, 1, 6.0, "envelope", None, True, False, False, True, "hydrogen"),
    (10, 32, 16, 2, 1, 6.0, "envelope", None, True, False, False, True, "hydrogen"),
    (10, 32, 16, 2, 1, 6.0, "envelope", None, True, True, True, True, "hydrogen"),
    # n_per_orb test
    (16, 16, 10, 1, 2, 6.0, "envelope", None, True, False, False, True, "hydrogen"),
    (16, 16, 10, 1, 2, 6.0, "envelope", None, True, True, True, True, "hydrogen"),
    (10, 32, 16, 2, 2, 6.0, "envelope", None, True, False, False, True, "hydrogen"),
    # max_orb test
    (16, 16, 10, 1, 2, 6.0, "envelope", "4p", True, False, False, True, "hydrogen"),
    (16, 16, 10, 1, 2, 6.0, "envelope", "3s", True, False, False, True, "hydrogen"),
    (16, 16, 10, 1, 2, 6.0, "envelope", "1s", True, False, False, True, "hydrogen"),
    (16, 16, 10, 1, 2, 6.0, "envelope", "4p", True, True, True, True, "hydrogen"),
    (10, 32, 16, 2, 1, 6.0, "envelope", "3s", True, False, False, True, "hydrogen"),
    # rbf_type test
    (16, 16, 10, 1, 1, 2.0, "envelope", None, True, False, False, True, "sphericalbessel"),
    (16, 16, 10, 1, 1, 2.0, "envelope", "4p", True, False, False, True, "sphericalbessel"),
    (16, 16, 10, 1, 2, 2.0, "envelope", None, True, False, False, True, "sphericalbessel"),
    (16, 16, 10, 1, 2, 2.0, "envelope", None, True, True, True, True, "sphericalbessel"),
    (16, 16, 10, 1, 2, 2.0, "envelope", "4p", True, True, True, True, "sphericalbessel"),
]


# @pytest.mark.model
@pytest.mark.parametrize(
    """emb_size, emb_size_coeff, emb_size_conv, out_size, n_per_orb, cutoff, cutoff_net, max_orb,
    elec_to_node, add_valence, extend_orb, is_extensive, rbf_type""",
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
):
    max_z = one_graph_data[GraphKeys.Z].max().item()
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
    )

    with torch.no_grad():
        out = model(one_graph_data)
        assert out.size() == (1, out_size)

        jit = torch.jit.export(model)
        assert torch.allclose(jit(one_graph_data), out)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    min_loss = float("inf")
    for _ in range(100):
        optimizer.zero_grad()
        out = model(one_graph_data)
        loss = F.mse_loss(out, torch.ones((1, out_size)))
        loss.backward()
        optimizer.step()
        min_loss = min(float(loss), min_loss)
    assert min_loss < 2
