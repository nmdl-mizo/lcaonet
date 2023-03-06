from __future__ import annotations  # type: ignore

import pytest
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torch_geometric.data import Data
from torch_scatter import scatter

from lcaonet.data import DataKeys
from lcaonet.model.lcaonet import EmbedElec, EmbedZ, LCAONet, PostProcess
from lcaonet.nn.cutoff import BaseCutoff, CosineCutoff, PolynomialCutoff


@pytest.fixture(scope="module")
def set_seed():
    seed_everything(42)


param_e_embed = [
    (2, 96, None, False),
    (2, 96, None, True),
    (2, 5, "3s", False),
    (2, 5, "3s", True),
    (32, 96, None, False),
    (32, 96, None, True),
    (32, 5, "3s", False),
    (32, 5, "3s", True),
]


@pytest.mark.parametrize("embed_dim, max_z, max_orb, extend_orb", param_e_embed)
def test_embed_elec(
    embed_dim: int,
    max_z: int,
    max_orb: str | None,
    extend_orb: bool,
):
    zs = torch.tensor([i for i in range(max_z + 1)])
    ez = EmbedZ(embed_dim, max_z)
    z_embed = ez(zs)
    ec = EmbedElec(embed_dim, 96, max_orb, extend_orb)
    coeffs = ec(zs, z_embed)
    # check embeding shape
    assert coeffs.size() == (zs.size(0), ec.n_orb, embed_dim)
    for i, z in enumerate(zs):
        # check padding_idx
        if not extend_orb:
            assert (coeffs[i, ec.elec[z] == 0, :] == torch.zeros_like(coeffs[i, ec.elec[z] == 0, :])).all()  # type: ignore # NOQA: E501
        if extend_orb:
            assert (coeffs[i, ec.elec[z] == 0, :] != torch.zeros_like(coeffs[i, ec.elec[z] == 0, :])).all()  # type: ignore # NOQA: E501


param_postprocess = [
    (1, None, False, None, True),  # default
    (1, None, False, None, False),  # no extensive
    (1, None, True, None, True),  # add mean
    (1, None, True, torch.ones(1), True),  # add mean (1)
    (1, None, True, torch.ones(1), False),  # add mean (1) and no extensive
    (1, torch.ones((100, 1)), False, None, True),  # add atomref
    (1, torch.ones((100, 1)), False, None, False),  # add atomref and no extensive
    (1, torch.ones((100, 1)), True, None, True),  # add atomref and mean
    (1, torch.ones((100, 1)), True, torch.ones(1), True),  # add atomref and mean (1)
    (1, torch.ones((100, 1)), True, torch.ones(1), False),  # add atomref and mean (1) and no extensive
    (10, torch.ones((100, 1)), True, torch.ones(1), True),  # add atomref and mean (1)
    (10, torch.ones((100, 1)), True, torch.ones(1), False),  # add atomref and mean (1) and no extensive
]


@pytest.mark.parametrize("out_dim, atomref, add_mean, mean, is_extensive", param_postprocess)
def test_postprocess(
    out_dim: int,
    atomref: torch.Tensor | None,
    add_mean: bool,
    mean: torch.Tensor | None,
    is_extensive: bool,
):
    n_batch, n_node = 10, 100
    out = torch.rand((n_batch, out_dim))
    zs = torch.randint(0, 100, (n_node,))
    batch_idx = torch.randint(0, n_batch, (n_node,))
    pp_layer = PostProcess(out_dim, atomref, add_mean, mean, is_extensive)

    out_pp = pp_layer(out, zs, batch_idx)

    assert out_pp.size() == (n_batch, out_dim)

    expected = out
    reduce = "sum" if is_extensive else "mean"
    if atomref is not None:
        expected += scatter(atomref[zs], batch_idx, dim=0, dim_size=n_batch, reduce=reduce)
    if add_mean:
        if mean is None:
            mean = torch.zeros(out_dim)
        else:
            mean = mean.unsqueeze(0).expand(n_node, -1)
            mean = scatter(mean, batch_idx, dim=0, dim_size=n_batch, reduce=reduce)
        expected += mean
    assert torch.allclose(out_pp, expected)


param_lcaonet = [
    (16, 16, 10, 1, 2.0, False, False, PolynomialCutoff, False),
    (16, 16, 10, 1, 2.0, False, True, PolynomialCutoff, False),
    (16, 16, 10, 1, 2.0, True, False, PolynomialCutoff, False),
    (16, 16, 10, 1, 2.0, True, True, PolynomialCutoff, False),
    (16, 16, 10, 1, 2.0, True, True, CosineCutoff, False),
    (16, 16, 10, 1, 2.0, True, True, None, False),
    (16, 16, 10, 1, 2.0, True, True, CosineCutoff, True),
    (16, 16, 10, 1, 2.0, True, True, None, True),
    (16, 16, 10, 2, 2.0, False, False, PolynomialCutoff, False),
    (16, 16, 10, 2, 2.0, False, True, PolynomialCutoff, False),
    (16, 16, 10, 2, 2.0, True, False, PolynomialCutoff, False),
    (16, 16, 10, 2, 2.0, True, True, PolynomialCutoff, False),
    (16, 16, 10, 2, 2.0, True, True, CosineCutoff, False),
    (16, 16, 10, 2, 2.0, True, True, None, False),
    (16, 16, 10, 1, None, False, False, None, False),
    (16, 16, 10, 1, None, False, True, None, False),
    (16, 16, 10, 1, None, True, False, None, False),
    (16, 16, 10, 1, None, True, True, None, False),
    (16, 16, 10, 1, None, True, True, PolynomialCutoff, False),
    (16, 16, 10, 1, None, True, True, CosineCutoff, False),
    (16, 16, 10, 1, None, True, True, PolynomialCutoff, True),
]


@pytest.mark.parametrize(
    "hidden_dim, coeffs_dim, conv_dim, out_dim, cutoff, extend_orb, add_valence, cutoff_net, postprocess", param_lcaonet
)
def test_LCAONet(
    one_graph_data: Data,
    hidden_dim: int,
    coeffs_dim: int,
    conv_dim: int,
    out_dim: int,
    cutoff: float | None,
    extend_orb: bool,
    add_valence: bool,
    cutoff_net: type[BaseCutoff] | None,
    postprocess: bool,
):
    max_z = one_graph_data[DataKeys.Atom_numbers].max().item()
    if cutoff_net is not None and cutoff is None:
        with pytest.raises(ValueError) as e:
            LCAONet(
                hidden_dim=hidden_dim,
                coeffs_dim=coeffs_dim,
                conv_dim=conv_dim,
                out_dim=out_dim,
                n_interaction=2,
                cutoff=cutoff,
                cutoff_net=cutoff_net,
                activation="Silu",
                add_valence=add_valence,
                extend_orb=extend_orb,
                weight_init="glorot_orthogonal",
                max_z=max_z,
                postprocess=postprocess,
            )
            assert str(e.value) == "cutoff_net must be specified when cutoff is not None"
    else:
        model = LCAONet(
            hidden_dim=hidden_dim,
            coeffs_dim=coeffs_dim,
            conv_dim=conv_dim,
            out_dim=out_dim,
            n_interaction=2,
            cutoff=cutoff,
            cutoff_net=cutoff_net,
            activation="Silu",
            add_valence=add_valence,
            extend_orb=extend_orb,
            weight_init="glorot_orthogonal",
            max_z=max_z,
            postprocess=postprocess,
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
