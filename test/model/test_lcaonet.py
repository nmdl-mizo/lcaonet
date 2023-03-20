from __future__ import annotations  # type: ignore

import pytest
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torch_geometric.data import Data
from torch_geometric.nn.inits import glorot_orthogonal
from torch_scatter import scatter

from lcaonet.atomistic import CustomNOrbAtomisticInformation
from lcaonet.data import DataKeys
from lcaonet.model.lcaonet import EmbedElec, EmbedNode, LCAONet, PostProcess
from lcaonet.nn.cutoff import BaseCutoff, CosineCutoff, PolynomialCutoff


@pytest.fixture(scope="module")
def set_seed():
    seed_everything(42)


param_EmbedElec = [
    (2, 96, None, False, 1),
    (2, 96, None, True, 1),
    (2, 96, "3s", False, 1),
    (2, 96, "3s", True, 1),
    (2, 5, "3s", False, 1),
    (2, 5, "3s", True, 1),
    (2, 5, "5p", False, 1),
    (2, 5, "5p", True, 1),
    (2, 96, None, False, 2),
    (2, 96, None, True, 2),
    (2, 96, "3s", False, 2),
    (2, 96, "3s", True, 2),
    (32, 96, None, False, 1),
    (32, 96, None, True, 1),
    (32, 5, "3s", False, 1),
    (32, 5, "3s", True, 1),
]


@pytest.mark.parametrize("embed_dim, max_z, max_orb, extend_orb, n_per_orb", param_EmbedElec)
def test_EmbedElec(
    embed_dim: int,
    max_z: int,
    max_orb: str | None,
    extend_orb: bool,
    n_per_orb: int,
):
    atom_info = CustomNOrbAtomisticInformation(max_z, max_orb, n_per_orb=n_per_orb)
    zs = torch.tensor([i for i in range(max_z + 1)])
    ec = EmbedElec(embed_dim, atom_info, extend_orb)
    coeffs = ec(zs)
    # check embeding shape
    assert coeffs.size() == (zs.size(0), ec.n_orb, embed_dim)
    for i, z in enumerate(zs):
        # check padding_idx
        if not extend_orb:
            assert (coeffs[i, ec.elec[z] == 0, :] == torch.zeros_like(coeffs[i, ec.elec[z] == 0, :])).all()  # type: ignore # NOQA: E501
        if extend_orb:
            assert (coeffs[i, ec.elec[z] == 0, :] != torch.zeros_like(coeffs[i, ec.elec[z] == 0, :])).all()  # type: ignore # NOQA: E501


param_EmbedNode = [
    (10, 10, True, 10),
    (10, 10, False, 10),
    (10, 10, False, None),
    (100, 10, True, 10),
]


@pytest.mark.parametrize("hidden_dim, z_dim, use_elec, e_dim", param_EmbedNode)
def test_EmbedNode(
    hidden_dim: int,
    z_dim: int,
    use_elec: bool,
    e_dim: int | None,
):
    node_embed = EmbedNode(hidden_dim, z_dim, use_elec, e_dim, weight_init=glorot_orthogonal)
    z_embed = torch.rand((100, z_dim))
    if use_elec:
        e_embed = torch.rand((100, 20, e_dim)) if e_dim else None
    else:
        e_embed = None
    node = node_embed(z_embed, e_embed)
    assert node.size() == (100, hidden_dim)


param_PostProcess = [
    (1, None, None, True),  # default
    (1, None, None, False),  # no extensive
    (1, None, None, True),  # add mean
    (1, None, torch.ones(1), True),  # add mean (1)
    (1, None, torch.ones(1), False),  # add mean (1) and no extensive
    (1, torch.ones((100, 1)), None, True),  # add atomref
    (1, torch.ones((100, 1)), None, False),  # add atomref and no extensive
    (1, torch.ones((100, 1)), None, True),  # add atomref and mean
    (1, torch.ones((100, 1)), torch.ones(1), True),  # add atomref and mean (1)
    (1, torch.ones((100, 1)), torch.ones(1), False),  # add atomref and mean (1) and no extensive
    (10, torch.ones((100, 1)), torch.ones(1), True),  # add atomref and mean (1)
    (10, torch.ones((100, 1)), torch.ones(1), False),  # add atomref and mean (1) and no extensive
]


@pytest.mark.parametrize("out_dim, atomref, mean, is_extensive", param_PostProcess)
def test_PostProcess(
    out_dim: int,
    atomref: torch.Tensor | None,
    mean: torch.Tensor | None,
    is_extensive: bool,
):
    n_batch, n_node = 10, 100
    out = torch.rand((n_batch, out_dim))
    zs = torch.randint(0, 100, (n_node,))
    batch_idx = torch.randint(0, n_batch, (n_node,))
    pp_layer = PostProcess(out_dim, is_extensive, atomref, mean)

    out_pp = pp_layer(out, zs, batch_idx)

    assert out_pp.size() == (n_batch, out_dim)

    expected = out
    reduce = "sum" if is_extensive else "mean"
    if atomref is not None:
        expected += scatter(atomref[zs], batch_idx, dim=0, dim_size=n_batch, reduce=reduce)
    if mean is None:
        mean = torch.zeros(out_dim)
    else:
        mean = mean.unsqueeze(0).expand(n_node, -1)
        mean = scatter(mean, batch_idx, dim=0, dim_size=n_batch, reduce=reduce)
    expected += mean
    assert torch.allclose(out_pp, expected)


param_LCAONet = [
    # cutoff check
    (16, 16, 10, 1, 2.0, True, True, True, None, None, 2, 7),
    (16, 16, 10, 1, 2.0, True, True, False, None, None, 2, 7),
    (16, 16, 10, 1, 2.0, True, False, True, None, None, 2, 7),
    (16, 16, 10, 1, 2.0, True, False, False, None, None, 2, 7),
    (16, 16, 10, 1, 2.0, False, True, True, None, None, 2, 7),
    (16, 16, 10, 1, 2.0, False, True, False, None, None, 2, 7),
    (16, 16, 10, 1, 2.0, False, False, True, None, None, 2, 7),
    (16, 16, 10, 1, 2.0, False, False, False, None, None, 2, 7),
    (16, 16, 10, 1, 2.0, True, True, True, None, torch.tensor([1.0]), 2, 7),
    (16, 16, 10, 1, 2.0, True, True, True, PolynomialCutoff, None, 2, 7),
    (16, 16, 10, 1, 2.0, True, True, True, PolynomialCutoff, torch.tensor([1.0]), 2, 7),
    (16, 16, 10, 1, 2.0, True, True, True, CosineCutoff, None, 2, 7),
    (16, 16, 10, 1, 2.0, True, True, True, CosineCutoff, torch.tensor([1.0]), 2, 7),
    # none cutoff check
    (16, 16, 10, 1, None, True, True, True, None, None, 2, 7),
    (16, 16, 10, 1, None, True, True, False, None, None, 2, 7),
    (16, 16, 10, 1, None, True, False, True, None, None, 2, 7),
    (16, 16, 10, 1, None, True, False, False, None, None, 2, 7),
    (16, 16, 10, 1, None, False, True, True, None, None, 2, 7),
    (16, 16, 10, 1, None, False, True, False, None, None, 2, 7),
    (16, 16, 10, 1, None, False, False, True, None, None, 2, 7),
    (16, 16, 10, 1, None, False, False, False, None, None, 2, 7),
    (16, 16, 10, 1, None, True, True, True, None, torch.tensor([1.0]), 2, 7),
    (16, 16, 10, 1, None, True, True, True, PolynomialCutoff, None, 2, 7),
    (16, 16, 10, 1, None, True, True, True, PolynomialCutoff, torch.tensor([1.0]), 2, 7),
    (16, 16, 10, 1, None, True, True, True, CosineCutoff, None, 2, 7),
    (16, 16, 10, 1, None, True, True, True, CosineCutoff, torch.tensor([1.0]), 2, 7),
    # out_dim and mean check
    (16, 16, 10, 2, None, True, True, True, None, None, 2, 7),
    (16, 16, 10, 2, None, True, True, True, None, torch.tensor([1.0, 1.0]), 2, 7),
    (16, 16, 10, 5, None, True, True, True, None, torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]), 2, 7),
    # dim check
    (16, 16, 10, 1, 2.0, True, True, True, None, None, 2, 7),
    (16, 8, 10, 1, 2.0, True, True, True, None, None, 2, 7),
    (20, 8, 10, 1, 2.0, True, True, True, None, None, 2, 7),
    (20, 8, 12, 1, 2.0, True, True, True, None, None, 2, 7),
    (20, 8, 12, 10, 2.0, True, True, True, None, None, 2, 7),
    (1, 1, 1, 1, 2.0, True, True, True, None, None, 2, 7),
    # basis check
    (16, 16, 10, 1, 2.0, True, True, True, None, None, 5, 7),
    (16, 16, 10, 1, 2.0, True, True, False, None, None, 5, 5),
    (16, 16, 10, 1, 2.0, True, False, True, None, None, 5, 1),
    (16, 16, 10, 1, 2.0, True, False, False, None, None, 2, 1),
    (16, 16, 10, 1, 2.0, False, True, True, None, None, 1, 7),
    (16, 16, 10, 1, 2.0, False, True, False, None, None, 1, 5),
    (16, 16, 10, 1, 2.0, False, False, True, None, None, 5, 7),
    (16, 16, 10, 1, 2.0, False, False, False, None, None, 2, 7),
]


@pytest.mark.parametrize(
    """
    hidden_dim, coeffs_dim, conv_dim, out_dim, cutoff, elec_to_node, extend_orb,
    add_valence, cutoff_net, mean, n_per_orb, n_spherical
    """,
    param_LCAONet,
)
def test_LCAONet(
    one_graph_data: Data,
    hidden_dim: int,
    coeffs_dim: int,
    conv_dim: int,
    out_dim: int,
    cutoff: float | None,
    elec_to_node: bool,
    extend_orb: bool,
    add_valence: bool,
    cutoff_net: type[BaseCutoff] | None,
    mean: torch.Tensor | None,
    n_per_orb: int,
    n_spherical: int,
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
                rbf_form="hydrogenradialwavefunctionbasis",
                n_per_orb=n_per_orb,
                n_spherical=n_spherical,
                cutoff=cutoff,
                cutoff_net=cutoff_net,
                max_z=max_z,
                max_orb=None,
                elec_to_node=elec_to_node,
                activation="Silu",
                add_valence=add_valence,
                extend_orb=extend_orb,
                weight_init="glorot_orthogonal",
                atomref=None,
                mean=mean,
            )
            assert str(e.value) == "cutoff_net must be specified when cutoff is not None"
    else:
        model = LCAONet(
            hidden_dim=hidden_dim,
            coeffs_dim=coeffs_dim,
            conv_dim=conv_dim,
            out_dim=out_dim,
            n_interaction=2,
            rbf_form="hydrogenradialwavefunctionbasis",
            n_per_orb=n_per_orb,
            n_spherical=n_spherical,
            cutoff=cutoff,
            cutoff_net=cutoff_net,
            max_z=max_z,
            max_orb=None,
            elec_to_node=elec_to_node,
            activation="Silu",
            add_valence=add_valence,
            extend_orb=extend_orb,
            weight_init="glorot_orthogonal",
            atomref=None,
            mean=mean,
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
