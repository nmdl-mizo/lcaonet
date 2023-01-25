import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from pyg_material.data import DataKeys
from pyg_material.model import DimeNet, DimeNetPlusPlus

testdata = [
    (16, 8, 4, 4, 1),
    (12, 8, 0, 2, 1),
    (12, 65, 0, 2, 1),
    (12, 4, 4, 4, 2),
]


@pytest.mark.parametrize("edge_message_dim, n_rad, n_sph, n_bilinear, out_dim", testdata)
def test_dimenet(
    one_graph_data: Data,
    edge_message_dim: int,
    n_rad: int,
    n_sph: int,
    n_bilinear: int,
    out_dim: int,
):
    max_z = one_graph_data[DataKeys.Atom_numbers].max().item() + 1
    if n_sph == 0:
        with pytest.raises(ValueError) as e:
            model = DimeNet(
                edge_message_dim=edge_message_dim,
                n_interaction=2,
                out_dim=out_dim,
                n_rad=n_rad,
                n_sph=n_sph,
                n_bilinear=n_bilinear,
                cutoff=2,
                envelope_exponent=5,
                aggr="add",
                weight_init="glorot_orthogonal",
                max_z=max_z,
            )
            assert str(e.value) == "n_sph must be greater than 0."
    elif n_rad > 64:
        with pytest.raises(ValueError) as e:
            model = DimeNet(
                edge_message_dim=edge_message_dim,
                n_interaction=2,
                out_dim=out_dim,
                n_rad=n_rad,
                n_sph=n_sph,
                n_bilinear=n_bilinear,
                cutoff=2,
                envelope_exponent=5,
                aggr="add",
                weight_init="glorot_orthogonal",
                max_z=max_z,
            )
            assert str(e.value) == "n_rad must be less than 64."
    else:
        model = DimeNet(
            edge_message_dim=edge_message_dim,
            n_interaction=2,
            out_dim=out_dim,
            n_rad=n_rad,
            n_sph=n_sph,
            n_bilinear=n_bilinear,
            cutoff=2,
            envelope_exponent=5,
            aggr="add",
            weight_init="glorot_orthogonal",
            max_z=max_z,
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
            # no batch
            loss = F.l1_loss(out, torch.ones((1, out_dim)))
            loss.backward()
            optimizer.step()
            min_loss = min(float(loss), min_loss)
        assert min_loss < 2


testdata_pp = [
    (16, 8, 4, 1),
    (12, 8, 0, 1),
    (12, 65, 0, 1),
    (12, 8, 4, 2),
]


@pytest.mark.parametrize("edge_message_dim, n_rad, n_sph, out_dim", testdata_pp)
def test_dimenet_plus_plus(
    one_graph_data: Data,
    edge_message_dim: int,
    n_rad: int,
    n_sph: int,
    out_dim: int,
):
    max_z = one_graph_data[DataKeys.Atom_numbers].max().item() + 1
    if n_sph == 0:
        with pytest.raises(ValueError) as e:
            model = DimeNetPlusPlus(
                edge_message_dim=edge_message_dim,
                n_interaction=2,
                out_dim=out_dim,
                n_rad=n_rad,
                n_sph=n_sph,
                edge_down_dim=8,
                basis_embed_dim=8,
                out_up_dim=12,
                cutoff=2,
                envelope_exponent=5,
                aggr="add",
                weight_init="glorot_orthogonal",
                max_z=max_z,
            )
            assert str(e.value) == "n_sph must be greater than 0."
    elif n_rad > 64:
        with pytest.raises(ValueError) as e:
            model = DimeNetPlusPlus(
                edge_message_dim=edge_message_dim,
                n_interaction=2,
                out_dim=out_dim,
                n_rad=n_rad,
                n_sph=n_sph,
                edge_down_dim=8,
                basis_embed_dim=8,
                out_up_dim=12,
                cutoff=2,
                envelope_exponent=5,
                aggr="add",
                weight_init="glorot_orthogonal",
                max_z=max_z,
            )
            assert str(e.value) == "n_rad must be under 64."
    else:
        model = DimeNetPlusPlus(
            edge_message_dim=edge_message_dim,
            n_interaction=2,
            out_dim=out_dim,
            n_rad=n_rad,
            n_sph=n_sph,
            edge_down_dim=8,
            basis_embed_dim=8,
            out_up_dim=12,
            cutoff=2,
            envelope_exponent=5,
            aggr="add",
            weight_init="glorot_orthogonal",
            max_z=max_z,
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
