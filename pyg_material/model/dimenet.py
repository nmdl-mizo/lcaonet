from __future__ import annotations  # type: ignore

from collections.abc import Callable
from math import pi as PI

import sympy as sym
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn.models.dimenet_utils import bessel_basis, real_sph_harm
from torch_scatter import scatter

from pyg_material.data import DataKeys
from pyg_material.model.base import BaseGNN
from pyg_material.nn import AtomicNum2Node, Dense, ResidualBlock, Swish
from pyg_material.utils import activation_resolver, init_resolver


class EnvelopeCutoff(nn.Module):
    """Polynomial cutoff network used in DimeNet.

    Args:
        exponent (int, optional): Order of the envelope function. Defaults to `5`.

    Notes:
        ref:
            [1] J. Klicpera et al., arXiv [cs.LG] (2020), (available at http://arxiv.org/abs/2003.03123).
    """

    def __init__(self, cutoff: float, exponent: int = 5):
        super().__init__()
        self.cutoff = cutoff
        self.p = exponent + 1
        # coeffs
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, dist: Tensor) -> Tensor:
        """Compute the inter atomic distances applying the cutoff function.

        Args:
            dist (Tensor): inter atomic distances normalized by cutoff radius of (n_edge) shape.

        Returns:
            Tensor: Cutoff values of (n_edge) shape.
        """
        p, a, b, c = self.p, self.a, self.b, self.c
        # calc polynomial
        dist_pow_p0 = dist.pow(p - 1)
        dist_pow_p1 = dist_pow_p0 * dist
        dist_pow_p2 = dist_pow_p1 * dist
        # Remove contributions beyond the cutoff radius
        return (1.0 / (dist + 1e-8) + a * dist_pow_p0 + b * dist_pow_p1 + c * dist_pow_p2) * (dist < 1.0).to(dist.dtype)


class BesselRBF(torch.nn.Module):
    """Layer that expand inter atomic distances in the Bessel radial basis.

    Args:
        n_rad (int): number of radial basis.
        cutoff (float, optional): cutoff radius. Defaults to `5.0`.
        envelope_exponent (int, optional): exponent of cutoff envelope function. Defaults to `5`.
    """

    def __init__(self, n_rad: int, cutoff: float = 5.0, envelope_exponent: int = 5):
        super().__init__()
        self.n_rad = n_rad
        self.cutoff = cutoff
        self.envelope_cutoff = EnvelopeCutoff(cutoff, envelope_exponent)
        self.freq = torch.nn.Parameter(torch.Tensor(n_rad))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            torch.arange(1, self.freq.numel() + 1, out=self.freq).mul_(PI)
        self.freq.requires_grad_()

    def forward(self, dist: Tensor) -> Tensor:
        """Compute expanded distances with Bessel radial basis.

        Args:
            dist (Tensor): interatomic distances shape of (n_edge).

        Returns:
            Tensor: expanded distances with Bessel RBFs shape of (n_edge, n_rad).
        """
        dist = dist.unsqueeze(-1) / self.cutoff
        return self.envelope_cutoff(dist) * (self.freq * dist).sin()


class BesselSBF(nn.Module):
    """Layer that expand inter atomic distances and angles in Bessel spherical
    and radial basis.

    Args:
        n_rad (int): number of radial basis.
        n_sph (int): number of spherical basis.
        cutoff (float, optional): cutoff radius. Defaults to `5.0`.
        envelope_exponent (int, optional): exponent of envelope cutoff fucntion. Defaults to `5`.
    """

    def __init__(self, n_rad: int, n_sph: int, cutoff: float = 5.0, envelope_exponent: int = 5):
        super().__init__()

        if n_sph <= 0:
            raise ValueError("n_sph must be greater than 0.")
        if n_rad > 64:
            raise ValueError("n_rad must be under 64.")
        self.n_rad = n_rad
        self.n_sph = n_sph
        self.cutoff = cutoff
        self.envelope_cutoff = EnvelopeCutoff(cutoff, envelope_exponent)

        bessel_forms = bessel_basis(n_sph, n_rad)
        sph_harm_forms = real_sph_harm(n_sph)
        self.sph_funcs = []
        self.bessel_funcs = []

        # make basis functions, number of basis functions is n_sph * n_rad
        x, theta = sym.symbols("x theta")
        modules = {"sin": torch.sin, "cos": torch.cos}
        for i in range(n_sph):
            if i == 0:
                sph1 = sym.lambdify([theta], sph_harm_forms[i][0], modules)(0)
                self.sph_funcs.append(lambda x: torch.zeros_like(x) + sph1)
            else:
                sph = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(sph)
            for j in range(n_rad):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    def forward(self, dist: Tensor, angle: Tensor, edge_idx_kj: torch.LongTensor) -> Tensor:
        """Compute expanded distances and angles with Bessel spherical and
        radial basis.

        Args:
            dist (Tensor): interatomic distances of (n_edge) shape.
            angle (Tensor): angles of triplets of (n_triplets) shape.
            edge_idx_kj (torch.LongTensor): edge index from atom k to j of (n_triplets) shape.

        Returns:
            Tensor: expanded distances and angles of (n_triplets, (n_sph x n_rad)) shape.
        """
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        # apply cutoff
        rbf = self.envelope_cutoff(dist).unsqueeze(-1) * rbf

        cbf = torch.stack([f(angle) for f in self.sph_funcs], dim=1)

        n, k = self.n_sph, self.n_rad
        return (rbf[edge_idx_kj].view(-1, n, k) * cbf.view(-1, n, 1)).view(-1, n * k)


class EdgeEmbed(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        n_rad: int,
        activation: nn.Module = Swish(beta=1.0),
        weight_init: Callable[[Tensor], Tensor] | None = glorot_orthogonal,
        **kwargs,
    ):
        super().__init__()

        self.rbf_lin = Dense(n_rad, edge_dim, False, weight_init, **kwargs)
        self.edge_embed = nn.Sequential(
            Dense(2 * node_dim + edge_dim, edge_dim, True, weight_init, **kwargs),
            activation,
        )

    def forward(self, x: Tensor, rbf: Tensor, idx_i: torch.LongTensor, idx_j: torch.LongTensor) -> Tensor:
        """Forward calculation of the initial edge embedding.

        Args:
            x (Tensor): node embedding vector of (n_node, node_dim) shape.
            rbf (Tensor): radial basis function of (n_edge, n_radial) shape.
            idx_i (LongTensor): index of the first node of the edge of (n_edge) shape.
            idx_j (LongTensor): index of the neighbor node of the edge of (n_edge) shape.

        Returns:
            Tensor: embedding edge message of (n_edge, edge_dim) shape.
        """
        rbf = self.rbf_lin(rbf)
        return self.edge_embed(torch.cat([x[idx_j], x[idx_i], rbf], dim=-1))


class DimNetInteraction(nn.Module):
    """The block to calculate the message interaction using Bessel Radial Basis
    and Bessel Spherical Basis. This block is used in the DimeNet.

    Args:
        edge_message_dim (int): dimension of edge message.
        n_rad (int): number of radial basis functions.
        n_sph (int): number of spherical basis functions.
        n_bilinear (int): number of bilinear layers.
        activation (nn.Module, optional): activation function. Defaults to `Swish(beta=1.0)`.
        weight_init (Callable, optional): weight initialization function. Defaults to `torch_geometric.nn.inits.glorot_orthogonal`.
    """  # NOQA: E501

    def __init__(
        self,
        edge_message_dim: int,
        n_rad: int,
        n_sph: int,
        n_bilinear: int,
        activation: nn.Module = Swish(beta=1.0),
        weight_init: Callable[[Tensor], Tensor] | None = glorot_orthogonal,
        **kwargs,
    ):
        super().__init__()
        # Dense transformation of basis
        self.rbf_lin = Dense(n_rad, edge_message_dim, False, weight_init, **kwargs)
        self.sbf_lin = Dense(n_sph * n_rad, n_bilinear, False, weight_init, **kwargs)

        # Dense transformations of input messages.
        self.kj_lin = nn.Sequential(
            Dense(edge_message_dim, edge_message_dim, True, weight_init, **kwargs),
            activation,
        )
        self.ji_lin = nn.Sequential(
            Dense(edge_message_dim, edge_message_dim, True, weight_init, **kwargs),
            activation,
        )

        # conbine rbf and sbf information
        self.bilinear = nn.Bilinear(n_bilinear, edge_message_dim, edge_message_dim, bias=False)

        # resnets
        self.res_before_skip = nn.Sequential(
            ResidualBlock(
                edge_message_dim,
                activation=activation,
                weight_init=weight_init,
                **kwargs,
            ),
            Dense(edge_message_dim, edge_message_dim, True, weight_init, **kwargs),
            activation,
        )
        self.res_after_skip = nn.Sequential(
            ResidualBlock(
                edge_message_dim,
                activation=activation,
                weight_init=weight_init,
                **kwargs,
            ),
            ResidualBlock(
                edge_message_dim,
                activation=activation,
                weight_init=weight_init,
                **kwargs,
            ),
        )

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.bilinear.weight, mean=0, std=2.0 / self.bilinear.weight.size(0))

    def forward(
        self,
        x: Tensor,
        rbf: Tensor,
        sbf: Tensor,
        edge_idx_kj: torch.LongTensor,
        edge_idx_ji: torch.LongTensor,
    ) -> Tensor:
        """Forward calculation of the interaction block.

        Args:
            x (Tensor): edge_embeddings of the graph of (num_edge, edge_message_dim) shape.
            rbf (Tensor): radial basis function of (num_edge, n_rad) shape.
            sbf (Tensor): spherical basis function of (num_edge, n_spherical) shape.
            edge_idx_kj (torch.LongTensor): edge index from atom k to j of (n_triplets) shape.
            edge_idx_ji (torch.LongTensor): edge index from atom j to i of (n_triplets) shape.

        Returns:
            Tensor: upadated edge message embedding of (num_edge, edge_message_dim) shape.
        """
        # linear transformation of basis
        rbf = self.rbf_lin(rbf)
        sbf = self.sbf_lin(sbf)
        # linear transformation of input messages
        x_ji = self.ji_lin(x)
        x_kj = self.kj_lin(x)
        # apply rbf and sbf to input messages
        x_kj = x_kj * rbf
        x_kj = self.bilinear(sbf, x_kj[edge_idx_kj])
        # summation k's messages of all j's neighbor
        x_kj = scatter(x_kj, edge_idx_ji, dim=0, dim_size=x.size(0))

        # linear transformation and skip connection to messages
        x = self.res_before_skip(x_ji + x_kj) + x
        return self.res_after_skip(x)


class DimeNetOutBlock(nn.Module):
    """The block to compute the node-wise proptery from edge embeddings. This
    block contains some Dense layers and aggregation block of all neighbors.
    This block is used in Dimenet.

    Args:
        edge_dim (int): number of input edge dimension.
        n_rad (int): number of radial basis function.
        out_dim (int, optional): number of output dimension. Defaults to `1`.
        n_layers (int, optional): number of Dense layers. Defaults to `3`.
        activation (nn.Module, optional): activation function. Defaults to `Swish(beta=1.0)`.
        aggr (str, optional): aggregation method. Defaults to `"add"`.
        weight_init (Callable[[Tensor], Tensor], optional): weight initialization method. Defaults to `glorot_orthogonal`.
    """  # NOQA: E501

    def __init__(
        self,
        edge_dim: int,
        n_rad: int,
        out_dim: int = 1,
        n_layers: int = 3,
        activation: nn.Module = Swish(beta=1.0),
        aggr: str = "add",
        weight_init: Callable[[Tensor], Tensor] | None = glorot_orthogonal,
        **kwargs,
    ):
        super().__init__()

        assert aggr == "add" or aggr == "mean"
        self.aggr = aggr
        # linear layer for radial basis
        self.rbf_lin = Dense(n_rad, edge_dim, False, weight_init, **kwargs)
        # linear layer for edge embedding
        denses: list[nn.Module] = []
        for _ in range(n_layers):
            denses.append(Dense(edge_dim, edge_dim, True, weight_init, **kwargs))
            denses.append(activation)
        denses.append(Dense(edge_dim, out_dim, False, weight_init, **kwargs))
        self.denses = nn.Sequential(*denses)

    def forward(
        self,
        x: Tensor,
        rbf: Tensor,
        idx_i: torch.LongTensor,
        num_nodes: int | None = None,
    ) -> Tensor:
        """Compute node-wise property from edge embeddings.

        Args:
            x (Tensor): edge embedding of (n_edge, edge_dim) shape.
            rbf (Tensor): radial basis function of (n_node, n_radial) shape.
            idx_i (torch.LongTensor): node index center atom i of (n_edge) shape.
            num_nodes (int, optional): number of node. Defaults to `None`.

        Returns:
            Tensor: node-wise properties shape of (n_node, out_dim).
        """
        x = self.rbf_lin(rbf) * x
        # add all neighbor atoms
        x = scatter(x, idx_i, dim=0, dim_size=num_nodes, reduce=self.aggr)
        return self.denses(x)


class DimeNet(BaseGNN):
    """DimeNet implemeted by using PyTorch Geometric. From atomic structure,
    predict graph-wise property such as formation energy.

    Args:
        edge_message_dim (int): edge message embedding dimension.
        n_interaction (int): number of interaction layers.
        out_dim (int): output dimension.
        n_rad (int): number of radial basis function.
        n_sph (int): number of spherical basis function.
        n_bilinear (int): embedding of spherical basis.
        activation (str, optional): name of activation fucntion. Defaults to `"swish"`.
        cutoff (float, optional): cutoff radius. Defaults to `5.0`.
        envelope_exponent (int, optional): exponent of envelope cutoff funcs. Defaults to `5`.
        aggr ("add" or "mean", optional): aggregation mehod. Defaults to `"add"`.
        weight_init (str, optional): name of weight initialization function. Defaults to `"glorot_orthogonal"`.
        max_z (int, optional): max atomic number. Defaults to `100`.

    Notes:
        PyTorch Geometric:
            https://pytorch-geometric.readthedocs.io/en/latest/

        DimeNet:
            [1] J. Klicpera et al., arXiv [cs.LG] (2020), (available at http://arxiv.org/abs/2003.03123).
            [2] https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/dimenet.html
            [3] https://github.com/gasteigerjo/dimenet
    """

    def __init__(
        self,
        edge_message_dim: int,
        n_interaction: int,
        out_dim: int,
        n_rad: int,
        n_sph: int,
        n_bilinear: int,
        activation: str = "swish",
        cutoff: float = 5.0,
        envelope_exponent: int = 5,
        aggr: str = "add",
        weight_init: str = "glorot_orthogonal",
        max_z: int | None = 100,
        **kwargs,
    ):
        super().__init__()
        act: nn.Module = activation_resolver(activation)
        wi: Callable[[Tensor], Tensor] = init_resolver(weight_init)

        self.edge_message_dim = edge_message_dim
        self.n_interaction = n_interaction
        self.out_dim = out_dim
        self.n_rad = n_rad
        self.sph = n_sph
        self.n_bilinear = n_bilinear
        self.cutoff = cutoff
        self.aggr = aggr

        # layers
        self.node_embed = AtomicNum2Node(edge_message_dim, max_z)
        self.edge_embed = EdgeEmbed(
            node_dim=edge_message_dim,
            edge_dim=edge_message_dim,
            n_rad=n_rad,
            activation=act,
            weight_init=wi,
            **kwargs,
        )
        self.rbf = BesselRBF(n_rad, cutoff, envelope_exponent)
        self.sbf = BesselSBF(n_rad, n_sph, cutoff, envelope_exponent)

        self.interactions = nn.ModuleList(
            [
                DimNetInteraction(
                    edge_message_dim=edge_message_dim,
                    n_rad=n_rad,
                    n_sph=n_sph,
                    n_bilinear=n_bilinear,
                    activation=act,
                    weight_init=wi,
                    **kwargs,
                )
                for _ in range(n_interaction)
            ]
        )

        self.outputs = nn.ModuleList(
            [
                DimeNetOutBlock(
                    edge_dim=edge_message_dim,
                    n_rad=n_rad,
                    out_dim=out_dim,
                    activation=act,
                    weight_init=wi,
                    aggr=aggr,
                    **kwargs,
                )
                for _ in range(n_interaction + 1)
            ]
        )

    def forward(self, batch: Batch) -> Tensor:
        batch_idx = batch.get(DataKeys.Batch_idx)
        pos = batch[DataKeys.Position]
        atom_numbers = batch[DataKeys.Atom_numbers]

        # calc atomic distances
        distances = self.calc_atomic_distances(batch)

        # get index and triplets
        (
            idx_i,
            idx_j,
            tri_idx_i,
            tri_idx_j,
            tri_idx_k,
            edge_idx_kj,
            edge_idx_ji,
        ) = self.get_triplets(batch)

        # calc angle each triplets
        # arctan is more stable than arccos
        pos_i = pos[tri_idx_i]
        pos_ji, pos_ki = pos[tri_idx_j] - pos_i, pos[tri_idx_k] - pos_i
        inner = (pos_ji * pos_ki).sum(dim=-1)
        outter = torch.cross(pos_ji, pos_ki).norm(dim=-1)
        angle = torch.atan2(outter, inner)

        # expand by radial and spherical basis
        rbf = self.rbf(distances)
        sbf = self.sbf(distances, angle, edge_idx_kj)

        # embedding and first output
        x = self.node_embed(atom_numbers)
        m = self.edge_embed(x, rbf, idx_i, idx_j)
        out = self.outputs[0](m, rbf, idx_i, num_nodes=atom_numbers.size(0))

        # interaction and outputs
        for ib, ob in zip(self.interactions, self.outputs[1:]):
            m = ib(m, rbf, sbf, edge_idx_kj, edge_idx_ji)
            out += ob(m, rbf, idx_i, num_nodes=atom_numbers.size(0))

        # aggregation each batch
        return out.sum(dim=0, keepdim=True) if batch_idx is None else scatter(out, batch_idx, dim=0, reduce=self.aggr)


# DimeNetPlusPlus
class DimNetPPInteraction(nn.Module):
    """The block to calculate the message interaction using Bessel Radial Basis
    and Bessel Spherical Basis by Hadamard product. This block is used in the
    DimeNetPlusPlus.

    Args:
        edge_message_dim (int): edge message embedding dimension.
        n_rad (int): number of radial basis.
        n_sph (int): number of spherical basis.
        edge_down_dim (int): edge down dimension for convolution calculation.
        basis_embed_dim (int): basis embedding dimension.
        activation (nn.Module, optional): activation function. Defaults to `Swish(beta=1.0)`.
        weight_init (Callable[[Tensor], Tensor], optional): weight initialization function.
            Defaults to `torch_geometric.nn.inits.glorot_orthogonal`.
    """

    def __init__(
        self,
        edge_message_dim: int,
        n_rad: int,
        n_sph: int,
        edge_down_dim: int,
        basis_embed_dim: int,
        activation: nn.Module = Swish(beta=1.0),
        weight_init: Callable[[Tensor], Tensor] | None = glorot_orthogonal,
        **kwargs,
    ):
        super().__init__()
        # Dense transformation of basis
        self.rbf_lin = nn.Sequential(
            Dense(n_rad, basis_embed_dim, bias=False, weight_init=weight_init, **kwargs),
            Dense(basis_embed_dim, edge_message_dim, False, weight_init, **kwargs),
        )
        self.sbf_lin = nn.Sequential(
            Dense(n_sph * n_rad, basis_embed_dim, False, weight_init, **kwargs),
            Dense(basis_embed_dim, edge_down_dim, False, weight_init, **kwargs),
        )

        # Dense transformations of input messages.
        self.kj_lin = nn.Sequential(
            Dense(edge_message_dim, edge_message_dim, True, weight_init, **kwargs),
            activation,
        )
        self.ji_lin = nn.Sequential(
            Dense(edge_message_dim, edge_message_dim, True, weight_init, **kwargs),
            activation,
        )

        # down and up projection of edge message embedding
        self.down_lin = nn.Sequential(
            Dense(edge_message_dim, edge_down_dim, False, weight_init, **kwargs),
            activation,
        )
        self.up_lin = nn.Sequential(
            Dense(edge_down_dim, edge_message_dim, False, weight_init, **kwargs),
            activation,
        )

        # resnets
        self.res_before_skip = nn.Sequential(
            ResidualBlock(
                edge_message_dim,
                activation=activation,
                weight_init=weight_init,
                **kwargs,
            ),
            Dense(edge_message_dim, edge_message_dim, True, weight_init, **kwargs),
            activation,
        )
        self.res_after_skip = nn.Sequential(
            ResidualBlock(
                edge_message_dim,
                activation=activation,
                weight_init=weight_init,
                **kwargs,
            ),
            ResidualBlock(
                edge_message_dim,
                activation=activation,
                weight_init=weight_init,
                **kwargs,
            ),
        )

    def forward(
        self,
        x: Tensor,
        rbf: Tensor,
        sbf: Tensor,
        edge_idx_kj: torch.LongTensor,
        edge_idx_ji: torch.LongTensor,
    ) -> Tensor:
        """Forward calculation of interaction block for Dimenet++.

        Args:
            x (Tensor): edge_embeddings of the graph of (num_edge, hidden_dim) shape.
            rbf (Tensor): radial basis function of (num_edge, n_rad) shape.
            sbf (Tensor): spherical basis function of (num_edge, n_sph) shape.
            edge_idx_kj (torch.LongTensor): edge index from atom k to j of (n_triplets) shape.
            edge_idx_ji (torch.LongTensor): edge index from atom j to i of (n_triplets) shape.

        Returns:
            Tensor: upadated edge message embedding of (num_edge, hidden_dim) shape.
        """
        # linear transformation of input messages
        x_ji = self.ji_lin(x)
        x_kj = self.kj_lin(x)

        # interaction with rbf by hadamard product
        rbf = self.rbf_lin(rbf)
        x_kj = x_kj * rbf

        # down projection
        x_kj = self.down_lin(x_kj)

        # interaction with sbf by hadamard product
        sbf = self.sbf_lin(sbf)
        x_kj = x_kj[edge_idx_kj] * sbf

        # summation k's messages of all j's neighbor and up projection
        x_kj = scatter(x_kj, edge_idx_ji, dim=0, dim_size=x.size(0))
        x_kj = self.up_lin(x_kj)

        # linear transformation and skip connection to messages
        x = self.res_before_skip(x_ji + x_kj) + x
        return self.res_after_skip(x)


class DimeNetPPOutBlock(nn.Module):
    """The block to compute the node-wise proptery from edge embeddings. This
    block contains some Dense layers and aggregation block of all neighbors.
    This block is used in DimenetPlusPlus.

    Args:
        edge_dim (int): the dimension of edge embedding.
        n_rad (int): the number of radial basis functions.
        out_dim (int, optional): the dimension of output. Defaults to `1`.
        out_up_dim (int, optional): the dimension of output up projection. Defaults to `256`.
        n_layers (int, optional): the number of Dense layers. Defaults to `3`.
        activation (nn.Module, optional): the activation function. Defaults to `Swish(beta=1.0)`.
        aggr (str, optional): the aggregation method. Defaults to `"add"`.
        weight_init (Callable[[Tensor], Tensor] or None, optional): the weight initialization method.
            Defaults to `torch_geometric.nn.inits.glorot_orthogonal`.
    """

    def __init__(
        self,
        edge_dim: int,
        n_rad: int,
        out_dim: int = 1,
        out_up_dim: int = 256,
        n_layers: int = 3,
        activation: nn.Module = Swish(beta=1.0),
        aggr: str = "add",
        weight_init: Callable[[Tensor], Tensor] | None = glorot_orthogonal,
        **kwargs,
    ):
        super().__init__()

        assert aggr == "add" or aggr == "mean"
        self.aggr = aggr

        # linear layer for radial basis
        self.rbf_lin = Dense(n_rad, edge_dim, False, weight_init, **kwargs)
        # up projection layer
        self.up_lin = Dense(edge_dim, out_up_dim, False, weight_init, **kwargs)
        # linear layer for edge embedding
        denses: list[nn.Module] = []
        for _ in range(n_layers):
            denses.append(Dense(out_up_dim, out_up_dim, True, weight_init, **kwargs))
            denses.append(activation)
        denses.append(Dense(out_up_dim, out_dim, False, weight_init, **kwargs))
        self.denses = nn.Sequential(*denses)

    def forward(
        self,
        x: Tensor,
        rbf: Tensor,
        idx_i: torch.LongTensor,
        num_nodes: int | None = None,
    ) -> Tensor:
        """Compute node-wise property from edge embeddings.

        Args:
            x (Tensor): edge embedding of (n_edge, edge_dim) shape.
            rbf (Tensor): radial basis function of (n_node, n_radial) shape.
            idx_i (torch.LongTensor): node index center atom i of (n_edge) shape.
            num_nodes (int or None, optional): number of node. Defaults to `None`.

        Returns:
            Tensor: node-wise properties of (n_node, out_dim) shape.
        """
        x = self.rbf_lin(rbf) * x
        # add all neighbor atoms
        x = scatter(x, idx_i, dim=0, dim_size=num_nodes, reduce=self.aggr)
        # up projection
        x = self.up_lin(x)
        return self.denses(x)


class DimeNetPlusPlus(BaseGNN):
    """DimeNet++ implemeted by using PyTorch Geometric. From atomic structure,
    predict graph-wise property such as formation energy.

    Args:
        edge_messag_dim (int): edge message embedding dimension.
        n_interaction (int): number of interaction layers.
        out_dim (int): output dimension.
        n_rad (int): number of radial basis function.
        n_sph (int): number of spherical basis function.
        edge_down_dim (int): edge down projection dimension. Defaults to  `64`.
        basis_embed_dim (int): basis embedding dimension. Defaults to `128`.
        out_up_dim (int): output up projection dimension. Defaults to `256`.
        activation (str, optional): name of activation fucntion. Defaults to `"swish"`.
        cutoff (float, optional): cutoff radius. Defaults to `5.0`.
        envelope_exponent (int, optional): exponent of envelope cutoff funcs. Defaults to `5.0`.
        aggr ("add" or "mean", optional): aggregation mehod. Defaults to `"add"`.
        weight_init (str, optional): name of weight initialization. Defaults to `"glorot_orthogonal"`.
        max_z (int, optional): max atomic number. Defaults to `100`.

    Notes:
        PyTorch Geometric:
            https://pytorch-geometric.readthedocs.io/en/latest/

        DimeNet++:
            [1] J. Gasteiger et al., arXiv [cs.LG] (2020), (available at http://arxiv.org/abs/2011.14115).
            [2] https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/dimenet.html
            [3] https://github.com/gasteigerjo/dimenet
    """

    def __init__(
        self,
        edge_message_dim: int,
        n_interaction: int,
        out_dim: int,
        n_rad: int,
        n_sph: int,
        edge_down_dim: int = 64,
        basis_embed_dim: int = 128,
        out_up_dim: int = 256,
        activation: str = "swish",
        cutoff: float = 5.0,
        envelope_exponent: int = 5,
        aggr: str = "add",
        weight_init: str = "glorot_orthogonal",
        max_z: int | None = 100,
        **kwargs,
    ):
        super().__init__()
        act: nn.Module = activation_resolver(activation)
        wi: Callable[[torch.Tensor], torch.Tensor] = init_resolver(weight_init)

        self.edge_message_dim = edge_message_dim
        self.n_interaction = n_interaction
        self.out_dim = out_dim
        self.n_rad = n_rad
        self.n_sph = n_sph
        self.edge_down_dim = edge_down_dim
        self.basis_embed_dim = basis_embed_dim
        self.out_up_dim = out_up_dim
        self.cutoff = cutoff
        self.aggr = aggr

        # layers
        self.node_embed = AtomicNum2Node(edge_message_dim, max_z)
        self.edge_embed = EdgeEmbed(
            node_dim=edge_message_dim,
            edge_dim=edge_message_dim,
            n_rad=n_rad,
            activation=act,
            weight_init=wi,
            **kwargs,
        )
        self.rbf = BesselRBF(n_rad, cutoff, envelope_exponent)
        self.sbf = BesselSBF(n_rad, n_sph, cutoff, envelope_exponent)

        self.interactions = nn.ModuleList(
            [
                DimNetPPInteraction(
                    edge_message_dim,
                    n_rad,
                    n_sph,
                    edge_down_dim,
                    basis_embed_dim,
                    act,
                    wi,
                    **kwargs,
                )
                for _ in range(n_interaction)
            ]
        )

        self.outputs = nn.ModuleList(
            [
                DimeNetPPOutBlock(
                    edge_dim=edge_message_dim,
                    n_rad=n_rad,
                    out_dim=out_dim,
                    out_up_dim=out_up_dim,
                    activation=act,
                    weight_init=wi,
                    aggr=aggr,
                    **kwargs,
                )
                for _ in range(n_interaction + 1)
            ]
        )

    def forward(self, batch: Batch) -> Tensor:
        batch_idx = batch.get(DataKeys.Batch_idx)
        pos = batch[DataKeys.Position]
        atom_numbers = batch[DataKeys.Atom_numbers]

        # calc atomic distances
        distances = self.calc_atomic_distances(batch)

        # get index and triplets
        (
            idx_i,
            idx_j,
            tri_idx_i,
            tri_idx_j,
            tri_idx_k,
            edge_idx_kj,
            edge_idx_ji,
        ) = self.get_triplets(batch)

        # calc angle each triplets
        # arctan is more stable than arccos
        pos_i = pos[tri_idx_i]
        pos_ji, pos_ki = pos[tri_idx_j] - pos_i, pos[tri_idx_k] - pos_i
        inner = (pos_ji * pos_ki).sum(dim=-1)
        outter = torch.cross(pos_ji, pos_ki).norm(dim=-1)
        angle = torch.atan2(outter, inner)

        # expand by radial and spherical basis
        rbf = self.rbf(distances)
        sbf = self.sbf(distances, angle, edge_idx_kj)

        # embedding and get first output
        x = self.node_embed(atom_numbers)
        m = self.edge_embed(x, rbf, idx_i, idx_j)
        out = self.outputs[0](m, rbf, idx_i, num_nodes=atom_numbers.size(0))

        # interaction and outputs
        for ib, ob in zip(self.interactions, self.outputs[1:]):
            m = ib(m, rbf, sbf, edge_idx_kj, edge_idx_ji)
            out += ob(m, rbf, idx_i, num_nodes=atom_numbers.size(0))

        # aggregation each batch
        return out.sum(dim=0, keepdim=True) if batch_idx is None else scatter(out, batch_idx, dim=0, reduce=self.aggr)
