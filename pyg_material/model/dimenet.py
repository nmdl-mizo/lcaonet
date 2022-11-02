from __future__ import annotations  # type: ignore

from collections.abc import Callable
from math import pi as PI

import sympy as sym
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn.models.dimenet_utils import bessel_basis, real_sph_harm
from torch_scatter import scatter

from pyg_material.data import DataKeys
from pyg_material.model.base import BaseGNN
from pyg_material.nn import AtomicNum2Node, Dense, ResidualBlock, Swish
from pyg_material.utils import activation_resolver, init_resolver


class EnvelopeCutoff(nn.Module):
    """EnvelopeCutoff Network.

    Args:
        exponent (int, optional): Order of the envelope function. Defaults to `5`.

    Notes:
        reference:
        [1] J. Klicpera et al., arXiv [cs.LG] (2020),
            (available at http://arxiv.org/abs/2003.03123).
    """

    def __init__(self, cutoff_radi: float, exponent: int = 5):
        super().__init__()
        self.cutoff_radi = cutoff_radi
        self.p = exponent + 1

    def forward(self, dist: Tensor) -> Tensor:
        """forward calculation of EnvelopeCutoffNetwork.

        Args:
            dist (Tensor): inter atomic distances normalized by cutoff radius
                shape of (n_edge).

        Returns:
            Tensor: Cutoff values shape of (n_edge).
        """
        p = self.p
        # coeffs
        a = -(p + 1) * (p + 2) / 2
        b = p * (p + 2)
        c = -p * (p + 1) / 2
        # calc polynomial
        dist_pow_p0 = dist.pow(p - 1)
        dist_pow_p1 = dist_pow_p0 * dist
        dist_pow_p2 = dist_pow_p1 * dist
        # Remove contributions beyond the cutoff radius
        return (1.0 / (dist + 1e-8) + a * dist_pow_p0 + b * dist_pow_p1 + c * dist_pow_p2) * (dist < 1.0).to(dist.dtype)


class BesselRBF(torch.nn.Module):
    """Bessel radial basis functions. Expand inter atomic distances by Bessel
    radial basis.

    Args:
        n_radial (int): number of radial basis.
        cutoff_radi (float, optional): cutoff radius. Defaults to `5.0`.
        envelope_exponent (int, optional): exponent of cutoff envelope function. Defaults to `5`.
    """

    def __init__(
        self,
        n_radial: int,
        cutoff_radi: float = 5.0,
        envelope_exponent: int = 5,
    ):
        super().__init__()
        self.cutoff_radi = cutoff_radi
        self.cutoff = EnvelopeCutoff(cutoff_radi, envelope_exponent)
        self.freq = torch.nn.Parameter(torch.Tensor(n_radial))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            torch.arange(1, self.freq.numel() + 1, out=self.freq).mul_(PI)
        self.freq.requires_grad_()

    def forward(self, dist: Tensor) -> Tensor:
        """Compute extended distances with Bessel basis.

        Args:
            dist (Tensor): interatomic distance values of (n_edge) shape.

        Returns:
            Tensor: extended distances of (n_edge x n_radial) shape.
        """
        dist = dist / self.cutoff_radi
        return self.cutoff(dist).unsqueeze(-1) * (self.freq * dist.unsqueeze(-1)).sin()


class BesselSBF(nn.Module):
    """Bessel Spherical basis functions. Expand inter atomic distances and
    angles by Bessel spherical and radial basis.

    Args:
        n_radial (int): number of radial basis.
        n_spherical (int): number of spherical basis.
        cutoff_radi (float, optional): cutoff radius. Defaults to `5.0`.
        envelope_exponent (int, optional): exponent of envelope cutoff fucntion. Defaults to `5`.
    """

    def __init__(
        self,
        n_radial: int,
        n_spherical: int,
        cutoff_radi: float = 5.0,
        envelope_exponent: int = 5,
    ):
        super().__init__()

        assert n_radial <= 64, "n_radial must be under 64"
        self.n_radial = n_radial
        self.n_spherical = n_spherical
        self.cutoff_radi = cutoff_radi
        self.cutoff = EnvelopeCutoff(cutoff_radi, envelope_exponent)

        bessel_forms = bessel_basis(n_spherical, n_radial)
        sph_harm_forms = real_sph_harm(n_spherical)
        self.sph_funcs = []
        self.bessel_funcs = []

        x, theta = sym.symbols("x theta")
        modules = {"sin": torch.sin, "cos": torch.cos}
        for i in range(n_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta], sph_harm_forms[i][0], modules)(0)
                self.sph_funcs.append(lambda x: torch.zeros_like(x) + sph1)
            else:
                sph = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(sph)
            for j in range(n_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    def forward(
        self,
        dist: Tensor,
        angle: Tensor,
        edge_idx_kj: torch.LongTensor,
    ) -> Tensor:
        """Extend distances and angles with Bessel spherical and radial basis.

        Args:
            dist (Tensor): interatomic distance values shape of (n_edge).
            angle (Tensor): angles of triplets shape of (n_triplets).
            edge_idx_kj (torch.LongTensor): edge index from atom k to j shape of (n_triplets).

        Returns:
            Tensor: extended distances and angles of (n_triplets x (n_spherical x n_radial)) shape.
        """
        dist = dist / self.cutoff_radi
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        # apply cutoff
        rbf = self.cutoff(dist).unsqueeze(-1) * rbf

        cbf = torch.stack([f(angle) for f in self.sph_funcs], dim=1)

        n, k = self.n_spherical, self.n_radial
        return (rbf[edge_idx_kj].view(-1, n, k) * cbf.view(-1, n, 1)).view(-1, n * k)


class EdgeEmbed(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        n_radial: int,
        activation: Callable[[Tensor], Tensor] = Swish(beta=1.0),
        weight_init: Callable[[Tensor], Tensor] = glorot_orthogonal,
        **kwargs,
    ):
        super().__init__()

        self.rbf_lin = Dense(
            n_radial,
            edge_dim,
            bias=False,
            weight_init=weight_init,
            **kwargs,
        )
        self.edge_embed = nn.Sequential(
            Dense(
                2 * node_dim + edge_dim,
                edge_dim,
                bias=True,
                weight_init=weight_init,
                **kwargs,
            ),
            activation,
        )

    def forward(
        self,
        x: Tensor,
        rbf: Tensor,
        idx_i: torch.LongTensor,
        idx_j: torch.LongTensor,
    ) -> Tensor:
        """Computed the initial edge embedding.

        Args:
            x (Tensor): node embedding vector shape of (n_node x node_dim).
            rbf (Tensor): radial basis function shape of (n_edge x n_radial).
            idx_i (LongTensor): index of the first node of the edge shape of (n_edge).
            idx_j (LongTensor): index of the second node of the edge shape of (n_edge).

        Returns:
            Tensor: embedding edge message shape of (n_edge x edge_dim).
        """
        rbf = self.rbf_lin(rbf)
        return self.edge_embed(torch.cat([x[idx_j], x[idx_i], rbf], dim=-1))


class DimNetInteraction(nn.Module):
    def __init__(
        self,
        edge_message_dim: int,
        n_radial: int,
        n_spherical: int,
        n_bilinear: int,
        activation: nn.Module = Swish(beta=1.0),
        weight_init: Callable[[Tensor], Tensor] = glorot_orthogonal,
        **kwargs,
    ):
        super().__init__()
        # Dense transformation of basis
        self.rbf_dense = Dense(
            n_radial,
            edge_message_dim,
            bias=False,
            weight_init=weight_init,
            **kwargs,
        )
        self.sbf_dense = Dense(
            n_spherical * n_radial,
            n_bilinear,
            bias=False,
            weight_init=weight_init,
            **kwargs,
        )

        # Dense transformations of input messages.
        self.kj_dense = nn.Sequential(
            Dense(
                edge_message_dim,
                edge_message_dim,
                bias=True,
                weight_init=weight_init,
                **kwargs,
            ),
            activation,
        )
        self.ji_dense = nn.Sequential(
            Dense(
                edge_message_dim,
                edge_message_dim,
                bias=True,
                weight_init=weight_init,
                **kwargs,
            ),
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
            Dense(
                edge_message_dim,
                edge_message_dim,
                bias=True,
                weight_init=weight_init,
                **kwargs,
            ),
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
        """The block to calculate the message interaction using Bessel Radial
        Basis and Bessel Spherical Basis. This block is used in the DimeNet.

        Args:
            x (Tensor): edge_embeddings of the graph shape of (num_edge x hidden_dim).
            rbf (Tensor): radial basis function shape of (num_edge x n_radial).
            sbf (Tensor): spherical basis function shape of (num_edge x n_spherical).
            edge_idx_kj (torch.LongTensor): edge index from atom k to j shape of (n_triplets).
            edge_idx_ji (torch.LongTensor): edge index from atom j to i shape of (n_triplets).

        Returns:
            Tensor: upadated edge message embedding shape of (num_edge x hidden_dim).
        """
        # linear transformation of basis
        rbf = self.rbf_dense(rbf)
        sbf = self.sbf_dense(sbf)
        # linear transformation of input messages
        x_ji = self.ji_dense(x)
        x_kj = self.kj_dense(x)
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
        n_radial (int): number of radial basis function.
        out_dim (int, optional): number of output dimension. Defaults to `1`.
        n_layers (int, optional): number of Dense layers. Defaults to `3`.
        activation (Callable[[Tensor], Tensor], optional): activation function. Defaults to `Swish(beta=1.0)`.
        aggr (str, optional): aggregation method. Defaults to `"add"`.
        weight_init (Callable[[Tensor], Tensor], optional): weight initialization method. Defaults to `glorot_orthogonal`.
    """  # NOQA: E501

    def __init__(
        self,
        edge_dim: int,
        n_radial: int,
        out_dim: int = 1,
        n_layers: int = 3,
        activation: Callable[[Tensor], Tensor] = Swish(beta=1.0),
        aggr: str = "add",
        weight_init: Callable[[Tensor], Tensor] = glorot_orthogonal,
        **kwargs,
    ):
        super().__init__()

        assert aggr == "add" or aggr == "mean"
        self.aggr = aggr
        # linear layer for radial basis
        self.rbf_dense = Dense(
            n_radial,
            edge_dim,
            bias=False,
            weight_init=weight_init,
            **kwargs,
        )
        # linear layer for edge embedding
        denses: list[nn.Module] = []
        for _ in range(n_layers):
            denses.append(
                Dense(
                    edge_dim,
                    edge_dim,
                    bias=True,
                    weight_init=weight_init,
                    **kwargs,
                )
            )
            denses.append(activation)
        denses.append(
            Dense(
                edge_dim,
                out_dim,
                bias=False,
                weight_init=weight_init,
                **kwargs,
            )
        )
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
            x (Tensor): edge embedding shape of (n_edge x edge_dim).
            rbf (Tensor): radial basis function shape of (n_node x n_radial).
            idx_i (torch.LongTensor): node index center atom i shape of (n_edge).
            num_nodes (int, optional): number of edge. Defaults to `None`.

        Returns:
            Tensor: node-wise properties shape of (n_node x out_dim).
        """
        x = self.rbf_dense(rbf) * x
        # add all neighbor atoms
        x = scatter(x, idx_i, dim=0, dim_size=num_nodes, reduce=self.aggr)
        return self.denses(x)


class DimeNet(BaseGNN):
    """DimeNet implemeted by using PyTorch Geometric. From atomic structure,
    predict global property such as energy.

    Args:
        edge_messag_dim (int): edge message embedding dimension.
        n_interaction (int): number of interaction layers.
        out_dim (int): output dimension.
        n_radial (int): number of radial basis function.
        n_spherical (int): number of spherical basis function.
        n_bilinear (int): embedding of spherical basis.
        activation (str, optional): activation fucntion. Defaults to `"swish"`.
        cutoff_radi (float, optional): cutoff radius. Defaults to `5.0`.
        envelope_exponent (int, optional): exponent of envelope cutoff funcs. Defaults to `5`.
        aggr ("add" or "mean", optional): aggregation mehod. Defaults to `"add"`.
        weight_init (str, optional): weight initialization. Defaults to `"glorot_orthogonal"`.
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
        n_radial: int,
        n_spherical: int,
        n_bilinear: int,
        activation: str = "swish",
        cutoff_radi: float = 4.0,
        envelope_exponent: int = 5,
        aggr: str = "add",
        weight_init: str = "glorot_orthogonal",
        max_z: int | None = 100,
        **kwargs,
    ):
        super().__init__()
        act = activation_resolver(activation)
        wi: Callable[[Tensor], Tensor] = init_resolver(weight_init)

        self.edge_message_dim = edge_message_dim
        self.n_interaction = n_interaction
        self.out_dim = out_dim
        self.n_radial = n_radial
        self.n_spherical = n_spherical
        self.n_bilinear = n_bilinear
        self.cutoff_radi = cutoff_radi
        self.aggr = aggr

        # layers
        self.node_embed = AtomicNum2Node(edge_message_dim, max_z)
        self.edge_embed = EdgeEmbed(
            node_dim=edge_message_dim,
            edge_dim=edge_message_dim,
            n_radial=n_radial,
            activation=act,
            weight_init=wi,
            **kwargs,
        )
        self.rbf = BesselRBF(n_radial, cutoff_radi, envelope_exponent)
        self.sbf = BesselSBF(n_spherical, n_radial, cutoff_radi, envelope_exponent)

        self.interactions = nn.ModuleList(
            [
                DimNetInteraction(
                    edge_message_dim=edge_message_dim,
                    n_radial=n_radial,
                    n_spherical=n_spherical,
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
                    n_radial=n_radial,
                    out_dim=out_dim,
                    activation=act,
                    weight_init=wi,
                    aggr=aggr,
                    **kwargs,
                )
                for _ in range(n_interaction + 1)
            ]
        )

    def forward(self, data_batch) -> Tensor:
        data_dict = self.get_data(data_batch, batch_index=True, position=True, atom_numbers=True)
        batch = data_dict[DataKeys.Batch]
        pos = data_dict[DataKeys.Position]
        atom_numbers = data_dict[DataKeys.Atom_numbers]
        # calc atomic distances
        distances = self.calc_atomic_distances(data_batch)
        # get triplets
        (
            idx_i,
            idx_j,
            triple_idx_i,
            triple_idx_j,
            triple_idx_k,
            edge_idx_kj,
            edge_idx_ji,
        ) = self.get_triplets(data_batch)
        # calc angle each triplets
        # arctan is more stable than arccos
        pos_i = pos[triple_idx_i]
        pos_ji, pos_ki = pos[triple_idx_j] - pos_i, pos[triple_idx_k] - pos_i
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
        return out.sum(dim=0) if batch is None else scatter(out, batch, dim=0, reduce=self.aggr)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"edge_message_dim={self.edge_message_dim}, "
            f"n_radial={self.n_radial}, "
            f"n_spherical={self.n_spherical}, "
            f"cutoff_radi={self.cutoff_radi}, "
            f"out_dim={self.out_dim}, "
            f"interaction_layers: {self.interactions[0].__class__.__name__} * {self.n_interaction})"
        )


# DimeNetPlusPlus
class DimNetPPInteraction(nn.Module):
    def __init__(
        self,
        edge_message_dim: int,
        n_radial: int,
        n_spherical: int,
        edge_down_dim: int,
        basis_embed_dim: int,
        activation: Callable[[Tensor], Tensor] = Swish(beta=1.0),
        weight_init: Callable[[Tensor], Tensor] = glorot_orthogonal,
        **kwargs,
    ):
        super().__init__()
        # Dense transformation of basis
        self.rbf_denses = nn.Sequential(
            Dense(n_radial, basis_embed_dim, bias=False, weight_init=weight_init, **kwargs),
            Dense(
                basis_embed_dim,
                edge_message_dim,
                bias=False,
                weight_init=weight_init,
                **kwargs,
            ),
        )
        self.sbf_denses = nn.Sequential(
            Dense(
                n_spherical * n_radial,
                basis_embed_dim,
                bias=False,
                weight_init=weight_init,
                **kwargs,
            ),
            Dense(
                basis_embed_dim,
                edge_down_dim,
                bias=False,
                weight_init=weight_init,
                **kwargs,
            ),
        )

        # Dense transformations of input messages.
        self.kj_dense = nn.Sequential(
            Dense(
                edge_message_dim,
                edge_message_dim,
                bias=True,
                weight_init=weight_init,
                **kwargs,
            ),
            activation,
        )
        self.ji_dense = nn.Sequential(
            Dense(
                edge_message_dim,
                edge_message_dim,
                bias=True,
                weight_init=weight_init,
                **kwargs,
            ),
            activation,
        )

        # down and up projection of edge message embedding
        self.down_dense = nn.Sequential(
            Dense(
                edge_message_dim,
                edge_down_dim,
                bias=False,
                weight_init=weight_init,
                **kwargs,
            ),
            activation,
        )
        self.up_dense = nn.Sequential(
            Dense(
                edge_down_dim,
                edge_message_dim,
                bias=False,
                weight_init=weight_init,
                **kwargs,
            ),
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
            Dense(
                edge_message_dim,
                edge_message_dim,
                bias=True,
                weight_init=weight_init,
                **kwargs,
            ),
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
        """The block to calculate the message interaction using Bessel Radial
        Basis and Bessel Spherical Basis by Hadamard product. This block is
        used in the DimeNetPlusPlus.

        Args:
            x (Tensor): edge_embeddings of the graph shape of (num_edge x hidden_dim).
            rbf (Tensor): radial basis function shape of (num_edge x n_radial).
            sbf (Tensor): spherical basis function shape of (num_edge x n_spherical).
            edge_idx_kj (torch.LongTensor): edge index from atom k to j shape of (n_triplets).
            edge_idx_ji (torch.LongTensor): edge index from atom j to i shape of (n_triplets).

        Returns:
            Tensor: upadated edge message embedding shape of (num_edge x hidden_dim).
        """
        # linear transformation of input messages
        x_ji = self.ji_dense(x)
        x_kj = self.kj_dense(x)

        # interaction with rbf by hadamard product
        rbf = self.rbf_denses(rbf)
        x_kj = x_kj * rbf

        # down projection
        x_kj = self.down_dense(x_kj)

        # interaction with sbf by hadamard product
        sbf = self.sbf_denses(sbf)
        x_kj = x_kj[edge_idx_kj] * sbf

        # summation k's messages of all j's neighbor and up projection
        x_kj = scatter(x_kj, edge_idx_ji, dim=0, dim_size=x.size(0))
        x_kj = self.up_dense(x_kj)

        # linear transformation and skip connection to messages
        x = self.res_before_skip(x_ji + x_kj) + x
        return self.res_after_skip(x)


class DimeNetPPOutBlock(nn.Module):
    """The block to compute the node-wise proptery from edge embeddings.

    This block contains some Dense layers and aggregation block of all
    neighbors. This block is used in DimenetPlusPlus.
    """

    def __init__(
        self,
        edge_dim: int,
        n_radial: int,
        out_dim: int = 1,
        out_up_dim: int = 256,
        n_layers: int = 3,
        activation: nn.Module = Swish(beta=1.0),
        aggr: str = "add",
        weight_init: Callable[[Tensor], Tensor] = glorot_orthogonal,
        **kwargs,
    ):
        super().__init__()

        assert aggr == "add" or aggr == "mean"
        self.aggr = aggr
        # linear layer for radial basis
        self.rbf_dense = Dense(
            n_radial,
            edge_dim,
            bias=False,
            weight_init=weight_init,
            **kwargs,
        )
        # up projection layer
        self.up_dense = Dense(
            edge_dim,
            out_up_dim,
            bias=False,
            weight_init=weight_init,
            **kwargs,
        )
        # linear layer for edge embedding
        denses = []
        for _ in range(n_layers):
            denses.append(
                Dense(
                    out_up_dim,
                    out_up_dim,
                    bias=True,
                    weight_init=weight_init,
                    **kwargs,
                )
            )
            denses.append(activation)
        denses.append(
            Dense(
                out_up_dim,
                out_dim,
                bias=False,
                weight_init=weight_init,
                **kwargs,
            )
        )
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
            x (Tensor): edge embedding shape of (n_edge x edge_dim).
            rbf (Tensor): radial basis function shape of (n_node x n_radial).
            idx_i (torch.LongTensor): node index center atom i shape of (n_edge).
            num_nodes (Optional[int], optional): number of edge. Defaults to `None`.

        Returns:
            Tensor: node-wise properties shape of (n_node x out_dim).
        """
        x = self.rbf_dense(rbf) * x
        # add all neighbor atoms
        x = scatter(x, idx_i, dim=0, dim_size=num_nodes, reduce=self.aggr)
        # up projection
        x = self.up_dense(x)
        return self.denses(x)


class DimeNetPlusPlus(BaseGNN):
    """DimeNet implemeted by using PyTorch Geometric. From atomic structure,
    predict global property such as energy.

    Args:
        edge_messag_dim (int): edge message embedding dimension.
        n_interaction (int): number of interaction layers.
        out_dim (int): output dimension.
        n_radial (int): number of radial basis function.
        n_spherical (int): number of spherical basis function.
        edge_down_dim (int): edge down projection dimension. Defaults to  `64`.
        basis_embed_dim (int): basis embedding dimension. Defaults to `128`.
        out_up_dim (int): output up projection dimension. Defaults to `256`.
        activation (str, optional): activation fucntion. Defaults to `"swish"`.
        cutoff_radi (float, optional): cutoff radius. Defaults to `5.0`.
        envelope_exponent (int, optional): exponent of envelope cutoff funcs. Defaults to `5`.
        aggr ("add" or "mean", optional): aggregation mehod. Defaults to `"add"`.
        weight_init (str, optional): weight initialization. Defaults to `"glorot_orthogonal"`.
        max_z (int, optional): max atomic number. Defaults to `100`.

    Notes:
        PyTorch Geometric:
        https://pytorch-geometric.readthedocs.io/en/latest/

        DimeNetPlusPlus:
        [1] J. Gasteiger et al., arXiv [cs.LG] (2020), (available at http://arxiv.org/abs/2011.14115).
        [2] https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/dimenet.html
        [3] https://github.com/gasteigerjo/dimenet
    """

    def __init__(
        self,
        edge_message_dim: int,
        n_interaction: int,
        out_dim: int,
        n_radial: int,
        n_spherical: int,
        edge_down_dim: int = 64,
        basis_embed_dim: int = 128,
        out_up_dim: int = 256,
        activation: str = "swish",
        cutoff_radi: float = 4.0,
        envelope_exponent: int = 5,
        aggr: str = "add",
        weight_init: str = "glorot_orthogonal",
        max_z: int | None = 100,
        **kwargs,
    ):
        super().__init__()
        act = activation_resolver(activation)
        wi: Callable[[torch.Tensor], torch.Tensor] = init_resolver(weight_init)

        self.edge_message_dim = edge_message_dim
        self.n_interaction = n_interaction
        self.out_dim = out_dim
        self.n_radial = n_radial
        self.n_spherical = n_spherical
        self.edge_down_dim = edge_down_dim
        self.basis_embed_dim = basis_embed_dim
        self.out_up_dim = out_up_dim
        self.cutoff_radi = cutoff_radi
        self.aggr = aggr

        # layers
        self.node_embed = AtomicNum2Node(edge_message_dim, max_z)
        self.edge_embed = EdgeEmbed(
            node_dim=edge_message_dim,
            edge_dim=edge_message_dim,
            n_radial=n_radial,
            activation=act,
            weight_init=wi,
            **kwargs,
        )
        self.rbf = BesselRBF(n_radial, cutoff_radi, envelope_exponent)
        self.sbf = BesselSBF(n_spherical, n_radial, cutoff_radi, envelope_exponent)

        self.interactions = nn.ModuleList(
            [
                DimNetPPInteraction(
                    edge_message_dim,
                    n_radial,
                    n_spherical,
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
                    n_radial=n_radial,
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

    def forward(self, data_batch) -> Tensor:
        data_dict = self.get_data(data_batch, batch_index=True, position=True, atom_numbers=True)
        batch = data_dict[DataKeys.Batch]
        pos = data_dict[DataKeys.Position]
        atom_numbers = data_dict[DataKeys.Atom_numbers]
        # calc atomic distances
        distances = self.calc_atomic_distances(data_batch)
        # get triplets
        (
            idx_i,
            idx_j,
            triple_idx_i,
            triple_idx_j,
            triple_idx_k,
            edge_idx_kj,
            edge_idx_ji,
        ) = self.get_triplets(data_batch)
        # calc angle each triplets
        # arctan is more stable than arccos
        pos_i = pos[triple_idx_i]
        pos_ji, pos_ki = pos[triple_idx_j] - pos_i, pos[triple_idx_k] - pos_i
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
        return out.sum(dim=0) if batch is None else scatter(out, batch, dim=0, reduce=self.aggr)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"edge_message_dim={self.edge_message_dim}, "
            f"n_radial={self.n_radial}, "
            f"n_spherical={self.n_spherical}, "
            f"cutoff_radi={self.cutoff_radi}, "
            f"out_dim={self.out_dim}, "
            f"interaction_layers: {self.interactions[0].__class__.__name__} * {self.n_interaction})"
        )
