from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor
import torch.nn as nn
from torch_scatter import scatter
from torch_geometric.nn.inits import glorot_orthogonal

from pyggnn.model.base import BaseGNN
from pyggnn.nn.activation import Swish
from pyggnn.nn.rbf import BesselRBF
from pyggnn.nn.abf import BesselSBF
from pyggnn.nn.node_embed import AtomicNum2Node
from pyggnn.nn.edge_embed import EdgeEmbed
from pyggnn.nn.base import Dense, ResidualBlock
from pyggnn.nn.edge_out import Edge2NodeProp1
from pyggnn.utils.resolve import activation_resolver, init_resolver


__all__ = ["DimeNet"]


class DimNetInteraction(nn.Module):
    def __init__(
        self,
        edge_message_dim: int,
        n_radial: int,
        n_spherical: int,
        n_bilinear: int,
        activation: Callable[[Tensor], Tensor] = Swish(beta=1.0),
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
        """
        The block to calculate the message interaction using Bessel Radial Basis and Bessel Spherical Basis.
        This block is used in the DimeNet.

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


class DimeNet(BaseGNN):
    """
    DimeNet implemeted by using PyTorch Geometric.
    From atomic structure, predict global property such as energy.

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
            share_weight (bool, optional): share weight parameter all interaction layers. Defaults to `False`.
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
        share_weight: bool = False,
        max_z: int | None = 100,
        **kwargs,
    ):
        super().__init__()
        act = activation_resolver(activation)
        weight_init = init_resolver(weight_init)

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
            weight_init=weight_init,
            **kwargs,
        )
        self.rbf = BesselRBF(n_radial, cutoff_radi, envelope_exponent)
        self.sbf = BesselSBF(n_spherical, n_radial, cutoff_radi, envelope_exponent)

        if share_weight:
            self.interactions = nn.ModuleList(
                [
                    DimNetInteraction(
                        edge_message_dim=edge_message_dim,
                        n_radial=n_radial,
                        n_spherical=n_spherical,
                        n_bilinear=n_bilinear,
                        activation=act,
                        weight_init=weight_init,
                        **kwargs,
                    )
                ]
                * n_interaction
            )
        else:
            self.interactions = nn.ModuleList(
                [
                    DimNetInteraction(
                        edge_message_dim=edge_message_dim,
                        n_radial=n_radial,
                        n_spherical=n_spherical,
                        n_bilinear=n_bilinear,
                        activation=act,
                        weight_init=weight_init,
                        **kwargs,
                    )
                    for _ in range(n_interaction)
                ]
            )

        self.outputs = nn.ModuleList(
            [
                Edge2NodeProp1(
                    edge_dim=edge_message_dim,
                    n_radial=n_radial,
                    out_dim=out_dim,
                    activation=act,
                    weight_init=weight_init,
                    aggr=aggr,
                    **kwargs,
                )
                for _ in range(n_interaction + 1)
            ]
        )

    def forward(self, data_batch) -> Tensor:
        batch, pos, atom_numbers = self.get_data(data_batch, batch_index=True, position=True, atom_numbers=True)
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

        # embedding and firset output
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
            f"cutoff={self.cutoff_radi}, "
            f"out_dim={self.out_dim}, "
            f"interaction_layers: {self.interactions[0].__class__.__name__} * {self.n_interaction})"
        )
