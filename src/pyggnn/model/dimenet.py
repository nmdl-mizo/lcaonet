from typing import Union, Any, Optional, Literal

import torch
from torch import Tensor
import torch.nn as nn
from torch_scatter import scatter

from pyggnn.model.base import BaseGNN
from pyggnn.nn.basis import BesselRB, BesselSB
from pyggnn.nn.embedding import EdgeEmbed
from pyggnn.nn.base import Dense
from pyggnn.nn.residual import ResidualBlock
from pyggnn.nn.out import Edge2NodeProperty
from pyggnn.data.datakeys import DataKeys
from pyggnn.utils.resolve import activation_resolver


__all__ = ["DimeNet"]


class InteractionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_radial: int,
        n_spherical: int,
        n_bilinear: int,
        activation: Union[Any, str] = "swish",
        **kwargs,
    ):
        super().__init__()
        act = activation_resolver(activation, **kwargs)

        self.rbf_lin = Dense(n_radial, hidden_dim, bias=False)
        self.sbf_lin = Dense(n_spherical * n_radial, n_bilinear, bias=False)

        # Dense transformations of input messages.
        self.lin_kj = nn.Sequential(
            Dense(
                hidden_dim,
                hidden_dim,
                bias=True,
                activation_name=activation,
                **kwargs,
            ),
            act,
        )
        self.lin_ji = nn.Sequential(
            Dense(
                hidden_dim,
                hidden_dim,
                bias=True,
                activation_name=activation,
                **kwargs,
            ),
            act,
        )

        # conbine rbf and sbf information
        self.bilinear = nn.Bilinear(n_bilinear, hidden_dim, hidden_dim, bias=False)

        # resnets
        self.res_before_skip = nn.Sequential(
            ResidualBlock(hidden_dim, activation=activation, **kwargs),
            Dense(
                hidden_dim,
                hidden_dim,
                bias=True,
                activation_name=activation,
                **kwargs,
            ),
            act,
        )
        self.res_after_skip = nn.Sequential(
            ResidualBlock(hidden_dim, activation=activation, **kwargs),
            ResidualBlock(hidden_dim, activation=activation, **kwargs),
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.rbf_lin.reset_parameters()
        self.sbf_lin.reset_parameters()
        for layer in self.lin_kj:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        for layer in self.lin_ji:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        self.bilinear.reset_parameters()
        for layer in self.res_before_skip:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        for layer in self.res_after_skip:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(
        self,
        x: Tensor,
        rbf: Tensor,
        sbf: Tensor,
        edge_idx_kj: torch.LongTensor,
        edge_idx_ji: torch.LongTensor,
    ) -> Tensor:
        """
        The block to calculate the message interaction using Bessel Radial Basis
        and Bessel Spherical Basis.
        This block is used in the DimeNet.

        Args:
            x (Tensor): edge_embeddings of the graph shape of (num_edge x hidden_dim).
            rbf (Tensor): radial basis function shape of (num_edge x n_radial).
            sbf (Tensor): spherical basis function shape of (num_edge x n_spherical).
            edge_idx_kj (torch.LongTensor): edge index from atom k to j
                shape of (n_triplets).
            edge_idx_ji (torch.LongTensor): edge index from atom j to i
                shape of (n_triplets).

        Returns:
            Tensor: upadated edge message embedding shape of (num_edge x hidden_dim).
        """
        # linear transformation of basis
        rbf = self.rbf_lin(rbf)
        sbf = self.sbf_lin(sbf)
        # linear transformation of input messages
        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x))
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

    Notes:
        PyTorch Geometric:
        https://pytorch-geometric.readthedocs.io/en/latest/

        DimeNet:
        [1] J. Klicpera et al., arXiv [cs.LG] (2020),
            (available at http://arxiv.org/abs/2003.03123).
        [2] https://github.com/pyg-team/pytorch_geometric
        [3] https://github.com/gasteigerjo/dimenet
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        n_interaction: int,
        out_dim: int,
        n_radial: int,
        n_spherical: int,
        n_bilinear: int,
        activation: Union[Any, str] = "swish",
        cutoff_radi: float = 4.0,
        envelope_exponent: int = 6,
        aggr: Literal["add", "mean"] = "add",
        share_weight: bool = False,
        max_z: Optional[int] = 100,
        **kwargs,
    ):
        """
        Args:
            node_dim (int): node embedding dimension.
            edge_dim (int): edge message embedding dimension.
            n_interaction (int): number of interaction layers.
            out_dim (int): output dimension.
            n_radial (int): number of radial basis function.
            n_spherical (int): number of spherical basis function.
            n_bilinear (int): embedding of spherical basis.
            activation (str or nn.Module, optional): activation fucntion.
                Defaults to `"swish"`.
            cutoff_radi (float, optional): cutoff radius. Defaults to `4.0`.
            envelope_exponent (int, optional): exponent of envelope cutoff funcs.
                Defaults to `6`.
            aggr ("add" or "mean", optional): aggregation mehod.
                Defaults to `"add"`.
            share_weight (bool, optional): share weight parameter all interaction layers.
                Defaults to `False`.
            max_z (int, optional): max atomic number. Defaults to `100`.
        """
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.n_interaction = n_interaction
        self.out_dim = out_dim
        self.n_radial = n_radial
        self.n_spherical = n_spherical
        self.n_bilinear = n_bilinear
        self.cutoff_radi = cutoff_radi
        self.aggr = aggr
        # layers
        self.rbf = BesselRB(n_radial, cutoff_radi, envelope_exponent)
        self.sbf = BesselSB(n_spherical, n_radial, cutoff_radi, envelope_exponent)
        self.embed_block = EdgeEmbed(
            node_dim,
            edge_dim,
            n_radial,
            activation,
            max_z,
            **kwargs,
        )
        if share_weight:
            self.interaction_blocks = nn.ModuleList(
                [
                    InteractionBlock(
                        hidden_dim=edge_dim,
                        n_radial=n_radial,
                        n_spherical=n_spherical,
                        n_bilinear=n_bilinear,
                        activation=activation,
                        **kwargs,
                    )
                    * n_interaction
                ]
            )
        else:
            self.interaction_blocks = nn.ModuleList(
                [
                    InteractionBlock(
                        hidden_dim=edge_dim,
                        n_radial=n_radial,
                        n_spherical=n_spherical,
                        n_bilinear=n_bilinear,
                        activation=activation,
                        **kwargs,
                    )
                    for _ in range(n_interaction)
                ]
            )
        self.output_blocks = nn.ModuleList(
            [
                Edge2NodeProperty(
                    hidden_dim=edge_dim,
                    n_radial=n_radial,
                    out_dim=out_dim,
                    activation=activation,
                    aggr=aggr,
                    **kwargs,
                )
                for _ in range(n_interaction + 1)
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.rbf.reset_parameters()
        self.sbf.reset_parameters()
        self.embed_block.reset_parameters()
        for ib in self.interaction_blocks:
            ib.reset_parameters()
        for ob in self.output_blocks:
            ob.reset_parameters()

    def forward(self, data_batch) -> Tensor:
        batch = data_batch[DataKeys.Batch]
        atomic_numbers = data_batch[DataKeys.Atomic_num]
        pos = data_batch[DataKeys.Pos]
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
        pos_j = pos[triple_idx_j]
        pos_ji, pos_kj = pos_j - pos[triple_idx_i], pos[triple_idx_k] - pos_j
        a = (pos_ji * pos_kj).sum(dim=-1)
        b = torch.cross(pos_ji, pos_kj).norm(dim=-1)
        angle = torch.atan2(b, a)
        # expand by radial and sperical basis
        rbf = self.rbf(distances)
        sbf = self.sbf(distances, angle, edge_idx_kj)

        # embedding
        x = self.embed_block(atomic_numbers, rbf, idx_i, idx_j)
        out = self.output_blocks[0](x, rbf, idx_i, num_nodes=atomic_numbers.size(0))

        # interaction
        for ib, ob in zip(self.interaction_blocks, self.output_blocks[1:]):
            x = ib(x, rbf, sbf, edge_idx_kj, edge_idx_ji)
            out += ob(x, rbf, idx_i, num_nodes=atomic_numbers.size(0))

        # output
        return (
            out.sum(dim=0)
            if batch is None
            else scatter(out, batch, dim=0, reduce=self.aggr)
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"node_dim={self.node_dim}, "
            f"edge_dim={self.edge_dim}, "
            f"n_interaction={self.n_interaction}, "
            f"cutoff={self.cutoff_radi}, "
            f"out_dim={self.out_dim})"
        )
