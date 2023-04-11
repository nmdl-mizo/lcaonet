from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch
from torch_scatter import scatter

from lcaonet.atomistic.info import ElecInfo
from lcaonet.data.datakeys import DataKeys
from lcaonet.model.base import BaseMPNN
from lcaonet.nn import Dense
from lcaonet.nn.cutoff import BaseCutoff
from lcaonet.nn.post import PostProcess
from lcaonet.nn.rbf import BaseRadialBasis
from lcaonet.nn.shbf import SphericalHarmonicsBasis
from lcaonet.utils.resolve import (
    activation_resolver,
    cutoffnet_resolver,
    init_resolver,
    rbf_resolver,
)


class EmbedZ(nn.Module):
    """The layer that embeds atomic numbers into latent vectors."""

    def __init__(self, embed_dim: int, max_z: int = 36):
        """
        Args:
            embed_dim (int): the dimension of embedding.
            max_z (int, optional): the maximum atomic number. Defaults to `36`.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.z_embed = nn.Embedding(max_z + 1, embed_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.z_embed.weight.data.uniform_(-math.sqrt(3), math.sqrt(3))

    def forward(self, z: Tensor) -> Tensor:
        """Forward calculation of EmbedZ.

        Args:
            z (torch.Tensor): the atomic numbers with (n_node) shape.

        Returns:
            z_embed (torch.Tensor): the embedding vectors with (n_node, embed_dim) shape.
        """
        return self.z_embed(z)


class EmbedElec(nn.Module):
    """The layer that embeds electron numbers into latent vectors.

    If `extend_orb=False`, then if the number of electrons in the ground
    state is zero, the orbital is a zero vector embedding.
    """

    def __init__(self, embed_dim: int, elec_info: ElecInfo, extend_orb: bool = False):
        """
        Args:
            embed_dim (int): the dimension of embedding.
            elec_info (lcaonet.atomistic.info.ElecInfo): the object that contains the information about the number of electrons.
            extend_orb (bool, optional): Whether to use an extended basis. Defaults to `False`.
        """  # NOQA: E501
        super().__init__()
        self.register_buffer("elec", elec_info.elec_table)
        self.n_orb = ElecInfo.n_orb
        self.embed_dim = embed_dim
        self.extend_orb = extend_orb

        self.e_embeds = nn.ModuleList(
            [nn.Embedding(m, embed_dim, padding_idx=None if extend_orb else 0) for m in elec_info.max_elec_idx]
        )

        self.reset_parameters()

    def reset_parameters(self):
        for ee in self.e_embeds:
            ee.weight.data.uniform_(-math.sqrt(3), math.sqrt(3))
            # set padding_idx to zero
            if not self.extend_orb:
                ee._fill_padding_idx_with_zero()

    def forward(self, z: Tensor) -> Tensor:
        """Forward calculation of EmbedElec.

        Args:
            z (torch.Tensor): the atomic numbers with (n_node) shape.

        Returns:
            e_embed (torch.Tensor): the embedding of electron numbers with (n_node, n_orb, embed_dim) shape.
        """
        # (n_node, n_orb)
        elec = self.elec[z]  # type: ignore
        # (n_orb, n_node)
        elec = torch.transpose(elec, 0, 1)
        # (n_orb, n_node, embed_dim)
        e_embed = torch.stack([ce(elec[i]) for i, ce in enumerate(self.e_embeds)], dim=0)
        # (n_node, n_orb, embed_dim)
        e_embed = torch.transpose(e_embed, 0, 1)

        return e_embed


class ValenceMask(nn.Module):
    """The layer that generates valence orbital mask.

    Only the coefficients for valence orbitals are set to 1, and the
    coefficients for all other orbitals (including inner-shell orbitals)
    are set to 0.
    """

    def __init__(self, embed_dim: int, elec_info: ElecInfo):
        """
        Args:
            embed_dim (int): the dimension of embedding.
            elec_info (lcaonet.atomistic.info.ElecInfo): the object that contains the information about the number of electrons.
        """  # NOQA: E501
        super().__init__()
        self.register_buffer("valence", elec_info.valence_table)
        self.n_orb = ElecInfo.n_orb

        self.embed_dim = embed_dim

    def forward(self, z: Tensor, idx_j: Tensor) -> Tensor:
        """Forward calculation of ValenceMask.

        Args:
            z (torch.Tensor): the atomic numbers with (n_node) shape.
            idx_j (torch.Tensor): the indices of the second node of each edge with (n_edge) shape.

        Returns:
            valence_mask (torch.Tensor): valence orbital mask with (n_edge, n_orb, embed_dim) shape.
        """
        valence_mask = self.valence[z]  # type: ignore
        return valence_mask.unsqueeze(-1).expand(-1, -1, self.embed_dim)[idx_j]


class EmbedNode(nn.Module):
    """The layer that embedds atomic numbers and electron numbers into node
    embedding vectors."""

    def __init__(
        self,
        hidden_dim: int,
        z_dim: int,
        use_elec: bool,
        e_dim: int | None = None,
        activation: nn.Module = nn.SiLU(),
        weight_init: Callable[[Tensor], Tensor] | None = None,
    ):
        """
        Args:
            hidden_dim (int): the dimension of node vector.
            z_dim (int): the dimension of atomic number embedding.
            use_elec (bool): whether to use electron number embedding.
            e_dim (int | None): the dimension of electron number embedding.
            activation (nn.Module, optional): the activation function. Defaults to `torch.nn.SiLU()`.
            weight_init (Callable[[torch.Tensor], torch.Tensor] | None, optional): the weight initialization function.
                Defaults to `None`.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.use_elec = use_elec
        if use_elec:
            assert e_dim is not None
            self.e_dim = e_dim
        else:
            self.e_dim = 0

        self.z_e_lin = nn.Sequential(
            activation,
            Dense(z_dim + self.e_dim, hidden_dim, True, weight_init),
            activation,
            Dense(hidden_dim, hidden_dim, False, weight_init),
        )

    def forward(self, z_embed: Tensor, e_embed: Tensor | None = None) -> Tensor:
        """Forward calculation of EmbedNode.

        Args:
            z_embed (torch.Tensor): the embedding of atomic numbers with (n_node, z_dim) shape.
            e_embed (torch.Tensor | None): the embedding of electron numbers with (n_node, n_orb, e_dim) shape.

        Returns:
            torch.Tensor: node embedding vectors with (n_node, hidden_dim) shape.
        """
        if self.use_elec:
            if e_embed is None:
                raise ValueError("e_embed must be set when use_elec is True.")
            z_e_embed = torch.cat([z_embed, e_embed.sum(1)], dim=-1)
        else:
            z_e_embed = z_embed
        return self.z_e_lin(z_e_embed)


class EmbedCoeffs(nn.Module):
    """The layer that embedds atomic numbers and electron numbers into
    coefficient embedding vectors."""

    def __init__(
        self,
        hidden_dim: int,
        z_dim: int,
        e_dim: int,
        activation: nn.Module = nn.SiLU(),
        weight_init: Callable[[Tensor], Tensor] | None = None,
    ):
        """
        Args:
            hidden_dim (int): the dimension of coefficient vector.
            z_dim (int): the dimension of atomic number embedding.
            e_dim (int): the dimension of electron number embedding.
            activation (nn.Module): the activation function. Defaults to `torch.nn.SiLU()`.
            weight_init (Callable[[Tensor], Tensor] | None): weight initialization func. Defaults to `None`.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.e_dim = e_dim

        self.z_lin = nn.Sequential(
            activation,
            Dense(2 * z_dim, hidden_dim, True, weight_init),
            activation,
            Dense(hidden_dim, hidden_dim, True, weight_init),
        )
        self.e_lin = nn.Sequential(
            activation,
            Dense(e_dim, hidden_dim, False, weight_init),
            activation,
            Dense(hidden_dim, hidden_dim, False, weight_init),
        )

    def forward(self, z_embed: Tensor, e_embed: Tensor, idx_i: Tensor, idx_j: Tensor) -> Tensor:
        """Forward calculation of EmbedCoeffs.

        Args:
            z_embed (torch.Tensor): the embedding of atomic numbers with (n_node, z_dim) shape.
            e_embed (torch.Tensor): the embedding of electron numbers with (n_node, n_orb, e_dim) shape.
            idx_i (torch.Tensor): the indices of center atoms with (n_edge) shape.
            idx_j (torch.Tensor): the indices of neighbor atoms with (n_edge) shape.

        Returns:
            coeff_embed (torch.Tensor): coefficient embedding vectors with (n_edge, n_orb, hidden_dim) shape.
        """
        z_embed = self.z_lin(torch.cat([z_embed[idx_i], z_embed[idx_j]], dim=-1))
        e_embed = self.e_lin(e_embed)[idx_j]
        return e_embed + e_embed * z_embed.unsqueeze(1)


class LCAOInteraction(nn.Module):
    """The layer that performs message-passing of LCAONet."""

    def __init__(
        self,
        hidden_dim: int,
        coeffs_dim: int,
        conv_dim: int,
        add_valence: bool = False,
        activation: nn.Module = nn.SiLU(),
        weight_init: Callable[[Tensor], Tensor] | None = None,
    ):
        """
        Args:
            hidden_dim (int): the dimension of node vector.
            coeffs_dim (int): the dimension of coefficient vectors.
            conv_dim (int): the dimension of embedding vectors at convolution.
            add_valence (bool, optional): whether to add the effect of valence orbitals. Defaults to `False`.
            activation (nn.Module, optional): the activation function. Defaults to `torch.nn.SiLU()`.
            weight_init (Callable[[torch.Tensor], torch.Tensor] | None, optional): the weight initialization function.
                Defaults to `None`.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.coeffs_dim = coeffs_dim
        self.conv_dim = conv_dim
        self.add_valence = add_valence

        self.node_before_lin = Dense(hidden_dim, 2 * conv_dim, True, weight_init)

        # No bias is used to keep 0 coefficient vectors at 0
        out_dim = 3 * conv_dim if add_valence else 2 * conv_dim
        self.coeffs_before_lin = nn.Sequential(
            activation,
            Dense(coeffs_dim, conv_dim, False, weight_init),
            activation,
            Dense(conv_dim, out_dim, False, weight_init),
        )

        three_out_dim = 2 * conv_dim if add_valence else conv_dim
        self.three_lin = nn.Sequential(
            activation,
            Dense(conv_dim, three_out_dim, True, weight_init),
        )

        self.node_lin = nn.Sequential(
            activation,
            Dense(conv_dim + conv_dim, conv_dim, True, weight_init),
            activation,
            Dense(conv_dim, conv_dim, True, weight_init),
        )
        self.node_after_lin = Dense(conv_dim, hidden_dim, False, weight_init)

    def forward(
        self,
        x: Tensor,
        cji: Tensor,
        valence_mask: Tensor | None,
        cutoff_w: Tensor | None,
        rb: Tensor,
        shb: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        tri_idx_k: Tensor,
        edge_idx_kj: torch.LongTensor,
        edge_idx_ji: torch.LongTensor,
    ) -> Tensor:
        """Forward calculation of LCAOConv.

        Args:
            x (torch.Tensor): node embedding vectors with (n_node, hidden_dim) shape.
            cji (torch.Tensor): coefficient vectors with (n_edge, n_orb, coeffs_dim) shape.
            valence_mask (torch.Tensor | None): valence orbital mask with (n_node, n_orb, conv_dim) shape.
            cutoff_w (torch.Tensor | None): cutoff weight with (n_edge) shape.
            rb (torch.Tensor): the radial basis with (n_edge, n_orb) shape.
            shb (torch.Tensor): the spherical harmonics basis with (n_triplets, n_orb) shape.
            idx_i (torch.Tensor): the indices of the first node of each edge with (n_edge) shape.
            idx_j (torch.Tensor): the indices of the second node of each edge with (n_edge) shape.
            tri_idx_k (torch.Tensor): the indices of the third node of each triplet with (n_triplets) shape.
            edge_idx_kj (torch.LongTensor): the edge index from atom k to j with (n_triplets) shape.
            edge_idx_ji (torch.LongTensor): the edge index from atom j to i with (n_triplets) shape.

        Returns:
            torch.Tensor: updated node embedding vectors with (n_node, hidden_dim) shape.
        """
        # Transformation of the node
        x_before = x
        x = self.node_before_lin(x)
        x, xk = torch.chunk(x, 2, dim=-1)

        # Transformation of the coefficient vectors
        cji = self.coeffs_before_lin(cji)
        if self.add_valence:
            cji, ckj = torch.split(cji, [2 * self.conv_dim, self.conv_dim], dim=-1)
        else:
            cji, ckj = torch.chunk(cji, 2, dim=-1)

        # cutoff
        if cutoff_w is not None:
            rb = rb * cutoff_w.unsqueeze(-1)

        # triple conv
        ckj = ckj[edge_idx_kj]
        ckj = F.normalize(ckj, dim=-1)
        # LCAO weight: summation of all orbitals multiplied by coefficient vectors
        three_body_orbs = torch.einsum("ed,edh->eh", rb[edge_idx_kj] * shb, ckj)
        three_body_orbs = F.normalize(three_body_orbs, dim=-1)
        # multiply node embedding
        xk = torch.sigmoid(xk[tri_idx_k])
        three_body_w = three_body_orbs * xk
        three_body_w = self.three_lin(scatter(three_body_w, edge_idx_ji, dim=0, dim_size=rb.size(0)))
        # threebody orbital information is injected to the coefficient vectors
        cji = cji + cji * three_body_w.unsqueeze(1)
        cji = F.normalize(cji, dim=-1)
        if self.add_valence:
            cji, cji_valence = torch.chunk(cji, 2, dim=-1)

        # LCAO weight: summation of all orbitals multiplied by coefficient vectors
        lcao_w = torch.einsum("ed,edh->eh", rb, cji)

        if self.add_valence:
            # valence contribution
            if valence_mask is None:
                raise ValueError("valence_mask must be provided when add_valence=True")
            valence_w = torch.einsum("ed,edh->eh", rb, cji_valence * valence_mask)
            lcao_w = lcao_w + valence_w

        lcao_w = F.normalize(lcao_w, dim=-1)

        x = x_before + self.node_after_lin(
            scatter(lcao_w * self.node_lin(torch.cat([x[idx_i], x[idx_j]], dim=-1)), idx_i, dim=0)
        )

        return x


class LCAOOut(nn.Module):
    """The output layer of LCAONet.

    Three-layer neural networks to output desired physical property
    values.
    """

    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        is_extensive: bool = True,
        activation: nn.Module = nn.SiLU(),
        weight_init: Callable[[Tensor], Tensor] | None = None,
    ):
        """
        Args:
            hidden_dim (int): the dimension of node embedding vectors.
            out_dim (int): the dimension of output property.
            is_extensive (bool): whether the output property is extensive or not. Defaults to `True`.
            activation (nn.Module): the activation function. Defaults to `nn.SiLU()`.
            weight_init (Callable[[torch.Tensor], torch.Tensor] | None): the weight initialization function.
                Defaults to `None`.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.is_extensive = is_extensive

        self.out_lin = nn.Sequential(
            activation,
            Dense(hidden_dim, hidden_dim, True, weight_init),
            activation,
            Dense(hidden_dim, hidden_dim // 2, True, weight_init),
            activation,
            Dense(hidden_dim // 2, out_dim, False, weight_init),
        )

    def forward(self, x: Tensor, batch_idx: Tensor | None) -> Tensor:
        """Forward calculation of LCAOOut.

        Args:
            x (torch.Tensor): node embedding vectors with (n_node, hidden_dim) shape.
            batch_idx (torch.Tensor | None): the batch indices of nodes with (n_node) shape.

        Returns:
            torch.Tensor: the output property values with (n_batch, out_dim) shape.
        """
        out = self.out_lin(x)
        if batch_idx is not None:
            return scatter(out, batch_idx, dim=0, reduce="sum" if self.is_extensive else "mean")
        if self.is_extensive:
            return out.sum(dim=0, keepdim=True)
        else:
            return out.mean(dim=0, keepdim=True)


class LCAONet(BaseMPNN):
    """
    LCAONet - MPNN including orbital interaction, physically motivatied by the LCAO method.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        coeffs_dim: int = 128,
        conv_dim: int = 128,
        out_dim: int = 1,
        n_interaction: int = 3,
        n_per_orb: int = 1,
        cutoff: float | None = None,
        rbf_type: str | type[BaseRadialBasis] = "hydrogen",
        cutoff_net: str | type[BaseCutoff] | None = None,
        max_z: int = 36,
        max_orb: str | None = None,
        elec_to_node: bool = True,
        add_valence: bool = False,
        extend_orb: bool = False,
        is_extensive: bool = True,
        activation: str = "SiLU",
        weight_init: str | None = "glorotorthogonal",
        atomref: Tensor | None = None,
        mean: Tensor | None = None,
    ):
        """
        Args:
            hidden_dim (int): the dimension of node embedding vectors. Defaults to `128`.
            coeffs_dim (int): the dimension of coefficient vectors. Defaults to `64`.
            conv_dim (int): the dimension of embedding vectors at convolution. Defaults to `64`.
            out_dim (int): the dimension of output property. Defaults to `1`.
            n_interaction (int): the number of interaction layers. Defaults to `3`.
            cutoff (float | None): the cutoff radius. Defaults to `None`.
                If specified, the basis functions are normalized within the cutoff radius.
                If `cutoff_net` is specified, the `cutoff` radius must be specified.
            rbf_type (str | type[lcaonet.nn.rbf.BaseRadialBasis]): the radial basis function or the name. Defaults to `hydrogen`.
            cutoff_net (str | type[lcaonet.nn.cutoff.BaseCutoff] | None): the cutoff network or the name Defaults to `None`.
            max_z (int): the maximum atomic number. Defaults to `36`.
            max_orb (str | None): the maximum orbital name like "2p". Defaults to `None`.
            elec_to_node (bool): whether to use electrons information to nodes embedding. Defaults to `True`.
            add_valence (bool): whether to add the effect of valence orbitals. Defaults to `False`.
            extend_orb (bool): whether to extend the basis set. Defaults to `False`. If `True`, MP is performed including unoccupied orbitals of the ground state.
            is_extensive (bool): whether to predict extensive property. Defaults to `True`.
            activation (str): the name of activation function. Defaults to `"SiLU"`.
            weight_init (str | None): the name of weight initialization function. Defaults to `"glorotorthogonal"`.
            atomref (torch.Tensor | None): the reference value of the output property with (max_z, out_dim) shape. Defaults to `None`.
            mean (torch.Tensor | None): the mean value of the output property with (out_dim) shape. Defaults to `None`.
        """  # NOQA: E501
        super().__init__()
        wi: Callable[[Tensor], Tensor] | None = init_resolver(weight_init) if weight_init is not None else None
        act: nn.Module = activation_resolver(activation)

        self.hidden_dim = hidden_dim
        self.coeffs_dim = coeffs_dim
        self.conv_dim = conv_dim
        self.out_dim = out_dim
        self.n_interaction = n_interaction
        self.cutoff = cutoff
        if cutoff_net is not None and cutoff is None:
            raise ValueError("cutoff must be specified when cutoff_net is used")
        self.cutoff_net = cutoff_net
        self.elec_to_node = elec_to_node
        self.add_valence = add_valence

        # electron information
        elec_info = ElecInfo(max_z, max_orb, n_per_orb)

        # calc basis layers
        self.rbf = rbf_resolver(rbf_type, cutoff=cutoff, elec_info=elec_info)
        self.shbf = SphericalHarmonicsBasis(elec_info)
        if cutoff_net:
            self.cn = cutoffnet_resolver(cutoff_net, cutoff=cutoff)

        # node and coefficient embedding layers
        z_embed_dim = self.hidden_dim + self.coeffs_dim
        self.node_e_embed_dim = hidden_dim if elec_to_node else 0
        e_embed_dim = self.node_e_embed_dim + self.coeffs_dim
        self.z_embed = EmbedZ(embed_dim=z_embed_dim, max_z=max_z)
        self.e_embed = EmbedElec(e_embed_dim, elec_info, extend_orb)
        self.node_embed = EmbedNode(hidden_dim, hidden_dim, elec_to_node, self.node_e_embed_dim, act, wi)
        self.coeff_embed = EmbedCoeffs(coeffs_dim, coeffs_dim, coeffs_dim, act, wi)
        if add_valence:
            self.valence_mask = ValenceMask(conv_dim, elec_info)

        # interaction layers
        self.int_layers = nn.ModuleList(
            [LCAOInteraction(hidden_dim, coeffs_dim, conv_dim, add_valence, act, wi) for _ in range(n_interaction)]
        )

        # output layers
        self.out_layer = LCAOOut(hidden_dim, out_dim, is_extensive, act, wi)
        self.pp_layer = PostProcess(out_dim, is_extensive, atomref, mean)

    def forward(self, batch: Batch) -> Tensor:
        """Forward calculation of LCAONet.

        Args:
            batch (Batch): the input graph batch data.

        Returns:
            torch.Tensor: the output property with (n_batch, out_dim) shape.
        """
        batch_idx: Tensor | None = batch.get(DataKeys.Batch_idx)
        pos = batch[DataKeys.Position]
        z = batch[DataKeys.Atom_numbers]
        idx_i, idx_j = batch[DataKeys.Edge_idx]

        # get triplets
        (
            _,
            _,
            tri_idx_i,
            tri_idx_j,
            tri_idx_k,
            edge_idx_kj,
            edge_idx_ji,
        ) = self.get_triplets(batch)

        # calc atomic distances
        distances = self.calc_atomic_distances(batch)

        # calc angles each triplets
        pos_j = pos[tri_idx_j]
        vec_ji, vec_jk = pos[tri_idx_i] - pos_j, pos[tri_idx_k] - pos_j
        inner = (vec_ji * vec_jk).sum(dim=-1)
        outter = torch.cross(vec_ji, vec_jk).norm(dim=-1)
        # arctan is more stable than arccos
        angle = torch.atan2(outter, inner)

        # calc basis
        rb = self.rbf(distances)
        shb = self.shbf(angle)
        cutoff_w = self.cn(distances) if self.cutoff_net else None

        # calc node and coefficient embedding vectors
        z_embed = self.z_embed(z)
        node_z, coeff_z = torch.split(z_embed, [self.hidden_dim, self.coeffs_dim], dim=-1)
        e_embed = self.e_embed(z)
        if self.elec_to_node:
            node_e, coeff_e = torch.split(e_embed, [self.node_e_embed_dim, self.coeffs_dim], dim=-1)
            x = self.node_embed(node_z, node_e)
        else:
            coeff_e = e_embed
            x = self.node_embed(node_z)
        cji = self.coeff_embed(coeff_z, coeff_e, idx_i, idx_j)

        # get valence mask coefficients
        valence_mask: Tensor | None = self.valence_mask(z, idx_j) if self.add_valence else None

        # calc interaction
        for inte in self.int_layers:
            x = inte(x, cji, valence_mask, cutoff_w, rb, shb, idx_i, idx_j, tri_idx_k, edge_idx_kj, edge_idx_ji)

        # output
        out = self.out_layer(x, batch_idx)
        out = self.pp_layer(out, z, batch_idx)

        return out
