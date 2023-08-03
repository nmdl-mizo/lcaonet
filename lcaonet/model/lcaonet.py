from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch
from torch_scatter import scatter
from torch_sparse import SparseTensor

from ..atomistic.info import ElecInfo
from ..data.keys import GraphKeys
from ..model.base import BaseMPNN
from ..nn import Dense
from ..nn.cutoff import BaseCutoff
from ..nn.embed import EmbedCoeffs, EmbedElec, EmbedNode, EmbedZ, ValenceMask
from ..nn.post import PostProcess
from ..nn.rbf import BaseRadialBasis
from ..nn.shbf import SphericalHarmonicsBasis
from ..utils.resolve import (
    activation_resolver,
    cutoffnet_resolver,
    init_resolver,
    rbf_resolver,
)


class LCAOEmbedding(nn.Module):
    def __init__(
        self,
        emb_size: int,
        emb_size_coeff: int,
        elec_info: ElecInfo,
        max_z: int,
        use_elec: bool,
        extend_orb: bool,
        activation: nn.Module = nn.SiLU(),
        weight_init: Callable[[Tensor], Tensor] | None = None,
    ):
        super().__init__()
        self.emb_size = emb_size
        self.emb_size_coeff = emb_size_coeff
        self.use_elec = use_elec

        emb_size_z = emb_size + emb_size_coeff
        self.z_embed = EmbedZ(emb_size_z, max_z)

        self.emb_size_node_e = emb_size if use_elec else 0
        emb_size_e = self.emb_size_node_e + emb_size_coeff
        self.e_embed = EmbedElec(emb_size_e, elec_info, extend_orb)

        self.node_embed = EmbedNode(emb_size, emb_size, use_elec, self.emb_size_node_e, activation, weight_init)
        self.coeff_embed = EmbedCoeffs(
            emb_size_coeff, emb_size_coeff, emb_size_coeff, elec_info.n_orb, activation, weight_init
        )

    def forward(self, z: Tensor, idx_s: Tensor, idx_t: Tensor) -> tuple[Tensor, Tensor]:
        # z: (N, emb_size_z)
        z_embed = self.z_embed(z)
        node_z, coeff_z = torch.split(z_embed, [self.emb_size, self.emb_size_coeff], dim=-1)

        # e: (N, n_orb, emb_size_e)
        e_embed = self.e_embed(z)
        if self.use_elec:
            node_e, coeff_e = torch.split(e_embed, [self.emb_size_node_e, self.emb_size_coeff], dim=-1)
            x = self.node_embed(node_z, node_e)
        else:
            coeff_e = e_embed
            x = self.node_embed(node_z)
        cst = self.coeff_embed(coeff_z, coeff_e, idx_s, idx_t)

        return x, cst


class LCAOInteraction(nn.Module):
    """The layer that performs message-passing of LCAONet."""

    def __init__(
        self,
        emb_size: int,
        emb_size_coeff: int,
        emb_size_conv: int,
        add_valence: bool = False,
        activation: nn.Module = nn.SiLU(),
        weight_init: Callable[[Tensor], Tensor] | None = None,
    ):
        """
        Args:
            emb_size (int): the size of node vector.
            emb_size_coeff (int): the size of coefficient vectors.
            emb_size_conv (int): the size of embedding vectors at convolution.
            add_valence (bool, optional): whether to add the effect of valence orbitals. Defaults to `False`.
            activation (nn.Module, optional): the activation function. Defaults to `torch.nn.SiLU()`.
            weight_init (Callable[[torch.Tensor], torch.Tensor] | None, optional): the weight initialization func. Defaults to `None`.
        """  # noqa: E501
        super().__init__()
        self.emb_size = emb_size
        self.emb_size_coeff = emb_size_coeff
        self.emb_size_conv = emb_size_conv
        self.add_valence = add_valence

        self.node_weight = Dense(emb_size, 2 * emb_size_conv, True, weight_init)

        # No bias is used to keep 0 coefficient vectors at 0
        out_size = 2 * emb_size_conv if add_valence else emb_size_conv
        self.f_coeffs = nn.Sequential(
            Dense(emb_size_coeff, emb_size_conv, False, weight_init),
            activation,
            Dense(emb_size_conv, out_size, False, weight_init),
            activation,
        )

        three_out_dim = 2 * emb_size_conv if add_valence else emb_size_conv
        self.f_three = nn.Sequential(
            Dense(emb_size_conv, three_out_dim, False, weight_init),
        )

        self.basis_weight = Dense(emb_size_conv, emb_size_conv, False, weight_init)

        self.f_node = nn.Sequential(
            Dense(2 * emb_size_conv, emb_size_conv, True, weight_init),
            activation,
            Dense(emb_size_conv, emb_size_conv, True, weight_init),
            activation,
        )
        self.out_weight = Dense(emb_size_conv, emb_size, False, weight_init)

    def forward(
        self,
        x: Tensor,
        cst: Tensor,
        valence_mask: Tensor | None,
        rb: Tensor,
        shb: Tensor,
        idx_s: Tensor,
        idx_t: Tensor,
        tri_idx_k: Tensor,
        edge_idx_ks: torch.LongTensor,
        edge_idx_st: torch.LongTensor,
    ) -> Tensor:
        """Forward calculation of LCAOConv.

        Args:
            x (torch.Tensor): node embedding vectors with (N, hidden_dim) shape.
            cst (torch.Tensor): coefficient vectors with (E, n_orb, coeffs_dim) shape.
            valence_mask (torch.Tensor | None): valence orbital mask with (E, n_orb, conv_dim) shape.
            rb (torch.Tensor): the radial basis with (E, n_orb) shape.
            shb (torch.Tensor): the spherical harmonics basis with (n_triplets, n_orb) shape.
            idx_s (torch.Tensor): the indices of the first node of each edge with (E) shape.
            idx_t (torch.Tensor): the indices of the second node of each edge with (E) shape.
            tri_idx_k (torch.Tensor): the indices of the third node of each triplet with (n_triplets) shape.
            edge_idx_ks (torch.LongTensor): the edge index from atom k to s with (n_triplets) shape.
            edge_idx_st (torch.LongTensor): the edge index from atom s to t with (n_triplets) shape.

        Returns:
            torch.Tensor: updated node embedding vectors with (N, hidden_dim) shape.
        """
        if self.add_valence and valence_mask is None:
            raise ValueError("valence_mask must be provided when add_valence=True")

        x_before = x
        N = x.size(0)

        # Transformation of the node vectors
        x, xk = torch.chunk(self.node_weight(x), 2, dim=-1)

        # Transformation of the coefficient vectors
        cst = self.f_coeffs(cst)

        # --- Threebody Message-passing ---
        cks = cst[edge_idx_ks]
        if self.add_valence:
            cks, cks_valence = torch.chunk(cks, 2, dim=-1)
            cks_valence = cks_valence * valence_mask[edge_idx_ks]  # type: ignore # Since mypy cannot determine that the Valencemask is not None # noqa: E501

        # threebody LCAO weight: summation of all orbitals multiplied by coefficient vectors
        three_body_orbs = rb[edge_idx_ks] * shb
        three_body_w = torch.einsum("ed,edh->eh", three_body_orbs, cks).contiguous()
        if self.add_valence:
            valence_w = torch.einsum("ed,edh->eh", three_body_orbs, cks_valence).contiguous()
            three_body_w = three_body_w + valence_w
        three_body_w = F.normalize(three_body_w, dim=-1)

        # multiply node embedding
        xk = torch.sigmoid(xk[tri_idx_k])
        three_body_w = three_body_w * xk
        three_body_w = scatter(three_body_w, edge_idx_st, dim=0, dim_size=rb.size(0))

        # threebody orbital information is injected to the coefficient vectors
        cst = cst + cst * self.f_three(three_body_w).unsqueeze(1)

        # --- Twobody Message-passings ---
        if self.add_valence:
            cst, cst_valence = torch.chunk(cst, 2, dim=-1)
            cst_valence = cst_valence * valence_mask

        # twobody LCAO weight: summation of all orbitals multiplied by coefficient vectors
        lcao_w = torch.einsum("ed,edh->eh", rb, cst).contiguous()
        if self.add_valence:
            valence_w = torch.einsum("ed,edh->eh", rb, cst_valence).contiguous()
            lcao_w = lcao_w + valence_w
        lcao_w = F.normalize(lcao_w, dim=-1)

        # Message-passing and update node embedding vector
        x = x_before + self.out_weight(
            scatter(
                self.basis_weight(lcao_w) * self.f_node(torch.cat([x[idx_s], x[idx_t]], dim=-1)),
                idx_s,
                dim=0,
                dim_size=N,
            )
        )

        return x


class LCAOOut(nn.Module):
    """The output layer of LCAONet.

    Three-layer neural networks to output desired physical property
    values.
    """

    def __init__(
        self,
        emb_size: int,
        out_size: int,
        is_extensive: bool = True,
        regress_forces: bool = False,
        direct_forces: bool = True,
        activation: nn.Module = nn.SiLU(),
        weight_init: Callable[[Tensor], Tensor] | None = None,
    ):
        """
        Args:
            emb_size (int): the size of node embedding vectors.
            out_size (int): the size of output property.
            is_extensive (bool): whether the output property is extensive or not. Defaults to `True`.
            regress_forces (bool): whether to regress inter atomic forces. Defaults to `False`.
            direct_forces (bool): whether to regress inter atomic forces directly. Defaults to `True`.
            activation (nn.Module): the activation function. Defaults to `nn.SiLU()`.
            weight_init (Callable[[torch.Tensor], torch.Tensor] | None): the weight initialization function.
                Defaults to `None`.
        """
        super().__init__()
        self.emb_size = emb_size
        self.out_size = out_size
        self.is_extensive = is_extensive
        self.regress_forces = regress_forces
        self.direct_forces = direct_forces

        self.out_lin = nn.Sequential(
            Dense(emb_size, emb_size, True, weight_init),
            activation,
            Dense(emb_size, emb_size // 2, True, weight_init),
            activation,
            Dense(emb_size // 2, out_size, False, weight_init),
        )

        if regress_forces and direct_forces:
            self.out_lin_force = nn.Sequential(
                Dense(2 * emb_size, emb_size, True, weight_init),
                activation,
                Dense(emb_size, emb_size // 2, True, weight_init),
                activation,
                Dense(emb_size // 2, 1, False, weight_init),
            )

    def forward(
        self,
        x: Tensor,
        batch_idx: Tensor | None,
        idx_s: Tensor,
        idx_t: Tensor,
        edge_vec_st: Tensor,
        pos: Tensor,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Forward calculation of LCAOOut.

        Args:
            x (torch.Tensor): node embedding vectors with (N, hidden_dim) shape.
            batch_idx (torch.Tensor | None): the batch indices of nodes with (N) shape.

        Returns:
            prop: the output property values with (B, out_dim) shape.
            force_s: the inter atomic forces with (N, 3) shape.
        """
        prop = self.out_lin(x)
        if batch_idx is not None:
            B = batch_idx.max().item() + 1
            prop = scatter(prop, batch_idx, dim=0, reduce="sum" if self.is_extensive else "mean", dim_size=B)
            return prop
        if self.is_extensive:
            prop = prop.sum(dim=0, keepdim=True)
        else:
            prop = prop.mean(dim=0, keepdim=True)
        if not self.regress_forces:
            return prop

        if self.direct_forces:
            N = x.size(0)
            force_st = self.out_lin_force(torch.cat([x[idx_s], x[idx_t]], dim=-1))  # (E, 1)
            force_st = force_st * edge_vec_st  # (E, 3)
            force_s = scatter(force_st, idx_s, dim=0, reduce="sum", dim_size=N)  # (N, 3)
            return prop, force_s

        if self.out_size > 1:
            force_s = torch.stack(
                [-torch.autograd.grad(prop[:, i].sum(), pos, create_graph=True)[0] for i in range(self.out_size)],
                dim=1,
            )  # (N, out_dim, 3)
            force_s = force_s.squeeze(1)  # (N, 3)
        else:
            force_s = -torch.autograd.grad(prop.sum(), pos, create_graph=True)[0]  # (N, 3)
        return prop, force_s


class LCAONet(BaseMPNN):
    """
    LCAONet - MPNN including orbital interaction, physically motivatied by the LCAO method.
    """

    def __init__(
        self,
        emb_size: int = 128,
        emb_size_coeff: int = 128,
        emb_size_conv: int = 128,
        out_size: int = 1,
        n_interaction: int = 3,
        n_per_orb: int = 1,
        cutoff: float = 6.0,
        rbf_type: str | type[BaseRadialBasis] = "hydrogen",
        cutoff_net: str | type[BaseCutoff] = "envelope",
        max_z: int = 36,
        min_orb: str | None = None,
        max_orb: str | None = None,
        elec_to_node: bool = True,
        add_valence: bool = False,
        extend_orb: bool = False,
        is_extensive: bool = True,
        activation: str = "SiLU",
        weight_init: str | None = "glorotorthogonal",
        atomref: Tensor | None = None,
        mean: Tensor | None = None,
        regress_forces: bool = False,
        direct_forces: bool = True,
    ):
        """
        Args:
            emb_size (int): the size of node embedding vectors. Defaults to `128`.
            emb_size_coeff (int): the size of coefficient vectors. Defaults to `64`.
            emb_size_conv (int): the size of embedding vectors at convolution. Defaults to `64`.
            out_size (int): the size of output property. Defaults to `1`.
            n_interaction (int): the number of interaction layers. Defaults to `3`.
            cutoff (float | None): the cutoff radius. Defaults to `None`.
                If specified, the basis functions are normalized within the cutoff radius.
                If `cutoff_net` is specified, the `cutoff` radius must be specified.
            rbf_type (str | type[lcaonet.nn.rbf.BaseRadialBasis]): the radial basis function or the name. Defaults to `hydrogen`.
            cutoff_net (str | type[lcaonet.nn.cutoff.BaseCutoff] | None): the cutoff network or the name Defaults to `None`.
            max_z (int): the maximum atomic number. Defaults to `36`.
            min_orb (str | None): the minimum orbital name like "2s". Defaults to `None`.
            max_orb (str | None): the maximum orbital name like "2p". Defaults to `None`.
            elec_to_node (bool): whether to use electrons information to nodes embedding. Defaults to `True`.
            add_valence (bool): whether to add the effect of valence orbitals. Defaults to `False`.
            extend_orb (bool): whether to extend the basis set. Defaults to `False`. If `True`, MP is performed including unoccupied orbitals of the ground state.
            is_extensive (bool): whether to predict extensive property. Defaults to `True`.
            activation (str): the name of activation function. Defaults to `"SiLU"`.
            weight_init (str | None): the name of weight initialization function. Defaults to `"glorotorthogonal"`.
            atomref (torch.Tensor | None): the reference value of the output property with (max_z, out_dim) shape. Defaults to `None`.
            mean (torch.Tensor | None): the mean value of the output property with (out_dim) shape. Defaults to `None`.
        """  # noqa: E501
        super().__init__()
        wi: Callable[[Tensor], Tensor] | None = init_resolver(weight_init) if weight_init is not None else None
        act: nn.Module = activation_resolver(activation)

        self.emb_size = emb_size
        self.emb_size_coeff = emb_size_coeff
        self.emb_size_conv = emb_size_conv
        self.out_size = out_size
        self.n_interaction = n_interaction
        self.cutoff = cutoff
        self.cutoff_net = cutoff_net
        self.elec_to_node = elec_to_node
        self.add_valence = add_valence
        self.regress_forces = regress_forces
        self.direct_forces = direct_forces

        # electron information
        elec_info = ElecInfo(max_z, max_orb, min_orb, n_per_orb)

        # basis layers
        cn = cutoffnet_resolver(cutoff_net, cutoff=cutoff)
        self.rbf = rbf_resolver(rbf_type, cutoff=cutoff, elec_info=elec_info, cutoff_net=cn)
        self.shbf = SphericalHarmonicsBasis(elec_info)

        # node and coefficient embedding layers
        self.emb_layer = LCAOEmbedding(emb_size, emb_size_coeff, elec_info, max_z, elec_to_node, extend_orb, act, wi)
        if add_valence:
            self.valence_mask = ValenceMask(emb_size_conv, elec_info)

        # interaction layers
        self.int_layers = nn.ModuleList(
            [
                LCAOInteraction(emb_size, emb_size_coeff, emb_size_conv, add_valence, act, wi)
                for _ in range(n_interaction)
            ]
        )

        # output layers
        self.out_layer = LCAOOut(emb_size, out_size, is_extensive, regress_forces, direct_forces, act, wi)
        self.pp_layer = PostProcess(out_size, is_extensive, atomref, mean)

    def calc_3body_angles(self, graph: Batch) -> Batch:
        """calculate 3body angles for periodic boundary conditions.

        Args:
            graph (torch_geometric.data.Batch): material graph batch.

        Returns:
            graph (torch_geometric.data.Batch): material graph batch with 3body angles:
                angles (torch.Tensor): angle of ijk with (n_triplets) shape.
        """
        pair_vec_st = graph.get(GraphKeys.Edge_vec_st)
        if pair_vec_st is None:
            raise ValueError("edge_vec_st is not calculated. Please run calc_atomic_distances(return_vec=True) first.")
        edge_idx_st, edge_idx_ks = graph[GraphKeys.Edge_idx_st_3b], graph[GraphKeys.Edge_idx_ks_3b]

        vec_st, vec_ks = pair_vec_st[edge_idx_st], pair_vec_st[edge_idx_ks]
        inner = (vec_st * vec_ks).sum(dim=-1)
        outter = torch.cross(vec_st, vec_ks).norm(dim=-1)
        # arctan is more stable than arccos
        angles = torch.atan2(outter, inner)

        graph[GraphKeys.Angles_3b] = angles
        return graph

    def get_triplets(self, graph: Batch) -> Batch:
        """Convert edge_index to triplets.

        Args:
            graph (torch_geometirc.data.Batch): material graph batch.

        Returns:
            graph (torch_geometric.data.Batch): material graph batch with 3body index:
                tri_idx_s (Tensor): index of atom i of (n_triplets) shape.
                tri_idx_t (Tensor): index of atom j of (n_triplets) shape.
                tri_idx_k (Tensor): index of atom k of (n_triplets) shape.
                edge_idx_ks (Tensor): edge index of center k to j of (n_triplets) shape.
                edge_idx_st (Tensor): edge index of center j to i of (n_triplets) shape.

        Notes:
            ref:
                https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/dimenet.html
        """
        # order is "source_to_traget"
        idx_s, idx_t = graph[GraphKeys.Edge_idx]

        value = torch.arange(idx_s.size(0), device=idx_s.device)
        num_nodes = graph[GraphKeys.Z].size(0)
        adj_t = SparseTensor(row=idx_t, col=idx_s, value=value, sparse_sizes=(num_nodes, num_nodes))
        adj_t_row = adj_t[idx_s]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        # Edge indices (k -> t) and (s -> t) for triplets.
        # The position of s in the pair edge_index becomes k and the position of i becomes j
        edge_idx_ks = adj_t_row.storage.value()
        edge_idx_st = adj_t_row.storage.row()
        mask = edge_idx_ks != edge_idx_st
        edge_idx_ks = edge_idx_ks[mask]
        edge_idx_st = edge_idx_st[mask]

        # Node indices (s, t, k) for triplets.
        tri_idx_s = idx_s.repeat_interleave(num_triplets)
        tri_idx_t = idx_t.repeat_interleave(num_triplets)
        tri_idx_k = adj_t_row.storage.col()
        # Remove i == k triplets.
        tri_idx_s, tri_idx_t, tri_idx_k = (tri_idx_s[mask], tri_idx_t[mask], tri_idx_k[mask])

        # graph[GraphKeys.Idx_i_3b] = tri_idx_s
        # graph[GraphKeys.Idx_j_3b] = tri_idx_t
        graph[GraphKeys.Idx_k_3b] = tri_idx_k
        graph[GraphKeys.Edge_idx_ks_3b] = edge_idx_ks
        graph[GraphKeys.Edge_idx_st_3b] = edge_idx_st
        return graph

    def forward(self, graph: Batch) -> Tensor | tuple[Tensor, Tensor]:
        """Forward calculation of LCAONet.

        Args:
            graph (Batch): the input graph batch data.

        Returns:
            torch.Tensor: the output property with (n_batch, out_dim) shape.
        """
        if self.regress_forces and not self.direct_forces:
            graph[GraphKeys.Pos].requires_grad_(True)

        # ---------- Get Graph information ----------
        batch_idx: Tensor | None = graph.get(GraphKeys.Batch_idx)
        z = graph[GraphKeys.Z]
        pos = graph[GraphKeys.Pos]
        # order is "source_to_target" i.e. [index_j, index_i]
        idx_s, idx_t = graph[GraphKeys.Edge_idx]

        # get triplets
        graph = self.get_triplets(graph)
        tri_idx_k = graph[GraphKeys.Idx_k_3b]
        edge_idx_ks = graph[GraphKeys.Edge_idx_ks_3b]
        edge_idx_st = graph[GraphKeys.Edge_idx_st_3b]

        # calc atomic distances
        graph = BaseMPNN.calc_atomic_distances(graph, return_vec=True)
        distances = graph[GraphKeys.Edge_dist]
        edge_vec_st = graph[GraphKeys.Edge_vec_st]

        # calc angles of each triplets
        graph = self.calc_3body_angles(graph)
        angles = graph[GraphKeys.Angles_3b]

        # ---------- Basis layers ----------
        rb = self.rbf(distances)
        shb = self.shbf(angles)

        # ---------- Embedding block ----------
        x, cst = self.emb_layer(z, idx_s, idx_t)

        # get valence mask coefficients
        valence_mask: Tensor | None = self.valence_mask(z, idx_t) if self.add_valence else None

        # ---------- Interaction blocks ----------
        for inte in self.int_layers:
            x = inte(x, cst, valence_mask, rb, shb, idx_s, idx_t, tri_idx_k, edge_idx_ks, edge_idx_st)

        # ---------- Output blocks ----------
        out = self.out_layer(x, batch_idx, idx_s, idx_t, edge_vec_st, pos)
        out = self.pp_layer(out, z, batch_idx)

        return out
