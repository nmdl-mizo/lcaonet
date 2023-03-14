from __future__ import annotations

import math
from collections.abc import Callable
from math import pi
from typing import Any

import sympy as sym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch
from torch_scatter import scatter

from lcaonet.atomistic.info import (
    BaseAtomisticInformation,
    ThreeBodyAtomisticInformation,
)
from lcaonet.data import DataKeys
from lcaonet.model.base import BaseGCNN
from lcaonet.nn import Dense
from lcaonet.nn.cutoff import BaseCutoff
from lcaonet.utils import activation_resolver, init_resolver, rbf_resolver


class SphericalHarmonicsBasis(nn.Module):
    """The layer that expand interatomic distance and angles with radial baisis
    functions and spherical harmonics functions."""

    def __init__(self, rbf: str = "hydrogenradialwavefunctionbasis", **kwargs):
        """
        Args:
            rbf (str): the name of radial basis functions. Defaults to `hydrogenradialwavefunctionbasis`.
        """
        super().__init__()
        self.radial_basis = rbf_resolver(rbf, **kwargs)

        # make spherical basis functions
        self.sph_funcs = self._calculate_symbolic_sh_funcs()

    @staticmethod
    def _y00(theta: Tensor, phi: Tensor) -> Tensor:
        r"""
        Spherical Harmonics with `l=m=0`.
        ..math::
            Y_0^0 = \frac{1}{2} \sqrt{\frac{1}{\pi}}

        Args:
            theta: the azimuthal angle.
            phi: the polar angle.

        Returns:
            `Y_0^0`: the spherical harmonics with `l=m=0`.
        """
        dtype = theta.dtype
        return (0.5 * torch.ones_like(theta) * math.sqrt(1.0 / pi)).to(dtype)

    def _calculate_symbolic_sh_funcs(self) -> list:
        """Calculate symbolic spherical harmonics functions.

        Returns:
            funcs (list[Callable]): the list of spherical harmonics functions.
        """
        funcs = []
        theta, phi = sym.symbols("theta phi")
        modules = {"sin": torch.sin, "cos": torch.cos, "conjugate": torch.conj, "sqrt": torch.sqrt, "exp": torch.exp}
        for nl in self.radial_basis.n_l_list:
            # !! only m=zero is used
            m_list = [0]
            for m in m_list:
                if nl[1] == 0:
                    funcs.append(SphericalHarmonicsBasis._y00)
                else:
                    func = sym.functions.special.spherical_harmonics.Znm(nl[1], m, theta, phi).expand(func=True)
                    func = sym.simplify(func).evalf()
                    funcs.append(sym.lambdify([theta, phi], func, modules))
        self.orig_funcs = funcs

        return funcs

    def forward(self, d: Tensor, angle: Tensor, edge_idx_kj: torch.LongTensor) -> Tensor:
        """Forward calculation of SphericalHarmonicsBasis.

        Args:
            d (torch.Tensor): the interatomic distance with (n_edge) shape.
            angle (torch.Tensor): the angles of triplets with (n_triplets) shape.
            edge_idx_kj (torch.LongTensor): the edge index from atom k to j with (n_triplets) shape.

        Returns:
            torch.Tensor: the expanded distance and angles of (n_triplets, n_orb) shape.
        """
        # (n_edge, n_orb)
        rbf = self.radial_basis(d)
        # (n_triplets, n_orb)
        sbf = torch.stack([f(angle, None) for f in self.sph_funcs], dim=1)

        # (n_triplets, n_orb)
        return rbf[edge_idx_kj] * sbf


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
    state is zero, the elec vector embedd into a zero vector.
    """

    def __init__(self, embed_dim: int, atom_info: BaseAtomisticInformation, extend_orb: bool = False):
        """
        Args:
            embed_dim (int): the dimension of embedding.
            atom_info (lcaonet.atomistic.info.BaseAtomisticInformation): the atomistic information.
            extend_orb (bool, optional): Whether to use an extended basis. Defaults to `False`.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.n_orb = atom_info.n_orb
        self.extend_orb = extend_orb

        self.register_buffer("elec", atom_info.get_elec_table)
        self.e_embeds = nn.ModuleList(
            [nn.Embedding(m, embed_dim, padding_idx=None if extend_orb else 0) for m in atom_info.get_max_elec_idx]
        )

        self.reset_parameters()

    def reset_parameters(self):
        for ee in self.e_embeds:
            ee.weight.data.uniform_(-math.sqrt(3), math.sqrt(3))
            # set padding_idx to zero
            if not self.extend_orb:
                ee._fill_padding_idx_with_zero()

    def forward(self, z: Tensor, z_embed: Tensor) -> Tensor:
        """Forward calculation of EmbedElec.

        Args:
            z (torch.Tensor): the atomic numbers with (n_node) shape.
            z_embed (torch.Tensor): the embedding of atomic numbers with (n_node, embed_dim) shape.

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

        # (n_node) -> (n_node, 1, embed_dim)
        z_embed = z_embed.unsqueeze(1)
        # inject atomic information to e_embed vectors
        e_embed = e_embed + e_embed * z_embed

        return e_embed


class ValenceMask(nn.Module):
    """The layer that generates valence orbital mask.

    Only the coefficients for valence orbitals are set to 1, and the
    coefficients for all other orbitals (including inner-shell orbitals)
    are set to 0.
    """

    def __init__(self, embed_dim: int, atom_info: BaseAtomisticInformation):
        """
        Args:
            embed_dim (int): the dimension of embedding.
            atom_info (lcaonet.atomistic.info.BaseAtomisticInformation): the atomistic information.
        """
        super().__init__()
        self.register_buffer("valence", atom_info.get_valence_table)
        self.n_orb = atom_info.n_orb

        self.embed_dim = embed_dim

    def forward(self, z: Tensor) -> Tensor:
        """Forward calculation of ValenceMask.

        Args:
            z (torch.Tensor): the atomic numbers with (n_node) shape.

        Returns:
            valence_mask (torch.Tensor): valence orbital mask with (n_node, n_orb, embed_dim) shape.
        """
        valence_mask = self.valence[z]  # type: ignore
        return valence_mask.unsqueeze(-1).expand(-1, -1, self.embed_dim)


class EmbedNode(nn.Module):
    """The layer that embeds atomic numbers and electron numbers into node
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


class LCAOConv(nn.Module):
    """The layer that performs LCAO convolution."""

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
        rbfs: Tensor,
        sbfs: Tensor,
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
            rbfs (torch.Tensor): the radial basis functions with (n_edge, n_orb) shape.
            sbfs (torch.Tensor): the spherical harmonics basis functions with (n_triplets, n_orb) shape.
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
            rbfs = rbfs * cutoff_w.unsqueeze(-1)

        # triple conv
        ckj = ckj[edge_idx_kj]
        ckj = F.normalize(ckj, dim=-1)
        # LCAO weight: summation of all orbitals multiplied by coefficient vectors
        three_body_orbs = torch.einsum("ed,edh->eh", sbfs * rbfs[edge_idx_kj], ckj)
        three_body_orbs = F.normalize(three_body_orbs, dim=-1)
        # multiply node embedding
        xk = torch.sigmoid(xk[tri_idx_k])
        three_body_w = three_body_orbs * xk
        three_body_w = self.three_lin(scatter(three_body_w, edge_idx_ji, dim=0, dim_size=rbfs.size(0)))
        # threebody orbital information is injected to the coefficient vectors
        cji = cji + cji * three_body_w.unsqueeze(1)
        cji = F.normalize(cji, dim=-1)
        if self.add_valence:
            cji, cji_valence = torch.chunk(cji, 2, dim=-1)

        # LCAO weight: summation of all orbitals multiplied by coefficient vectors
        lcao_w = torch.einsum("ed,edh->eh", rbfs, cji)

        if self.add_valence:
            # valence contribution
            if valence_mask is None:
                raise ValueError("valence_mask must be provided when add_valence=True")
            valence_w = torch.einsum("ed,edh->eh", rbfs, cji_valence * valence_mask)
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


class PostProcess(nn.Module):
    def __init__(
        self,
        out_dim: int,
        atomref: Tensor | None = None,
        add_mean: bool = False,
        mean: Tensor | None = None,
        is_extensive: bool = True,
    ):
        """postprocess the output property values.

        Add atom reference property and mean value to the output property values.

        Args:
            out_dim (int): output property dimension.
            atomref (torch.Tensor | None, optional): atom reference property values with (n_node, out_dim) shape.
                Defaults to `None`.
            add_mean (bool, optional): Whether to add mean value to the output property values.
                Defaults to `False`.
            mean (torch.Tensor | None, optional): mean value of the output property values with (out_dim) shape.
                Defaults to `None`.
            is_extensive (bool, optional): whether the output property is extensive or not. Defaults to `True`.
        """
        super().__init__()
        self.out_dim = out_dim
        if atomref is None:
            atomref = torch.zeros((100, out_dim))
        self.register_buffer("atomref", atomref)
        self.add_mean = add_mean
        self.register_buffer("mean", mean if mean else torch.zeros(out_dim))
        self.is_extensive = is_extensive

    def forward(self, out: Tensor, z: Tensor, batch_idx: Tensor | None) -> Tensor:
        """Forward calculation of PostProcess.

        Args:
            out (torch.Tensor): Output property values with (n_batch, out_dim) shape.
            z (torch.Tensor): Atomic numbers with (n_node) shape.
            batch_idx (torch.Tensor | None): The batch indices of nodes with (n_node) shape.

        Returns:
            torch.Tensor: Offset output property values with (n_batch, out_dim) shape.
        """
        if self.add_mean:
            mean = self.mean  # type: ignore
            if self.is_extensive:
                mean = mean.unsqueeze(0).expand(z.size(0), -1)  # type: ignore
                mean = (
                    mean.sum(dim=0, keepdim=True)
                    if batch_idx is None
                    else scatter(mean, batch_idx, dim=0, reduce="sum")
                )
            out = out + mean

        aref = self.atomref[z]  # type: ignore
        if self.is_extensive:
            aref = aref.sum(dim=0, keepdim=True) if batch_idx is None else scatter(aref, batch_idx, dim=0, reduce="sum")
        else:
            aref = (
                aref.mean(dim=0, keepdim=True) if batch_idx is None else scatter(aref, batch_idx, dim=0, reduce="mean")
            )
        return out + aref


class LCAONet(BaseGCNN):
    """
    LCAONet - GCNN including orbital interaction, physically motivatied by the LCAO method.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        coeffs_dim: int = 128,
        conv_dim: int = 128,
        out_dim: int = 1,
        n_interaction: int = 3,
        rbf_form: str = "hydrogenradialwavefunctionbasis",
        rbf_kwargs: dict[str, Any] = {},
        cutoff: float | None = None,
        cutoff_net: type[BaseCutoff] | None = None,
        max_z: int = 36,
        max_orb: str | None = None,
        elec_to_node: bool = True,
        add_valence: bool = False,
        extend_orb: bool = False,
        is_extensive: bool = True,
        activation: str = "SiLU",
        weight_init: str | None = "glorotorthogonal",
        postprocess: bool = False,
        **kwargs,
    ):
        """
        Args:
            hidden_dim (int): the dimension of node embedding vectors. Defaults to `128`.
            coeffs_dim (int): the dimension of coefficient vectors. Defaults to `64`.
            conv_dim (int): the dimension of embedding vectors at convolution. Defaults to `64`.
            out_dim (int): the dimension of output property. Defaults to `1`.
            n_interaction (int): the number of interaction layers. Defaults to `3`.
            rbf_form (str): the form of radial basis functions. Defaults to `"hydrogenradialwavefunctionbasis"`.
            rbf_kwargs (dict[str, Any]): the keyword arguments of radial basis functions. Defaults to `{}`.
            cutoff (float | None): the cutoff radius. Defaults to `None`.
                If specified, the basis functions are normalized within the cutoff radius.
                If `cutoff_net` is specified, the `cutoff` radius must be specified.
            cutoff_net (type[lcaonet.nn.cutoff.BaseCutoff] | None): the cutoff network. Defaults to `None`.
            max_z (int): the maximum atomic number. Defaults to `36`.
            max_orb (str | None): the maximum orbital name like "2p". Defaults to `None`.
            elec_to_node (bool): whether to use electrons information to nodes embedding. Defaults to `True`.
            add_valence (bool): whether to add the effect of valence orbitals. Defaults to `False`.
            extend_orb (bool): whether to extend the basis set. Defaults to `False`.
                If `True`, message-passing is performed including unoccupied orbitals of the ground state.
            is_extensive (bool): whether to predict extensive property. Defaults to `True`.
            activation (str): the name of activation function. Defaults to `"SiLU"`.
            weight_init (str | None): the name of weight initialization function. Defaults to `"glorotorthogonal"`.
            postprocess (bool): whether to use postprocess. Defaults to `False`.
        """
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
        self.postprocess = postprocess

        # basis layers
        rbf_kwargs["cutoff"] = cutoff
        self.rbf = rbf_resolver(rbf_form, **rbf_kwargs)
        self.sbf = SphericalHarmonicsBasis(rbf_form, **rbf_kwargs)
        if cutoff_net:
            self.cn = cutoff_net(cutoff)  # type: ignore

        # atomistic information
        atom_info = ThreeBodyAtomisticInformation(max_z, max_orb, limit_n_orb=self.rbf.limit_n_orb)

        # node and coefficient embedding layers
        self.node_z_embed_dim = 64  # fix
        self.node_e_embed_dim = 64 if elec_to_node else 0  # fix
        self.e_embed_dim = self.coeffs_dim + self.node_e_embed_dim
        self.z_embed_dim = self.e_embed_dim + self.node_z_embed_dim
        self.z_embed = EmbedZ(embed_dim=self.z_embed_dim, max_z=max_z)
        self.e_embed = EmbedElec(self.e_embed_dim, atom_info, extend_orb)
        self.node_embed = EmbedNode(hidden_dim, self.node_z_embed_dim, elec_to_node, self.node_e_embed_dim, act, wi)
        if add_valence:
            self.valence_mask = ValenceMask(conv_dim, atom_info)

        # interaction layers
        self.int_layers = nn.ModuleList(
            [LCAOConv(hidden_dim, coeffs_dim, conv_dim, add_valence, act, wi) for _ in range(n_interaction)]
        )

        # output layer
        self.out_layer = LCAOOut(hidden_dim, out_dim, is_extensive, act, wi)

        if postprocess:
            self.pp_layer = PostProcess(out_dim=out_dim, is_extensive=is_extensive, **kwargs)

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
        pos_i = pos[tri_idx_i]
        vec_ji, vec_ki = pos[tri_idx_j] - pos_i, pos[tri_idx_k] - pos_i
        inner = (vec_ji * vec_ki).sum(dim=-1)
        outter = torch.cross(vec_ji, vec_ki).norm(dim=-1)
        # arctan is more stable than arccos
        angle = torch.atan2(outter, inner)

        # calc basis
        rbfs = self.rbf(distances)
        sbfs = self.sbf(distances, angle, edge_idx_kj)
        cutoff_w = self.cn(distances) if self.cutoff_net else None

        # calc node and coefficient embedding vectors
        z_embed = self.z_embed(z)
        z_embed1, z_embed2 = torch.split(z_embed, [self.e_embed_dim, self.node_z_embed_dim], dim=-1)

        e_embed = self.e_embed(z, z_embed1)
        e_embed1, e_embed2 = torch.split(e_embed, [self.coeffs_dim, self.node_e_embed_dim], dim=-1)

        cji = e_embed1[idx_j] + e_embed1[idx_i] * e_embed1[idx_j]
        x = self.node_embed(z_embed2, e_embed2)

        # valence mask coefficients
        valence_mask: Tensor | None = self.valence_mask(z)[idx_j] if self.add_valence else None

        # calc interaction
        for inte in self.int_layers:
            x = inte(x, cji, valence_mask, cutoff_w, rbfs, sbfs, idx_i, idx_j, tri_idx_k, edge_idx_kj, edge_idx_ji)

        # output
        out = self.out_layer(x, batch_idx)

        if self.postprocess:
            out = self.pp_layer(out, z, batch_idx)

        return out
