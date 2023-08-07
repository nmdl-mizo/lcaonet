from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor

from ..atomistic.info import ElecInfo
from ..nn.base import Dense


class EmbedZ(nn.Module):
    """The layer that embeds atomic numbers into latent vectors."""

    def __init__(self, emb_size: int, max_z: int = 94):
        """
        Args:
            emb_size (int): the size of embedding vector.
            max_z (int, optional): the maximum atomic number. Defaults to `94`.
        """
        super().__init__()
        self.emb_size = emb_size
        self.z_embed = nn.Embedding(max_z, emb_size)

        self.reset_parameters()

    def reset_parameters(self):
        self.z_embed.weight.data.uniform_(-math.sqrt(3), math.sqrt(3))

    def forward(self, z: Tensor) -> Tensor:
        """Forward calculation of EmbedZ.

        Args:
            z (torch.Tensor): the atomic numbers with (N) shape.

        Returns:
            z_emb (torch.Tensor): the embedding vectors with (N, emb_size) shape.
        """
        return self.z_embed(z - 1)


class EmbedElec(nn.Module):
    """The layer that embeds electron numbers into latent vectors.

    If `extend_orb=False`, then if the number of electrons in the ground
    state is zero, the orbital is a zero vector embedding.
    """

    def __init__(self, emb_size: int, elec_info: ElecInfo, extend_orb: bool = False):
        """
        Args:
            emb_size (int): the size of embedding vector.
            elec_info (lcaonet.atomistic.info.ElecInfo): the object that contains the information about the number of electrons.
            extend_orb (bool, optional): Whether to use an extended basis. Defaults to `False`.
        """  # noqa: E501
        super().__init__()
        self.register_buffer("elec", elec_info.elec_table)
        self.n_orb = elec_info.n_orb
        self.emb_size = emb_size
        self.extend_orb = extend_orb

        self.e_embeds = nn.ModuleList()
        min_idx = elec_info.min_orb_idx if elec_info.min_orb_idx else -1
        for i, max_e in enumerate(elec_info.max_elec_idx):
            if i <= min_idx or extend_orb:
                padding_idx = None
            else:
                padding_idx = 0
            self.e_embeds.append(nn.Embedding(max_e, emb_size, padding_idx=padding_idx))

        self.reset_parameters()

    def reset_parameters(self):
        for ee in self.e_embeds:
            ee.weight.data.uniform_(-math.sqrt(2), math.sqrt(2))
            # set padding_idx to zero
            ee._fill_padding_idx_with_zero()

    def forward(self, z: Tensor) -> Tensor:
        """Forward calculation of EmbedElec.

        Args:
            z (torch.Tensor): the atomic numbers with (N) shape.

        Returns:
            e_embed (torch.Tensor): the embedding of electron numbers with (N, n_orb, emb_size) shape.
        """
        # (n_node, n_orb)
        elec = self.elec[z]  # type: ignore # Since mypy cannot determine that the elec is a tensor
        # (n_orb, n_node)
        elec = torch.transpose(elec, 0, 1)
        # (n_orb, n_node, embed_dim)
        e_emb = torch.stack([ce(elec[i]) for i, ce in enumerate(self.e_embeds)], dim=0)
        # (n_node, n_orb, embed_dim)
        e_emb = torch.transpose(e_emb, 0, 1)

        return e_emb


class ValenceMask(nn.Module):
    """The layer that generates valence orbital mask.

    Only the coefficients for valence orbitals are set to 1, and the
    coefficients for all other orbitals (including inner-shell orbitals)
    are set to 0.
    """

    def __init__(self, emb_size: int, elec_info: ElecInfo):
        """
        Args:
            emb_size (int): the size of embedding vector.
            elec_info (lcaonet.atomistic.info.ElecInfo): the object that contains the information about the number of electrons.
        """  # noqa: E501
        super().__init__()
        self.register_buffer("valence", elec_info.valence_table)
        self.n_orb = elec_info.n_orb

        self.emb_size = emb_size

    def forward(self, z: Tensor, idx_j: Tensor) -> Tensor:
        """Forward calculation of ValenceMask.

        Args:
            z (torch.Tensor): the atomic numbers with (N) shape.
            idx_j (torch.Tensor): the indices of the second node of each edge with (E) shape.

        Returns:
            valence_mask (torch.Tensor): valence orbital mask with (E, n_orb, emb_size) shape.
        """
        valence_mask = self.valence[z]  # type: ignore # Since mypy cannot determine that the valence is a tensor
        return valence_mask.unsqueeze(-1).expand(-1, -1, self.emb_size)[idx_j]


class EmbedNode(nn.Module):
    """The layer that embedds atomic numbers and electron numbers into node
    embedding vectors."""

    def __init__(
        self,
        emb_size: int,
        emb_size_z: int,
        use_elec: bool,
        emb_size_e: int | None = None,
        activation: nn.Module = nn.SiLU(),
        weight_init: Callable[[Tensor], Tensor] | None = None,
    ):
        """
        Args:
            emb_size (int): the size of embedding vector.
            emb_size_z (int): the size of atomic number embedding.
            use_elec (bool): whether to use electron number embedding.
            emb_size_e (int | None): the size of electron number embedding.
            activation (nn.Module, optional): the activation function. Defaults to `torch.nn.SiLU()`.
            weight_init (Callable[[torch.Tensor], torch.Tensor] | None, optional): the weight initialization function. Defaults to `None`.
        """  # noqa: E501
        super().__init__()
        self.emb_size = emb_size
        self.emb_size_z = emb_size_z
        self.use_elec = use_elec
        if use_elec:
            assert emb_size_e is not None
            self.emb_size_e = emb_size_e
        else:
            self.emb_size_e = 0

        hid_size = max(emb_size, (emb_size_z + self.emb_size_e) // 2)
        self.f_enc = nn.Sequential(
            Dense(emb_size_z + self.emb_size_e, hid_size, True, weight_init),
            activation,
            Dense(hid_size, emb_size, True, weight_init),
            activation,
        )
        self.bn = nn.BatchNorm1d(emb_size)

    def forward(self, z_emb: Tensor, e_emb: Tensor | None = None) -> Tensor:
        """Forward calculation of EmbedNode.

        Args:
            z_emb (torch.Tensor): the embedding of atomic numbers with (N, emb_size_z) shape.
            e_emb (torch.Tensor | None): the embedding of electron numbers with (N, n_orb, emb_size_e) shape.

        Returns:
            node_emb (torch.Tensor): the node embedding vectors with (N, emb_size) shape.
        """
        if self.use_elec:
            if e_emb is None:
                raise ValueError("e_emb must be set when use_elec is True.")
            e_emb = e_emb.sum(1) / math.sqrt(e_emb.shape[1])
            z_e_emb = torch.cat([z_emb, e_emb], dim=-1)
        else:
            z_e_emb = z_emb
        return self.bn(self.f_enc(z_e_emb))


class EmbedCoeffs(nn.Module):
    """The layer that embedds atomic numbers and electron numbers into
    coefficient embedding vectors."""

    def __init__(
        self,
        emb_size: int,
        emb_size_z: int,
        emb_size_e: int,
        n_orb: int,
        activation: nn.Module = nn.SiLU(),
        weight_init: Callable[[Tensor], Tensor] | None = None,
    ):
        """
        Args:
            emb_size (int): the size of coefficient vector.
            emb_size_z (int): the size of atomic number embedding.
            emb_size_e (int): the size of electron number embedding.
            activation (nn.Module): the activation function. Defaults to `torch.nn.SiLU()`.
            weight_init (Callable[[Tensor], Tensor] | None): weight initialization func. Defaults to `None`.
        """
        super().__init__()
        self.emb_size = emb_size
        self.emb_size_z = emb_size_z
        self.emb_size_e = emb_size_e

        self.f_z = nn.Sequential(
            Dense(2 * emb_size_z, emb_size, False, weight_init),
        )
        self.f_e = nn.Sequential(
            Dense(emb_size_e, emb_size, False, weight_init),
            activation,
            Dense(emb_size, emb_size, False, weight_init),
            activation,
        )
        self.bn = nn.BatchNorm1d(emb_size * n_orb)

    def forward(self, z_emb: Tensor, e_emb: Tensor, idx_s: Tensor, idx_t: Tensor) -> Tensor:
        """Forward calculation of EmbedCoeffs.

        Args:
            z_emb (torch.Tensor): the embedding of atomic numbers with (N, emb_size_z) shape.
            e_emb (torch.Tensor): the embedding of electron numbers with (N, n_orb, emb_size_e) shape.
            idx_s (torch.Tensor): the indices of center atoms with (E) shape.
            idx_t (torch.Tensor): the indices of neighbor atoms with (E) shape.

        Returns:
            coeff_emb (torch.Tensor): the coefficient embedding vectors with (E, n_orb, emb_size) shape.
        """
        z_emb = self.f_z(torch.cat([z_emb[idx_s], z_emb[idx_t]], dim=-1))
        e_emb = self.f_e(e_emb)[idx_t]
        coeff_emb = e_emb + e_emb * z_emb.unsqueeze(1)
        return self.bn(coeff_emb.reshape(coeff_emb.size(0), -1)).reshape(coeff_emb.size())
