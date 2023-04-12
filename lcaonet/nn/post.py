from __future__ import annotations

import torch.nn as nn
from torch import Tensor
from torch_scatter import scatter


class PostProcess(nn.Module):
    """postprocess the output property values.

    Add atom reference property and mean value to the network output
    values.
    """

    def __init__(
        self,
        out_dim: int,
        is_extensive: bool = True,
        atomref: Tensor | None = None,
        mean: Tensor | None = None,
    ):
        """
        Args:
            out_dim (int): output property dimension.
            is_extensive (bool): whether the output property is extensive or not. Defaults to `True`.
            atomref (torch.Tensor | None): atom reference values with (max_z, out_dim) shape. Defaults to `None`.
            mean (torch.Tensor | None): mean value of the output property with (out_dim) shape. Defaults to `None`.
        """
        super().__init__()
        self.out_dim = out_dim
        self.is_extensive = is_extensive
        # atom ref
        self.register_buffer("atomref", atomref)
        # mean and std
        self.register_buffer("mean", mean)

    def extra_repr(self) -> str:
        return "is_extensive={}, atomref={}, mean={}".format(
            self.is_extensive,
            self.atomref if self.atomref is not None else None,
            self.mean if self.mean is not None else None,
        )

    def forward(self, out: Tensor, z: Tensor, batch_idx: Tensor | None) -> Tensor:
        """Forward calculation of PostProcess.

        Args:
            out (torch.Tensor): Output property values with (n_batch, out_dim) shape.
            z (torch.Tensor): Atomic numbers with (n_node) shape.
            batch_idx (torch.Tensor | None): The batch indices of nodes with (n_node) shape.

        Returns:
            torch.Tensor: Offset output property values with (n_batch, out_dim) shape.
        """
        if self.atomref is not None:
            aref = self.atomref[z]  # type: ignore # Since mypy cannot determine that the atomref is a tensor
            if self.is_extensive:
                aref = (
                    aref.sum(dim=0, keepdim=True)
                    if batch_idx is None
                    else scatter(aref, batch_idx, dim=0, reduce="sum")
                )
            else:
                aref = (
                    aref.mean(dim=0, keepdim=True)
                    if batch_idx is None
                    else scatter(aref, batch_idx, dim=0, reduce="mean")
                )
            out = out + aref

        if self.mean is not None:
            mean = self.mean  # type: ignore # Since mypy cannot determine that the mean is a tensor
            if self.is_extensive:
                mean = mean.unsqueeze(0).expand(z.size(0), -1)  # type: ignore # Since mypy cannot determine that the mean is a tensor # NOQA: E501
                mean = (
                    mean.sum(dim=0, keepdim=True)
                    if batch_idx is None
                    else scatter(mean, batch_idx, dim=0, reduce="sum")
                )
            out = out + mean

        return out
