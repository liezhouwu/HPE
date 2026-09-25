"""Original ViT-MAE training wrapper for the ViT-csi-small backbone.

The masked-autoencoder definition itself stays in
``pose_ssl/pretrain_methods.py`` (the implementation behind the ViT report line);
this module only adapts it to the pretraining runner contract:
``forward(csi) -> (loss, log)`` plus ``export_encoder_state_dict()``, where the
exported state is exactly the ViT-csi-small backbone used by downstream
fine-tuning (``build_pose_model('vit_csi_small')``).

Masking uses the module's global RNG, so an interrupted run resumes faithfully
through the checkpoint's stored torch RNG state; no extra state contract is
needed here.
"""

from __future__ import annotations

import math
from numbers import Real

from torch import Tensor, nn

from pose_ssl.model import ViTCSIEncoder
from pose_ssl.pretrain_methods import MAE

_INPUT_HEIGHT = 114
_INPUT_WIDTH = 10


class ViTMAEMethod(nn.Module):
    """Masked autoencoding: 75% patch masking, per-patch normalized reconstruction."""

    def __init__(
        self,
        *,
        mask_ratio: float = 0.75,
        decoder_dim: int = 256,
        decoder_depth: int = 2,
        decoder_heads: int = 4,
        patch_h: int = 6,
        patch_w: int = 5,
    ) -> None:
        super().__init__()
        if isinstance(mask_ratio, bool) or not isinstance(mask_ratio, Real):
            raise TypeError("mask_ratio must be a finite scalar")
        mask_ratio = float(mask_ratio)
        if not math.isfinite(mask_ratio) or not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must be strictly between 0 and 1")
        for name, value in (
            ("decoder_dim", decoder_dim),
            ("decoder_depth", decoder_depth),
            ("decoder_heads", decoder_heads),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value, span in (
            ("patch_h", patch_h, _INPUT_HEIGHT),
            ("patch_w", patch_w, _INPUT_WIDTH),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            if span % value:
                raise ValueError(f"{name} must divide the ViT CSI span of {span}")

        self.mask_ratio = mask_ratio
        self.decoder_dim = decoder_dim
        self.decoder_depth = decoder_depth
        self.decoder_heads = decoder_heads
        self.patch_h = patch_h
        self.patch_w = patch_w

        # The encoder grid must match the masking grid: the encoder owns the patch
        # projection and the positional embedding, so both sides get the same patch size.
        self.backbone = ViTCSIEncoder(patch_h=patch_h, patch_w=patch_w)
        self.mae = MAE(
            self.backbone,
            mask_ratio=mask_ratio,
            decoder_dim=decoder_dim,
            decoder_depth=decoder_depth,
            decoder_heads=decoder_heads,
            patch_h=patch_h,
            patch_w=patch_w,
        )
        if self.backbone.nh * self.backbone.nw != self.mae.n_patches:
            raise ValueError(
                "encoder patch grid does not match the masking grid: "
                f"{self.backbone.nh}x{self.backbone.nw} vs {self.mae.n_patches}"
            )

    def forward(self, csi: Tensor) -> tuple[Tensor, dict[str, float]]:
        """Return ``(loss, log)`` for one CSI batch of shape ``(B, 3, 114, 10)``."""

        return self.mae(csi)

    def export_encoder_state_dict(self) -> dict[str, Tensor]:
        """Export the fine-tune-ready ViT-csi-small encoder state."""

        return {
            name: value.detach().clone() for name, value in self.mae.vit.state_dict().items()
        }
