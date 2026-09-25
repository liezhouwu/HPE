"""Original masked autoencoding (MAE) for the complete MetaFi-R34 encoder.

The official MAE implementation masks a high ratio of input patches, encodes the
visible patches only, and reconstructs the raw input with a lightweight decoder
plus learnable mask tokens.  The MetaFi encoder is a convolutional ResNet-34
that cannot skip patches, so this module keeps the original recipe -- 75% random
patch masking, per-patch normalized reconstruction targets, loss on masked
patches only, learnable mask tokens, small asymmetric decoder -- and applies the
masking on the CSI input instead: masked patches are zeroed before the encoder
sees the frame.

Unlike :class:`pose_ssl.metafi.mfm.MetaFiMFM` there is no teacher encoder, no EMA
update and no feature-space objective: this is the plain masked autoencoder.

Masking is applied on the input because the official encoder is convolutional, so
its BatchNorm statistics are estimated on partially zeroed frames.  This matches
the established MFM contract; ``mask_ratio`` is the knob that bounds the shift.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from numbers import Real

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mmfi_wifi.metafi_encoder import MetaFiEncoder

from .base import MetaFiSSLMethod, SSLStepOutput
from .pretrain_data import PretrainBatch


_INPUT_SUBCARRIERS = 114
_INPUT_TIME_STEPS = 10
_DEVICE_KEY = re.compile(r"^(?:cpu|cuda:[0-9]+)$")


def _validate_generator_device(generator: torch.Generator, device: torch.device) -> None:
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator")
    generator_device = torch.device(generator.device)
    input_device = torch.device(device)
    if str(generator_device) != str(input_device):
        raise ValueError(
            "generator.device must exactly match the input tensor device; "
            f"got generator.device={generator_device} and input device={input_device}"
        )


def _validate_csi(x: Tensor) -> None:
    if not isinstance(x, Tensor) or x.ndim != 4 or x.shape[1] != 3:
        raise ValueError("CSI must have shape (B, 3, H, W)")
    if x.shape[0] < 1 or x.shape[2] < 1 or x.shape[3] < 1:
        raise ValueError("CSI must have non-empty batch and spatial dimensions")
    if not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError("CSI must contain only finite floating-point values")


def patchify(csi: Tensor, patch_height: int, patch_width: int) -> Tensor:
    """Flatten CSI into non-overlapping patches of all antennas.

    Returns ``(B, rows * columns, 3 * patch_height * patch_width)`` with row-major
    patch order, matching the decoder output layout.
    """

    _validate_csi(csi)
    batch, channels, height, width = csi.shape
    rows, columns = height // patch_height, width // patch_width
    patches = csi.reshape(batch, channels, rows, patch_height, columns, patch_width)
    patches = patches.permute(0, 2, 4, 3, 5, 1)
    return patches.reshape(batch, rows * columns, patch_height * patch_width * channels)


def patch_mask_to_pixel_mask(patch_mask: Tensor, patch_height: int, patch_width: int) -> Tensor:
    """Expand a ``(B, rows, columns)`` patch mask to a boolean ``(B, 1, H, W)`` mask."""

    if not isinstance(patch_mask, Tensor) or patch_mask.ndim != 3:
        raise ValueError("patch mask must have shape (B, rows, columns)")
    if patch_mask.dtype != torch.bool:
        raise ValueError("patch mask must have boolean dtype")
    pixel_mask = patch_mask.unsqueeze(1)
    pixel_mask = pixel_mask.repeat_interleave(patch_height, dim=2)
    pixel_mask = pixel_mask.repeat_interleave(patch_width, dim=3)
    return pixel_mask


def sample_patch_mask(
    batch_size: int,
    row_count: int,
    column_count: int,
    mask_ratio: float,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[Tensor, int]:
    """Sample the original MAE patch mask and return ``(keep_mask, keep_count)``.

    ``keep_mask`` is ``True`` for visible patches and has shape
    ``(B, row_count, column_count)``.  Patches are permuted with the official
    ``argsort(noise)``/``argsort(permutation)`` pairing, so the visible subset is
    uniformly random and reproducible from the generator state.
    """

    if batch_size < 1 or row_count < 1 or column_count < 1:
        raise ValueError("batch size and patch grid dimensions must be positive")
    _validate_generator_device(generator, device)
    patch_count = row_count * column_count
    keep_count = max(1, min(patch_count - 1, int(patch_count * (1.0 - mask_ratio))))
    noise = torch.rand(batch_size, patch_count, device=device, generator=generator)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    keep = torch.zeros(batch_size, patch_count, dtype=torch.bool, device=device)
    keep[:, :keep_count] = True
    keep = torch.gather(keep, 1, ids_restore)
    return keep.view(batch_size, row_count, column_count), keep_count


def normalize_patches(patches: Tensor) -> Tensor:
    """Return the per-patch normalized reconstruction target of the original MAE."""

    if not isinstance(patches, Tensor) or patches.ndim != 3:
        raise ValueError("patches must have shape (B, n_patches, patch_pixels)")
    values = patches.float()
    mean = values.mean(dim=-1, keepdim=True)
    variance = values.var(dim=-1, keepdim=True, unbiased=True)
    return (values - mean) / (variance + 1e-6).sqrt()


def masked_patch_reconstruction_loss(
    predicted: Tensor,
    patches: Tensor,
    patch_mask: Tensor,
    *,
    patch_norm: bool = True,
) -> tuple[Tensor, Tensor]:
    """MSE over masked patches only, with a per-patch normalized target.

    Returns ``(masked_loss, visible_loss)``.  Only ``masked_loss`` is optimized;
    the visible-region value is reported for monitoring.
    """

    if not isinstance(predicted, Tensor) or predicted.ndim != 3:
        raise ValueError("predicted patches must have shape (B, n_patches, patch_pixels)")
    if patches.shape != predicted.shape:
        raise ValueError("predicted and target patches must have the same shape")
    if patch_mask.dtype != torch.bool or patch_mask.shape != patches.shape[:2]:
        raise ValueError("patch mask must have shape (B, n_patches) and boolean dtype")
    if patch_mask.device != predicted.device:
        raise ValueError("patch mask and predicted patches must use the same device")
    if not isinstance(patch_norm, bool):
        raise TypeError("patch_norm must be a boolean")
    target = normalize_patches(patches) if patch_norm else patches.float()
    per_patch = (predicted.float() - target.detach()).square().mean(dim=-1)
    weights = patch_mask.float()
    masked_loss = (per_patch * weights).sum() / weights.sum().clamp_min(1.0)
    visible = (~patch_mask).float()
    visible_loss = (per_patch * visible).sum() / visible.sum().clamp_min(1.0)
    return masked_loss, visible_loss


class MetaFiMAE(MetaFiSSLMethod):
    """Masked autoencoding: encode the masked CSI, reconstruct masked patches."""

    _EXTRA_STATE_SCHEMA_VERSION = 1

    def __init__(
        self,
        encoder: MetaFiEncoder,
        *,
        mask_ratio: float = 0.75,
        patch_height: int = 6,
        patch_width: int = 5,
        decoder_hidden_dim: int = 256,
        patch_norm: bool = True,
        use_mask_token: bool = True,
        augmentation_seed: int = 0,
    ) -> None:
        super().__init__()
        if not isinstance(encoder, MetaFiEncoder):
            raise TypeError("encoder must be a MetaFiEncoder")
        if isinstance(mask_ratio, bool) or not isinstance(mask_ratio, Real):
            raise TypeError("mask_ratio must be a finite scalar")
        mask_ratio = float(mask_ratio)
        if not math.isfinite(mask_ratio) or not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must be strictly between 0 and 1")
        for name, value, span in (
            ("patch_height", patch_height, _INPUT_SUBCARRIERS),
            ("patch_width", patch_width, _INPUT_TIME_STEPS),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            if span % value:
                raise ValueError(f"{name} must divide the official CSI span of {span}")
        if isinstance(decoder_hidden_dim, bool) or not isinstance(decoder_hidden_dim, int) or decoder_hidden_dim < 1:
            raise ValueError("decoder_hidden_dim must be a positive integer")
        if not isinstance(patch_norm, bool):
            raise TypeError("patch_norm must be a boolean")
        if not isinstance(use_mask_token, bool):
            raise TypeError("use_mask_token must be a boolean")
        if isinstance(augmentation_seed, bool) or not isinstance(augmentation_seed, int):
            raise TypeError("augmentation_seed must be an integer")

        self.mask_ratio = mask_ratio
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.patch_norm = patch_norm
        self.use_mask_token = use_mask_token
        self.augmentation_seed = augmentation_seed

        self.encoder = encoder
        self.decoder_embed = nn.Conv2d(encoder.feature_channels, decoder_hidden_dim, kernel_size=1)
        self.decoder_act = nn.GELU()
        # The original decoder lets tokens exchange information across patches; a single
        # 3x3 block on the patch grid is the convolutional equivalent.
        self.decoder_block = nn.Conv2d(
            decoder_hidden_dim, decoder_hidden_dim, kernel_size=3, padding=1
        )
        self.decoder_pred = nn.Conv2d(
            decoder_hidden_dim, 3 * patch_height * patch_width, kernel_size=1
        )
        self.mask_token = (
            nn.Parameter(torch.zeros(1, decoder_hidden_dim, 1, 1)) if use_mask_token else None
        )
        if self.mask_token is not None:
            nn.init.trunc_normal_(self.mask_token, std=0.02)

        self._augmentation_generators: dict[str, torch.Generator] = {}
        self._pending_augmentation_generator_states: dict[str, Tensor] = {}

    @staticmethod
    def _generator_key(device: torch.device) -> str:
        return str(torch.device(device))

    @staticmethod
    def _validate_generator_state(key: object, state: object) -> Tensor:
        if not isinstance(key, str) or _DEVICE_KEY.fullmatch(key) is None:
            raise ValueError(f"invalid MetaFiMAE generator state device key: {key!r}")
        try:
            device = torch.device(key)
            generator = torch.Generator(device=device)
        except (RuntimeError, TypeError, ValueError) as error:
            raise ValueError(f"invalid MetaFiMAE generator state device key: {key!r}") from error
        if not isinstance(state, Tensor) or state.dtype != torch.uint8 or state.ndim != 1:
            raise ValueError(f"invalid MetaFiMAE generator state for device {key!r}")
        restored = state.detach().cpu().clone()
        try:
            generator.set_state(restored)
        except RuntimeError as error:
            raise ValueError(f"invalid MetaFiMAE generator state for device {key!r}") from error
        return restored

    def _make_generator(self, device: torch.device) -> torch.Generator:
        normalized = torch.device(device)
        key = self._generator_key(normalized)
        generator = torch.Generator(device=normalized)
        pending = self._pending_augmentation_generator_states.pop(key, None)
        if pending is None:
            generator.manual_seed(self.augmentation_seed)
        else:
            try:
                generator.set_state(pending)
            except RuntimeError as error:
                raise ValueError(f"invalid MetaFiMAE generator state for device {key!r}") from error
        return generator

    def _default_generator(self, device: torch.device) -> torch.Generator:
        key = self._generator_key(device)
        generator = self._augmentation_generators.get(key)
        if generator is None:
            generator = self._make_generator(device)
            self._augmentation_generators[key] = generator
        return generator

    def get_extra_state(self) -> dict[str, object]:
        states = {
            key: generator.get_state().detach().cpu().clone()
            for key, generator in self._augmentation_generators.items()
        }
        states.update(
            {
                key: state.detach().cpu().clone()
                for key, state in self._pending_augmentation_generator_states.items()
            }
        )
        return {
            "schema_version": self._EXTRA_STATE_SCHEMA_VERSION,
            "generator_states": dict(sorted(states.items())),
            "mask_ratio": self.mask_ratio,
            "patch_height": self.patch_height,
            "patch_width": self.patch_width,
            "patch_norm": self.patch_norm,
        }

    def set_extra_state(self, state: object) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("MetaFiMAE extra state must be a mapping")
        if state.get("schema_version") != self._EXTRA_STATE_SCHEMA_VERSION:
            raise ValueError("unsupported MetaFiMAE extra state schema")
        try:
            mask_ratio = float(state.get("mask_ratio"))
        except (TypeError, ValueError) as error:
            raise ValueError("MetaFiMAE extra state requires mask_ratio") from error
        if not math.isclose(mask_ratio, self.mask_ratio, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("MetaFiMAE mask_ratio mismatch")
        for name, expected in (("patch_height", self.patch_height), ("patch_width", self.patch_width)):
            if state.get(name) != expected:
                raise ValueError(f"MetaFiMAE {name} mismatch")
        if state.get("patch_norm") is not self.patch_norm:
            raise ValueError("MetaFiMAE patch_norm mismatch")
        raw_states = state.get("generator_states")
        if not isinstance(raw_states, Mapping):
            raise ValueError("MetaFiMAE extra state requires generator_states")
        restored = {
            key: self._validate_generator_state(key, value)
            for key, value in raw_states.items()
        }
        self._augmentation_generators = {}
        self._pending_augmentation_generator_states = dict(sorted(restored.items()))

    def _decode(
        self,
        features: Tensor,
        row_count: int,
        column_count: int,
        patch_mask: Tensor,
    ) -> Tensor:
        """Decode encoder features into per-patch CSI predictions ``(B, N, P)``."""

        hidden = self.decoder_act(self.decoder_embed(features))
        tokens = F.interpolate(
            hidden, size=(row_count, column_count), mode="bilinear", align_corners=False
        )
        if self.mask_token is not None:
            # Masked positions carry the learnable token only, as in the original MAE;
            # the 3x3 block below is what lets them read the visible context.
            token = self.mask_token.to(tokens.dtype).expand_as(tokens)
            tokens = torch.where(patch_mask.unsqueeze(1), token, tokens)
        prediction = self.decoder_pred(self.decoder_act(self.decoder_block(tokens)))
        return prediction.flatten(2).transpose(1, 2)

    def forward(
        self,
        batch: PretrainBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> SSLStepOutput:
        if not isinstance(batch, PretrainBatch):
            raise TypeError("batch must be a PretrainBatch")
        if batch.anchor.shape[0] < 2:
            raise ValueError("MetaFiMAE requires batch size >= 2 for stable BatchNorm statistics")
        anchor = batch.anchor
        _validate_csi(anchor)
        if anchor.shape[2] % self.patch_height or anchor.shape[3] % self.patch_width:
            raise ValueError(
                "CSI spatial dimensions must be divisible by the patch size "
                f"({self.patch_height}, {self.patch_width})"
            )
        if generator is None:
            generator = self._default_generator(anchor.device)
        else:
            _validate_generator_device(generator, anchor.device)

        row_count = anchor.shape[2] // self.patch_height
        column_count = anchor.shape[3] // self.patch_width
        keep_mask, keep_count = sample_patch_mask(
            anchor.shape[0],
            row_count,
            column_count,
            self.mask_ratio,
            device=anchor.device,
            generator=generator,
        )
        patch_mask = ~keep_mask
        pixel_mask = patch_mask_to_pixel_mask(patch_mask, self.patch_height, self.patch_width)
        masked_anchor = anchor.masked_fill(pixel_mask.expand(-1, anchor.shape[1], -1, -1), 0.0)

        features = self.encoder(masked_anchor.unsqueeze(1)).feature_map
        predicted = self._decode(features, row_count, column_count, patch_mask)
        if not torch.isfinite(predicted).all():
            raise ValueError("MetaFiMAE predicted patches must be finite")
        patches = patchify(anchor, self.patch_height, self.patch_width)
        flat_mask = patch_mask.reshape(anchor.shape[0], row_count * column_count)
        loss, visible_loss = masked_patch_reconstruction_loss(
            predicted, patches, flat_mask, patch_norm=self.patch_norm
        )
        if not torch.isfinite(loss):
            raise ValueError("MetaFiMAE loss must be finite")
        return SSLStepOutput(
            loss=loss,
            metrics={
                "loss": float(loss.detach()),
                "masked_patch_loss": float(loss.detach()),
                "visible_patch_loss": float(visible_loss.detach()),
                "mask_ratio": float(patch_mask.float().mean().detach()),
                "n_keep": float(keep_count),
                "patch_count": float(row_count * column_count),
            },
        )

    def export_encoder_state_dict(self) -> dict[str, Tensor]:
        """Export only the trainable encoder for downstream fine-tuning."""

        return {name: value.detach().clone() for name, value in self.encoder.state_dict().items()}
