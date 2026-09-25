"""Masked feature modeling for the complete MetaFi-R34 encoder."""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mmfi_wifi.metafi_encoder import MetaFiEncoder

from .base import MetaFiSSLMethod, SSLStepOutput
from .pretrain_data import PretrainBatch


MFMObjective = Literal["feature", "raw", "hybrid"]
_MASK_MODES = {"joint", "time", "subcarrier"}
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


@dataclass(frozen=True)
class StructuredCSIMask:
    """A synchronized, structured mask for ``(B, 3, subcarrier, time)`` CSI."""

    mask_ratio: float = 0.40
    modes: tuple[str, ...] = ("joint",)

    def __post_init__(self) -> None:
        if isinstance(self.mask_ratio, bool) or not isinstance(self.mask_ratio, (int, float)):
            raise TypeError("mask_ratio must be a finite scalar")
        ratio = float(self.mask_ratio)
        if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
            raise ValueError("mask_ratio must be between 0 and 1")
        modes = tuple(self.modes)
        if not modes:
            raise ValueError("modes must contain at least one mask mode")
        unknown = set(modes).difference(_MASK_MODES)
        if unknown:
            raise ValueError(f"unknown structured CSI mask modes: {sorted(unknown)!r}")
        object.__setattr__(self, "mask_ratio", ratio)
        object.__setattr__(self, "modes", modes)

    @classmethod
    def from_mapping(cls, config: Mapping[str, object] | "StructuredCSIMask") -> "StructuredCSIMask":
        if isinstance(config, cls):
            return config
        if not isinstance(config, Mapping):
            raise TypeError("mask config must be a mapping or StructuredCSIMask")
        allowed = {"mask_ratio", "modes"}
        unknown = set(config).difference(allowed)
        if unknown:
            raise ValueError(f"unknown structured CSI mask fields: {sorted(unknown)!r}")
        modes = config.get("modes", ("joint",))
        if isinstance(modes, str):
            modes = (modes,)
        if not isinstance(modes, (tuple, list)):
            raise TypeError("mask modes must be a sequence of strings")
        return cls(
            mask_ratio=config.get("mask_ratio", 0.40),  # type: ignore[arg-type]
            modes=tuple(modes),  # type: ignore[arg-type]
        )

    @staticmethod
    def _sample_start(
        *, dimension: int, width: int, device: torch.device, generator: torch.Generator
    ) -> int:
        if width >= dimension:
            return 0
        return int(torch.randint(dimension - width + 1, (1,), device=device, generator=generator).item())

    def _sample_one(self, height: int, width: int, device: torch.device, generator: torch.Generator) -> Tensor:
        mask = torch.zeros((height, width), dtype=torch.bool, device=device)
        ratio = self.mask_ratio
        if ratio == 0:
            return mask
        if ratio == 1:
            return torch.ones_like(mask)
        mode_index = int(torch.randint(len(self.modes), (1,), device=device, generator=generator).item())
        mode = self.modes[mode_index]
        target_area = max(1, int(round(ratio * height * width)))
        if mode == "time":
            block_width = max(1, min(width, int(round(ratio * width))))
            start = self._sample_start(dimension=width, width=block_width, device=device, generator=generator)
            mask[:, start : start + block_width] = True
        elif mode == "subcarrier":
            block_height = max(1, min(height, int(round(ratio * height))))
            start = self._sample_start(dimension=height, width=block_height, device=device, generator=generator)
            mask[start : start + block_height, :] = True
        else:
            block_height = max(1, min(height, int(round(math.sqrt(target_area * height / width)))))
            block_width = max(1, min(width, int(round(target_area / block_height))))
            start_h = self._sample_start(
                dimension=height, width=block_height, device=device, generator=generator
            )
            start_w = self._sample_start(
                dimension=width, width=block_width, device=device, generator=generator
            )
            mask[start_h : start_h + block_height, start_w : start_w + block_width] = True
        return mask

    def sample(self, x: Tensor, generator: torch.Generator) -> Tensor:
        """Generate a boolean ``(B,1,H,W)`` mask shared by all antennas."""

        _validate_csi(x)
        _validate_generator_device(generator, x.device)
        masks = [
            self._sample_one(x.shape[2], x.shape[3], x.device, generator)
            for _ in range(x.shape[0])
        ]
        return torch.stack(masks, dim=0).unsqueeze(1)

    def apply(self, x: Tensor, mask: Tensor) -> Tensor:
        """Zero the masked locations while preserving the antenna dimension."""

        _validate_csi(x)
        if mask.shape != (x.shape[0], 1, x.shape[2], x.shape[3]) or mask.dtype != torch.bool:
            raise ValueError(
                "mask must have shape (B,1,H,W) and boolean dtype, "
                f"got shape={tuple(mask.shape)}, dtype={mask.dtype}"
            )
        if mask.device != x.device:
            raise ValueError("mask and CSI must use the same device")
        return x.masked_fill(mask.expand(-1, x.shape[1], -1, -1), 0)

    def __call__(self, x: Tensor, generator: torch.Generator) -> Tensor:
        return self.sample(x, generator)


def _validate_feature_pair(predicted: Tensor, target: Tensor, mask: Tensor) -> None:
    if predicted.ndim != 4 or target.shape != predicted.shape:
        raise ValueError("predicted and target feature maps must have the same rank-4 shape")
    if mask.ndim != 4 or mask.shape[0] != predicted.shape[0] or mask.shape[1] != 1:
        raise ValueError("feature mask must have shape (B,1,H,W)")
    if mask.device != predicted.device:
        raise ValueError("feature mask and feature maps must use the same device")


def feature_mask_to_feature_map(input_mask: Tensor, feature_shape: tuple[int, ...]) -> Tensor:
    """将三天线共享 CSI 掩码映射到官方单分支 ResNet 特征图。

    官方骨架会在输入端把三根天线沿宽度拼接为 30 列，再缩放为网络输入。
    因此先把共享掩码复制三次，再直接缩放到 encoder 的 (17, 4) 特征图。
    """

    if input_mask.ndim != 4 or input_mask.shape[1] != 1:
        raise ValueError("input_mask must have shape (B,1,H,W)")
    if len(feature_shape) != 4 or input_mask.shape[0] != feature_shape[0]:
        raise ValueError("input mask and feature map batch sizes must match")
    feature_height, feature_width = feature_shape[-2:]
    if feature_height < 1 or feature_width < 1:
        raise ValueError("feature map dimensions must be positive")
    official_input_mask = input_mask.float().repeat(1, 1, 1, 3)
    return F.interpolate(official_input_mask, size=(feature_height, feature_width), mode="nearest")


def feature_reconstruction_loss(
    predicted: Tensor,
    target: Tensor,
    input_mask: Tensor,
    *,
    masked_region_weight: float = 2.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return whole-map cosine, weighted MSE, and their feature objective."""

    _validate_feature_pair(predicted, target, input_mask)
    if not isinstance(masked_region_weight, (int, float)) or isinstance(masked_region_weight, bool):
        raise TypeError("masked_region_weight must be a finite scalar")
    masked_region_weight = float(masked_region_weight)
    if not math.isfinite(masked_region_weight) or masked_region_weight < 1.0:
        raise ValueError("masked_region_weight must be finite and at least 1")
    predicted_f = predicted.float()
    target_f = target.float().detach()
    downsampled = feature_mask_to_feature_map(input_mask, tuple(predicted.shape))
    weights = 1.0 + (masked_region_weight - 1.0) * downsampled
    mse = ((predicted_f - target_f).square() * weights).mean()
    cosine = 1.0 - F.cosine_similarity(
        predicted_f.flatten(1), target_f.flatten(1), dim=1, eps=1e-8
    ).mean()
    return cosine + 0.1 * mse, cosine, mse


class _RawCSIDecoder(nn.Module):
    """Small feature-map decoder used only by the explicit raw/hybrid variants."""

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim < 1:
            raise ValueError("raw_decoder_hidden_dim must be a positive integer")
        self.net = nn.Sequential(
            nn.Conv2d(512, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 3, kernel_size=1),
        )

    def forward(self, feature_map: Tensor, output_size: tuple[int, int]) -> Tensor:
        x = self.net(feature_map)
        return F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)


class MetaFiMFM(MetaFiSSLMethod):
    """MetaFi masked feature modeling with a frozen EMA teacher encoder."""

    _EXTRA_STATE_SCHEMA_VERSION = 1

    def __init__(
        self,
        encoder: MetaFiEncoder,
        *,
        objective: MFMObjective = "feature",
        mask: StructuredCSIMask | Mapping[str, object] | None = None,
        mask_ratio: float = 0.40,
        mask_modes: tuple[str, ...] = ("joint",),
        augmentation_seed: int = 0,
        ema_momentum: float = 0.996,
        ema_schedule: str = "constant",
        predictor_hidden_dim: int = 256,
        projector_hidden_dim: int | None = None,
        masked_region_weight: float = 2.0,
        collapse_std_threshold: float = 1e-6,
        collapse_window: int = 3,
        raw_decoder_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if not isinstance(encoder, MetaFiEncoder):
            raise TypeError("encoder must be a MetaFiEncoder")
        if objective not in ("feature", "raw", "hybrid"):
            raise ValueError("objective must be one of: feature, raw, hybrid")
        if not isinstance(augmentation_seed, int) or isinstance(augmentation_seed, bool):
            raise TypeError("augmentation_seed must be an integer")
        if not isinstance(ema_momentum, (int, float)) or isinstance(ema_momentum, bool):
            raise TypeError("ema_momentum must be a finite scalar in [0,1)")
        ema_momentum = float(ema_momentum)
        if not math.isfinite(ema_momentum) or not 0.0 <= ema_momentum < 1.0:
            raise ValueError("ema_momentum must be a finite scalar in [0,1)")
        if ema_schedule != "constant":
            raise ValueError("only the constant EMA momentum schedule is supported")
        if projector_hidden_dim is not None:
            predictor_hidden_dim = projector_hidden_dim
        if not isinstance(collapse_std_threshold, (int, float)) or isinstance(collapse_std_threshold, bool):
            raise TypeError("collapse_std_threshold must be a finite non-negative scalar")
        collapse_std_threshold = float(collapse_std_threshold)
        if not math.isfinite(collapse_std_threshold) or collapse_std_threshold < 0:
            raise ValueError("collapse_std_threshold must be a finite non-negative scalar")
        if isinstance(collapse_window, bool) or not isinstance(collapse_window, int) or collapse_window < 1:
            raise ValueError("collapse_window must be a positive integer")

        self.objective: MFMObjective = objective
        self.augmentation_seed = augmentation_seed
        self.ema_momentum = ema_momentum
        self.ema_schedule = ema_schedule
        self.mask = StructuredCSIMask(mask_ratio, mask_modes) if mask is None else StructuredCSIMask.from_mapping(mask)
        self.masked_region_weight = float(masked_region_weight)
        if not math.isfinite(self.masked_region_weight) or self.masked_region_weight < 1.0:
            raise ValueError("masked_region_weight must be finite and at least 1")
        self.collapse_std_threshold = collapse_std_threshold
        self.collapse_window = collapse_window

        self.encoder = encoder
        self.teacher_encoder = copy.deepcopy(encoder)
        self.teacher_encoder.requires_grad_(False)
        self.teacher_encoder.eval()
        self.predictor = nn.Sequential(
            nn.Conv2d(512, predictor_hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(predictor_hidden_dim, 512, kernel_size=1),
        )
        self.raw_decoder = (
            _RawCSIDecoder(raw_decoder_hidden_dim)
            if objective in ("raw", "hybrid")
            else None
        )
        self.register_buffer("ema_step", torch.zeros((), dtype=torch.long))
        self.register_buffer("collapse_streak", torch.zeros((), dtype=torch.long))
        self._augmentation_generators: dict[str, torch.Generator] = {}
        self._pending_augmentation_generator_states: dict[str, Tensor] = {}

    def train(self, mode: bool = True) -> "MetaFiMFM":
        super().train(mode)
        self.teacher_encoder.eval()
        return self

    @staticmethod
    def _generator_key(device: torch.device) -> str:
        return str(torch.device(device))

    @staticmethod
    def _validate_generator_state(key: object, state: object) -> Tensor:
        if not isinstance(key, str) or _DEVICE_KEY.fullmatch(key) is None:
            raise ValueError(f"invalid MetaFiMFM generator state device key: {key!r}")
        try:
            device = torch.device(key)
            generator = torch.Generator(device=device)
        except (RuntimeError, TypeError, ValueError) as error:
            raise ValueError(f"invalid MetaFiMFM generator state device key: {key!r}") from error
        if not isinstance(state, Tensor) or state.dtype != torch.uint8 or state.ndim != 1:
            raise ValueError(f"invalid MetaFiMFM generator state for device {key!r}")
        restored = state.detach().cpu().clone()
        try:
            generator.set_state(restored)
        except RuntimeError as error:
            raise ValueError(f"invalid MetaFiMFM generator state for device {key!r}") from error
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
                raise ValueError(f"invalid MetaFiMFM generator state for device {key!r}") from error
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
            "ema_schedule": self.ema_schedule,
            "ema_momentum": self.ema_momentum,
            "collapse_streak": int(self.collapse_streak.item()),
        }

    def set_extra_state(self, state: object) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("MetaFiMFM extra state must be a mapping")
        if state.get("schema_version") != self._EXTRA_STATE_SCHEMA_VERSION:
            raise ValueError("unsupported MetaFiMFM extra state schema")
        if state.get("ema_schedule") != self.ema_schedule:
            raise ValueError("MetaFiMFM EMA schedule mismatch")
        try:
            momentum = float(state.get("ema_momentum"))
        except (TypeError, ValueError) as error:
            raise ValueError("MetaFiMFM extra state requires ema_momentum") from error
        if not math.isclose(momentum, self.ema_momentum, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("MetaFiMFM EMA momentum mismatch")
        raw_states = state.get("generator_states")
        if not isinstance(raw_states, Mapping):
            raise ValueError("MetaFiMFM extra state requires generator_states")
        restored = {
            key: self._validate_generator_state(key, value)
            for key, value in raw_states.items()
        }
        streak = state.get("collapse_streak")
        if isinstance(streak, bool) or not isinstance(streak, int) or streak < 0:
            raise ValueError("MetaFiMFM collapse_streak must be a non-negative integer")
        self._augmentation_generators = {}
        self._pending_augmentation_generator_states = dict(sorted(restored.items()))
        self.collapse_streak.fill_(streak)

    @staticmethod
    def _encode(encoder: MetaFiEncoder, csi: Tensor) -> Tensor:
        _validate_csi(csi)
        return encoder(csi.unsqueeze(1)).feature_map

    @staticmethod
    def _masked_raw_loss(predicted: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        if predicted.shape != target.shape:
            raise ValueError("raw decoder output and CSI target must have the same shape")
        expanded = mask.expand(-1, target.shape[1], -1, -1).float()
        denominator = expanded.sum().clamp_min(1.0)
        return ((predicted.float() - target.float().detach()).square() * expanded).sum() / denominator

    def _diagnose(self, teacher_features: Tensor, student_features: Tensor) -> tuple[float, float, float]:
        teacher_std_tensor = teacher_features.float().std(unbiased=False)
        student_std_tensor = student_features.float().std(unbiased=False)
        if not torch.isfinite(teacher_std_tensor) or not torch.isfinite(student_std_tensor):
            raise ValueError("MetaFiMFM feature standard deviation must be finite")
        teacher_std = float(teacher_std_tensor.detach())
        student_std = float(student_std_tensor.detach())
        low = (
            self.collapse_std_threshold > 0
            and (teacher_std < self.collapse_std_threshold or student_std < self.collapse_std_threshold)
        )
        if low:
            self.collapse_streak.add_(1)
        else:
            self.collapse_streak.zero_()
        warning = float(int(self.collapse_streak.item()) >= self.collapse_window)
        return teacher_std, student_std, warning

    def ema_momentum_for_step(self) -> float:
        return self.ema_momentum

    @torch.no_grad()
    def ema_update(self) -> None:
        """Move teacher parameters/buffers toward the current student state."""

        momentum = self.ema_momentum_for_step()
        teacher_parameters = dict(self.teacher_encoder.named_parameters())
        student_parameters = dict(self.encoder.named_parameters())
        if teacher_parameters.keys() != student_parameters.keys():
            raise RuntimeError("student and teacher encoder parameter structures differ")
        for name, teacher_parameter in teacher_parameters.items():
            teacher_parameter.mul_(momentum).add_(student_parameters[name].detach(), alpha=1.0 - momentum)
        teacher_buffers = dict(self.teacher_encoder.named_buffers())
        student_buffers = dict(self.encoder.named_buffers())
        if teacher_buffers.keys() != student_buffers.keys():
            raise RuntimeError("student and teacher encoder buffer structures differ")
        for name, teacher_buffer in teacher_buffers.items():
            student_buffer = student_buffers[name].detach()
            if torch.is_floating_point(teacher_buffer):
                teacher_buffer.mul_(momentum).add_(student_buffer, alpha=1.0 - momentum)
            else:
                teacher_buffer.copy_(student_buffer)
        self.ema_step.add_(1)
        self.teacher_encoder.eval()

    update_teacher = ema_update
    momentum_update = ema_update

    def forward(
        self,
        batch: PretrainBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> SSLStepOutput:
        if not isinstance(batch, PretrainBatch):
            raise TypeError("batch must be a PretrainBatch")
        if batch.anchor.shape[0] < 2:
            raise ValueError("MetaFiMFM requires batch size >= 2 for stable BatchNorm statistics")
        anchor = batch.anchor
        _validate_csi(anchor)
        if generator is None:
            generator = self._default_generator(anchor.device)
        else:
            _validate_generator_device(generator, anchor.device)
        input_mask = self.mask.sample(anchor, generator)
        masked_anchor = self.mask.apply(anchor, input_mask)
        zero = anchor.new_zeros(())
        teacher_std = 0.0
        student_std = 0.0
        collapse_warning = 0.0
        cosine_loss = zero
        mse_loss = zero
        feature_loss = zero
        student_features: Tensor | None = None

        if self.objective in ("feature", "hybrid"):
            with torch.no_grad():
                self.teacher_encoder.eval()
                teacher_features = self._encode(self.teacher_encoder, anchor)
            student_features = self._encode(self.encoder, masked_anchor)
            predicted_features = self.predictor(student_features)
            teacher_std, student_std, collapse_warning = self._diagnose(
                teacher_features, student_features
            )
            if not torch.isfinite(predicted_features).all():
                raise ValueError("MetaFiMFM predicted features must be finite")

            feature_loss, cosine_loss, mse_loss = feature_reconstruction_loss(
                predicted_features,
                teacher_features,
                input_mask,
                masked_region_weight=self.masked_region_weight,
            )
        if self.objective in ("raw", "hybrid"):
            assert self.raw_decoder is not None
            if student_features is None:
                student_features = self._encode(self.encoder, masked_anchor)
            raw_prediction = self.raw_decoder(
                student_features, (anchor.shape[2], anchor.shape[3])
            )
            raw_loss = self._masked_raw_loss(raw_prediction, anchor, input_mask)
        else:
            raw_loss = zero
        if self.objective == "feature":
            loss = feature_loss
        elif self.objective == "raw":
            loss = raw_loss
        else:
            loss = feature_loss + 0.1 * raw_loss
        if not torch.isfinite(loss):
            raise ValueError("MetaFiMFM loss must be finite")
        return SSLStepOutput(
            loss=loss,
            metrics={
                "loss": float(loss.detach()),
                "teacher_feature_std": teacher_std,
                "student_feature_std": student_std,
                "cosine_loss": float(cosine_loss.detach()),
                "mse_loss": float(mse_loss.detach()),
                "raw_loss": float(raw_loss.detach()),
                "mask_ratio": float(input_mask.float().mean().detach()),
                "ema_momentum": self.ema_momentum,
                "collapse_warning": collapse_warning,
                "low_std_streak": float(self.collapse_streak.item()),
                "feature_objective_computed": float(self.objective in ("feature", "hybrid")),
            },
        )

    def export_encoder_state_dict(self) -> dict[str, Tensor]:
        """Export only the trainable student encoder for downstream fine-tuning."""

        return {name: value.detach().clone() for name, value in self.encoder.state_dict().items()}
