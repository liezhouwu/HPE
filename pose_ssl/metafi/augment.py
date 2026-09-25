"""Position-preserving CSI augmentations for MetaFi-R34 SSL pretraining.

The three CSI antenna channels have fixed physical identities.  This module
therefore never reorders channels and applies spatial masks and time shifts
synchronously to every antenna of each sample.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Mapping, Protocol

import torch
from torch import Tensor


class _AugmentCallable(Protocol):
    def forward(self, x: Tensor, generator: torch.Generator) -> Tensor: ...


@dataclass(frozen=True)
class PositionPreservingAugmentConfig:
    """Validated settings for the main positioning-safe augmentation policy."""

    jitter_sigma: float = 0.001
    amplitude_scale_range: tuple[float, float] = (0.95, 1.05)
    time_mask_width: int = 1
    subcarrier_mask_width: int = 4
    max_time_shift: int = 1

    @classmethod
    def from_mapping(
        cls, config: Mapping[str, object] | "PositionPreservingAugmentConfig"
    ) -> "PositionPreservingAugmentConfig":
        if isinstance(config, cls):
            return config
        if not isinstance(config, Mapping):
            raise TypeError("augmentation config must be a mapping")
        if config.get("channel_shuffle", False):
            raise ValueError(
                "channel_shuffle is forbidden in the main positioning-preserving "
                "augmentation policy"
            )

        allowed = {
            "jitter_sigma",
            "amplitude_scale_range",
            "time_mask_width",
            "subcarrier_mask_width",
            "max_time_shift",
            "channel_shuffle",
        }
        unknown = set(config).difference(allowed)
        if unknown:
            raise ValueError(f"unknown main augmentation config fields: {sorted(unknown)!r}")

        scale_range = config.get("amplitude_scale_range", cls.amplitude_scale_range)
        # YAML 没有 tuple 字面量，允许用户用 [min, max] 配置范围。
        if isinstance(scale_range, list):
            scale_range = tuple(scale_range)
        values = {
            "jitter_sigma": config.get("jitter_sigma", cls.jitter_sigma),
            "amplitude_scale_range": scale_range,
            "time_mask_width": config.get("time_mask_width", cls.time_mask_width),
            "subcarrier_mask_width": config.get(
                "subcarrier_mask_width", cls.subcarrier_mask_width
            ),
            "max_time_shift": config.get("max_time_shift", cls.max_time_shift),
        }
        result = cls(**values)  # type: ignore[arg-type]
        result._validate()
        return result

    def _validate(self) -> None:
        if not isinstance(self.jitter_sigma, Real) or not math.isfinite(self.jitter_sigma):
            raise ValueError("jitter_sigma must be finite")
        if self.jitter_sigma < 0:
            raise ValueError("jitter_sigma must be non-negative")

        scale_range = self.amplitude_scale_range
        if not isinstance(scale_range, tuple) or len(scale_range) != 2:
            raise ValueError("amplitude_scale_range must be a (min, max) tuple")
        lower, upper = scale_range
        if not all(isinstance(value, Real) and math.isfinite(value) for value in (lower, upper)):
            raise ValueError("amplitude_scale_range values must be finite")
        if lower <= 0 or upper < lower:
            raise ValueError("amplitude_scale_range must be positive and ordered")

        for name, value in (
            ("time_mask_width", self.time_mask_width),
            ("subcarrier_mask_width", self.subcarrier_mask_width),
            ("max_time_shift", self.max_time_shift),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


class PositionPreservingAugment:
    """Apply main MetaFi SSL transforms without changing antenna identity.

    Inputs are batches shaped ``(B, 3, subcarriers, time)``.  Random choices
    are per sample, but a choice is shared by all three antenna channels.
    """

    def __init__(
        self, config: Mapping[str, object] | PositionPreservingAugmentConfig
    ) -> None:
        self.config = PositionPreservingAugmentConfig.from_mapping(config)

    def forward(self, x: Tensor, generator: torch.Generator) -> Tensor:
        """Return one deterministic augmented CSI view using ``generator``."""

        self._validate_input(x, generator)
        result = x.clone()
        result = self._apply_jitter(result, generator)
        result = self._apply_amplitude_scaling(result, generator)
        result = self._apply_time_mask(result, generator)
        result = self._apply_subcarrier_mask(result, generator)
        return self._apply_time_shift(result, generator)

    def __call__(self, x: Tensor, generator: torch.Generator) -> Tensor:
        return self.forward(x, generator)

    @staticmethod
    def _validate_input(x: Tensor, generator: torch.Generator) -> None:
        if not isinstance(x, Tensor) or x.ndim != 4:
            raise ValueError("CSI batch must have shape (B, 3, subcarriers, time)")
        if x.shape[0] < 1 or x.shape[1] != 3 or x.shape[2] < 1 or x.shape[3] < 1:
            raise ValueError("CSI batch must have non-empty B/3/H/W dimensions")
        if not x.is_floating_point():
            raise ValueError("CSI batch must use a floating-point dtype")
        if not isinstance(generator, torch.Generator):
            raise TypeError("generator must be a torch.Generator")

        generator_device = torch.device(generator.device)
        input_device = x.device
        devices_match = generator_device.type == input_device.type and (
            generator_device.index is None
            or generator_device.index == input_device.index
        )
        if not devices_match:
            raise ValueError(
                "generator.device must match the input tensor device; "
                f"got generator.device={generator_device} and input tensor "
                f"device={input_device}. Create the generator with "
                "torch.Generator(device=x.device)."
            )

    @staticmethod
    def _sample_starts(
        *, count: int, dimension: int, width: int, device: torch.device, generator: torch.Generator
    ) -> Tensor:
        if width > dimension:
            raise ValueError(f"mask width {width} exceeds input dimension {dimension}")
        return torch.randint(
            0,
            dimension - width + 1,
            (count,),
            device=device,
            generator=generator,
        )

    def _apply_jitter(self, x: Tensor, generator: torch.Generator) -> Tensor:
        sigma = self.config.jitter_sigma
        if sigma == 0:
            return x
        return x + torch.randn(
            x.shape, dtype=x.dtype, device=x.device, generator=generator
        ) * sigma

    def _apply_amplitude_scaling(self, x: Tensor, generator: torch.Generator) -> Tensor:
        lower, upper = self.config.amplitude_scale_range
        if lower == upper == 1.0:
            return x
        factors = torch.rand(
            (x.shape[0], 1, 1, 1), dtype=x.dtype, device=x.device, generator=generator
        )
        factors = factors * (upper - lower) + lower
        return x * factors

    def _apply_time_mask(self, x: Tensor, generator: torch.Generator) -> Tensor:
        width = self.config.time_mask_width
        if width == 0:
            return x
        starts = self._sample_starts(
            count=x.shape[0], dimension=x.shape[-1], width=width, device=x.device, generator=generator
        )
        result = x.clone()
        for sample_index, start in enumerate(starts.tolist()):
            result[sample_index, :, :, start : start + width] = 0
        return result

    def _apply_subcarrier_mask(self, x: Tensor, generator: torch.Generator) -> Tensor:
        width = self.config.subcarrier_mask_width
        if width == 0:
            return x
        starts = self._sample_starts(
            count=x.shape[0], dimension=x.shape[-2], width=width, device=x.device, generator=generator
        )
        result = x.clone()
        for sample_index, start in enumerate(starts.tolist()):
            result[sample_index, :, start : start + width, :] = 0
        return result

    def _apply_time_shift(self, x: Tensor, generator: torch.Generator) -> Tensor:
        maximum = self.config.max_time_shift
        if maximum == 0:
            return x
        # Draw only non-zero shifts.  ``max_time_shift=1`` is exactly a ±1 shift.
        raw = torch.randint(
            0, 2 * maximum, (x.shape[0],), device=x.device, generator=generator
        )
        shifts = raw - maximum
        shifts = torch.where(shifts >= 0, shifts + 1, shifts)

        result = torch.zeros_like(x)
        for sample_index, shift in enumerate(shifts.tolist()):
            if shift > 0:
                result[sample_index, :, :, shift:] = x[sample_index, :, :, :-shift]
            else:
                result[sample_index, :, :, :shift] = x[sample_index, :, :, -shift:]
        return result


def build_two_views(
    x: Tensor, augment: _AugmentCallable, generator: torch.Generator
) -> tuple[Tensor, Tensor]:
    """Build two independently augmented, reproducible views from one CSI batch."""

    return augment.forward(x, generator), augment.forward(x, generator)
