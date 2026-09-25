"""Relative-position prediction adapted to the complete MetaFi-R34 encoder.

The task uses a fixed 3x3 CSI grid.  Time is right-padded from 10 to 12 so
that every grid cell has width four; the center cell and one labelled
surrounding cell are resized independently back to the MetaFi CSI geometry.
Antenna channels are never shuffled or mixed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mmfi_wifi.metafi_encoder import MetaFiEncoder

from .base import MetaFiSSLMethod, SSLStepOutput
from .pretrain_data import PretrainBatch


GRID_SIZE = 3
CENTER_POSITION = (1, 1)
# Row-major 3x3 positions excluding the center.  The index is the classifier
# target and is intentionally frozen by exhaustive coordinate-coded tests.
RELATIVE_POSITIONS: tuple[tuple[int, int], ...] = tuple(
    (row, column)
    for row in range(GRID_SIZE)
    for column in range(GRID_SIZE)
    if (row, column) != CENTER_POSITION
)
N_RELATIVE_CLASSES = len(RELATIVE_POSITIONS)
_INPUT_HEIGHT = 114
_INPUT_TIME = 10
_PADDED_TIME = 12
_GRID_HEIGHT = _INPUT_HEIGHT // GRID_SIZE
_GRID_TIME = _PADDED_TIME // GRID_SIZE


def _validate_csi(x: Tensor) -> None:
    if not isinstance(x, Tensor) or x.ndim != 4:
        raise ValueError("CSI must have shape (B, 3, 114, 10)")
    if x.shape[0] < 1 or tuple(x.shape[1:]) != (3, _INPUT_HEIGHT, _INPUT_TIME):
        raise ValueError(
            "CSI must have shape (B, 3, 114, 10), "
            f"got {tuple(x.shape)}"
        )
    if not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError("CSI must contain only finite floating-point values")


def _validate_class_index(class_index: int) -> int:
    if isinstance(class_index, bool) or not isinstance(class_index, int):
        raise TypeError("class_index must be an integer in [0, 7]")
    if not 0 <= class_index < N_RELATIVE_CLASSES:
        raise ValueError("class_index must be an integer in [0, 7]")
    return class_index


def _resize_grid_cell(x_padded: Tensor, row: int, column: int) -> Tensor:
    patch = x_padded[
        :,
        :,
        row * _GRID_HEIGHT : (row + 1) * _GRID_HEIGHT,
        column * _GRID_TIME : (column + 1) * _GRID_TIME,
    ]
    # Bilinear interpolation acts independently per N/C plane, so it cannot
    # mix the three physical antenna channels.
    return F.interpolate(
        patch,
        size=(_INPUT_HEIGHT, _INPUT_TIME),
        mode="bilinear",
        align_corners=False,
    )


def extract_relative_patches(x: Tensor, class_index: int) -> tuple[Tensor, Tensor]:
    """Return resized center and labelled-neighbor CSI patches.

    ``class_index`` uses :data:`RELATIVE_POSITIONS`, a row-major ordering of
    the eight non-center cells in a 3x3 grid.  The input time dimension is
    right-padded from 10 to 12 before splitting into 38x4 source cells.
    """

    _validate_csi(x)
    class_index = _validate_class_index(class_index)
    x_padded = F.pad(x, (0, _PADDED_TIME - _INPUT_TIME))
    center = _resize_grid_cell(x_padded, *CENTER_POSITION)
    neighbor = _resize_grid_cell(x_padded, *RELATIVE_POSITIONS[class_index])
    return center, neighbor


def _validate_generator_device(generator: torch.Generator, device: torch.device) -> None:
    """Require an exact, concrete generator/input device match."""

    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator")
    generator_device = torch.device(generator.device)
    input_device = torch.device(device)
    if generator_device.type == "cuda" and generator_device.index is None:
        raise ValueError(
            "generator.device must identify an exact CUDA device such as 'cuda:0', "
            "not bare 'cuda'"
        )
    if str(generator_device) != str(input_device):
        raise ValueError(
            "generator.device must match the input tensor device exactly; "
            f"got generator.device={generator_device} and input tensor device={input_device}."
        )


def _validate_generator_state(key: object, state: object) -> Tensor:
    """Validate a canonical device key and a usable serialized RNG state."""

    if not isinstance(key, str):
        raise ValueError("RelPos generator-state key must be a string")
    try:
        device = torch.device(key)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"invalid RelPos generator-state key {key!r}") from error
    if key != str(device):
        raise ValueError(
            "RelPos generator-state key must be canonical: "
            f"expected {str(device)!r}, got {key!r}"
        )
    if device.type == "cuda" and device.index is None:
        raise ValueError(
            "RelPos generator-state key must identify an exact CUDA device "
            "such as 'cuda:0', not bare 'cuda'"
        )
    if not isinstance(state, Tensor) or state.dtype != torch.uint8 or state.ndim != 1:
        raise ValueError(
            "RelPos generator states must be rank-1 uint8 tensors"
        )

    restored_state = state.detach().cpu().clone()
    try:
        generator = torch.Generator(device=device)
        if str(torch.device(generator.device)) != key:
            raise ValueError(
                "RelPos generator-state key does not reconstruct exactly: "
                f"expected {str(torch.device(generator.device))!r}, got {key!r}"
            )
        generator.set_state(restored_state)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError(
            f"invalid RelPos generator state for device {key!r}"
        ) from error
    return restored_state


class RelPosMetaFi(MetaFiSSLMethod):
    """Predict a fixed relative CSI-grid position with full MetaFi encoders.

    The primary loss is 8-way relative-position cross entropy.  Optional
    temporal consistency compares the anchor logits with logits computed from
    only valid same-sequence neighboring frames; unavailable edge placeholders
    are never encoded.
    """

    def __init__(
        self,
        encoder: MetaFiEncoder,
        *,
        hidden_dim: int = 256,
        temporal_consistency: bool = False,
        temporal_coefficient: float = 0.2,
        augmentation_config: object | None = None,
        augmentation_seed: int | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(encoder, MetaFiEncoder):
            raise TypeError("encoder must be a MetaFiEncoder")
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim < 1:
            raise ValueError("hidden_dim must be a positive integer")
        if not isinstance(temporal_consistency, bool):
            raise TypeError("temporal_consistency must be a bool")
        if not isinstance(temporal_coefficient, Real) or isinstance(temporal_coefficient, bool):
            raise TypeError("temporal_coefficient must be a finite non-negative scalar")
        if not math.isfinite(float(temporal_coefficient)) or float(temporal_coefficient) < 0:
            raise ValueError("temporal_coefficient must be a finite non-negative scalar")
        # RelPos intentionally uses no stochastic CSI augmentation: changing
        # grid-cell content would confound its coordinate label.  Rejecting
        # non-empty augmentation settings prevents callers from assuming an
        # unimplemented transform was applied.
        if augmentation_config not in (None, {}):
            raise ValueError("RelPosMetaFi does not support CSI augmentation")
        if augmentation_seed is None:
            augmentation_seed = 0
        elif isinstance(augmentation_seed, bool) or not isinstance(augmentation_seed, int):
            raise TypeError("augmentation_seed must be an integer when provided")

        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(encoder.feature_channels * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, N_RELATIVE_CLASSES),
        )
        self.temporal_consistency = temporal_consistency
        self.temporal_coefficient = float(temporal_coefficient)
        self.augmentation_seed = augmentation_seed
        # Class-label sampling is stochastic method state even though RelPos
        # intentionally applies no CSI augmentation.  Keep one stream per
        # tensor device so the default forward(batch) API is reproducible.
        self._class_sampling_generators: dict[str, torch.Generator] = {}
        self._pending_class_sampling_generator_states: dict[str, Tensor] = {}

    @staticmethod
    def _generator_key(device: torch.device) -> str:
        return str(torch.device(device))

    def _make_generator(self, device: torch.device) -> torch.Generator:
        normalized = torch.device(device)
        key = self._generator_key(normalized)
        generator = torch.Generator(device=normalized)
        pending_state = self._pending_class_sampling_generator_states.pop(key, None)
        if pending_state is None:
            generator.manual_seed(self.augmentation_seed)
        else:
            try:
                generator.set_state(pending_state)
            except (RuntimeError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid RelPos generator state for device {key!r}"
                ) from error
        return generator

    def _default_generator(self, device: torch.device) -> torch.Generator:
        key = self._generator_key(device)
        generator = self._class_sampling_generators.get(key)
        if generator is None:
            generator = self._make_generator(device)
            self._class_sampling_generators[key] = generator
        return generator

    def get_extra_state(self) -> dict[str, object]:
        """Serialize default class-sampling RNG streams."""

        generator_states = {
            key: generator.get_state().detach().cpu().clone()
            for key, generator in self._class_sampling_generators.items()
        }
        generator_states.update(
            {
                key: state.detach().cpu().clone()
                for key, state in self._pending_class_sampling_generator_states.items()
            }
        )
        return {
            "schema_version": 1,
            "generator_states": dict(sorted(generator_states.items())),
        }

    def set_extra_state(self, state: object) -> None:
        """Restore default class-sampling RNG streams fail-closed."""

        if not isinstance(state, Mapping):
            raise ValueError("RelPos extra state must be a mapping")
        if state.get("schema_version") != 1:
            raise ValueError("unsupported RelPos class-sampling RNG state schema")
        raw_generator_states = state.get("generator_states")
        if not isinstance(raw_generator_states, Mapping):
            raise ValueError("RelPos extra state requires generator_states")
        restored = {
            key: _validate_generator_state(key, value)
            for key, value in raw_generator_states.items()
        }
        self._class_sampling_generators = {}
        self._pending_class_sampling_generator_states = dict(sorted(restored.items()))

    @staticmethod
    def _encode_patch_pair(center: Tensor, neighbor: Tensor, encoder: MetaFiEncoder) -> Tensor:
        center_features = encoder(center.unsqueeze(1)).global_vector
        neighbor_features = encoder(neighbor.unsqueeze(1)).global_vector
        return torch.cat((center_features, neighbor_features), dim=1)

    def _logits_for(self, csi: Tensor, class_index: int) -> Tensor:
        center, neighbor = extract_relative_patches(csi, class_index)
        return self.classifier(self._encode_patch_pair(center, neighbor, self.encoder))

    @staticmethod
    def _validate_batch(batch: PretrainBatch) -> None:
        if not isinstance(batch, PretrainBatch):
            raise TypeError("batch must be a PretrainBatch")
        _validate_csi(batch.anchor)
        if batch.anchor.shape[0] < 2:
            raise ValueError("RelPosMetaFi requires batch size >= 2 for MetaFi encoder BatchNorm")
        if not (
            len(batch.neighbors)
            == len(batch.neighbor_offsets)
            == len(batch.neighbor_available)
        ):
            raise ValueError("PretrainBatch neighbor fields must have matching lengths")

    def _sample_class_index(
        self,
        *,
        device: torch.device,
        generator: torch.Generator | None,
        class_index: int | None,
    ) -> int:
        if class_index is not None:
            return _validate_class_index(class_index)
        if generator is None:
            generator = self._default_generator(device)
        else:
            _validate_generator_device(generator, device)
        sampled = torch.randint(
            N_RELATIVE_CLASSES,
            (1,),
            device=device,
            generator=generator,
        )
        return int(sampled.item())

    def _temporal_loss(self, batch: PretrainBatch, *, class_index: int, anchor_logits: Tensor) -> Tensor:
        if not self.temporal_consistency:
            return anchor_logits.new_zeros(())

        losses: list[Tensor] = []
        batch_size = batch.anchor.shape[0]
        for neighbor, available in zip(batch.neighbors, batch.neighbor_available, strict=True):
            if available.shape != (batch_size,):
                raise ValueError("neighbor availability must have shape (batch_size,)")
            available_indices = torch.nonzero(
                available.to(device=batch.anchor.device, dtype=torch.bool),
                as_tuple=False,
            ).flatten()
            # A single compact neighbor cannot safely traverse the training-mode
            # MetaFi BatchNorm layers.  Skipping it is preferable to encoding a
            # cross-sequence/zero placeholder merely to pad the batch.
            if available_indices.numel() < 2:
                continue
            neighbor_csi = neighbor.index_select(0, available_indices)
            neighbor_logits = self._logits_for(neighbor_csi, class_index)
            target_probabilities = anchor_logits.index_select(0, available_indices).detach().softmax(dim=1)
            losses.append(F.mse_loss(neighbor_logits.softmax(dim=1), target_probabilities))
        if not losses:
            return anchor_logits.new_zeros(())
        return torch.stack(losses).mean()

    def forward(
        self,
        batch: PretrainBatch,
        *,
        class_index: int | None = None,
        generator: torch.Generator | None = None,
    ) -> SSLStepOutput:
        """Compute relative-position CE and optional same-sequence consistency."""

        self._validate_batch(batch)
        selected_class = self._sample_class_index(
            device=batch.anchor.device,
            generator=generator,
            class_index=class_index,
        )
        logits = self._logits_for(batch.anchor, selected_class)
        labels = torch.full(
            (batch.anchor.shape[0],),
            selected_class,
            device=logits.device,
            dtype=torch.long,
        )
        relative_position_loss = F.cross_entropy(logits, labels)
        temporal_loss = self._temporal_loss(
            batch,
            class_index=selected_class,
            anchor_logits=logits,
        )
        loss = relative_position_loss + self.temporal_coefficient * temporal_loss
        return SSLStepOutput(
            loss=loss,
            metrics={
                "loss": float(loss.detach()),
                "relative_position_loss": float(relative_position_loss.detach()),
                "temporal_loss": float(temporal_loss.detach()),
                "temporal_enabled": float(self.temporal_consistency),
                "position_class": float(selected_class),
            },
        )

    def export_encoder_state_dict(self) -> dict[str, Tensor]:
        return {name: value.detach().clone() for name, value in self.encoder.state_dict().items()}