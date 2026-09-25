"""Position-preserving SwAV for the complete MetaFi-R34 encoder."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mmfi_wifi.metafi_encoder import MetaFiEncoder

from .augment import PositionPreservingAugment, PositionPreservingAugmentConfig
from .base import MetaFiSSLMethod, SSLStepOutput
from .pretrain_data import PretrainBatch
from .projectors import Projector
from .temporal_positive import temporal_weight


_VIEW_COUNTS = {"8gb": (2, 2), "high_resource": (2, 4)}
_DEFAULT_LOCAL_MASK_CONFIG: dict[str, object] = {
    "time_mask_width": 2,
    "subcarrier_mask_width": 8,
}


def _positive_finite(name: str, value: object) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a positive finite scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite scalar")
    return result


def _nonnegative_finite(name: str, value: object) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a finite non-negative scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite non-negative scalar")
    return result


def _generator_key(device: torch.device) -> str:
    return str(torch.device(device))


def _validate_generator_state(key: object, state: object) -> Tensor:
    if not isinstance(key, str):
        raise ValueError("augmentation generator-state key must be a string")
    try:
        device = torch.device(key)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"invalid augmentation generator-state key {key!r}") from error
    if key != str(device):
        raise ValueError("augmentation generator-state key must be canonical")
    if device.type == "cuda" and device.index is None:
        raise ValueError("augmentation generator-state key must identify an exact CUDA device")
    if not isinstance(state, Tensor) or state.dtype != torch.uint8 or state.ndim != 1:
        raise ValueError("augmentation generator states must be rank-1 uint8 tensors")
    restored = state.detach().cpu().clone()
    try:
        generator = torch.Generator(device=device)
        if str(generator.device) != key:
            raise ValueError("generator device does not reconstruct exactly")
        generator.set_state(restored)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"invalid augmentation generator state for device {key!r}") from error
    return restored


@dataclass(frozen=True)
class SwAVViewPolicy:
    """Resource-bound number of global and local positioning-safe views."""

    global_views: int
    local_views: int
    local_mask_config: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.global_views != 2:
            raise ValueError("SwAV requires exactly two global views")
        if isinstance(self.local_views, bool) or not isinstance(self.local_views, int):
            raise TypeError("local_views must be an integer")
        if self.local_views < 1:
            raise ValueError("local_views must be at least one")
        if not isinstance(self.local_mask_config, Mapping):
            raise TypeError("local_mask_config must be a mapping")
        config = dict(self.local_mask_config)
        PositionPreservingAugmentConfig.from_mapping(config)
        object.__setattr__(self, "local_mask_config", MappingProxyType(config))

    @classmethod
    def from_resource_profile(
        cls,
        profile: str,
        *,
        local_mask_config: Mapping[str, object] | None = None,
    ) -> "SwAVViewPolicy":
        try:
            global_views, local_views = _VIEW_COUNTS[profile]
        except KeyError as error:
            valid = ", ".join(sorted(_VIEW_COUNTS))
            raise ValueError(f"unknown SwAV resource profile {profile!r}; valid profiles: {valid}") from error
        return cls(
            global_views=global_views,
            local_views=local_views,
            local_mask_config=(
                _DEFAULT_LOCAL_MASK_CONFIG if local_mask_config is None else local_mask_config
            ),
        )


def sinkhorn_assignments(scores: Tensor, *, epsilon: float, iterations: int) -> Tensor:
    """Port the repository's current balanced Sinkhorn assignment convention.

    As in :func:`pose_ssl.pretrain_methods.sinkhorn`, the returned matrix has
    prototype columns with unit mass and sample rows with mass ``K / B``.
    This scale is deliberately retained because the existing SwAV objective
    consumes these assignments directly in swapped cross entropy.
    """

    if not isinstance(scores, Tensor) or scores.ndim != 2:
        raise ValueError("scores must be a rank-2 tensor")
    batch_size, prototypes = scores.shape
    if batch_size < 1 or prototypes < 1:
        raise ValueError("scores must have non-empty dimensions")
    if not torch.isfinite(scores).all():
        raise ValueError("scores must be finite")
    epsilon = _positive_finite("epsilon", epsilon)
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")

    work_dtype = torch.float64 if scores.dtype == torch.float64 else torch.float32
    scaled = (scores / epsilon).to(work_dtype)
    # Subtracting a scalar is mathematically neutral and prevents exp overflow.
    q = torch.exp(scaled - scaled.amax()).transpose(0, 1)
    q /= q.sum().clamp_min(torch.finfo(work_dtype).tiny)
    for _ in range(iterations):
        q /= q.sum(dim=0, keepdim=True).clamp_min(torch.finfo(work_dtype).tiny)
        q /= prototypes
        q /= q.sum(dim=1, keepdim=True).clamp_min(torch.finfo(work_dtype).tiny)
        q /= batch_size
    result = (q * batch_size).transpose(0, 1).to(scores.dtype)
    if not torch.isfinite(result).all():
        raise RuntimeError("Sinkhorn produced non-finite assignments")
    return result


def temporal_prototype_consistency(
    target_assignments: Tensor,
    neighbor_scores: Tensor,
    weights: Tensor,
    *,
    temperature: float = 0.1,
) -> Tensor:
    """Weighted, detached-global-assignment cross entropy for neighbors."""

    if target_assignments.ndim != 2 or neighbor_scores.ndim != 2:
        raise ValueError("prototype tensors must be rank-2")
    if target_assignments.shape != neighbor_scores.shape:
        raise ValueError("prototype tensors must have matching shapes")
    if weights.shape != (target_assignments.shape[0],):
        raise ValueError("weights must have shape (batch_size,)")
    if not torch.isfinite(target_assignments).all() or not torch.isfinite(neighbor_scores).all():
        raise ValueError("prototype tensors must be finite")
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("weights must be finite and non-negative")
    temperature = _positive_finite("temperature", temperature)
    weight_sum = weights.sum()
    if float(weight_sum.detach()) == 0.0:
        return neighbor_scores.sum() * 0.0
    loss = -(target_assignments.detach() * F.log_softmax(neighbor_scores / temperature, dim=1)).sum(dim=1)
    return (loss * weights.to(loss.dtype)).sum() / weight_sum


class SwAVMetaFi(MetaFiSSLMethod):
    """SwAV with sequential complete-MetaFi view encoding."""

    def __init__(
        self,
        encoder: MetaFiEncoder,
        *,
        view_policy: SwAVViewPolicy | None = None,
        augmentation_config: Mapping[str, object] | PositionPreservingAugmentConfig | None = None,
        augmentation_seed: int = 0,
        projector_hidden_dim: int = 512,
        projector_out_dim: int = 128,
        n_prototypes: int = 256,
        temperature: float = 0.1,
        sinkhorn_epsilon: float = 0.05,
        sinkhorn_iterations: int = 3,
        temporal_lambda: float = 0.2,
        temporal_weights: Mapping[int, float] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(encoder, MetaFiEncoder):
            raise TypeError("encoder must be a MetaFiEncoder")
        if isinstance(augmentation_seed, bool) or not isinstance(augmentation_seed, int):
            raise TypeError("augmentation_seed must be an integer")
        if isinstance(n_prototypes, bool) or not isinstance(n_prototypes, int) or n_prototypes < 1:
            raise ValueError("n_prototypes must be a positive integer")
        if isinstance(sinkhorn_iterations, bool) or not isinstance(sinkhorn_iterations, int) or sinkhorn_iterations < 1:
            raise ValueError("sinkhorn_iterations must be a positive integer")
        if view_policy is None:
            view_policy = SwAVViewPolicy.from_resource_profile("8gb")
        if not isinstance(view_policy, SwAVViewPolicy):
            raise TypeError("view_policy must be a SwAVViewPolicy")
        raw_weights = {} if temporal_weights is None else temporal_weights
        if not isinstance(raw_weights, Mapping):
            raise TypeError("temporal_weights must be a mapping")
        weights = {distance: temporal_weight(distance, {distance: weight}) for distance, weight in raw_weights.items()}

        self.encoder = encoder
        self.projector = Projector(encoder.feature_channels, projector_hidden_dim, projector_out_dim)
        self.prototypes = nn.Linear(projector_out_dim, n_prototypes, bias=False)
        self.view_policy = view_policy
        self.global_augment = PositionPreservingAugment({} if augmentation_config is None else augmentation_config)
        self.local_augment = PositionPreservingAugment(view_policy.local_mask_config)
        self.augmentation_seed = augmentation_seed
        self.temperature = _positive_finite("temperature", temperature)
        self.sinkhorn_epsilon = _positive_finite("sinkhorn_epsilon", sinkhorn_epsilon)
        self.sinkhorn_iterations = sinkhorn_iterations
        self.temporal_lambda = _nonnegative_finite("temporal_lambda", temporal_lambda)
        self.temporal_weights = dict(sorted(weights.items()))
        self._augmentation_generators: dict[str, torch.Generator] = {}
        self._pending_augmentation_generator_states: dict[str, Tensor] = {}

    def _make_generator(self, device: torch.device) -> torch.Generator:
        key = _generator_key(device)
        generator = torch.Generator(device=device)
        pending = self._pending_augmentation_generator_states.pop(key, None)
        if pending is None:
            generator.manual_seed(self.augmentation_seed)
        else:
            generator.set_state(pending)
        return generator

    def _default_generator(self, device: torch.device) -> torch.Generator:
        key = _generator_key(device)
        if key not in self._augmentation_generators:
            self._augmentation_generators[key] = self._make_generator(device)
        return self._augmentation_generators[key]

    def get_extra_state(self) -> dict[str, object]:
        states = {key: generator.get_state().detach().cpu().clone() for key, generator in self._augmentation_generators.items()}
        states.update({key: value.detach().cpu().clone() for key, value in self._pending_augmentation_generator_states.items()})
        return {"schema_version": 1, "generator_states": dict(sorted(states.items()))}

    def set_extra_state(self, state: object) -> None:
        if not isinstance(state, Mapping) or state.get("schema_version") != 1:
            raise ValueError("invalid SwAV augmentation RNG extra state")
        raw_states = state.get("generator_states")
        if not isinstance(raw_states, Mapping):
            raise ValueError("SwAV augmentation RNG extra state needs generator_states")
        pending: dict[str, Tensor] = {}
        for key, value in raw_states.items():
            if key in pending:
                raise ValueError("duplicate SwAV augmentation generator-state key")
            pending[str(key)] = _validate_generator_state(key, value)
        self._augmentation_generators = {}
        self._pending_augmentation_generator_states = pending

    @staticmethod
    def _validate_csi(csi: Tensor) -> Tensor:
        if not isinstance(csi, Tensor) or csi.ndim != 4 or csi.shape[1] != 3:
            raise ValueError("SwAVMetaFi expects CSI shape (B, 3, H, W)")
        return csi

    def make_views(self, csi: Tensor, generator: torch.Generator) -> tuple[Tensor, ...]:
        source = self._validate_csi(csi)
        views = [self.global_augment(source, generator) for _ in range(self.view_policy.global_views)]
        views.extend(self.local_augment(source, generator) for _ in range(self.view_policy.local_views))
        return tuple(views)

    def _encode_project(self, csi: Tensor) -> Tensor:
        """Encode one CSI view. This method intentionally receives no concatenated views."""
        features = self.encoder(self._validate_csi(csi).unsqueeze(1)).global_vector
        if features.shape[0] == 1 and self.training:
            projected = self.projector(torch.cat((features, features), dim=0))[:1]
        else:
            projected = self.projector(features)
        return self.prototypes(F.normalize(projected, dim=1))

    @torch.no_grad()
    def _normalize_prototypes(self) -> None:
        self.prototypes.weight.copy_(F.normalize(self.prototypes.weight, dim=1))

    def _swapped_prediction(self, assignments: Sequence[Tensor], scores: Sequence[Tensor]) -> Tensor:
        terms: list[Tensor] = []
        for score_index, score in enumerate(scores):
            for global_index, target in enumerate(assignments):
                if score_index != global_index:
                    terms.append(-(target.detach() * F.log_softmax(score / self.temperature, dim=1)).sum(dim=1).mean())
        if not terms:
            raise RuntimeError("SwAV needs at least two global views")
        return torch.stack(terms).mean()

    def _temporal_consistency(self, batch: PretrainBatch, target: Tensor, generator: torch.Generator) -> Tensor:
        if self.temporal_lambda == 0 or not batch.neighbors:
            return target.sum() * 0.0
        if not (len(batch.neighbors) == len(batch.neighbor_offsets) == len(batch.neighbor_available)):
            raise ValueError("PretrainBatch neighbor fields must have matching lengths")
        blocks: list[tuple[Tensor, Tensor]] = []
        batch_size = batch.anchor.shape[0]
        for neighbor, offset, available in zip(batch.neighbors, batch.neighbor_offsets, batch.neighbor_available, strict=True):
            if available.shape != (batch_size,):
                raise ValueError("neighbor availability must have shape (batch_size,)")
            raw_weight = temporal_weight(offset, self.temporal_weights)
            available = available.to(device=batch.anchor.device, dtype=torch.bool)
            if raw_weight == 0 or not available.any():
                continue
            indices = torch.nonzero(available, as_tuple=False).flatten()
            scores = self._encode_project(self.local_augment(neighbor.index_select(0, indices), generator))
            aligned_scores = torch.zeros((batch_size, scores.shape[1]), device=scores.device, dtype=scores.dtype)
            aligned_scores.index_copy_(0, indices, scores)
            aligned_weights = torch.zeros(batch_size, device=scores.device, dtype=scores.dtype)
            aligned_weights.index_fill_(0, indices, raw_weight)
            blocks.append((aligned_scores, aligned_weights))
        if not blocks:
            return target.sum() * 0.0
        total = torch.stack([weights for _, weights in blocks]).sum(dim=0)
        parts = []
        for scores, raw_weights in blocks:
            normalized = torch.where(total > 0, raw_weights / total.clamp_min(torch.finfo(total.dtype).tiny), torch.zeros_like(raw_weights))
            parts.append(temporal_prototype_consistency(target, scores, normalized, temperature=self.temperature))
        return self.temporal_lambda * torch.stack(parts).sum()

    def forward(self, batch: PretrainBatch, *, generator: torch.Generator | None = None) -> SSLStepOutput:
        if not isinstance(batch, PretrainBatch):
            raise TypeError("batch must be a PretrainBatch")
        if batch.anchor.shape[0] < 2:
            raise ValueError("SwAVMetaFi requires batch size >= 2 for Projector BatchNorm")
        if generator is None:
            generator = self._default_generator(batch.anchor.device)
        self._normalize_prototypes()
        # Generate and encode each view immediately.  No list of raw CSI views
        # and no concatenated all-view tensor is retained across encoder calls.
        scores: list[Tensor] = []
        anchor = self._validate_csi(batch.anchor)
        for _ in range(self.view_policy.global_views):
            scores.append(self._encode_project(self.global_augment(anchor, generator)))
        for _ in range(self.view_policy.local_views):
            scores.append(self._encode_project(self.local_augment(anchor, generator)))
        assignments = [
            sinkhorn_assignments(
                score.detach(),
                epsilon=self.sinkhorn_epsilon,
                iterations=self.sinkhorn_iterations,
            )
            for score in scores[: self.view_policy.global_views]
        ]
        swapped = self._swapped_prediction(assignments, scores)
        temporal_target = torch.stack(assignments).mean(dim=0).detach()
        temporal = self._temporal_consistency(batch, temporal_target, generator)
        loss = swapped + temporal
        return SSLStepOutput(loss=loss, metrics={"loss": float(loss.detach()), "swapped_loss": float(swapped.detach()), "temporal_loss": float(temporal.detach()), "global_views": float(self.view_policy.global_views), "local_views": float(self.view_policy.local_views)})

    def export_encoder_state_dict(self) -> dict[str, Tensor]:
        return {name: value.detach().clone() for name, value in self.encoder.state_dict().items()}
