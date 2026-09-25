"""MoCo adapted to the complete MetaFi-R34 encoder."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from numbers import Real

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mmfi_wifi.metafi_encoder import MetaFiEncoder

from .augment import PositionPreservingAugment, PositionPreservingAugmentConfig, build_two_views
from .base import MetaFiSSLMethod, SSLStepOutput
from .pretrain_data import PretrainBatch
from .projectors import Projector
from .temporal_positive import temporal_weight


def _require_positive_finite(name: str, value: object) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a positive finite scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite scalar")
    return result


def _require_nonnegative_finite(name: str, value: object) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a finite non-negative scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite non-negative scalar")
    return result


def _validate_embedding_matrix(name: str, value: Tensor, *, batch_size: int, dimension: int) -> None:
    if not isinstance(value, Tensor) or value.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 embedding tensor")
    if value.shape != (batch_size, dimension):
        raise ValueError(
            f"{name} must have shape ({batch_size}, {dimension}), got {tuple(value.shape)}"
        )
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")


def _as_weight_vector(
    weight: float | Tensor,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if isinstance(weight, Tensor):
        if weight.ndim == 0:
            values = weight.to(device=device, dtype=dtype).expand(batch_size)
        elif weight.shape == (batch_size,):
            values = weight.to(device=device, dtype=dtype)
        else:
            raise ValueError(
                "temporal weights must be scalar or shape (batch_size,), "
                f"got {tuple(weight.shape)}"
            )
    elif isinstance(weight, Real) and not isinstance(weight, bool):
        values = torch.full((batch_size,), float(weight), device=device, dtype=dtype)
    else:
        raise TypeError("temporal weights must be finite non-negative scalars or tensors")
    if not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("temporal weights must be finite and non-negative")
    return values


def moco_weighted_info_nce(
    query: Tensor,
    primary_keys: Tensor,
    queue: Tensor,
    *,
    temporal_keys: Sequence[Tensor],
    temporal_weights: Sequence[float | Tensor],
    temperature: float,
    temporal_lambda: float,
) -> Tensor:
    """MoCo InfoNCE with one primary and weak, weighted temporal positives.

    The primary same-frame key always has positive coefficient ``1``.  For an
    anchor with available temporal keys, their raw per-anchor weights are
    normalized and assigned total positive coefficient ``temporal_lambda``.
    Queue rows are always negatives: they never enter the positive log-sum.
    """

    if not isinstance(query, Tensor) or query.ndim != 2:
        raise ValueError("query must be a rank-2 embedding tensor")
    batch_size, dimension = query.shape
    if batch_size < 1 or dimension < 1:
        raise ValueError("query must have non-empty batch and feature dimensions")
    _validate_embedding_matrix("query", query, batch_size=batch_size, dimension=dimension)
    _validate_embedding_matrix(
        "primary_keys", primary_keys, batch_size=batch_size, dimension=dimension
    )
    if query.device != primary_keys.device or query.dtype != primary_keys.dtype:
        raise ValueError("query and primary_keys must use the same device and dtype")
    if not isinstance(queue, Tensor) or queue.ndim != 2 or queue.shape[1] != dimension:
        raise ValueError(f"queue must have shape (queue_size, {dimension})")
    if queue.shape[0] < 1 or queue.device != query.device or queue.dtype != query.dtype:
        raise ValueError("queue must be non-empty and use query's device and dtype")
    if not torch.isfinite(queue).all():
        raise ValueError("queue must contain only finite values")
    if len(temporal_keys) != len(temporal_weights):
        raise ValueError("temporal_keys and temporal_weights must have the same length")

    temperature = _require_positive_finite("temperature", temperature)
    temporal_lambda = _require_nonnegative_finite("temporal_lambda", temporal_lambda)

    primary_logits = (query * primary_keys).sum(dim=1, keepdim=True) / temperature
    negative_logits = query @ queue.detach().T / temperature

    temporal_logits: list[Tensor] = []
    temporal_weight_rows: list[Tensor] = []
    if temporal_lambda > 0:
        for index, (key, weight) in enumerate(zip(temporal_keys, temporal_weights, strict=True)):
            _validate_embedding_matrix(
                f"temporal_keys[{index}]", key, batch_size=batch_size, dimension=dimension
            )
            if key.device != query.device or key.dtype != query.dtype:
                raise ValueError("temporal keys must use query's device and dtype")
            weights = _as_weight_vector(
                weight,
                batch_size=batch_size,
                device=query.device,
                dtype=query.dtype,
            )
            temporal_logits.append((query * key).sum(dim=1) / temperature)
            temporal_weight_rows.append(weights)

    # Preserve classic MoCo exactly if no usable temporal coefficient exists.
    if not temporal_logits:
        logits = torch.cat((primary_logits, negative_logits), dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=query.device)
        return F.cross_entropy(logits, labels)

    temporal_logits_matrix = torch.stack(temporal_logits, dim=1)
    raw_weights = torch.stack(temporal_weight_rows, dim=1)
    available_total = raw_weights.sum(dim=1)
    coefficient_scale = torch.where(
        available_total > 0,
        torch.full_like(available_total, temporal_lambda) / available_total,
        torch.zeros_like(available_total),
    )
    temporal_coefficients = raw_weights * coefficient_scale[:, None]

    # A temporal key is a candidate in the partition function only when it is
    # actually available for this anchor.  Edge placeholders therefore cannot
    # become accidental negatives.
    unavailable_logits = torch.full_like(temporal_logits_matrix, float("-inf"))
    valid_temporal_logits = torch.where(
        temporal_coefficients > 0, temporal_logits_matrix, unavailable_logits)
    denominator = torch.logsumexp(
        torch.cat((primary_logits, negative_logits, valid_temporal_logits), dim=1), dim=1
    )

    positive_logits = torch.cat((primary_logits, valid_temporal_logits), dim=1)
    positive_coefficients = torch.cat(
        (torch.ones((batch_size, 1), device=query.device, dtype=query.dtype), temporal_coefficients),
        dim=1,
    )
    positive_log_coefficients = torch.where(
        positive_coefficients > 0,
        positive_coefficients.log(),
        torch.full_like(positive_coefficients, float("-inf")),
    )
    numerator = torch.logsumexp(positive_logits + positive_log_coefficients, dim=1)
    return (denominator - numerator).mean()


class MoCoMetaFi(MetaFiSSLMethod):
    """Position-preserving MoCo over the complete MetaFi-R34 encoder.

    Query-side modules receive gradients.  Key-side modules are EMA teachers,
    remain gradient-free, and are persisted alongside the negative queue so a
    later runner can resume exactly at an epoch boundary.
    """

    _EXTRA_STATE_SCHEMA_VERSION = 1

    def __init__(
        self,
        encoder: MetaFiEncoder,
        *,
        projector_dim: int = 128,
        projector_hidden_dim: int = 512,
        queue_size: int = 65_536,
        momentum: float = 0.999,
        temperature: float = 0.2,
        temporal_lambda: float = 0.2,
        temporal_weights: Mapping[int, float] | None = None,
        augmentation_config: Mapping[str, object] | PositionPreservingAugmentConfig | None = None,
        augmentation_seed: int = 0,
    ) -> None:
        super().__init__()
        if not isinstance(encoder, MetaFiEncoder):
            raise TypeError("encoder must be a MetaFiEncoder")
        if not isinstance(projector_dim, int) or projector_dim <= 0:
            raise ValueError("projector_dim must be a positive integer")
        if not isinstance(projector_hidden_dim, int) or projector_hidden_dim <= 0:
            raise ValueError("projector_hidden_dim must be a positive integer")
        if not isinstance(queue_size, int) or queue_size <= 0:
            raise ValueError("queue_size must be a positive integer")
        if isinstance(momentum, bool) or not isinstance(momentum, Real):
            raise TypeError("momentum must be a finite scalar in [0, 1)")
        if not math.isfinite(float(momentum)) or not 0.0 <= float(momentum) < 1.0:
            raise ValueError("momentum must be a finite scalar in [0, 1)")
        if isinstance(augmentation_seed, bool) or not isinstance(augmentation_seed, int):
            raise TypeError("augmentation_seed must be an integer")

        raw_temporal_weights = {} if temporal_weights is None else temporal_weights
        if not isinstance(raw_temporal_weights, Mapping):
            raise TypeError("temporal_weights must be a mapping")
        validated_temporal_weights: dict[int, float] = {}
        for distance, weight in raw_temporal_weights.items():
            validated_temporal_weights[distance] = temporal_weight(distance, {distance: weight})

        self.encoder_q = encoder
        self.projector_q = Projector(
            in_dim=encoder.feature_channels,
            hidden_dim=projector_hidden_dim,
            out_dim=projector_dim,
        )
        self.encoder_k = copy.deepcopy(encoder)
        self.projector_k = copy.deepcopy(self.projector_q)
        self._freeze_key_modules()
        self.encoder_k.eval()
        self.projector_k.eval()

        self.queue_size = queue_size
        self.momentum = float(momentum)
        self.temperature = _require_positive_finite("temperature", temperature)
        self.temporal_lambda = _require_nonnegative_finite("temporal_lambda", temporal_lambda)
        self.temporal_weights = dict(sorted(validated_temporal_weights.items()))
        self.augment = PositionPreservingAugment(
            {} if augmentation_config is None else augmentation_config
        )
        self.augmentation_seed = augmentation_seed

        self.register_buffer("queue", F.normalize(torch.randn(queue_size, projector_dim), dim=1))
        self.register_buffer("queue_ptr", torch.zeros((), dtype=torch.long))
        # The constant schedule itself is in extra state; this step tracks
        # schedule progress and is part of ordinary state_dict buffers.
        self.register_buffer("momentum_step", torch.zeros((), dtype=torch.long))

        self._augmentation_generators: dict[str, torch.Generator] = {}
        self._pending_augmentation_generator_states: dict[str, Tensor] = {}

    def _freeze_key_modules(self) -> None:
        for module in (self.encoder_k, self.projector_k):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    @staticmethod
    def _generator_key(device: torch.device) -> str:
        return str(torch.device(device))

    @staticmethod
    def _validate_generator_state(key: object, state: object) -> Tensor:
        if not isinstance(key, str):
            raise ValueError("augmentation generator-state key must be a string")
        try:
            device = torch.device(key)
        except (RuntimeError, TypeError, ValueError) as error:
            raise ValueError(f"invalid augmentation generator-state key {key!r}") from error
        if key != str(device):
            raise ValueError(
                "augmentation generator-state key must be canonical: "
                f"expected {str(device)!r}, got {key!r}"
            )
        if device.type == "cuda" and device.index is None:
            raise ValueError(
                "augmentation generator-state key must identify an exact CUDA device "
                "such as 'cuda:0', not bare 'cuda'"
            )
        if not isinstance(state, Tensor) or state.dtype != torch.uint8 or state.ndim != 1:
            raise ValueError("augmentation generator states must be rank-1 uint8 tensors")
        restored = state.detach().cpu().clone()
        try:
            generator = torch.Generator(device=device)
            if str(generator.device) != key:
                raise ValueError("generator device does not match canonical state key")
            generator.set_state(restored)
        except (RuntimeError, TypeError, ValueError) as error:
            raise ValueError(f"invalid augmentation generator state for device {key!r}") from error
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
                raise ValueError(f"invalid augmentation generator state for device {key!r}") from error
        return generator

    def _default_generator(self, device: torch.device) -> torch.Generator:
        key = self._generator_key(device)
        generator = self._augmentation_generators.get(key)
        if generator is None:
            generator = self._make_generator(device)
            self._augmentation_generators[key] = generator
        return generator

    def get_extra_state(self) -> dict[str, object]:
        """Persist fixed schedule metadata and method-owned augmentation RNGs."""

        states = {
            key: generator.get_state().detach().cpu().clone()
            for key, generator in self._augmentation_generators.items()
        }
        states.update(
            {
                key: value.detach().cpu().clone()
                for key, value in self._pending_augmentation_generator_states.items()
            }
        )
        return {
            "schema_version": self._EXTRA_STATE_SCHEMA_VERSION,
            "momentum_schedule": "constant",
            "momentum": self.momentum,
            "temperature": self.temperature,
            "temporal_lambda": self.temporal_lambda,
            "augmentation_seed": self.augmentation_seed,
            "generator_states": dict(sorted(states.items())),
        }

    def set_extra_state(self, state: object) -> None:
        """Restore runner-resumable queue/EMA auxiliary state fail-closed."""

        if not isinstance(state, Mapping):
            raise ValueError("MoCoMetaFi extra state must be a mapping")
        if state.get("schema_version") != self._EXTRA_STATE_SCHEMA_VERSION:
            raise ValueError("unsupported MoCoMetaFi method state schema")
        if state.get("momentum_schedule") != "constant":
            raise ValueError("unsupported MoCoMetaFi momentum schedule")
        for name, expected in (
            ("momentum", self.momentum),
            ("temperature", self.temperature),
            ("temporal_lambda", self.temporal_lambda),
        ):
            value = state.get(name)
            if not isinstance(value, Real) or isinstance(value, bool) or float(value) != expected:
                raise ValueError(f"MoCoMetaFi method state has incompatible {name}")
        if state.get("augmentation_seed") != self.augmentation_seed:
            raise ValueError("MoCoMetaFi method state has incompatible augmentation_seed")
        raw_states = state.get("generator_states")
        if not isinstance(raw_states, Mapping):
            raise ValueError("MoCoMetaFi extra state requires generator_states")
        restored = {
            key: self._validate_generator_state(key, value)
            for key, value in raw_states.items()
        }
        self._augmentation_generators = {}
        self._pending_augmentation_generator_states = dict(sorted(restored.items()))

    @staticmethod
    def _for_augmentation(csi: Tensor) -> Tensor:
        if not isinstance(csi, Tensor) or csi.ndim != 4 or csi.shape[1] != 3:
            raise ValueError("MetaFi pretraining CSI must have shape (B, 3, H, W)")
        return csi

    def _query_embeddings(self, csi: Tensor) -> Tensor:
        features = self.encoder_q(csi.unsqueeze(1)).global_vector
        return F.normalize(self.projector_q(features), dim=1)

    @torch.no_grad()
    def _key_embeddings(self, csi: Tensor) -> Tensor:
        # ``Module.train()`` on the outer method must not update teacher BN
        # statistics; the momentum teacher is an evaluation-mode target.
        self.encoder_k.eval()
        self.projector_k.eval()
        features = self.encoder_k(csi.unsqueeze(1)).global_vector
        return F.normalize(self.projector_k(features), dim=1)

    @torch.no_grad()
    def momentum_update(self) -> None:
        """Apply the fixed EMA schedule to key parameters and BN buffers."""

        for query_module, key_module in (
            (self.encoder_q, self.encoder_k),
            (self.projector_q, self.projector_k),
        ):
            query_parameters = dict(query_module.named_parameters())
            key_parameters = dict(key_module.named_parameters())
            if query_parameters.keys() != key_parameters.keys():
                raise RuntimeError("query/key parameter structures do not match")
            for name, key_parameter in key_parameters.items():
                key_parameter.mul_(self.momentum).add_(
                    query_parameters[name].detach(), alpha=1.0 - self.momentum
                )

            query_buffers = dict(query_module.named_buffers())
            key_buffers = dict(key_module.named_buffers())
            if query_buffers.keys() != key_buffers.keys():
                raise RuntimeError("query/key buffer structures do not match")
            for name, key_buffer in key_buffers.items():
                query_buffer = query_buffers[name].detach()
                if torch.is_floating_point(key_buffer):
                    key_buffer.mul_(self.momentum).add_(query_buffer, alpha=1.0 - self.momentum)
                else:
                    key_buffer.copy_(query_buffer)
        self.momentum_step.add_(1)

    @torch.no_grad()
    def enqueue_keys(self, keys: Tensor) -> None:
        """Insert normalized primary keys in circular order and advance the pointer."""

        if not isinstance(keys, Tensor) or keys.ndim != 2 or keys.shape[1] != self.queue.shape[1]:
            raise ValueError(
                f"keys must have shape (batch_size, {self.queue.shape[1]})"
            )
        if keys.shape[0] < 1 or keys.device != self.queue.device or keys.dtype != self.queue.dtype:
            raise ValueError("keys must be non-empty and match the queue device and dtype")
        if not torch.isfinite(keys).all():
            raise ValueError("keys must contain only finite values")
        pointer = int(self.queue_ptr.item())
        for key in keys.detach():
            self.queue[pointer].copy_(key)
            pointer = (pointer + 1) % self.queue_size
        self.queue_ptr.fill_(pointer)

    def forward(
        self,
        batch: PretrainBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> SSLStepOutput:
        """Compute MoCo primary/temporal positives and update EMA/queue state."""

        if not isinstance(batch, PretrainBatch):
            raise TypeError("batch must be a PretrainBatch")
        batch_size = batch.anchor.shape[0]
        if batch_size < 2:
            raise ValueError("MoCoMetaFi requires batch size >= 2 for Projector BatchNorm")
        if not (
            len(batch.neighbors)
            == len(batch.neighbor_offsets)
            == len(batch.neighbor_available)
        ):
            raise ValueError("PretrainBatch neighbor fields must have matching lengths")
        if generator is None:
            generator = self._default_generator(batch.anchor.device)

        anchor = self._for_augmentation(batch.anchor)
        query_view, key_view = build_two_views(anchor, self.augment, generator)
        query = self._query_embeddings(query_view)

        temporal_key_blocks: list[Tensor] = []
        temporal_weight_blocks: list[Tensor] = []
        with torch.no_grad():
            self.momentum_update()
            primary_keys = self._key_embeddings(key_view)
            for neighbor, offset, available in zip(
                batch.neighbors,
                batch.neighbor_offsets,
                batch.neighbor_available,
                strict=True,
            ):
                if available.shape != (batch_size,):
                    raise ValueError("neighbor availability must have shape (batch_size,)")
                raw_weight = temporal_weight(offset, self.temporal_weights)
                available = available.to(device=batch.anchor.device, dtype=torch.bool)
                if raw_weight == 0 or not available.any():
                    continue
                indices = torch.nonzero(available, as_tuple=False).flatten()
                neighbor_csi = self._for_augmentation(neighbor.index_select(0, indices))
                compact_keys = self._key_embeddings(self.augment(neighbor_csi, generator))
                aligned_keys = torch.zeros(
                    (batch_size, compact_keys.shape[1]),
                    device=compact_keys.device,
                    dtype=compact_keys.dtype,
                )
                aligned_keys.index_copy_(0, indices, compact_keys)
                aligned_weights = torch.zeros(
                    batch_size, device=compact_keys.device, dtype=compact_keys.dtype
                )
                aligned_weights.index_fill_(0, indices, raw_weight)
                temporal_key_blocks.append(aligned_keys)
                temporal_weight_blocks.append(aligned_weights)

        loss = moco_weighted_info_nce(
            query,
            primary_keys,
            self.queue.detach().clone(),
            temporal_keys=temporal_key_blocks,
            temporal_weights=temporal_weight_blocks,
            temperature=self.temperature,
            temporal_lambda=self.temporal_lambda,
        )
        with torch.no_grad():
            self.enqueue_keys(primary_keys)
        return SSLStepOutput(
            loss=loss,
            metrics={
                "loss": float(loss.detach()),
                "momentum": self.momentum,
                "momentum_step": float(self.momentum_step.item()),
                "queue_ptr": float(self.queue_ptr.item()),
            },
        )

    def export_encoder_state_dict(self) -> dict[str, Tensor]:
        """Export only the query MetaFiEncoder for downstream pose fine-tuning."""

        return {
            name: value.detach().clone()
            for name, value in self.encoder_q.state_dict().items()
        }
