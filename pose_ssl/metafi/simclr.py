"""SimCLR adapted to the complete MetaFi-R34 encoder."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real

import torch
import torch.nn.functional as F
from torch import Tensor

from mmfi_wifi.metafi_encoder import MetaFiEncoder
from pose_ssl.pretrain_methods import nt_xent_loss

from .augment import PositionPreservingAugment, PositionPreservingAugmentConfig, build_two_views
from .base import MetaFiSSLMethod, SSLStepOutput
from .pretrain_data import PretrainBatch
from .projectors import Projector
from .temporal_positive import temporal_weight


def _validate_embeddings(name: str, embeddings: Tensor, *, batch_size: int, dimension: int) -> None:
    if not isinstance(embeddings, Tensor) or embeddings.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 embedding tensor")
    if embeddings.shape != (batch_size, dimension):
        raise ValueError(
            f"{name} must have shape ({batch_size}, {dimension}), got {tuple(embeddings.shape)}"
        )
    if not torch.isfinite(embeddings).all():
        raise ValueError(f"{name} must contain only finite values")


def _positive_weight_vector(
    weight: float | Tensor,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if isinstance(weight, Tensor):
        if weight.ndim == 0:
            result = weight.to(device=device, dtype=dtype).expand(batch_size)
        elif weight.shape == (batch_size,):
            result = weight.to(device=device, dtype=dtype)
        else:
            raise ValueError(
                "neighbor weights must be scalar or shape (batch_size,), "
                f"got {tuple(weight.shape)}"
            )
    elif isinstance(weight, Real) and not isinstance(weight, bool):
        result = torch.full((batch_size,), float(weight), device=device, dtype=dtype)
    else:
        raise TypeError("neighbor weights must be finite non-negative scalars or tensors")
    if not torch.isfinite(result).all() or (result < 0).any():
        raise ValueError("neighbor weights must be finite and non-negative")
    return result


def multi_positive_nt_xent(
    anchor_a: Tensor,
    anchor_b: Tensor,
    neighbors: Sequence[Tensor],
    neighbor_weights: Sequence[float | Tensor],
    temperature: float,
) -> Tensor:
    """Return SimCLR's NT-Xent objective with weak, weighted temporal positives.

    ``anchor_a`` and ``anchor_b`` are the L2-normalized views of the same CSI
    frames.  Each neighbor tensor is batch-aligned with them.  A zero weight
    omits that sample entirely: it is neither a positive nor a negative, which
    is how edge placeholders from :class:`PretrainBatch` are excluded.
    """

    if not isinstance(anchor_a, Tensor) or anchor_a.ndim != 2:
        raise ValueError("anchor_a must be a rank-2 embedding tensor")
    batch_size, dimension = anchor_a.shape
    if batch_size < 1 or dimension < 1:
        raise ValueError("anchor embeddings must have non-empty batch and feature dimensions")
    _validate_embeddings("anchor_a", anchor_a, batch_size=batch_size, dimension=dimension)
    _validate_embeddings("anchor_b", anchor_b, batch_size=batch_size, dimension=dimension)
    if anchor_a.device != anchor_b.device or anchor_a.dtype != anchor_b.dtype:
        raise ValueError("anchor_a and anchor_b must use the same device and dtype")
    if not isinstance(temperature, Real) or isinstance(temperature, bool):
        raise TypeError("temperature must be a positive finite scalar")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a positive finite scalar")
    if len(neighbors) != len(neighbor_weights):
        raise ValueError("neighbors and neighbor_weights must have the same length")

    # Exact legacy behavior is important for the no-temporal-positive ablation.
    if not neighbors:
        return nt_xent_loss(anchor_a, anchor_b, temperature)

    candidate_blocks: list[Tensor] = [anchor_a, anchor_b]
    positives_by_anchor: list[list[tuple[int, Tensor]]] = [
        [(batch_size + index, anchor_a.new_tensor(1.0))] for index in range(batch_size)
    ] + [
        [(index, anchor_a.new_tensor(1.0))] for index in range(batch_size)
    ]

    next_candidate_index = 2 * batch_size
    for neighbor_number, (neighbor, weight) in enumerate(zip(neighbors, neighbor_weights, strict=True)):
        _validate_embeddings(
            f"neighbors[{neighbor_number}]",
            neighbor,
            batch_size=batch_size,
            dimension=dimension,
        )
        if neighbor.device != anchor_a.device or neighbor.dtype != anchor_a.dtype:
            raise ValueError("neighbor embeddings must use the anchor device and dtype")
        weights = _positive_weight_vector(
            weight,
            batch_size=batch_size,
            device=anchor_a.device,
            dtype=anchor_a.dtype,
        )
        available_indices = torch.nonzero(weights > 0, as_tuple=False).flatten()
        if available_indices.numel() == 0:
            continue
        compact_neighbor = neighbor.index_select(0, available_indices)
        candidate_blocks.append(compact_neighbor)
        for compact_index, anchor_index in enumerate(available_indices.tolist()):
            candidate_index = next_candidate_index + compact_index
            positive_weight = weights[anchor_index]
            positives_by_anchor[anchor_index].append((candidate_index, positive_weight))
            positives_by_anchor[batch_size + anchor_index].append((candidate_index, positive_weight))
        next_candidate_index += compact_neighbor.shape[0]

    candidates = torch.cat(candidate_blocks, dim=0)
    anchor_vectors = torch.cat((anchor_a, anchor_b), dim=0)
    logits = anchor_vectors @ candidates.T / temperature
    anchor_count = 2 * batch_size
    logits[torch.arange(anchor_count, device=logits.device), torch.arange(anchor_count, device=logits.device)] = float("-inf")
    log_probabilities = F.log_softmax(logits, dim=1)

    losses: list[Tensor] = []
    for anchor_index, positives in enumerate(positives_by_anchor):
        positive_indices = torch.tensor(
            [candidate_index for candidate_index, _ in positives],
            dtype=torch.long,
            device=log_probabilities.device,
        )
        weights = torch.stack([weight for _, weight in positives])
        normalized_weights = weights / weights.sum()
        # Multi-positive InfoNCE uses the probability mass of the weighted
        # positive set. Averaging individual negative log-probabilities would
        # penalize a weak, imperfect temporal neighbor as if it were a second
        # required exact match.
        log_weights = normalized_weights.log()
        losses.append(-torch.logsumexp(log_weights + log_probabilities[anchor_index, positive_indices], dim=0))
    return torch.stack(losses).mean()


def _align_neighbor_weights(
    weights: Tensor,
    available_indices: Tensor,
    *,
    batch_size: int,
    embedding: Tensor,
) -> Tensor:
    """把时间正样本权重对齐到 AMP 投影表示的设备与 dtype。"""
    aligned = torch.zeros(batch_size, device=embedding.device, dtype=embedding.dtype)
    aligned.index_copy_(
        0,
        available_indices.to(device=embedding.device, dtype=torch.long),
        weights.to(device=embedding.device, dtype=embedding.dtype),
    )
    return aligned


class SimCLRMetaFi(MetaFiSSLMethod):
    """Position-preserving SimCLR for complete MetaFi-R34 representations."""

    def __init__(
        self,
        encoder: MetaFiEncoder,
        *,
        temperature: float = 0.1,
        temporal_weights: Mapping[int, float] | None = None,
        augmentation_config: Mapping[str, object] | PositionPreservingAugmentConfig | None = None,
        augmentation_seed: int = 0,
        projector_hidden_dim: int = 512,
        projector_out_dim: int = 128,
    ) -> None:
        super().__init__()
        if not isinstance(encoder, MetaFiEncoder):
            raise TypeError("encoder must be a MetaFiEncoder")
        if not isinstance(temperature, Real) or isinstance(temperature, bool):
            raise TypeError("temperature must be a positive finite scalar")
        if not math.isfinite(float(temperature)) or float(temperature) <= 0:
            raise ValueError("temperature must be a positive finite scalar")
        if isinstance(augmentation_seed, bool) or not isinstance(augmentation_seed, int):
            raise TypeError("augmentation_seed must be an integer")
        raw_temporal_weights = {} if temporal_weights is None else temporal_weights
        if not isinstance(raw_temporal_weights, Mapping):
            raise TypeError("temporal_weights must be a mapping")
        validated_temporal_weights: dict[int, float] = {}
        for distance, weight in raw_temporal_weights.items():
            # Call the shared validator rather than duplicating temporal semantics.
            validated_temporal_weights[distance] = temporal_weight(distance, {distance: weight})

        self.encoder = encoder
        self.projector = Projector(
            in_dim=encoder.feature_channels,
            hidden_dim=projector_hidden_dim,
            out_dim=projector_out_dim,
        )
        self.temperature = float(temperature)
        self.temporal_weights = dict(sorted(validated_temporal_weights.items()))
        self.augment = PositionPreservingAugment(
            {} if augmentation_config is None else augmentation_config
        )
        self.augmentation_seed = augmentation_seed
        # Default augmentation RNG is method-owned so callers that use
        # ``method(batch)`` get one reproducible stream rather than a reset
        # stream on every optimizer step.  These are deliberately ordinary
        # attributes: torch.Generator is not an nn.Module parameter/buffer.
        self._augmentation_generators: dict[str, torch.Generator] = {}
        self._pending_augmentation_generator_states: dict[str, Tensor] = {}

    @staticmethod
    def _generator_key(device: torch.device) -> str:
        """Return the canonical key for the device that owns an RNG stream."""

        return str(torch.device(device))

    @staticmethod
    def _validate_generator_state(key: object, state: object) -> Tensor:
        """Validate a canonical, usable generator device key and its RNG state."""

        if not isinstance(key, str):
            raise ValueError("augmentation generator-state key must be a string")
        try:
            device = torch.device(key)
        except (RuntimeError, TypeError, ValueError) as error:
            raise ValueError(
                f"invalid augmentation generator-state key {key!r}"
            ) from error
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
            raise ValueError(
                "augmentation generator states must be rank-1 uint8 tensors"
            )

        restored_state = state.detach().cpu().clone()
        try:
            generator = torch.Generator(device=device)
            if key != str(generator.device):
                raise ValueError(
                    "augmentation generator-state key does not reconstruct exactly: "
                    f"expected {str(generator.device)!r}, got {key!r}"
                )
            generator.set_state(restored_state)
        except (RuntimeError, TypeError, ValueError) as error:
            raise ValueError(
                f"invalid augmentation generator state for device {key!r}"
            ) from error
        return restored_state

    def _make_generator(self, device: torch.device) -> torch.Generator:
        """Create one device-matched generator, restoring a pending state if any."""

        normalized = torch.device(device)
        key = self._generator_key(normalized)
        generator = torch.Generator(device=normalized)
        pending_state = self._pending_augmentation_generator_states.pop(key, None)
        if pending_state is None:
            generator.manual_seed(self.augmentation_seed)
        else:
            try:
                generator.set_state(pending_state)
            except RuntimeError as error:
                raise ValueError(
                    f"invalid augmentation generator state for device {key!r}"
                ) from error
        return generator

    def _default_generator(self, device: torch.device) -> torch.Generator:
        key = self._generator_key(device)
        generator = self._augmentation_generators.get(key)
        if generator is None:
            generator = self._make_generator(device)
            self._augmentation_generators[key] = generator
        return generator

    def get_extra_state(self) -> dict[str, object]:
        """Serialize default augmentation RNG streams with this method state."""

        generator_states = {
            key: generator.get_state().detach().cpu().clone()
            for key, generator in self._augmentation_generators.items()
        }
        generator_states.update(
            {
                key: state.detach().cpu().clone()
                for key, state in self._pending_augmentation_generator_states.items()
            }
        )
        return {
            "schema_version": 1,
            "generator_states": dict(sorted(generator_states.items())),
        }

    def set_extra_state(self, state: object) -> None:
        """Restore serialized default augmentation streams fail-closed."""

        if not isinstance(state, Mapping):
            raise ValueError("SimCLRMetaFi extra state must be a mapping")
        if state.get("schema_version") != 1:
            raise ValueError("unsupported SimCLRMetaFi augmentation RNG state schema")
        raw_generator_states = state.get("generator_states")
        if not isinstance(raw_generator_states, Mapping):
            raise ValueError("SimCLRMetaFi extra state requires generator_states")
        restored = {
            key: self._validate_generator_state(key, value)
            for key, value in raw_generator_states.items()
        }
        self._augmentation_generators = {}
        self._pending_augmentation_generator_states = dict(sorted(restored.items()))

    @staticmethod
    def _for_augmentation(csi: Tensor) -> Tensor:
        if csi.ndim != 4 or csi.shape[1] != 3:
            raise ValueError("MetaFi pretraining CSI must have shape (B, 3, H, W)")
        return csi

    def _encode(self, csi: Tensor) -> Tensor:
        if csi.ndim != 4 or csi.shape[1] != 3:
            raise ValueError("MetaFi encoder CSI must have shape (B, 3, H, W)")
        return self.encoder(csi.unsqueeze(1)).global_vector

    def _project_normalize(self, features: Tensor) -> Tensor:
        return F.normalize(self.projector(features), dim=1)

    def forward(
        self,
        batch: PretrainBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> SSLStepOutput:
        """Compute same-frame SimCLR plus only available weak temporal positives."""

        if not isinstance(batch, PretrainBatch):
            raise TypeError("batch must be a PretrainBatch")
        if batch.anchor.shape[0] < 2:
            raise ValueError("SimCLRMetaFi requires batch size >= 2 for Projector BatchNorm")
        if generator is None:
            generator = self._default_generator(batch.anchor.device)
        anchor_csi = self._for_augmentation(batch.anchor)
        view_a, view_b = build_two_views(anchor_csi, self.augment, generator)
        anchor_features_a = self._encode(view_a)
        anchor_features_b = self._encode(view_b)

        neighbor_feature_blocks: list[tuple[Tensor, Tensor, Tensor]] = []
        if not (
            len(batch.neighbors)
            == len(batch.neighbor_offsets)
            == len(batch.neighbor_available)
        ):
            raise ValueError("PretrainBatch neighbor fields must have matching lengths")
        for neighbor, offset, available in zip(
            batch.neighbors,
            batch.neighbor_offsets,
            batch.neighbor_available,
            strict=True,
        ):
            if available.shape != (batch.anchor.shape[0],):
                raise ValueError("neighbor availability must have shape (batch_size,)")
            available = available.to(device=batch.anchor.device, dtype=torch.bool)
            weight = temporal_weight(offset, self.temporal_weights)
            if weight == 0 or not available.any():
                continue
            available_indices = torch.nonzero(available, as_tuple=False).flatten()
            # Deliberately do not encode edge placeholders.  They must not become
            # positives or negatives under any batch composition.
            available_neighbor = neighbor.index_select(0, available_indices)
            neighbor_view = self.augment(self._for_augmentation(available_neighbor), generator)
            neighbor_features = self._encode(neighbor_view)
            neighbor_feature_blocks.append((neighbor_features, available_indices, torch.full(
                (available_indices.numel(),),
                weight,
                device=neighbor_features.device,
                dtype=neighbor_features.dtype,
            )))

        # The projector contains BatchNorm1d. Project all same-frame and
        # available temporal features together, so an edge batch with exactly
        # one available neighbor never substitutes a placeholder merely to
        # satisfy BatchNorm batch-size requirements.
        projected_features = self._project_normalize(
            torch.cat(
                [anchor_features_a, anchor_features_b]
                + [features for features, _, _ in neighbor_feature_blocks],
                dim=0,
            )
        )
        batch_size = batch.anchor.shape[0]
        embedding_a = projected_features[:batch_size]
        embedding_b = projected_features[batch_size : 2 * batch_size]
        cursor = 2 * batch_size
        neighbor_embeddings: list[Tensor] = []
        neighbor_weights: list[Tensor] = []
        for features, available_indices, weights in neighbor_feature_blocks:
            compact_embedding = projected_features[cursor : cursor + features.shape[0]]
            cursor += features.shape[0]
            aligned_embedding = torch.zeros(
                (batch_size, compact_embedding.shape[1]),
                device=compact_embedding.device,
                dtype=compact_embedding.dtype,
            )
            aligned_embedding.index_copy_(0, available_indices, compact_embedding)
            aligned_weights = _align_neighbor_weights(
                weights,
                available_indices,
                batch_size=batch_size,
                embedding=compact_embedding,
            )
            neighbor_embeddings.append(aligned_embedding)
            neighbor_weights.append(aligned_weights)

        loss = multi_positive_nt_xent(
            embedding_a,
            embedding_b,
            neighbor_embeddings,
            neighbor_weights,
            self.temperature,
        )
        return SSLStepOutput(loss=loss, metrics={"loss": float(loss.detach())})

    def export_encoder_state_dict(self) -> dict[str, Tensor]:
        """Export only the complete downstream MetaFiEncoder state."""

        return {name: value.detach().clone() for name, value in self.encoder.state_dict().items()}
