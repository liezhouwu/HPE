"""CSI-only, sequence-safe samples for MetaFi SSL pretraining.

This module deliberately indexes only CSI files.  It neither stores label paths nor
uses ``MMFi_Dataset`` because that dataset's frame path reads ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path
from typing import Collection, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from mmfi_wifi.data import _load_csi_frame, _load_packed_csi
from mmfi_wifi.sequence_keys import SequenceKey


_FRAME_COUNT = 297
_PACKED_FILENAME = "wifi-csi-packed.npy"
_SHA256_HEXDIGEST = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class PretrainSample:
    """One CSI anchor and its valid same-sequence temporal neighbors."""

    anchor: Tensor
    neighbors: tuple[Tensor, ...]
    sequence_key: SequenceKey
    frame_index: int
    neighbor_indices: tuple[int, ...]
    neighbor_sequence_keys: tuple[SequenceKey, ...]


@dataclass(frozen=True)
class PretrainBatch:
    """Collated pretraining samples with per-offset availability masks.

    Missing edge neighbors are represented by a zero placeholder whose matching
    ``neighbor_available`` entry is false.  They are never substituted with a
    frame from another sequence.
    """

    anchor: Tensor
    neighbors: tuple[Tensor, ...]
    neighbor_offsets: tuple[int, ...]
    sequence_keys: tuple[SequenceKey, ...]
    frame_indices: Tensor
    neighbor_available: tuple[Tensor, ...]


@dataclass(frozen=True)
class _FrameRecord:
    sequence_key: SequenceKey
    frame_index: int
    packed_path: Path | None
    frame_path: Path


def _canonical_cache_identity(
    *,
    protocol: str,
    split: str,
    scope: str,
    manifest_fingerprint: str,
    neighbor_offsets: tuple[int, ...],
    sequence_keys: tuple[SequenceKey, ...],
) -> str:
    """Return a path-independent identity for CSI record/cache reuse."""

    payload = {
        "version": 1,
        "protocol": protocol,
        "split": split,
        "scope": scope,
        "manifest_fingerprint": manifest_fingerprint,
        "neighbor_offsets": list(neighbor_offsets),
        "sequence_keys": [
            {"scene": key.scene, "subject": key.subject, "action": key.action}
            for key in sequence_keys
        ],
    }
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _frame_path(root: Path, sequence_key: SequenceKey, frame_index: int) -> Path:
    """Return the CSI frame path without consulting any label metadata."""

    return (
        root
        / sequence_key.scene
        / sequence_key.subject
        / sequence_key.action
        / "wifi-csi"
        / f"frame{frame_index + 1:03d}.mat"
    )


def _validate_offsets(neighbor_offsets: tuple[int, ...]) -> tuple[int, ...]:
    offsets = tuple(neighbor_offsets)
    if len(set(offsets)) != len(offsets):
        raise ValueError("neighbor_offsets must not contain duplicates")
    if 0 in offsets:
        raise ValueError("neighbor_offsets must not include the anchor offset 0")
    return offsets


def _require_identity_field(name: str, value: str | None) -> str:
    """Reject cache identities that are not explicitly manifest-bound."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_manifest_fingerprint(value: str | None) -> str:
    """Require the canonical lower-case SHA-256 digest of a data manifest."""

    fingerprint = _require_identity_field("manifest_fingerprint", value)
    if _SHA256_HEXDIGEST.fullmatch(fingerprint) is None:
        raise ValueError(
            "manifest_fingerprint must be a 64-character lower-case hexadecimal SHA-256 digest"
        )
    return fingerprint


def _has_per_frame_source(frame_path: Path) -> bool:
    """Return whether the shared CSI loader can read this conventional path.

    ``_load_csi_frame`` receives the conventional ``.mat`` path but first
    prefers a sibling ``.npy`` cache.  The dataset must therefore accept
    either representation when no sequence-level packed cache is present.
    """

    return frame_path.is_file() or frame_path.with_suffix(".npy").is_file()


class MetaFiPretrainDataset(Dataset[PretrainSample]):
    """Frame-level CSI dataset that cannot traverse sequence or label boundaries."""

    def __init__(
        self,
        dataset_root: str,
        sequence_keys: Collection[SequenceKey],
        neighbor_offsets: tuple[int, ...] = (-2, -1, 1, 2),
        *,
        protocol: str | None = None,
        split: str | None = None,
        scope: str | None = None,
        manifest_fingerprint: str | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.sequence_keys = tuple(sorted(frozenset(sequence_keys)))
        if not self.sequence_keys:
            raise ValueError("sequence_keys must not be empty")
        self.neighbor_offsets = _validate_offsets(neighbor_offsets)
        self.protocol = _require_identity_field("protocol", protocol)
        self.split = _require_identity_field("split", split)
        self.scope = _require_identity_field("scope", scope)
        self.manifest_fingerprint = _require_manifest_fingerprint(manifest_fingerprint)
        self.cache_identity = _canonical_cache_identity(
            protocol=self.protocol,
            split=self.split,
            scope=self.scope,
            manifest_fingerprint=self.manifest_fingerprint,
            neighbor_offsets=self.neighbor_offsets,
            sequence_keys=self.sequence_keys,
        )
        self._records = self._build_records()

    def _build_records(self) -> tuple[_FrameRecord, ...]:
        records: list[_FrameRecord] = []
        for sequence_key in self.sequence_keys:
            first_frame = _frame_path(self.dataset_root, sequence_key, 0)
            packed_path = first_frame.with_name(_PACKED_FILENAME)
            use_packed = packed_path.is_file()
            if not use_packed and not _has_per_frame_source(first_frame):
                raise FileNotFoundError(
                    f"No CSI source exists for sequence {sequence_key}: "
                    f"expected {packed_path}, {first_frame}, or "
                    f"{first_frame.with_suffix('.npy')}"
                )
            for frame_index in range(_FRAME_COUNT):
                frame_path = _frame_path(self.dataset_root, sequence_key, frame_index)
                if not use_packed and not _has_per_frame_source(frame_path):
                    raise FileNotFoundError(
                        f"Missing CSI frame: {frame_path} or "
                        f"{frame_path.with_suffix('.npy')}"
                    )
                records.append(
                    _FrameRecord(
                        sequence_key=sequence_key,
                        frame_index=frame_index,
                        packed_path=packed_path if use_packed else None,
                        frame_path=frame_path,
                    )
                )
        return tuple(records)

    def __len__(self) -> int:
        return len(self._records)

    @staticmethod
    def _to_tensor(csi) -> Tensor:
        return torch.as_tensor(csi, dtype=torch.float32).contiguous()

    def _load_record(self, record: _FrameRecord) -> Tensor:
        if record.packed_path is not None:
            csi = _load_packed_csi(str(record.packed_path), record.frame_index)
        else:
            csi = _load_csi_frame(str(record.frame_path))
        return self._to_tensor(csi)

    def __getitem__(self, index: int) -> PretrainSample:
        record = self._records[index]
        anchor = self._load_record(record)
        neighbor_indices: list[int] = []
        neighbors: list[Tensor] = []
        for offset in self.neighbor_offsets:
            neighbor_index = record.frame_index + offset
            if not 0 <= neighbor_index < _FRAME_COUNT:
                continue
            neighbor_record = _FrameRecord(
                sequence_key=record.sequence_key,
                frame_index=neighbor_index,
                packed_path=record.packed_path,
                frame_path=_frame_path(self.dataset_root, record.sequence_key, neighbor_index),
            )
            neighbor_indices.append(neighbor_index)
            neighbors.append(self._load_record(neighbor_record))
        neighbor_sequence_keys = (record.sequence_key,) * len(neighbor_indices)
        return PretrainSample(
            anchor=anchor,
            neighbors=tuple(neighbors),
            sequence_key=record.sequence_key,
            frame_index=record.frame_index,
            neighbor_indices=tuple(neighbor_indices),
            neighbor_sequence_keys=neighbor_sequence_keys,
        )


def collate_pretrain_samples(samples: Sequence[PretrainSample]) -> PretrainBatch:
    """Group variable-length neighbors by temporal offset with availability masks."""

    if not samples:
        raise ValueError("cannot collate an empty pretraining batch")

    anchor = torch.stack([sample.anchor for sample in samples])
    offsets = tuple(
        sorted(
            {
                neighbor_index - sample.frame_index
                for sample in samples
                for neighbor_index in sample.neighbor_indices
            }
        )
    )
    neighbors_by_offset: list[Tensor] = []
    availability_by_offset: list[Tensor] = []
    for offset in offsets:
        per_sample: list[Tensor] = []
        available: list[bool] = []
        for sample in samples:
            neighbor_map = {
                neighbor_index - sample.frame_index: neighbor
                for neighbor_index, neighbor in zip(sample.neighbor_indices, sample.neighbors, strict=True)
            }
            neighbor = neighbor_map.get(offset)
            if neighbor is None:
                per_sample.append(torch.zeros_like(sample.anchor))
                available.append(False)
            else:
                per_sample.append(neighbor)
                available.append(True)
        neighbors_by_offset.append(torch.stack(per_sample))
        availability_by_offset.append(torch.tensor(available, dtype=torch.bool))

    return PretrainBatch(
        anchor=anchor,
        neighbors=tuple(neighbors_by_offset),
        neighbor_offsets=offsets,
        sequence_keys=tuple(sample.sequence_key for sample in samples),
        frame_indices=torch.tensor([sample.frame_index for sample in samples], dtype=torch.long),
        neighbor_available=tuple(availability_by_offset),
    )
