"""Labeled-only output-axis statistics for MetaFi fine-tuning.

This module intentionally opens only the explicitly selected sequences' ground
truth files.  It does not construct ``MMFi_Dataset`` or scan the dataset tree,
so unlabeled internal-train, select, and test labels cannot affect statistics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Collection

import numpy as np

from mmfi_wifi.metafi_decoder import AxisStats
from mmfi_wifi.sequence_keys import SequenceKey


def _normalise_labeled_keys(
    labeled_keys: Collection[SequenceKey],
) -> frozenset[SequenceKey]:
    keys = frozenset(labeled_keys)
    if not keys:
        raise ValueError("labeled_keys 不能为空")
    if any(not isinstance(key, SequenceKey) for key in keys):
        raise TypeError("labeled_keys 只能包含 SequenceKey")
    return keys


def fingerprint_keys(labeled_keys: Collection[SequenceKey]) -> str:
    """Return a stable SHA-256 fingerprint for logical labeled sequence keys."""

    keys = _normalise_labeled_keys(labeled_keys)
    payload = [[key.scene, key.subject, key.action] for key in sorted(keys)]
    canonical = json.dumps(
        payload,
        sort_keys=False,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _ground_truth_path(dataset_root: Path, key: SequenceKey) -> Path:
    return dataset_root / key.scene / key.subject / key.action / "ground_truth.npy"


def compute_axis_stats(
    dataset_root: str,
    labeled_keys: Collection[SequenceKey],
) -> AxisStats:
    """Stream three-axis pose statistics from *only* explicitly labeled keys.

    The population standard deviation is used because these values initialize
    decoder output affine parameters; all statistics are accumulated in
    ``float64`` to avoid precision loss across the full sequence collection.
    """

    if not isinstance(dataset_root, str) or not dataset_root.strip():
        raise ValueError("dataset_root 必须是非空字符串")

    keys = _normalise_labeled_keys(labeled_keys)
    root = Path(dataset_root)
    total = np.zeros(3, dtype=np.float64)
    squared_total = np.zeros(3, dtype=np.float64)
    count = 0

    for key in sorted(keys):
        gt_path = _ground_truth_path(root, key)
        if not gt_path.is_file():
            raise FileNotFoundError(f"找不到 labeled sequence 的 ground_truth.npy: {gt_path}")

        ground_truth = np.load(gt_path, mmap_mode="r")
        if ground_truth.ndim < 2 or ground_truth.shape[-1] != 3:
            raise ValueError(f"ground_truth 必须以坐标轴 3 结尾: {gt_path}")
        values = np.asarray(ground_truth, dtype=np.float64).reshape(-1, 3)
        if values.size == 0:
            raise ValueError(f"ground_truth 不能为空: {gt_path}")
        if not np.isfinite(values).all():
            raise ValueError(f"ground_truth 必须全部为有限值: {gt_path}")

        total += values.sum(axis=0, dtype=np.float64)
        squared_total += np.square(values, dtype=np.float64).sum(axis=0, dtype=np.float64)
        count += values.shape[0]

    if count == 0:
        raise ValueError("labeled ground_truth 没有坐标值")

    mean = total / count
    variance = squared_total / count - np.square(mean)
    # Floating-point cancellation may produce a tiny negative value near zero;
    # a non-positive final standard deviation remains a fail-closed error.
    variance = np.maximum(variance, 0.0)
    std = np.sqrt(variance)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("axis statistics 必须全部为有限值")
    if np.any(std <= 0.0):
        raise ValueError("axis statistics 的标准差必须全部大于 0")

    return AxisStats(
        mean=tuple(float(value) for value in mean),
        std=tuple(float(value) for value in std),
        source_fingerprint=fingerprint_keys(keys),
    )
