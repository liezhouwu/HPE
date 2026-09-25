"""Typed, compatibility-preserving pose evaluation metrics.

The mathematical implementations remain in :mod:`mmfi_wifi.metrics`.  This
module gives their three reporting values stable names without changing the
legacy evaluation semantics used by the training engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .metrics import evaluate_pose, evaluate_pose_official, mpjpe_mm


@dataclass(frozen=True)
class PoseMetrics:
    """The three pose errors reported by a MetaFi evaluation, in millimetres."""

    absolute_mpjpe_mm: float
    pelvis_mpjpe_mm: float
    pa_mpjpe_mm: float


@dataclass(frozen=True)
class EvaluationOutput:
    """Typed output from :func:`mmfi_wifi.engine.evaluate`.

    ``as_legacy_tuple`` and iteration intentionally retain the historic six
    value unpacking contract while internal callers migrate to named fields.
    The legacy order is ``loss, mpjpe, pa_mpjpe, pelvis_mpjpe, preds, gts``.
    """

    average_loss: float
    metrics: PoseMetrics
    predictions: np.ndarray
    targets: np.ndarray

    def as_legacy_tuple(self) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
        """Return the historic ``engine.evaluate`` tuple in its original order."""
        return (
            self.average_loss,
            self.metrics.absolute_mpjpe_mm,
            self.metrics.pa_mpjpe_mm,
            self.metrics.pelvis_mpjpe_mm,
            self.predictions,
            self.targets,
        )

    def __iter__(self) -> Iterator[object]:
        """Allow legacy six-value unpacking while callers transition to fields."""
        return iter(self.as_legacy_tuple())


def evaluate_pose_triplet(pred: np.ndarray, gt: np.ndarray, target_space: str) -> PoseMetrics:
    """Evaluate using precisely the legacy engine metric formulas.

    For ``absolute`` targets, the first value uses the MMFi public-code
    absolute-coordinate MPJPE.  For ``root_relative`` targets it keeps the
    historic engine behavior: the first value is a pelvis-aligned MPJPE.
    """
    if target_space == "absolute":
        absolute_mpjpe_mm, pa_mpjpe_mm = evaluate_pose_official(pred, gt)
        pelvis_mpjpe_mm = mpjpe_mm(pred, gt, already_root_relative=False)
    elif target_space == "root_relative":
        absolute_mpjpe_mm, pa_mpjpe_mm = evaluate_pose(
            pred, gt, already_root_relative=False,
        )
        pelvis_mpjpe_mm = absolute_mpjpe_mm
    else:
        raise ValueError(
            "target_space must be 'absolute' or 'root_relative', "
            f"got {target_space!r}"
        )

    return PoseMetrics(
        absolute_mpjpe_mm=float(absolute_mpjpe_mm),
        pelvis_mpjpe_mm=float(pelvis_mpjpe_mm),
        pa_mpjpe_mm=float(pa_mpjpe_mm),
    )
