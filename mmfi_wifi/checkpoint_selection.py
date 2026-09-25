"""Independent selection of the three MetaFi pose-error checkpoints."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
from enum import Enum
from typing import Final

from .pose_metrics import PoseMetrics


class CheckpointRole(str, Enum):
    """The metric used to select a saved best-model checkpoint."""

    ABSOLUTE = "absolute"
    PELVIS = "pelvis"
    PA = "pa"


CHECKPOINT_FILENAMES: Final[dict[CheckpointRole, str]] = {
    CheckpointRole.ABSOLUTE: "best_absolute.pth",
    CheckpointRole.PELVIS: "best_pelvis.pth",
    CheckpointRole.PA: "best_pa.pth",
}


ROLE_METRIC_NAMES: Final[dict[CheckpointRole, str]] = {
    CheckpointRole.ABSOLUTE: "absolute_mpjpe_mm",
    CheckpointRole.PELVIS: "pelvis_mpjpe_mm",
    CheckpointRole.PA: "pa_mpjpe_mm",
}


@dataclass(frozen=True)
class BestRecord:
    """One metric-specific best selection and its complete validation triplet."""

    epoch: int
    value: float
    metrics: PoseMetrics


def metric_value_for_role(role: CheckpointRole, metrics: PoseMetrics) -> float:
    """Return the selection value for ``role`` from a complete metric triplet."""
    return float(getattr(metrics, ROLE_METRIC_NAMES[role]))


def metrics_as_dict(metrics: PoseMetrics) -> dict[str, float]:
    """Serialize all three validation metrics without mixing checkpoint roles."""
    return {
        "absolute_mpjpe_mm": float(metrics.absolute_mpjpe_mm),
        "pelvis_mpjpe_mm": float(metrics.pelvis_mpjpe_mm),
        "pa_mpjpe_mm": float(metrics.pa_mpjpe_mm),
    }


def checkpoint_selection_metadata(
    role: CheckpointRole,
    record: BestRecord,
) -> dict[str, object]:
    """Return role/selection fields required in every best checkpoint state."""
    return {
        "checkpoint_role": role.value,
        "selected_epoch": int(record.epoch),
        "selected_metric": ROLE_METRIC_NAMES[role],
        "selected_value": float(record.value),
        "metrics": metrics_as_dict(record.metrics),
    }


class BestCheckpointTracker:
    """Track independent minima for Absolute, Pelvis, and PA validation error."""

    def __init__(self) -> None:
        self.best: dict[CheckpointRole, BestRecord] = {}

    def state_dict(self) -> dict[str, dict[str, object]]:
        """Return a complete, JSON-like snapshot for epoch-boundary resume.

        A partially populated tracker is unsafe to resume because a missing role
        would treat the next validation as its first best and overwrite its
        checkpoint.  Training validates at epoch zero, so every persisted
        epoch-boundary state must contain all three role records.
        """
        missing = [role.value for role in CheckpointRole if role not in self.best]
        if missing:
            raise ValueError(
                "cannot persist incomplete best checkpoint tracker: "
                + ", ".join(missing)
            )
        return {
            role.value: {
                "epoch": int(record.epoch),
                "value": float(record.value),
                "metrics": metrics_as_dict(record.metrics),
            }
            for role, record in self.best.items()
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore all role records, rejecting incomplete or malformed state."""
        if not isinstance(state, Mapping):
            raise ValueError("best checkpoint tracker state must be a mapping")

        expected = {role.value for role in CheckpointRole}
        actual = set(state)
        if actual != expected:
            raise ValueError(
                "best checkpoint tracker roles mismatch: "
                f"expected {sorted(expected)}, got {sorted(actual)}"
            )

        restored: dict[CheckpointRole, BestRecord] = {}
        for role in CheckpointRole:
            raw_record = state[role.value]
            if not isinstance(raw_record, Mapping):
                raise ValueError(f"best checkpoint record for {role.value} must be a mapping")
            try:
                epoch = int(raw_record["epoch"])
                value = float(raw_record["value"])
                raw_metrics = raw_record["metrics"]
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid best checkpoint record for {role.value}"
                ) from error
            if epoch < 0 or not math.isfinite(value):
                raise ValueError(f"invalid epoch/value for {role.value} checkpoint record")
            if not isinstance(raw_metrics, Mapping):
                raise ValueError(f"metrics for {role.value} checkpoint record must be a mapping")
            try:
                metrics = PoseMetrics(
                    absolute_mpjpe_mm=float(raw_metrics["absolute_mpjpe_mm"]),
                    pelvis_mpjpe_mm=float(raw_metrics["pelvis_mpjpe_mm"]),
                    pa_mpjpe_mm=float(raw_metrics["pa_mpjpe_mm"]),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid metrics for {role.value} checkpoint record"
                ) from error
            if not all(math.isfinite(metric) for metric in metrics_as_dict(metrics).values()):
                raise ValueError(f"non-finite metrics for {role.value} checkpoint record")
            expected_value = metric_value_for_role(role, metrics)
            if value != expected_value:
                raise ValueError(
                    f"selection value does not match metrics for {role.value} checkpoint record"
                )
            restored[role] = BestRecord(epoch=epoch, value=value, metrics=metrics)

        self.best = restored

    def update(
        self,
        epoch: int,
        metrics: PoseMetrics,
        save: Callable[[CheckpointRole, BestRecord], None],
    ) -> tuple[CheckpointRole, ...]:
        """Save and remember every role whose validation value strictly improves.

        The callback runs before the tracker commits its in-memory record, so a
        failed checkpoint write cannot make the tracker claim an unsaved best.
        """
        updated: list[CheckpointRole] = []
        for role in CheckpointRole:
            candidate = BestRecord(
                epoch=int(epoch),
                value=metric_value_for_role(role, metrics),
                metrics=metrics,
            )
            previous = self.best.get(role)
            if previous is None or candidate.value < previous.value:
                save(role, candidate)
                self.best[role] = candidate
                updated.append(role)
        return tuple(updated)
