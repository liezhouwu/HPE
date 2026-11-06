from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch


SCENE_TO_ID = {"E01": 0, "E02": 1, "E03": 2}


def scene_to_environment_id(scene: str) -> int:
    try:
        return SCENE_TO_ID[scene]
    except KeyError as exc:
        raise ValueError(f"scene {scene!r} is not a source environment") from exc


def records_to_environment_ids(records: Iterable[Any]) -> torch.Tensor:
    ids = []
    for record in records:
        scene = record.sequence_key.scene
        ids.append(scene_to_environment_id(scene))
    return torch.tensor(ids, dtype=torch.long)


def validate_source_only_records(records: Iterable[Any]) -> None:
    for record in records:
        scene = record.sequence_key.scene
        if scene not in SCENE_TO_ID:
            raise ValueError(
                f"EIL-JEPA source-only pretraining received {scene!r}; "
                "E04 must remain held out."
            )


def validate_csi_shape(x: torch.Tensor) -> None:
    if x.ndim == 5 and x.shape[1] == 1:
        x = x[:, 0]
    if tuple(x.shape[1:]) != (3, 114, 10):
        raise ValueError(f"expected CSI shape (B,3,114,10), got {tuple(x.shape)}")

