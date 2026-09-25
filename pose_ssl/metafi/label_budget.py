"""Deterministic sequence-level label budgets for MetaFi SSL experiments."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Collection, Mapping, TypeVar

import torch

from mmfi_wifi.sequence_keys import SequenceKey


LABEL_MANIFEST_SCHEMA_VERSION: int = 1
LABEL_MANIFEST_ALGORITHM_VERSION: str = "metafi-label-budget-v1"
_SUPPORTED_SPLITS = frozenset({"random_split", "cross_subject_split", "cross_scene_split"})
_S3_TRAIN_SCENES: tuple[str, str, str] = ("E01", "E02", "E03")
_T = TypeVar("_T")


def _normalise_keys(
    keys: Collection[SequenceKey], *, field_name: str = "internal_train_keys"
) -> frozenset[SequenceKey]:
    result = frozenset(keys)
    if any(not isinstance(key, SequenceKey) for key in result):
        raise TypeError(f"{field_name} 只能包含 SequenceKey")
    if not result:
        raise ValueError(f"{field_name} 不能为空")
    return result


def _validate_positive_int(value: int, *, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} 必须是 int")
    if value < 1:
        raise ValueError(f"{name} 必须大于 0")


def _validate_seed(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed 必须是 int")


def _shuffled(values: Collection[_T], generator: torch.Generator) -> list[_T]:
    ordered = sorted(values)
    order = torch.randperm(len(ordered), generator=generator).tolist()
    return [ordered[index] for index in order]


def _group_by_action(keys: Collection[SequenceKey]) -> dict[str, list[SequenceKey]]:
    groups: dict[str, list[SequenceKey]] = defaultdict(list)
    for key in keys:
        groups[key.action].append(key)
    return {action: sorted(action_keys) for action, action_keys in groups.items()}


def _sample_cross_subject(
    action_keys: Collection[SequenceKey], k: int, generator: torch.Generator
) -> list[SequenceKey]:
    by_subject: dict[str, list[SequenceKey]] = defaultdict(list)
    for key in action_keys:
        by_subject[key.subject].append(key)
    if k > len(by_subject):
        raise ValueError(
            "cross_subject_split 要求每个动作使用不同被试；"
            f"请求 k={k}，可用被试数={len(by_subject)}"
        )
    return [
        _shuffled(by_subject[subject], generator)[0]
        for subject in _shuffled(by_subject.keys(), generator)[:k]
    ]


def _sample_cross_scene(
    action_keys: Collection[SequenceKey], k: int, generator: torch.Generator
) -> list[SequenceKey]:
    by_scene: dict[str, list[SequenceKey]] = defaultdict(list)
    for key in action_keys:
        by_scene[key.scene].append(key)
    unexpected = sorted(set(by_scene) - set(_S3_TRAIN_SCENES))
    if unexpected:
        raise ValueError(
            "cross_scene_split 的训练侧只能包含 E01/E02/E03，"
            f"实际还包含 {unexpected}"
        )

    base, remainder = divmod(k, len(_S3_TRAIN_SCENES))
    candidates = [
        scene for scene in _S3_TRAIN_SCENES
        if len(by_scene.get(scene, ())) >= base + 1
    ]
    if any(len(by_scene.get(scene, ())) < base for scene in _S3_TRAIN_SCENES) or len(candidates) < remainder:
        capacities = {scene: len(by_scene.get(scene, ())) for scene in _S3_TRAIN_SCENES}
        raise ValueError(
            "cross_scene_split 无法在 E01/E02/E03 间保持场景均衡："
            f"k={k}, capacities={capacities}"
        )

    quotas = {scene: base for scene in _S3_TRAIN_SCENES}
    for scene in _shuffled(candidates, generator)[:remainder]:
        quotas[scene] += 1
    return [
        key
        for scene in _S3_TRAIN_SCENES
        for key in _shuffled(by_scene[scene], generator)[:quotas[scene]]
    ]


def sample_kshot(
    internal_train_keys: Collection[SequenceKey],
    k: int,
    seed: int,
    split: str,
) -> frozenset[SequenceKey]:
    """Select exactly ``k`` complete sequences per action.

    S2 chooses different subjects within each action whenever ``k`` is no
    larger than that action's number of training subjects.  S3 distributes
    each action's quota over E01/E02/E03, with scene counts differing by at
    most one.  Impossible requests fail before a partial manifest is returned.
    """

    _validate_positive_int(k, name="k")
    _validate_seed(seed)
    if split not in _SUPPORTED_SPLITS:
        raise ValueError(f"不支持的 split: {split!r}")

    keys = _normalise_keys(internal_train_keys)
    by_action = _group_by_action(keys)
    smallest_pool = min(len(action_keys) for action_keys in by_action.values())
    if k > smallest_pool:
        raise ValueError(
            "k 不能大于最小每个动作池；"
            f"请求 k={k}，最小每个动作池={smallest_pool}"
        )

    generator = torch.Generator().manual_seed(seed)
    selected: set[SequenceKey] = set()
    for action in sorted(by_action):
        action_keys = by_action[action]
        if split == "cross_subject_split":
            chosen = _sample_cross_subject(action_keys, k, generator)
        elif split == "cross_scene_split":
            chosen = _sample_cross_scene(action_keys, k, generator)
        else:
            chosen = _shuffled(action_keys, generator)[:k]
        if len(chosen) != k:
            raise RuntimeError(f"动作 {action} 未选择恰好 {k} 条完整序列")
        selected.update(chosen)

    expected = k * len(by_action)
    if len(selected) != expected:
        raise RuntimeError("K-shot 抽样产生重复 sequence key")
    return frozenset(selected)


def sample_subject_budget(
    internal_train_keys: Collection[SequenceKey], n_subjects: int, seed: int
) -> frozenset[SequenceKey]:
    """Select every complete sequence for a deterministic set of full subjects."""

    _validate_positive_int(n_subjects, name="n_subjects")
    _validate_seed(seed)
    keys = _normalise_keys(internal_train_keys)
    all_actions = frozenset(key.action for key in keys)
    actions_by_subject: dict[str, set[str]] = defaultdict(set)
    for key in keys:
        actions_by_subject[key.subject].add(key.action)
    eligible_subjects = sorted(
        subject
        for subject, subject_actions in actions_by_subject.items()
        if subject_actions == all_actions
    )
    if n_subjects > len(eligible_subjects):
        raise ValueError(
            f"n_subjects={n_subjects} 超过可用完整 subject 数 {len(eligible_subjects)}"
        )

    generator = torch.Generator().manual_seed(seed)
    chosen_subjects = set(_shuffled(eligible_subjects, generator)[:n_subjects])
    selected = frozenset(key for key in keys if key.subject in chosen_subjects)
    if not selected:
        raise RuntimeError("subject budget 抽样结果为空")
    return selected


def _serialise_keys(keys: Collection[SequenceKey]) -> list[list[str]]:
    return [[key.scene, key.subject, key.action] for key in sorted(keys)]


@dataclass(frozen=True)
class LabelManifest:
    """Immutable, path-independent K-shot or subject-budget selection."""

    mode: str
    selected_keys: Collection[SequenceKey]
    fingerprint: str = field(init=False)
    algorithm_version: str = field(
        default=LABEL_MANIFEST_ALGORITHM_VERSION,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str) or not self.mode:
            raise ValueError("mode 必须是非空字符串")
        object.__setattr__(
            self,
            "selected_keys",
            _normalise_keys(self.selected_keys, field_name="selected_keys"),
        )
        canonical = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        object.__setattr__(self, "fingerprint", hashlib.sha256(canonical.encode("utf-8")).hexdigest())

    @classmethod
    def create(cls, mode: str, selected_keys: Collection[SequenceKey]) -> "LabelManifest":
        return cls(mode=mode, selected_keys=selected_keys)

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": LABEL_MANIFEST_SCHEMA_VERSION,
            "algorithm_version": LABEL_MANIFEST_ALGORITHM_VERSION,
            "mode": self.mode,
            "selected_keys": _serialise_keys(self.selected_keys),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.canonical_payload()
        payload["fingerprint"] = self.fingerprint
        return payload

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
