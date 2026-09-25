"""Immutable sequence identities and deterministic sequence-level partitioning."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Collection, Mapping

import torch


@dataclass(frozen=True, order=True)
class SequenceKey:
    """Identity of one indivisible MM-Fi CSI/video sequence."""

    scene: str
    subject: str
    action: str


def sequence_key_from_item(item: Mapping[str, Any]) -> SequenceKey:
    """Create a :class:`SequenceKey` from a MM-Fi ``data_list`` item."""

    return SequenceKey(
        scene=item["scene"],
        subject=item["subject"],
        action=item["action"],
    )


def partition_train_select(
    keys: Collection[SequenceKey],
    val_fraction: float,
    seed: int,
) -> tuple[frozenset[SequenceKey], frozenset[SequenceKey]]:
    """Deterministically split sequence keys into action-stratified train/select sets.

    The quota allocation deliberately matches the legacy engine implementation:
    sorted per-action keys, one seeded torch generator, and maximum-remainder
    allocation of the requested total select sequence count.
    """

    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction 必须在 (0,1), 实际 {val_fraction}")

    all_keys = frozenset(keys)
    if len(all_keys) < 2:
        raise ValueError("至少需要 2 个序列才能划分 train/select")

    by_action: dict[str, list[SequenceKey]] = defaultdict(list)
    for key in all_keys:
        by_action[key.action].append(key)

    generator = torch.Generator().manual_seed(seed)
    shuffled: dict[str, list[SequenceKey]] = {}
    for action in sorted(by_action):
        action_keys = sorted(by_action[action])
        order = torch.randperm(len(action_keys), generator=generator).tolist()
        shuffled[action] = [action_keys[index] for index in order]

    target_total = min(
        max(round(len(all_keys) * val_fraction), 1),
        len(all_keys) - 1,
    )
    exact = {action: len(action_keys) * val_fraction for action, action_keys in shuffled.items()}
    quotas = {action: int(exact[action]) for action in shuffled}
    remaining = target_total - sum(quotas.values())
    ranked = sorted(shuffled, key=lambda action: (-(exact[action] - quotas[action]), action))
    for action in ranked[:remaining]:
        quotas[action] += 1

    select_keys = frozenset(
        key
        for action, action_keys in shuffled.items()
        for key in action_keys[:quotas[action]]
    )
    train_keys = all_keys - select_keys
    if not train_keys or not select_keys:
        raise RuntimeError("序列级划分产生空 train/select 集")
    return train_keys, select_keys
