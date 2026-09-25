"""严格 SSL 预训练的小样本 sequence 子集。

按动作固定比例抽取完整 sequence；S3 再尽量平衡 E01/E02/E03，
抽样结果由 manifest 固化，所有 SSL 方法可复用同一批无标签 CSI。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection
import math

import torch

from mmfi_wifi.data_manifest import DataManifest
from mmfi_wifi.sequence_keys import SequenceKey


def _validate_fraction(fraction: float) -> float:
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        raise TypeError("pretrain_sequence_fraction 必须是数值")
    fraction = float(fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("pretrain_sequence_fraction 必须在 (0, 1] 内")
    return fraction


def _shuffle(values: list[SequenceKey], generator: torch.Generator) -> list[SequenceKey]:
    order = torch.randperm(len(values), generator=generator).tolist()
    return [values[index] for index in order]


def _sample_action_keys(
    keys: list[SequenceKey],
    quota: int,
    *,
    split: str,
    generator: torch.Generator,
) -> list[SequenceKey]:
    if quota >= len(keys):
        return list(keys)
    if split != "cross_scene_split":
        return _shuffle(sorted(keys), generator)[:quota]

    # S3 训练侧包含 E01/E02/E03；按场景轮转抽样，避免小样本只来自一个场景。
    by_scene: dict[str, list[SequenceKey]] = defaultdict(list)
    for key in keys:
        by_scene[key.scene].append(key)
    selected: list[SequenceKey] = []
    # 场景必须按名字固定顺序消费随机数：dict/frozenset 的迭代序随字符串哈希逐进程变化，
    # 否则同一 (清单, fraction, seed) 在不同进程会抽出不同子集，子集清单也就无法复算。
    pools = {scene: _shuffle(sorted(by_scene[scene]), generator) for scene in sorted(by_scene)}
    while len(selected) < quota and any(pools.values()):
        for scene in sorted(pools):
            if pools[scene] and len(selected) < quota:
                selected.append(pools[scene].pop())
    return selected


def sample_pretrain_sequences(
    internal_train_keys: Collection[SequenceKey],
    *,
    fraction: float,
    seed: int,
    split: str,
) -> frozenset[SequenceKey]:
    """按动作分层抽取完整无标签 sequence。

    每个动作的名额为 round(action_pool × fraction)，至少一条；这样 P1/P2/P3
    都保持动作覆盖，而不是把少量样本集中在少数动作上。
    """

    fraction = _validate_fraction(fraction)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed 必须是整数")
    keys = frozenset(internal_train_keys)
    if not keys or any(not isinstance(key, SequenceKey) for key in keys):
        raise ValueError("internal_train_keys 必须是非空 SequenceKey 集合")

    by_action: dict[str, list[SequenceKey]] = defaultdict(list)
    for key in keys:
        by_action[key.action].append(key)
    generator = torch.Generator().manual_seed(seed)
    selected: set[SequenceKey] = set()
    for action in sorted(by_action):
        action_keys = by_action[action]
        quota = max(1, round(len(action_keys) * fraction))
        selected.update(_sample_action_keys(action_keys, quota, split=split, generator=generator))
    return frozenset(selected)


def make_small_pretrain_manifest(
    manifest: DataManifest,
    *,
    fraction: float,
    seed: int,
) -> DataManifest:
    """复制数据边界，仅将 pretrain_keys 替换为固定分层子集。"""

    if manifest.scope != "strict":
        raise ValueError("小样本入口当前仅支持 strict SSL")
    selected = sample_pretrain_sequences(
        manifest.internal_train_keys, fraction=fraction, seed=seed, split=manifest.split
    )
    return DataManifest(
        protocol=manifest.protocol,
        split=manifest.split,
        seed=seed,
        scope=manifest.scope,
        official_train_keys=manifest.official_train_keys,
        internal_train_keys=manifest.internal_train_keys,
        pretrain_keys=selected,
        select_keys=manifest.select_keys,
        test_keys=manifest.test_keys,
    )


def selection_summary(manifest: DataManifest) -> dict[str, int]:
    """返回总量与每动作选择数量，用于记录小样本计划。"""

    counts: dict[str, int] = defaultdict(int)
    for key in manifest.pretrain_keys:
        counts[key.action] += 1
    return {
        "pretrain_sequences": len(manifest.pretrain_keys),
        "actions": len(counts),
        "min_sequences_per_action": min(counts.values()),
        "max_sequences_per_action": max(counts.values()),
    }
