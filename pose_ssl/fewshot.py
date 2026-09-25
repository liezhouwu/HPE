"""
pose_ssl.fewshot — 实验三少样本标注抽样
==========================================
从训练序列池按 (scene, subject, action) 整序列抽样 fraction 比例:
  - 同序列 297 帧绝不跨集合 (防泄漏, 与序列级划分同一原则)
  - 按动作分层 + 最大余数法分配名额 (保证动作覆盖均衡)
  - 固定 seed, 所有方法共用同一份抽样 (公平对比前提)
  - select 集不受影响 (模型选择口径跨分数一致)

接受 MMFi_Dataset 或包裹它的 torch Subset (官方划分的 train 侧是 Subset)。
"""

import torch
from collections import defaultdict

from .metafi.label_budget import LabelManifest, sample_kshot, sample_subject_budget


def _sequence_key(item):
    return item['scene'], item['subject'], item['action']


def sample_train_sequences(train_ds, fraction, seed=42):
    """返回 (Subset, metadata)。Subset 基于最底层 MMFi_Dataset。"""
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction 必须在 (0,1], 实际 {fraction}")

    # 穿透 Subset 拿到底层 dataset 与池内索引
    pool_indices = list(range(len(train_ds)))
    base = train_ds
    while isinstance(base, torch.utils.data.Subset):
        pool_indices = [base.indices[i] for i in pool_indices]
        base = base.dataset
    items = base.data_list

    groups = defaultdict(list)          # seq_key -> [pool 内位置]
    by_action = defaultdict(list)       # action -> [seq_key]
    for pos, idx in enumerate(pool_indices):
        key = _sequence_key(items[idx])
        if key not in groups:
            by_action[key[2]].append(key)
        groups[key].append(idx)
    if len(groups) < 2:
        raise ValueError("序列池过小, 无法抽样")

    generator = torch.Generator().manual_seed(seed)
    shuffled = {}
    for action in sorted(by_action):
        keys = sorted(by_action[action])
        order = torch.randperm(len(keys), generator=generator).tolist()
        shuffled[action] = [keys[i] for i in order]

    # 总名额 + 按动作最大余数法分配
    target_total = min(max(round(len(groups) * fraction), 1), len(groups) - 1)
    exact = {a: len(keys) * fraction for a, keys in shuffled.items()}
    quotas = {a: int(exact[a]) for a in shuffled}
    remaining = target_total - sum(quotas.values())
    ranked = sorted(shuffled, key=lambda a: (-(exact[a] - quotas[a]), a))
    for action in ranked[:remaining]:
        quotas[action] += 1

    selected_keys = set()
    for action, keys in shuffled.items():
        selected_keys.update(keys[:quotas[action]])
    selected_indices = sorted(i for key in selected_keys for i in groups[key])
    if not selected_indices:
        raise RuntimeError("抽样结果为空")

    metadata = {
        'fraction': fraction, 'seed': seed,
        'pool_sequences': len(groups),
        'sampled_sequences': len(selected_keys),
        'sampled_samples': len(selected_indices),
        'sampled_keys': [list(k) for k in sorted(selected_keys)],
    }
    return torch.utils.data.Subset(base, selected_indices), metadata

# New action-balanced samplers deliberately coexist with the legacy fraction
# sampler above.  Re-exporting them here keeps legacy callers working while
# requiring new experiments to opt into the new manifest-based API explicitly.
from .metafi.label_budget import LabelManifest, sample_kshot, sample_subject_budget
