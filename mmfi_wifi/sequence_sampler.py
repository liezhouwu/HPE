"""序列均衡采样器，用于少量标注序列的过拟合诊断。"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator
from torch.utils.data import Sampler, Subset

from .sequence_keys import sequence_key_from_item


class SequenceBalancedSampler(Sampler[int]):
    """每个 epoch 从每条序列采样固定数量帧，并交错输出。"""

    def __init__(
        self,
        dataset,
        frames_per_sequence: int = 16,
        seed: int = 42,
    ) -> None:
        if frames_per_sequence < 1:
            raise ValueError("frames_per_sequence 必须为正数")
        if isinstance(dataset, Subset):
            source = dataset.dataset
            source_indices = list(dataset.indices)
        else:
            source = dataset
            source_indices = list(range(len(dataset)))
        if not hasattr(source, "data_list"):
            raise TypeError("SequenceBalancedSampler 需要 Dataset.data_list")

        groups: dict[object, list[int]] = defaultdict(list)
        for local_index, source_index in enumerate(source_indices):
            key = sequence_key_from_item(source.data_list[source_index])
            groups[key].append(local_index)
        if not groups:
            raise ValueError("SequenceBalancedSampler 不能处理空 Dataset")
        self.groups = tuple(tuple(indices) for _, indices in sorted(groups.items()))
        self.frames_per_sequence = int(frames_per_sequence)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.groups) * self.frames_per_sequence

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        selected: list[list[int]] = []
        for group in self.groups:
            if len(group) >= self.frames_per_sequence:
                values = rng.sample(list(group), self.frames_per_sequence)
            else:
                values = [rng.choice(group) for _ in range(self.frames_per_sequence)]
            rng.shuffle(values)
            selected.append(values)

        # 交错不同序列，避免一个 batch 被同一条序列占满。
        result: list[int] = []
        for frame_index in range(self.frames_per_sequence):
            order = list(range(len(selected)))
            rng.shuffle(order)
            result.extend(selected[group_index][frame_index] for group_index in order)
        return iter(result)
