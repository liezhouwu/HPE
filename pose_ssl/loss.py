"""
pose_ssl.loss — 姿态回归损失 (实验二/三共用)
================================================
L = MSE(pred, gt) + bone_weight * BoneLoss(pred, gt)

BoneLoss: 骨骼长度一致性损失。对命名骨骼拓扑中的关节边分别计算预测与 GT 的
关节间欧氏长度，并取 L1 差。坐标为米制，典型骨长 0.2–0.7m，损失量级约 0.05，
乘 bone_weight=0.05 后与 MSE 同量级。

``legacy15`` 固化了旧实验实际使用的 15 条边。它不是完整的 COCO-17 图：耳关节
索引 3 和 4 不受骨长约束。不要在受控消融中替换该图，否则会额外改变实验变量。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn

# COCO-17 风格树形骨架的旧实验 15 条边（关节索引与 mmfi_wifi.metrics.KP_NAMES 一致）。
# 头-肩-臂 x2，肩-髋 x2，髋间，髋-膝-踝 x2；耳关节 3、4 不受约束。
COCO_BONES = [
    (0, 1), (0, 2), (0, 5), (0, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# Named topology contract for BoneLoss ablations.  Preserve the legacy math exactly.
BONE_SETS: dict[str, tuple[tuple[int, int], ...]] = {
    "legacy15": tuple(COCO_BONES),
}


def _resolve_bone_set(bone_set: str) -> tuple[tuple[int, int], ...]:
    """Return a named topology or fail closed for an unsupported name."""
    try:
        return BONE_SETS[bone_set]
    except KeyError as exc:
        available = ", ".join(sorted(BONE_SETS))
        raise ValueError(
            f"unknown bone_set {bone_set!r}; expected one of: {available}"
        ) from exc


def bone_lengths(
    pose: torch.Tensor,
    bones: Sequence[tuple[int, int]],
) -> torch.Tensor:
    """Return per-edge Euclidean lengths for a ``(B, 17, 3)`` pose batch."""
    i = torch.tensor([bone[0] for bone in bones], device=pose.device)
    j = torch.tensor([bone[1] for bone in bones], device=pose.device)
    return torch.norm(pose[:, i] - pose[:, j], dim=-1)


class BoneLoss(nn.Module):
    """L1 difference between predicted and target bone lengths for one named topology."""

    def __init__(self, bone_set: str = "legacy15") -> None:
        super().__init__()
        self.bone_set = bone_set
        self.bones = _resolve_bone_set(bone_set)

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        lp = bone_lengths(pred, self.bones)
        lg = bone_lengths(gt, self.bones)
        return torch.abs(lp - lg).mean()


class PoseLoss(nn.Module):
    """MSE plus a weighted named BoneLoss topology.

    The default ``legacy15`` topology is the exact 15-edge graph used by the
    pre-existing experiments.  BoneLoss ablations must select a named topology.
    """

    def __init__(self, bone_weight: float = 0.05, bone_set: str = "legacy15") -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.bone = BoneLoss(bone_set=bone_set)
        self.bone_weight = bone_weight
        self.bone_set = bone_set

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mse(pred, target) + self.bone_weight * self.bone(pred, target)

def build_ablation_pose_loss(config: Mapping[str, object]) -> PoseLoss:
    """Build BoneLoss ablation loss from a config with an explicit topology.

    This boundary intentionally does not inherit :class:`PoseLoss`'s legacy
    ``bone_set`` default. Every controlled BoneLoss ablation must record the
    topology it uses, so an omitted or blank name is rejected before training.
    """
    if not isinstance(config, Mapping):
        raise ValueError("ablation loss config must be a mapping with bone_set")

    bone_set = config.get("bone_set")
    if not isinstance(bone_set, str) or not bone_set.strip():
        raise ValueError(
            "BoneLoss ablation config must contain a nonempty bone_set"
        )

    bone_weight = config.get("bone_weight", 0.05)
    return PoseLoss(bone_weight=bone_weight, bone_set=bone_set)
