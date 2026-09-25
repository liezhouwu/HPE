"""与 benchmark/models/mynetwork.py 对齐的姿态解码器。

保持官方卷积、Tanh 和 (1,4) 平均池化顺序；没有额外的 Transformer、
Dropout 或输出仿射层。
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class AxisStats:
    """兼容旧训练入口的坐标统计量占位类型。"""

    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    source_fingerprint: str = "official-skeleton"


LEGACY_AXIS_STATS = AxisStats((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))


class MetaFiPoseDecoder(nn.Module):
    """benchmark posenet 的卷积姿态头。"""

    def __init__(self, dropout_p: float = 0.0, target_space: str = "absolute", axis_stats: AxisStats | None = None) -> None:
        super().__init__()
        self.dropout_p = float(dropout_p)  # 保留参数接口，官方骨架不使用 Dropout。
        self.target_space = target_space
        self.axis_stats = axis_stats
        self.decode = nn.Sequential(
            nn.Conv2d(512, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.m = nn.AvgPool2d((1, 4))
        self.bn1 = nn.BatchNorm2d(3)  # benchmark 中定义但未参与 forward，保留参数骨架。
        self.bn2 = nn.BatchNorm2d(512)
        self.rl = nn.ReLU(inplace=True)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        if feature_map.ndim != 4 or feature_map.shape[1:] != (512, 17, 4):
            raise ValueError(f"expected feature map (B,512,17,4), got {tuple(feature_map.shape)}")
        x = self.decode(feature_map)
        x = self.m(x).squeeze(dim=3)
        return torch.transpose(x, 1, 2)
