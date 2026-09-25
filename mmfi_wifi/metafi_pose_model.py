"""官方 MetaFi++ 编码器与姿态头的组合模型。

该模型是 try3 中复现、监督对照和 SSL 微调共同使用的唯一下游骨架。
"""

import torch
import torch.nn as nn

from .metafi_decoder import AxisStats, MetaFiPoseDecoder
from .metafi_encoder import MetaFiEncoder


class MetaFiPoseModel(nn.Module):
    def __init__(
        self,
        dropout_p: float = 0.0,
        target_space: str = "absolute",
        axis_stats: AxisStats | None = None,
        encoder: MetaFiEncoder | None = None,
        decoder: MetaFiPoseDecoder | None = None,
        pretrained_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder or MetaFiEncoder(pretrained_backbone=pretrained_backbone)
        self.decoder = decoder or MetaFiPoseDecoder(dropout_p, target_space, axis_stats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x).feature_map)
