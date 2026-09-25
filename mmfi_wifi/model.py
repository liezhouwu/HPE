"""try3 的官方骨架模型入口。

posenet 与 benchmark/models/mynetwork.py 使用相同的 CSI 展平、ResNet-34 和
卷积姿态头结构；额外参数只为保持原训练脚本调用接口兼容。
"""

import torch.nn as nn

from .metafi_decoder import LEGACY_AXIS_STATS
from .metafi_pose_model import MetaFiPoseModel

GT_AXIS_MEAN = LEGACY_AXIS_STATS.mean
GT_AXIS_STD = LEGACY_AXIS_STATS.std


class posenet(MetaFiPoseModel):
    def __init__(
        self,
        dropout_p: float = 0.0,
        target_space: str = "absolute",
        axis_stats=None,
        pretrained_backbone: bool = False,
    ) -> None:
        super().__init__(
            dropout_p=dropout_p,
            target_space=target_space,
            axis_stats=axis_stats,
            pretrained_backbone=pretrained_backbone,
        )


def weights_init(module: nn.Module) -> None:
    """沿用 benchmark 的卷积和 BatchNorm 初始化规则。"""
    if isinstance(module, nn.Conv2d):
        nn.init.xavier_normal_(module.weight.data)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
