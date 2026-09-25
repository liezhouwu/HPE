"""与 benchmark/models/mynetwork.py 对齐的 MetaFi++ 编码器。

官方实现先把三根天线拼成一张 CSI 特征图，再送入单个 ResNet-34。
SSL 和监督训练都复用这一编码器，保证下游骨架一致。
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Resize


@dataclass(frozen=True)
class EncoderOutput:
    """官方 ResNet 主干输出的空间特征与全局向量。"""

    feature_map: torch.Tensor
    global_vector: torch.Tensor


class MetaFiEncoder(nn.Module):
    """benchmark 中 posenet 的单分支 ResNet-34 编码部分。"""

    feature_channels: int = 512

    def __init__(self, pretrained_backbone: bool = False) -> None:
        super().__init__()
        weights = torchvision.models.ResNet34_Weights.DEFAULT if pretrained_backbone else None
        backbone = torchvision.models.resnet34(weights=weights)
        self.encoder_conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder_bn1 = backbone.bn1
        self.encoder_relu = backbone.relu
        self.encoder_maxpool = backbone.maxpool  # 官方定义保留；forward 同样不使用。
        self.encoder_layer1 = backbone.layer1
        self.encoder_layer2 = backbone.layer2
        self.encoder_layer3 = backbone.layer3
        self.encoder_layer4 = backbone.layer4
        self.resize = Resize([136, 32])

    @staticmethod
    def _to_official_image(x: torch.Tensor) -> torch.Tensor:
        """将 (B,1,3,114,10) CSI 变为官方网络的 (B,1,114,30) 输入图。"""
        if x.ndim != 5 or x.shape[1:] != (1, 3, 114, 10):
            raise ValueError(f"expected CSI shape (B,1,3,114,10), got {tuple(x.shape)}")
        return torch.flatten(torch.transpose(x, 2, 3), 3, 4)

    def forward(self, x: torch.Tensor) -> EncoderOutput:
        x = self.resize(self._to_official_image(x))
        x = self.encoder_conv1(x)
        x = self.encoder_bn1(x)
        x = self.encoder_relu(x)
        x = self.encoder_layer1(x)
        x = self.encoder_layer2(x)
        x = self.encoder_layer3(x)
        feature_map = self.encoder_layer4(x)
        return EncoderOutput(feature_map, feature_map.mean(dim=(2, 3)))
