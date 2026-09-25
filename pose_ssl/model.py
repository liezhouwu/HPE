"""
pose_ssl — SSL 对比实验的模型定义 (实验二/三共用)
====================================================
实验二 Sup 对照组与后续 SSL 微调共用同一套下游模型, 仅编码器初始化不同
(单变量原则)。

Backbone:
  resnet18 : torchvision ResNet-18, conv1 改为 in_channels=3 ——
             3 天线直接作为 3 个输入通道, CSI (3,114,10) 视作图像
             (subcarrier=H, time=W), 与 ssl.pdf "CSI as image" 惯例一致。
  vit_csi_small : ViT (embed 768 / 6 层 / 6 头 / FFN 3072, patch 6x5 →
             38 tokens + CLS), 按 ssl.pdf Appendix A.5 的 csi-small 规格。
             供 MAE 预训练与 Sup-ViT 对照使用。

回归头 PoseHead: Linear(in→256)-BN-ReLU-Dropout → (256→128) → (128→51),
reshape (17,3)。末层 Linear 无 BN (避免绝对坐标的 BN 阻塞问题,
见 mmfi_wifi/model.py 头注), bias 用训练集 GT 逐轴均值初始化。

输入:  (B, 1, 3, 114, 10)  (engine 数据管线输出后 unsqueeze(1) 的形态)
输出:  (B, 17, 3) 绝对相机系坐标 (米)
"""

import math

import torch
import torch.nn as nn
import torchvision

from mmfi_wifi.model import GT_AXIS_MEAN


class ResNet18Encoder(nn.Module):
    """ResNet-18, 首层 conv 适配 3 通道 CSI '图像'。输出 (B, 512)。"""

    def __init__(self, in_channels=3):
        super().__init__()
        net = torchvision.models.resnet18(weights=None)
        net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                              padding=3, bias=False)
        self.features = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3, net.layer4)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = 512

    def forward(self, x):
        x = self.features(x)
        return torch.flatten(self.pool(x), 1)


class ViTCSIEncoder(nn.Module):
    """ViT-csi-small (ssl.pdf A.5): 768 维 / 6 层 / 6 头 / FFN 3072。
    CSI (3,114,10) 按 6x5 patch 切分 → 19x2=38 tokens + CLS。输出 (B, 768)。
    """

    def __init__(self, img_h=114, img_w=10, patch_h=6, patch_w=5, in_channels=3,
                 embed_dim=768, depth=6, num_heads=6, ffn_dim=3072):
        super().__init__()
        assert img_h % patch_h == 0 and img_w % patch_w == 0, \
            f"patch 尺寸 {patch_h}x{patch_w} 无法整除输入 {img_h}x{img_w}"
        self.nh, self.nw = img_h // patch_h, img_w // patch_w
        self.patch_proj = nn.Conv2d(in_channels, embed_dim,
                                    kernel_size=(patch_h, patch_w),
                                    stride=(patch_h, patch_w))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        n_tokens = self.nh * self.nw + 1
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ffn_dim,
            activation='gelu', batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.out_dim = embed_dim
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_proj(x)                       # (B, E, nh, nw)
        x = x.flatten(2).transpose(1, 2)             # (B, nh*nw, E)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.norm(self.blocks(x))
        return x[:, 0]

    def patch_tokens(self, x):
        """不含 CLS 的 patch token 嵌入 (MAE 预训练用)。
        返回 (B, nh*nw, E), 加了 patch 部分的 position embedding。"""
        x = self.patch_proj(x).flatten(2).transpose(1, 2)   # (B, n, E)
        return x + self.pos_embed[:, 1:]

    def encode_visible(self, x, ids_keep):
        """仅编码可见 token (MAE 掩码预训练)。ids_keep: (B, n_keep) 长整型。
        返回 (B, n_keep, E)。与 forward 共享全部权重 (state_dict 不变)。"""
        B, n_keep = ids_keep.shape
        tokens = self.patch_tokens(x)                        # (B, n, E)
        idx = ids_keep[:, :, None].expand(B, n_keep, tokens.shape[-1])
        tokens = torch.gather(tokens, 1, idx)
        return self.norm(self.blocks(tokens))


class PoseHead(nn.Module):
    """MLP 回归头: in→256(BN+ReLU+Dropout)→128→51 → reshape (17,3)。
    末层无 BN; bias 初始化为 GT 逐轴均值 (17 关节共享), 训练从正确量级起步。
    """

    def __init__(self, in_dim, dropout_p=0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 51)
        self._init_output()

    def _init_output(self):
        nn.init.xavier_normal_(self.fc3.weight)
        shift = torch.tensor(GT_AXIS_MEAN, dtype=torch.float32)
        with torch.no_grad():
            self.fc3.bias.copy_(shift.repeat(17))

    def forward(self, f):
        x = self.drop(torch.relu(self.bn1(self.fc1(f))))
        x = torch.relu(self.fc2(x))
        return self.fc3(x).view(-1, 17, 3)


class PoseModel(nn.Module):
    """backbone + PoseHead。输入 (B,1,3,114,10) → 输出 (B,17,3)。"""

    def __init__(self, backbone='resnet18', dropout_p=0.3):
        super().__init__()
        if backbone == 'resnet18':
            self.encoder = ResNet18Encoder(in_channels=3)
        elif backbone == 'vit_csi_small':
            self.encoder = ViTCSIEncoder()
        else:
            raise ValueError(f"未知 backbone: {backbone!r}")
        self.backbone = backbone
        self.head = PoseHead(self.encoder.out_dim, dropout_p)

    def forward(self, x):
        x = x.squeeze(1)          # (B,1,3,114,10) -> (B,3,114,10)
        f = self.encoder(x)
        return self.head(f)


def _linear_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None and m.bias.abs().sum() > 0:
            pass  # fc3 bias 已用 GT 均值初始化, 不覆盖
        elif m.bias is not None:
            nn.init.zeros_(m.bias)


def build_pose_model(backbone='resnet18', dropout_p=0.3, target_space='absolute'):
    """模型工厂 (签名与 engine 的 model_factory 约定一致)。
    仅支持 absolute 官方口径 (实验二起统一)。
    """
    if target_space != 'absolute':
        raise ValueError("pose_ssl 模型仅支持 target_space='absolute'")
    model = PoseModel(backbone=backbone, dropout_p=dropout_p)
    model.apply(_linear_xavier)   # head 内未手工初始化的 Linear 用 Xavier
    return model
