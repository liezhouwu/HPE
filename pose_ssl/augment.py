"""
pose_ssl.augment — SSL 预训练数据增强
========================================
按 ssl.pdf Appendix B 的 ResNet 配置:
  SimCLR / MoCo : JitterCSI (sigma=0.001)
  SwAV          : ChannelShuffle (全局视图) + 局部裁剪 (multi-crop)
所有增强在 GPU 张量上按 batch 执行, 输入形状 (B, 3, 114, 10)。
"""

import torch
import torch.nn.functional as F


def jitter_csi(x, sigma=0.001):
    """高斯噪声 (JitterCSI)。x: (B, 3, 114, 10)。"""
    return x + torch.randn_like(x) * sigma


def channel_shuffle(x):
    """沿天线(通道)维随机置换, 每个样本独立。x: (B, 3, 114, 10)。"""
    B, C = x.shape[0], x.shape[1]
    perm = torch.stack([torch.randperm(C, device=x.device) for _ in range(B)])
    idx = perm.view(B, C, 1, 1).expand(B, C, *x.shape[2:])
    return torch.gather(x, 1, idx)


def subcarrier_crop_resize(x, crop_h):
    """随机裁剪 subcarrier 行窗口并双线性插值回原高度 (SwAV 局部视图)。
    x: (B, 3, 114, 10) -> (B, 3, 114, 10), 内容为 crop_h 行窗口的上采样。
    """
    B, C, H, W = x.shape
    if crop_h >= H:
        return jitter_csi(x, 1e-4)
    top = torch.randint(0, H - crop_h + 1, (B,), device=x.device)
    rows = torch.arange(crop_h, device=x.device)
    idx = (top[:, None] + rows[None, :])            # (B, crop_h)
    idx = idx[:, None, :, None].expand(B, C, crop_h, W)
    crops = torch.gather(x, 2, idx)                 # (B, C, crop_h, W)
    return F.interpolate(crops, size=(H, W), mode='bilinear', align_corners=False)


def time_pad_to(x, target_w=12):
    """时间维右侧零填充到 target_w (Rel-Pos 3x3 网格需要 10→12)。"""
    W = x.shape[-1]
    if W >= target_w:
        return x
    return F.pad(x, (0, target_w - W))
