"""
pose_ssl.pretrain_methods — 5 个自监督预训练方法
====================================================
按 ssl.pdf (ACM TOSN 2025) 的实现规格移植到 MMFi CSI (3,114,10):

  SimCLR  : 双视图 InfoNCE, JitterCSI σ=0.001, temp=0.1
  MoCo    : 动量编码器 + queue(65536), momentum=0.999, temp=0.2, JitterCSI
  SwAV    : multi-crop (2 全局 ChannelShuffle + 4 局部 subcarrier 裁剪),
            prototypes=5005 (论文 A.3), Sinkhorn 交换预测
  RelPos  : 3×3 网格 (时间维 10→12 零填充), 中心+外围 patch 对,
            patch 双线性缩放回 (114,10) 编码, 8 分类
  MAE     : ViT-csi-small, 75% patch 掩码, 轻量解码器重建 (per-patch 归一化)

统一接口: forward(x) -> (loss, log_dict), x: (B, 3, 114, 10)。
微调时只取其中的 encoder (state_dict 与 pose_ssl.model 的编码器完全一致)。
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .augment import jitter_csi, channel_shuffle, subcarrier_crop_resize, time_pad_to


class Projector(nn.Module):
    """SimCLR 式投影头: Linear-BN-ReLU-Linear。"""

    def __init__(self, in_dim, hidden_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        return self.net(x)


def nt_xent_loss(z1, z2, temperature):
    """NT-Xent (InfoNCE 对称形式)。z1/z2: (B, D) 已 L2 归一化。"""
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)                       # (2B, D)
    sim = z @ z.t() / temperature                        # (2B, 2B)
    # 去掉自身对角
    mask_self = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask_self, float('-inf'))
    # 正样本对: i <-> i+B
    pos_idx = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    labels = pos_idx
    return F.cross_entropy(sim, labels)


class SimCLR(nn.Module):
    def __init__(self, encoder, proj_dim=128, temperature=0.1, jitter_sigma=0.001):
        super().__init__()
        self.encoder = encoder
        self.projector = Projector(encoder.out_dim, encoder.out_dim, proj_dim)
        self.temperature = temperature
        self.sigma = jitter_sigma

    def forward(self, x):
        v1 = jitter_csi(x, self.sigma)
        v2 = jitter_csi(x, self.sigma)
        z1 = F.normalize(self.projector(self.encoder(v1)), dim=1)
        z2 = F.normalize(self.projector(self.encoder(v2)), dim=1)
        loss = nt_xent_loss(z1, z2, self.temperature)
        return loss, {'loss': loss.item()}


class MoCo(nn.Module):
    def __init__(self, encoder, proj_dim=128, queue_size=65536, momentum=0.999,
                 temperature=0.2, jitter_sigma=0.001):
        super().__init__()
        self.encoder_q = encoder
        self.projector_q = Projector(encoder.out_dim, encoder.out_dim, proj_dim)
        self.encoder_k = copy.deepcopy(encoder)
        self.projector_k = copy.deepcopy(self.projector_q)
        for p in self.encoder_k.parameters():
            p.requires_grad = False
        for p in self.projector_k.parameters():
            p.requires_grad = False
        self.momentum = momentum
        self.temperature = temperature
        self.sigma = jitter_sigma
        self.register_buffer('queue', torch.randn(proj_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        self.queue_size = queue_size

    @torch.no_grad()
    def _momentum_update(self):
        for pq, pk in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            pk.data = pk.data * self.momentum + pq.data * (1.0 - self.momentum)
        for pq, pk in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            pk.data = pk.data * self.momentum + pq.data * (1.0 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        B = keys.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + B <= self.queue_size:
            self.queue[:, ptr:ptr + B] = keys.t()
        else:  # 环绕写入
            rem = self.queue_size - ptr
            self.queue[:, ptr:] = keys.t()[:, :rem]
            self.queue[:, :B - rem] = keys.t()[:, rem:]
        ptr = (ptr + B) % self.queue_size
        self.queue_ptr[0] = ptr

    def forward(self, x):
        v1 = jitter_csi(x, self.sigma)
        v2 = jitter_csi(x, self.sigma)
        q = F.normalize(self.projector_q(self.encoder_q(v1)), dim=1)
        with torch.no_grad():
            self._momentum_update()
            k = F.normalize(self.projector_k(self.encoder_k(v2)), dim=1)
        l_pos = (q * k).sum(dim=1, keepdim=True)          # (B, 1)
        l_neg = q @ self.queue.clone().detach()           # (B, K)
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=x.device)
        loss = F.cross_entropy(logits, labels)
        with torch.no_grad():
            self._dequeue_and_enqueue(k)
        return loss, {'loss': loss.item()}


@torch.no_grad()
def sinkhorn(scores, eps, n_iters):
    """Sinkhorn-Knopp 归一化 (官方 SwAV 实现)。scores: (B, K) -> codes (B, K)。"""
    Q = torch.exp(scores / eps).t()                       # (K, B)
    K, B = Q.shape
    Q /= Q.sum()
    for _ in range(n_iters):
        Q /= Q.sum(dim=0, keepdim=True)
        Q /= K
        Q /= Q.sum(dim=1, keepdim=True)
        Q /= B
    Q *= B
    return Q.t()                                          # (B, K)


class SwAV(nn.Module):
    # 原型数说明: ssl.pdf A.3 写 5005, 但 Sinkhorn codes 的行和 ≈ K/B,
    # 只有 batch >= K 时损失量级才正常 (~ln K)。本项目 bs=256, 取 K=256
    # (与旧实验一致) 使 K/B≈1; 偏离论文值已在实验文档中声明。
    def __init__(self, encoder, proj_dim=128, n_prototypes=256, temperature=0.1,
                 eps=0.05, sinkhorn_iters=3, n_local_crops=4, local_crop_h=57,
                 jitter_sigma=0.001):
        super().__init__()
        self.encoder = encoder
        self.projector = Projector(encoder.out_dim, encoder.out_dim, proj_dim)
        self.prototypes = nn.Linear(proj_dim, n_prototypes, bias=False)
        self.temperature = temperature
        self.eps = eps
        self.sinkhorn_iters = sinkhorn_iters
        self.n_local = n_local_crops
        self.local_crop_h = local_crop_h
        self.sigma = jitter_sigma

    def _embed(self, v):
        return F.normalize(self.projector(self.encoder(v)), dim=1)

    def forward(self, x):
        # 原型权重归一化 (SwAV 标准约束)
        with torch.no_grad():
            W = self.prototypes.weight.data
            self.prototypes.weight.data = W / W.norm(dim=1, keepdim=True).clamp(min=1e-8)

        g1 = jitter_csi(channel_shuffle(x), self.sigma)
        g2 = jitter_csi(channel_shuffle(x), self.sigma)
        views = [g1, g2] + [subcarrier_crop_resize(x, self.local_crop_h)
                            for _ in range(self.n_local)]
        z = [self._embed(v) for v in views]
        scores = [self.prototypes(zi) for zi in z]

        with torch.no_grad():
            q1 = sinkhorn(scores[0].detach(), self.eps, self.sinkhorn_iters)
            q2 = sinkhorn(scores[1].detach(), self.eps, self.sinkhorn_iters)

        def cross(q, s):
            return -torch.mean(torch.sum(
                q * F.log_softmax(s / self.temperature, dim=1), dim=1))

        loss = cross(q2, scores[0]) + cross(q1, scores[1])
        local_loss = sum(cross(q1, s) + cross(q2, s) for s in scores[2:])
        loss = loss + local_loss / self.n_local
        return loss, {'loss': loss.item()}


class RelPos(nn.Module):
    """相对位置预测: CSI 填充至 (114,12) 后切 3×3 网格, 中心 patch 与随机
    外围 patch 各自缩放回 (114,10) 编码, 拼接特征做 8 分类。"""

    GRID = 3
    N_CLASSES = 8
    # 外围 8 个位置的 (row, col), 行主序去掉中心 (1,1)
    POSITIONS = [(r, c) for r in range(3) for c in range(3) if (r, c) != (1, 1)]

    def __init__(self, encoder, hidden_dim=256, jitter_sigma=0.001):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(encoder.out_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.N_CLASSES))
        self.sigma = jitter_sigma

    def _patch(self, x_pad, r, c):
        """x_pad: (B,3,114,12) -> (r,c) 网格 patch, 缩放回 (114,10)。"""
        h = x_pad.shape[2] // self.GRID          # 38
        w = x_pad.shape[3] // self.GRID          # 4
        p = x_pad[:, :, r * h:(r + 1) * h, c * w:(c + 1) * w]
        return F.interpolate(p, size=(114, 10), mode='bilinear',
                             align_corners=False)

    def forward(self, x):
        B = x.shape[0]
        x_pad = time_pad_to(x, 12)
        pos_idx = torch.randint(self.N_CLASSES, (1,)).item()
        r, c = self.POSITIONS[pos_idx]
        center = self._patch(x_pad, 1, 1)
        neigh = self._patch(x_pad, r, c)
        f_c = self.encoder(jitter_csi(center, self.sigma))
        f_n = self.encoder(jitter_csi(neigh, self.sigma))
        logits = self.classifier(torch.cat([f_c, f_n], dim=1))
        labels = torch.full((B,), pos_idx, dtype=torch.long, device=x.device)
        loss = F.cross_entropy(logits, labels)
        return loss, {'loss': loss.item(), 'pos_class': pos_idx}


class MAE(nn.Module):
    """掩码自编码: ViT 编码 25% 可见 patch, 轻量解码器重建全部 patch
    (per-patch 归一化后的像素 MSE)。编码器与 ViTCSIEncoder 权重完全共享。"""

    def __init__(self, vit, mask_ratio=0.75, decoder_dim=256, decoder_depth=2,
                 decoder_heads=4, patch_h=6, patch_w=5):
        super().__init__()
        self.vit = vit
        self.mask_ratio = mask_ratio
        self.patch_h, self.patch_w = patch_h, patch_w
        self.n_patches = (114 // patch_h) * (10 // patch_w)   # 19*2 = 38
        embed_dim = vit.out_dim
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos = nn.Parameter(
            torch.zeros(1, self.n_patches, decoder_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim, nhead=decoder_heads,
            dim_feedforward=decoder_dim * 4, activation='gelu',
            batch_first=True, norm_first=True)
        self.decoder_blocks = nn.TransformerEncoder(layer, num_layers=decoder_depth)
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, 3 * patch_h * patch_w)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos, std=0.02)

    def patchify(self, x):
        """(B,3,114,10) -> (B, n_patches, 3*ph*pw), token 顺序与 patch_proj 一致。"""
        B, C, H, W = x.shape
        nh, nw = H // self.patch_h, W // self.patch_w
        p = x.view(B, C, nh, self.patch_h, nw, self.patch_w)
        p = p.permute(0, 2, 4, 3, 5, 1)              # (B, nh, nw, ph, pw, C)
        return p.reshape(B, nh * nw, self.patch_h * self.patch_w * C)

    def forward(self, x):
        B = x.shape[0]
        N = self.n_patches
        patches = self.patchify(x)                            # (B, N, P)
        # per-patch 归一化 (重建目标)
        mean = patches.mean(dim=-1, keepdim=True)
        var = patches.var(dim=-1, keepdim=True, unbiased=True)
        target = (patches - mean) / (var + 1e-6).sqrt()

        n_keep = max(1, int(N * (1 - self.mask_ratio)))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :n_keep]

        z = self.vit.encode_visible(x, ids_keep)              # (B, n_keep, E)
        z = self.decoder_embed(z)                             # (B, n_keep, D)
        mask_tokens = self.mask_token.expand(B, N - n_keep, -1)
        z_full = torch.cat([z, mask_tokens], dim=1)           # (B, N, D)
        idx = ids_restore[:, :, None].expand(B, N, z.shape[-1])
        z_full = torch.gather(z_full, 1, idx)                 # 还原原始顺序
        z_full = z_full + self.decoder_pos
        z_full = self.decoder_norm(self.decoder_blocks(z_full))
        pred = self.decoder_pred(z_full)                      # (B, N, P)

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)                              # 每 patch 平均
        mask = torch.ones(B, N, device=x.device)
        mask[:, :n_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)             # 与被掩码位置对齐
        loss = (loss * mask).sum() / mask.sum()
        return loss, {'loss': loss.item(), 'n_keep': n_keep}
