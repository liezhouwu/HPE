from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from WIFIJEPA.src.model import StructuredViTEncoder


# MM-Fi H36M 17-joint topology used by the existing evaluator.
BONES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12),
    (8, 13), (13, 14), (14, 15),
)
PARTS: tuple[tuple[int, ...], ...] = (
    (0,), (1, 2, 3), (4, 5, 6), (7, 8, 9, 10), (11, 12), (13, 14, 15, 16),
)
PART_COUNTS = (20, 30, 25, 25, 25, 25)


def root_relative(pose: Tensor) -> Tensor:
    return pose - pose[:, :1]


def limb_length_loss(pred: Tensor, target: Tensor) -> Tensor:
    values = []
    for left, right in BONES:
        pred_len = (pred[:, left] - pred[:, right]).norm(dim=-1)
        target_len = (target[:, left] - target[:, right]).norm(dim=-1)
        values.append((pred_len - target_len).abs())
    return torch.stack(values, dim=1).mean()


class SkeletonGraphAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, depth: int = 2, dropout: float = 0.0):
        super().__init__()
        self.part_adjacency = torch.tensor(
            [[1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0],
             [1, 0, 0, 1, 0, 0], [1, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 1]],
            dtype=torch.bool,
        )
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=dim * 4,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, part_tokens: Tensor) -> Tensor:
        # part_tokens is (B, six parts, D); the mask blocks unrelated limbs.
        mask = ~self.part_adjacency.to(part_tokens.device)
        x = part_tokens
        for block in self.blocks:
            x = block(x, src_key_padding_mask=None, src_mask=mask)
        return self.norm(x)


class RePosStructure(nn.Module):
    """Stage 1: body structure branch trained only on root-relative pose."""

    def __init__(
        self,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        n_time: int = 10,
        n_links: int = 3,
        n_subcarriers: int = 114,
        query_depth: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = StructuredViTEncoder(
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            ffn_dim=ffn_dim, n_time=n_time, n_links=n_links,
        )
        self.part_queries = nn.Parameter(torch.randn(1, 6, embed_dim) * 0.02)
        self.part_expanders = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, count * embed_dim), nn.GELU())
            for count in PART_COUNTS
        ])
        self.part_pos = nn.ParameterList([
            nn.Parameter(torch.randn(1, count, embed_dim) * 0.02) for count in PART_COUNTS
        ])
        cross_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.part_cross = nn.TransformerDecoder(cross_layer, num_layers=1)
        self.sga = SkeletonGraphAttention(embed_dim, heads=num_heads, depth=2, dropout=dropout)
        self.joint_queries = nn.Parameter(torch.randn(1, 17, embed_dim) * 0.02)
        joint_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.joint_decoder = nn.TransformerDecoder(joint_layer, num_layers=query_depth)
        self.joint_norm = nn.LayerNorm(embed_dim)
        self.pose_head = nn.Linear(embed_dim, 3)
        self.gcn = nn.Linear(embed_dim, embed_dim)
        self.register_buffer("joint_adjacency", self._make_joint_adjacency(), persistent=False)

    @staticmethod
    def _make_joint_adjacency() -> Tensor:
        adjacency = torch.eye(17)
        for left, right in BONES:
            adjacency[left, right] = 1
            adjacency[right, left] = 1
        degree = adjacency.sum(dim=1, keepdim=True).clamp_min(1.0)
        return adjacency / degree

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 5 and x.shape[1] == 1:
            x = x.squeeze(1)
        encoded = self.encoder(x, return_tokens=True)
        memory = encoded[:, 1:]
        queries = self.part_queries.expand(x.shape[0], -1, -1)
        parts = self.part_cross(queries, memory)
        expanded = []
        for index, (expand, pos) in enumerate(zip(self.part_expanders, self.part_pos, strict=True)):
            count = PART_COUNTS[index]
            expanded.append(expand(parts[:, index]).view(x.shape[0], count, -1) + pos)
        body_tokens = torch.cat(expanded, dim=1)
        part_features = torch.stack([
            body_tokens[:, start:start + count].mean(dim=1)
            for start, count in self._part_ranges()
        ], dim=1)
        part_features = self.sga(part_features)
        joint_queries = self.joint_queries.expand(x.shape[0], -1, -1)
        decoded = self.joint_decoder(joint_queries, part_features)
        decoded = self.joint_norm(decoded)
        # One lightweight skeleton mixing step before the coordinate head.
        decoded = decoded + self.gcn(torch.einsum("ij,bjd->bid", self.joint_adjacency, decoded))
        relative = self.pose_head(decoded)
        return root_relative(relative)

    @staticmethod
    def _part_ranges() -> tuple[tuple[int, int], ...]:
        ranges = []
        start = 0
        for count in PART_COUNTS:
            ranges.append((start, start + count))
            start += count
        return tuple(ranges)


class ASPN(nn.Module):
    """Stage 2 amplitude spatial prior: learned phase + beamforming + IFFT."""

    def __init__(self, n_time: int = 10, n_links: int = 3, n_subcarriers: int = 114, n_angle: int = 90, n_delay: int = 64, phase_mode: str = "paper"):
        super().__init__()
        self.n_time = int(n_time)
        self.n_links = int(n_links)
        self.n_subcarriers = int(n_subcarriers)
        self.n_angle = int(n_angle)
        self.n_delay = int(n_delay)
        self.phase_mode = str(phase_mode)
        if self.phase_mode not in {"paper", "per_link"}:
            raise ValueError("phase_mode must be paper or per_link")
        phase_in = n_links * n_time if self.phase_mode == "paper" else n_time
        phase_out = n_links * n_time if self.phase_mode == "paper" else 1
        self.phase_net = nn.Sequential(
            nn.Conv1d(phase_in, 32, kernel_size=5, padding=2), nn.GELU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(16, phase_out, kernel_size=1), nn.Tanh(),
        )
        angle = torch.linspace(-math.pi / 2, math.pi / 2, n_angle)
        rx = torch.arange(n_links, dtype=torch.float32)
        steering = torch.exp(1j * torch.outer(torch.sin(angle), rx))
        self.register_buffer("steering", steering, persistent=False)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv2d(16, 8, kernel_size=3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 8)), nn.Flatten(),
        )
        self.xy_head = nn.Sequential(nn.Linear(8 * 4 * 8, 128), nn.GELU(), nn.Linear(128, 2))
        self.z_head = nn.Sequential(nn.Linear(8 * 4 * 8, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x: Tensor) -> Tensor:
        features = self.forward_features(x)
        return torch.cat([self.xy_head(features), self.z_head(features)], dim=-1)

    def forward_features(self, x: Tensor) -> Tensor:
        heatmap = self.forward_heatmap(x)
        return self.heatmap_head(heatmap)

    def forward_heatmap(self, x: Tensor) -> Tensor:
        if x.ndim == 5 and x.shape[1] == 1:
            x = x.squeeze(1)
        amp = x.permute(0, 1, 3, 2)
        if self.phase_mode == "paper":
            phase = self.phase_net(amp.reshape(amp.shape[0], self.n_links * self.n_time, self.n_subcarriers))
            phase = phase.view(amp.shape[0], self.n_links, self.n_time, self.n_subcarriers).permute(0, 1, 3, 2)
        else:
            phase = self.phase_net(amp.reshape(amp.shape[0] * self.n_links, self.n_time, self.n_subcarriers))
            phase = phase.view(amp.shape[0], self.n_links, self.n_subcarriers, 1).expand(-1, -1, -1, self.n_time)
        signal = x.to(torch.float32) * torch.exp(1j * math.pi * phase)
        # Learned beamforming over the physical link dimension.
        angular = torch.einsum("blst,al->bast", signal, self.steering.conj())
        delay = torch.fft.ifft(angular, n=self.n_delay, dim=2)
        return delay.abs().square().mean(dim=-1, keepdim=False).unsqueeze(1)


class RePos(nn.Module):
    """Full two-stage RePos model. Stage 1 and ASPN are trained separately."""

    def __init__(self, **kwargs):
        super().__init__()
        structure_kwargs = dict(kwargs)
        structure_kwargs.pop("n_angle", None)
        structure_kwargs.pop("phase_mode", None)
        self.structure = RePosStructure(**structure_kwargs)
        self.aspn = ASPN(
            n_time=int(kwargs.get("n_time", 10)),
            n_links=int(kwargs.get("n_links", 3)),
            n_subcarriers=int(kwargs.get("n_subcarriers", 114)),
            n_angle=int(kwargs.get("n_angle", 90)),
            phase_mode=str(kwargs.get("phase_mode", "paper")),
        )

    def forward_relative(self, x: Tensor) -> Tensor:
        return self.structure(x)

    def forward_root(self, x: Tensor) -> Tensor:
        return self.aspn(x)

    def forward_absolute(self, x: Tensor) -> Tensor:
        return self.forward_relative(x) + self.forward_root(x)[:, None, :]


def stage1_loss(pred_relative: Tensor, target_absolute: Tensor, limb_weight: float = 0.1) -> tuple[Tensor, dict]:
    target_relative = root_relative(target_absolute)
    l1 = F.l1_loss(pred_relative, target_relative)
    l2 = F.mse_loss(pred_relative, target_relative)
    limb = limb_length_loss(pred_relative, target_relative)
    loss = l1 + l2 + float(limb_weight) * limb
    return loss, {"loss": float(loss.detach()), "l1": float(l1.detach()), "l2": float(l2.detach()), "limb": float(limb.detach())}


def stage2_loss(pred_root: Tensor, pred_relative: Tensor, target_absolute: Tensor, root_weight: float = 1.0) -> tuple[Tensor, dict]:
    target_root = target_absolute[:, 0]
    pred_absolute = pred_relative.detach() + pred_root[:, None, :]
    l1 = F.l1_loss(pred_absolute, target_absolute)
    l2 = F.mse_loss(pred_absolute, target_absolute)
    root = F.l1_loss(pred_root, target_root)
    loss = l1 + l2 + float(root_weight) * root
    return loss, {"loss": float(loss.detach()), "absolute_l1": float(l1.detach()), "absolute_l2": float(l2.detach()), "root_l1": float(root.detach())}


if __name__ == "__main__":
    torch.manual_seed(0)
    model = RePos(embed_dim=64, depth=2, num_heads=8, ffn_dim=256, query_depth=1)
    x = torch.randn(2, 3, 114, 10)
    y = torch.randn(2, 17, 3)
    rel = model.forward_relative(x)
    root = model.forward_root(x)
    assert rel.shape == (2, 17, 3) and root.shape == (2, 3)
    loss, _ = stage1_loss(rel, y)
    assert torch.isfinite(loss)
    print(tuple(rel.shape), tuple(root.shape), float(loss))





