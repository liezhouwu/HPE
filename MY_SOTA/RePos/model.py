from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

# MM-Fi H36M 17-joint topology used by the existing evaluator.
BONES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12),
    (12, 13), (8, 14), (14, 15), (15, 16),
)
PARTS: tuple[tuple[int, ...], ...] = (
    (10,), (0, 7, 8, 9), (11, 12, 13), (14, 15, 16), (4, 5, 6), (1, 2, 3),
)
_JOINT_PART = [0] * 17
for _part, _joints in enumerate(PARTS):
    for _joint in _joints:
        _JOINT_PART[_joint] = _part
JOINT_PART = tuple(_JOINT_PART)
PART_COUNTS = (20, 30, 25, 25, 25, 25)


class PaperCSIEncoder(nn.Module):
    """RePos paper stem: 3-layer CNN, 570 tokens, 4-layer Transformer."""

    def __init__(self, n_time: int = 10, n_links: int = 3,
                 n_subcarriers: int = 114, embed_dim: int = 256,
                 depth: int = 4, num_heads: int = 8, ffn_dim: int = 1024):
        super().__init__()
        if n_time % 2:
            raise ValueError("paper encoder requires an even time dimension")
        self.n_time = n_time
        self.n_links = n_links
        self.n_subcarriers = n_subcarriers
        self.out_dim = embed_dim
        self.stem = nn.Sequential(
            # MM-Fi exposes the ten temporal samples inside one frame; unlike
            # the paper's multi-window tensor they remain a spatial axis here.
            nn.Conv2d(n_links, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.GELU(),
            nn.Conv2d(128, embed_dim, (3, 3), stride=(1, 2), padding=1), nn.GELU(),
        )
        self.n_tokens = n_subcarriers * (n_time // 2)
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ffn_dim,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: Tensor, return_tokens: bool = False) -> Tensor:
        if x.ndim == 5:
            x = x.squeeze(1)
        if x.ndim != 4 or tuple(x.shape[1:]) != (self.n_links, self.n_subcarriers, self.n_time):
            raise ValueError(
                f"expected CSI shape (B,{self.n_links},{self.n_subcarriers},{self.n_time})"
            )
        image = x
        tokens = self.stem(image).flatten(2).transpose(1, 2)
        encoded = self.norm(self.blocks(tokens + self.pos))
        return encoded if return_tokens else encoded.mean(dim=1)


def root_relative(pose: Tensor) -> Tensor:
    return pose - pose[:, :1]


def limb_length_loss(pred: Tensor, target: Tensor) -> Tensor:
    values = []
    for left, right in BONES:
        pred_len = (pred[:, left] - pred[:, right]).norm(dim=-1)
        target_len = (target[:, left] - target[:, right]).norm(dim=-1)
        values.append((pred_len - target_len).abs())
    return torch.stack(values, dim=1).mean()


class SkeletonGraphAttentionBlock(nn.Module):
    """One paper SGA block: intra-part attention, inter-part attention, FFN."""

    def __init__(self, dim: int, heads: int, dropout: float, intra_mask: Tensor, inter_mask: Tensor):
        super().__init__()
        self.intra_norm = nn.LayerNorm(dim)
        self.inter_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.intra = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.inter = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))
        self.register_buffer("intra_mask", intra_mask, persistent=False)
        self.register_buffer("inter_mask", inter_mask, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        y = self.intra_norm(x)
        x = x + self.intra(y, y, y, attn_mask=self.intra_mask)[0]
        y = self.inter_norm(x)
        x = x + self.inter(y, y, y, attn_mask=self.inter_mask)[0]
        return x + self.ffn(self.ffn_norm(x))


class SkeletonGraphAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, depth: int = 4, dropout: float = 0.0):
        super().__init__()
        part_ids = torch.cat([torch.full((count,), index, dtype=torch.long) for index, count in enumerate(PART_COUNTS)])
        # PARTS is ordered as head, torso, left/right arms, left/right legs.
        # The paper's skeletal graph makes the torso the hub: it connects to
        # the head and all four limbs; each peripheral part only connects to
        # itself and the torso.
        part_adjacency = torch.tensor(
            [[1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1], [0, 1, 1, 0, 0, 0],
             [0, 1, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1]], dtype=torch.bool,
        )
        same_part = part_ids[:, None].eq(part_ids[None, :])
        connected = part_adjacency[part_ids[:, None], part_ids[None, :]]
        self.blocks = nn.ModuleList([
            SkeletonGraphAttentionBlock(dim, heads, dropout, ~same_part, ~connected)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, part_tokens: Tensor) -> Tensor:
        for block in self.blocks:
            part_tokens = block(part_tokens)
        return self.norm(part_tokens)


class ChebGraphConv(nn.Module):
    def __init__(self, dim: int, order: int = 3):
        super().__init__()
        self.weights = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(order)])

    def forward(self, x: Tensor, adjacency: Tensor) -> Tensor:
        identity = torch.eye(adjacency.shape[0], device=x.device, dtype=x.dtype)
        laplacian = identity - adjacency.to(dtype=x.dtype)
        outputs = [x]
        if len(self.weights) > 1:
            outputs.append(torch.einsum("ij,bjd->bid", laplacian, x))
        for _ in range(2, len(self.weights)):
            outputs.append(2.0 * torch.einsum("ij,bjd->bid", laplacian, outputs[-1]) - outputs[-2])
        return sum(layer(value) for layer, value in zip(self.weights, outputs))


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
        sga_depth: int = 4,
        gcn_layers: int = 3,
        joint_attention_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = PaperCSIEncoder(
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            ffn_dim=ffn_dim, n_time=n_time, n_links=n_links,
            n_subcarriers=n_subcarriers,
        )
        self.part_queries = nn.Parameter(torch.randn(1, 6, embed_dim) * 0.02)
        self.part_expanders = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, count * embed_dim), nn.GELU())
            for count in PART_COUNTS
        ])
        self.part_projection = nn.Linear(embed_dim, embed_dim)
        self.part_pos = nn.ParameterList([
            nn.Parameter(torch.randn(1, count, embed_dim) * 0.02) for count in PART_COUNTS
        ])
        cross_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.part_cross = nn.TransformerDecoder(cross_layer, num_layers=1)
        self.sga = SkeletonGraphAttention(embed_dim, heads=num_heads, depth=sga_depth, dropout=dropout)
        self.joint_queries = nn.Parameter(torch.randn(1, 17, embed_dim) * 0.02)
        self.joint_part_embed = nn.Parameter(torch.randn(6, embed_dim) * 0.02)
        joint_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.joint_decoder = nn.TransformerDecoder(joint_layer, num_layers=query_depth)
        self.joint_norm = nn.LayerNorm(embed_dim)
        self.pose_head = nn.Linear(embed_dim, 3)
        self.gcn = nn.ModuleList([ChebGraphConv(embed_dim, order=3) for _ in range(gcn_layers)])
        self.joint_attention = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads, dim_feedforward=ffn_dim,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            ) for _ in range(joint_attention_layers)
        ])
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
        memory = self.encoder(x, return_tokens=True)
        queries = self.part_queries.expand(x.shape[0], -1, -1)
        parts = self.part_cross(queries, memory)
        expanded = []
        for index, (expand, pos) in enumerate(zip(self.part_expanders, self.part_pos, strict=True)):
            count = PART_COUNTS[index]
            expanded_tokens = expand(parts[:, index]).view(x.shape[0], count, -1)
            expanded.append(self.part_projection(expanded_tokens) + pos)
        body_tokens = torch.cat(expanded, dim=1)
        body_tokens = self.sga(body_tokens)
        pool_queries = self.part_queries.expand(x.shape[0], -1, -1)
        part_features = []
        for index, (start, end) in enumerate(self._part_ranges()):
            tokens = body_tokens[:, start:end]
            scores = (tokens * pool_queries[:, index:index + 1]).sum(dim=-1) / math.sqrt(tokens.shape[-1])
            weights = scores.softmax(dim=1).unsqueeze(-1)
            part_features.append((tokens * weights).sum(dim=1))
        part_features = torch.stack(part_features, dim=1)
        joint_queries = self.joint_queries.expand(x.shape[0], -1, -1)
        part_ids = torch.as_tensor(JOINT_PART, device=x.device)
        joint_queries = joint_queries + self.joint_part_embed[part_ids][None]
        decoded = self.joint_decoder(joint_queries, part_features)
        decoded = self.joint_norm(decoded)
        for index, layer in enumerate(self.gcn):
            decoded = decoded + F.gelu(layer(decoded, self.joint_adjacency))
            if index < len(self.joint_attention):
                decoded = self.joint_attention[index](decoded)
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

    def __init__(self, n_time: int = 10, n_links: int = 3, n_subcarriers: int = 114, n_angle: int = 90, n_delay: int = 64, phase_mode: str = "paper", delay_mode: str = "legacy_truncate"):
        super().__init__()
        self.n_time = int(n_time)
        self.n_links = int(n_links)
        self.n_subcarriers = int(n_subcarriers)
        self.n_angle = int(n_angle)
        self.n_delay = int(n_delay)
        self.phase_mode = str(phase_mode)
        self.delay_mode = str(delay_mode)
        if self.phase_mode not in {"paper", "per_link"}:
            raise ValueError("phase_mode must be paper or per_link")
        if self.delay_mode not in {"legacy_truncate", "full_spectrum"}:
            raise ValueError("delay_mode must be legacy_truncate or full_spectrum")
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
        fft_size = self.n_subcarriers if self.delay_mode == "full_spectrum" else self.n_delay
        delay = torch.fft.ifft(angular, n=fft_size, dim=2)
        heatmap = delay.abs().square().mean(dim=-1).unsqueeze(1)
        if self.delay_mode == "full_spectrum" and fft_size != self.n_delay:
            heatmap = F.adaptive_avg_pool2d(heatmap, (self.n_angle, self.n_delay))
        return heatmap


class RePos(nn.Module):
    """Full two-stage RePos model. Stage 1 and ASPN are trained separately."""

    def __init__(self, **kwargs):
        super().__init__()
        structure_kwargs = dict(kwargs)
        structure_kwargs.pop("n_angle", None)
        structure_kwargs.pop("phase_mode", None)
        structure_kwargs.pop("delay_mode", None)
        self.structure = RePosStructure(**structure_kwargs)
        self.aspn = ASPN(
            n_time=int(kwargs.get("n_time", 10)),
            n_links=int(kwargs.get("n_links", 3)),
            n_subcarriers=int(kwargs.get("n_subcarriers", 114)),
            n_angle=int(kwargs.get("n_angle", 90)),
            phase_mode=str(kwargs.get("phase_mode", "paper")),
            delay_mode=str(kwargs.get("delay_mode", "legacy_truncate")),
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
    return loss, {"loss": loss.detach(), "l1": l1.detach(), "l2": l2.detach(), "limb": limb.detach()}


def stage2_loss(pred_root: Tensor, pred_relative: Tensor, target_absolute: Tensor, root_weight: float = 1.0) -> tuple[Tensor, dict]:
    target_root = target_absolute[:, 0]
    pred_absolute = pred_relative.detach() + pred_root[:, None, :]
    l1 = F.l1_loss(pred_absolute, target_absolute)
    l2 = F.mse_loss(pred_absolute, target_absolute)
    root = F.l1_loss(pred_root, target_root)
    loss = l1 + l2 + float(root_weight) * root
    return loss, {"loss": loss.detach(), "absolute_l1": l1.detach(), "absolute_l2": l2.detach(), "root_l1": root.detach()}


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





