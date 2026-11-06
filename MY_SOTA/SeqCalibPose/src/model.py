from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

CSI_SHAPE = (3, 114, 10)
PARENTS = (-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 8, 13, 14, 15)


def _check(x: Tensor) -> Tensor:
    if x.ndim != 4 or tuple(x.shape[1:]) != CSI_SHAPE:
        raise ValueError(f"expected (B,{CSI_SHAPE}), got {tuple(x.shape)}")
    return x


class LinkSpectrumEncoder(nn.Module):
    """Preserve link, subcarrier and intra-frame structure until fusion."""

    def __init__(self, dim: int = 160) -> None:
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(4, 48, (9, 3), padding=(4, 1)), nn.GELU(),
            nn.Conv2d(48, 80, (5, 3), padding=(2, 1)), nn.GELU(),
            nn.Conv2d(80, 96, (3, 3), padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((12, 5)), nn.Flatten(),
            nn.Linear(96 * 12 * 5, dim), nn.LayerNorm(dim),
        )
        self.link_id = nn.Parameter(torch.empty(1, 3, dim))
        nn.init.trunc_normal_(self.link_id, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 3, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.cross_link = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.frame_norm = nn.LayerNorm(dim)

    def forward(self, amplitude: Tensor, raw: Tensor, phase: Tensor) -> tuple[Tensor, Tensor]:
        b = amplitude.shape[0]
        # Phase enters as a unit complex pair. No angle unwrap or destructive
        # per-Rx detrending is applied to this first candidate.
        csi = torch.stack((amplitude, torch.log1p(raw.abs()), phase.cos(), phase.sin()), dim=2)
        link = self.local(csi.reshape(b * 3, 4, 114, 10)).reshape(b, 3, -1)
        tokens = self.frame_norm(self.cross_link(link + self.link_id))
        return tokens, tokens.mean(dim=1)


class SkeletonGraphDecoder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.joints = nn.Parameter(torch.empty(1, 17, dim))
        nn.init.trunc_normal_(self.joints, std=0.02)
        self.decode = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim), nn.GELU())
        self.direction = nn.Linear(dim, 3)
        self.length = nn.Linear(dim, 1)
        adjacency = torch.zeros(17, 17)
        for j, p in enumerate(PARENTS):
            if p >= 0:
                adjacency[j, p] = adjacency[p, j] = 0.5
        self.register_buffer("adjacency", adjacency, persistent=False)

    def forward(self, feature: Tensor) -> Tensor:
        h = self.decode(feature[:, None] + self.joints)
        h = h + torch.einsum("ij,bjd->bid", self.adjacency, h)
        direction = F.normalize(self.direction(h), dim=-1, eps=1e-6)
        length = F.softplus(self.length(h))
        xyz = [feature.new_zeros((feature.shape[0], 1, 3))]
        for j in range(1, 17):
            p = PARENTS[j]
            xyz.append(xyz[p] + direction[:, j:j + 1] * length[:, j:j + 1])
        return torch.cat(xyz, dim=1)


@dataclass
class SeqCalibOutput:
    pose: Tensor
    relative: Tensor
    root: Tensor
    link_weights: Tensor


class SeqCalibPose(nn.Module):
    """Single-frame CSI pose with task-partitioned, input-conditioned root calibration."""

    def __init__(self, dim: int = 160, root_residual_limit: float = 0.45) -> None:
        super().__init__()
        self.encoder = LinkSpectrumEncoder(dim)
        self.body_adapter = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))
        self.body = SkeletonGraphDecoder(dim)
        self.link_gate = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.root = nn.Sequential(nn.LayerNorm(dim + 14), nn.Linear(dim + 14, dim), nn.GELU(),
                                  nn.Linear(dim, dim // 2), nn.GELU())
        self.root_center = nn.Linear(dim // 2, 3)
        nn.init.zeros_(self.root_center.weight)
        nn.init.zeros_(self.root_center.bias)
        self.register_buffer("root_prior", torch.tensor([0.0, 0.02, 3.13]))
        self.residual_limit = float(root_residual_limit)

    @staticmethod
    def _stats(raw: Tensor) -> Tensor:
        mean = raw.mean((2, 3))
        std = raw.std((2, 3), unbiased=False)
        lo = raw.amin((2, 3))
        hi = raw.amax((2, 3))
        ratios = mean[:, 1:] / mean[:, :1].clamp_min(1e-4)
        return torch.cat((mean, std, lo, hi, ratios), dim=-1)

    def forward(self, amplitude: Tensor, raw: Tensor, phase: Tensor) -> SeqCalibOutput:
        amplitude, raw, phase = _check(amplitude), _check(raw), _check(phase)
        tokens, pooled = self.encoder(amplitude, raw, phase)
        weights = torch.softmax(self.link_gate(tokens).squeeze(-1), dim=-1)
        body_feature = self.body_adapter((tokens * weights.unsqueeze(-1)).sum(1))
        relative = self.body(body_feature)
        # Root uses unpooled link evidence and raw propagation statistics.
        root_feature = self.root(torch.cat((pooled, self._stats(raw)), dim=-1))
        center = self.root_center(root_feature)
        # Bound frame-specific correction around the source-coordinate prior.
        # This keeps a few-shot source fit from producing unbounded OOD shifts.
        root = self.root_prior[None] + self.residual_limit * torch.tanh(center)
        return SeqCalibOutput(relative + root[:, None], relative, root, weights)
