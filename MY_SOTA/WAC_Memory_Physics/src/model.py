from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

IMPLEMENTATION_VERSION = 2
CSI_SHAPE = (3, 114, 10)
PARENTS = (-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 8, 13, 14, 15)
EDGES = tuple((p, j) for j, p in enumerate(PARENTS) if p >= 0)
SPECTRAL_BINS = 6


def check_csi(x: Tensor) -> Tensor:
    if x.ndim != 4 or tuple(x.shape[1:]) != CSI_SHAPE:
        raise ValueError(f"expected CSI (B,3,114,10), got {tuple(x.shape)}")
    return x


def physical_descriptor(raw: Tensor) -> Tensor:
    # Mean finite differences telescope to endpoints; RMS retains interior variation.
    z = torch.log1p(check_csi(raw).float().abs())
    spectrum = z.mean(3)
    slope = spectrum.diff(dim=-1)
    curvature = slope.diff(dim=-1)
    return torch.cat([
        z.mean((2, 3)), z.std((2, 3), correction=0),
        (z - z.mean(1, keepdim=True)).mean((2, 3)),
        slope.square().mean(-1).clamp_min(1e-12).sqrt(),
        curvature.square().mean(-1).clamp_min(1e-12).sqrt(),
        spectrum.std(-1, correction=0), z.std(3, correction=0).mean(-1),
    ], dim=1)


def pretrain_target(raw: Tensor) -> Tensor:
    """CSI-only target retaining link order and local spectral structure."""
    z = torch.log1p(check_csi(raw).float().abs())
    band_mean = F.adaptive_avg_pool1d(z.mean(3), SPECTRAL_BINS).flatten(1)
    band_var = F.adaptive_avg_pool1d(z.var(3, correction=0), SPECTRAL_BINS).flatten(1)
    return torch.cat([physical_descriptor(raw), band_mean, band_var], dim=-1)


class PhysicalEncoder(nn.Module):
    def __init__(self, dim: int = 128, descriptor_dim: int = 21) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, (9, 3), padding=(4, 1)), nn.GELU(),
            nn.Conv2d(32, 64, (5, 3), padding=(2, 1)), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
        )
        self.proj = nn.Sequential(
            nn.Linear(3 * 64 * SPECTRAL_BINS + descriptor_dim, dim),
            nn.GELU(), nn.LayerNorm(dim),
        )
        self.body = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))
        self.propagation = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))

    def forward(self, normalized: Tensor, raw: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        normalized, raw = check_csi(normalized), check_csi(raw)
        d = physical_descriptor(raw)
        z = torch.log1p(raw.float().abs())
        # Pair normalized/raw inputs WITHIN each link, preserving link order.
        link_input = torch.stack([normalized, z], dim=2).flatten(0, 1)
        h = F.adaptive_avg_pool2d(self.conv(link_input), (SPECTRAL_BINS, 1))
        fused = self.proj(torch.cat([h.reshape(len(raw), -1), d], dim=-1))
        return self.body(fused), self.propagation(fused), d


class PretrainModel(nn.Module):
    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.encoder = PhysicalEncoder(dim)
        target_dim = 21 + 6 * SPECTRAL_BINS
        self.reconstruct = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, target_dim))
        self.nuisance = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 3))
        self.register_buffer("target_mean", torch.zeros(target_dim))
        self.register_buffer("target_std", torch.ones(target_dim))

    def forward(self, x: Tensor, raw: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        body, prop, _ = self.encoder(x, raw)
        # Reconstruction trains body; propagation retains nuisance information.
        return self.reconstruct(body), self.nuisance(prop), body


class RelativeDecoder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.joint = nn.Parameter(torch.zeros(1, 17, dim))
        nn.init.trunc_normal_(self.joint, std=.02)
        self.g1, self.g2 = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.dir, self.length = nn.Linear(dim, 3), nn.Linear(dim, 1)
        adj = torch.zeros(17, 17)
        for p, c in EDGES:
            adj[p, c] = adj[c, p] = .5
        self.register_buffer("adj", adj, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        h = F.gelu(self.g1(x[:, None] + self.joint))
        h = F.gelu(self.g2(h + torch.einsum("ij,bjd->bid", self.adj, h)))
        direction = F.normalize(self.dir(h), dim=-1, eps=1e-6)
        length = F.softplus(self.length(h))
        out = [x.new_zeros(len(x), 3)]
        for j in range(1, 17):
            out.append(out[PARENTS[j]] + direction[:, j] * length[:, j])
        return torch.stack(out, 1)


class MemoryHead(nn.Module):
    def __init__(self, dim: int, memory_keys: Tensor, memory_coords: Tensor,
                 memory_desc: Tensor, memory_seq: Tensor, top_k: int = 8,
                 max_align: float = .35, temperature: float = .1,
                 physical_weight: float = .3) -> None:
        super().__init__()
        n = len(memory_coords)
        if n < 2 or memory_coords.shape != (n, 3) or memory_keys.shape != (n, dim):
            raise ValueError("invalid or empty source memory")
        if memory_desc.shape != (n, 21) or memory_seq.shape != (n,):
            raise ValueError("memory descriptor/sequence shape mismatch")
        if not 0 < top_k <= n or temperature <= 0 or not 0 <= physical_weight <= 1 or max_align <= 0:
            raise ValueError("invalid retrieval settings")
        self.top_k, self.max_align = int(top_k), float(max_align)
        self.temperature, self.physical_weight = float(temperature), float(physical_weight)
        for name, tensor in [("memory_keys", memory_keys), ("memory_coords", memory_coords),
                             ("memory_desc", memory_desc)]:
            if not torch.isfinite(tensor).all():
                raise ValueError(f"nonfinite {name}")
            self.register_buffer(name, tensor.detach().float().clone())
        self.register_buffer("memory_seq", memory_seq.detach().long().clone())
        # Fit exclusively on labeled SOURCE memory.
        self.register_buffer("desc_mean", memory_desc.float().mean(0))
        self.register_buffer("desc_std", memory_desc.float().std(0, correction=0).clamp_min(1e-3))
        self.metric = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        nn.init.zeros_(self.metric[-1].weight)
        nn.init.zeros_(self.metric[-1].bias)
        self.descriptor_proj = nn.Linear(21, 21, bias=False)
        nn.init.eye_(self.descriptor_proj.weight)
        self.refine = nn.Sequential(nn.Linear(dim + 3, dim), nn.GELU(), nn.Linear(dim, 3))
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def embed(self, x: Tensor) -> Tensor:
        return F.normalize(x + self.metric(x), dim=-1, eps=1e-6)

    def embed_descriptor(self, d: Tensor) -> Tensor:
        z = ((d - self.desc_mean) / self.desc_std).clamp(-10, 10)
        return F.normalize(self.descriptor_proj(z), dim=-1, eps=1e-6)

    def forward(self, prop: Tensor, descriptor: Tensor,
                exclude_seq: Optional[Tensor] = None) -> dict[str, Tensor]:
        # Both sides use the SAME trainable projection of a fixed encoder space.
        # Keep retrieval, softmax and coordinate expectations in fp32 under AMP.
        with torch.autocast(device_type=prop.device.type, enabled=False):
            prop, descriptor = prop.float(), descriptor.float()
            learned = self.embed(prop) @ self.embed(self.memory_keys).t()
            physical = self.embed_descriptor(descriptor) @ self.embed_descriptor(self.memory_desc).t()
            score = ((1 - self.physical_weight) * learned + self.physical_weight * physical) / self.temperature
            valid = torch.ones_like(score, dtype=torch.bool)
            if exclude_seq is not None:
                valid = self.memory_seq[None] != exclude_seq[:, None]
                if (valid.sum(-1) < self.top_k).any():
                    raise ValueError("too few memory entries after leave-one-sequence-out masking")
                score = score.masked_fill(~valid, -torch.inf)
            top_score, idx = torch.topk(score, self.top_k, -1)
            weight = F.softmax(top_score, -1)
            anchor = (weight[..., None] * self.memory_coords[idx]).sum(1)
            soft_anchor = F.softmax(score, -1) @ self.memory_coords
        # A single bounded correction avoids two unidentifiable additive offsets.
        align = torch.tanh(self.refine(torch.cat([prop, anchor], -1))).float() * self.max_align
        return dict(root=anchor + align, anchor=anchor, align=align, score=score,
                    idx=idx, weight=weight, valid=valid, soft_anchor=soft_anchor)


@dataclass
class PoseOutput:
    pose: Tensor
    relative: Tensor
    root: Tensor
    anchor: Tensor
    align: Tensor
    score: Tensor
    idx: Tensor
    weight: Tensor
    valid: Tensor
    soft_anchor: Tensor


class PhysAlignPose(nn.Module):
    def __init__(self, dim: int, memory_keys: Tensor, memory_coords: Tensor,
                 memory_desc: Tensor, memory_seq: Tensor, top_k: int = 8,
                 max_align: float = .35, temperature: float = .1,
                 physical_weight: float = .3, neighbor_sigma: float = .15,
                 positive_k: int = 64, retrieval_weight: float = .1,
                 anchor_weight: float = 1.) -> None:
        super().__init__()
        if neighbor_sigma <= 0 or positive_k < 1:
            raise ValueError("invalid coordinate neighborhood")
        self.encoder = PhysicalEncoder(dim)
        self.relative = RelativeDecoder(dim)
        self.memory = MemoryHead(dim, memory_keys, memory_coords, memory_desc, memory_seq,
                                 top_k, max_align, temperature, physical_weight)
        self.neighbor_sigma, self.positive_k = float(neighbor_sigma), int(positive_k)
        self.retrieval_weight, self.anchor_weight = float(retrieval_weight), float(anchor_weight)

    def configure_finetuning(self) -> None:
        # Freeze cached-key trunk/propagation; train body and shared retrieval metric.
        self.encoder.requires_grad_(False)
        self.encoder.body.requires_grad_(True)

    def forward(self, x: Tensor, raw: Tensor, exclude_seq: Optional[Tensor] = None) -> PoseOutput:
        body, prop, descriptor = self.encoder(x, raw)
        rel = self.relative(body).float()
        m = self.memory(prop, descriptor, exclude_seq)
        return PoseOutput(m["root"][:, None] + rel, rel, m["root"], m["anchor"],
                          m["align"], m["score"], m["idx"], m["weight"], m["valid"], m["soft_anchor"])

    @torch.no_grad()
    def neighbor_target(self, root: Tensor, valid: Tensor) -> Tensor:
        d2 = (root.float()[:, None] - self.memory.memory_coords[None]).square().sum(-1)
        d2 = d2.masked_fill(~valid, torch.inf)
        distances, indices = torch.topk(d2, min(self.positive_k, d2.shape[1]), largest=False)
        weights = F.softmax(-distances / (2 * self.neighbor_sigma ** 2), -1)
        if not torch.isfinite(weights).all():
            raise ValueError("query has no eligible source-memory neighbors")
        return torch.zeros_like(d2).scatter_(1, indices, weights)

    def loss(self, out: PoseOutput, target: Tensor) -> tuple[Tensor, dict[str, float]]:
        with torch.autocast(device_type=target.device.type, enabled=False):
            target = target.float()
            root, rel = target[:, 0], target - target[:, :1]
            p = self.neighbor_target(root, out.valid)
            # Avoid 0 * -inf at excluded sequence entries.
            logq = F.log_softmax(out.score, -1).masked_fill(~out.valid, 0.)
            retrieval_l = (p * (p.clamp_min(1e-12).log() - logq)).sum(-1).mean()
            pose_l = (out.pose - target).norm(dim=-1).mean()
            root_l = (out.root - root).norm(dim=-1).mean()
            rel_l = F.smooth_l1_loss(out.relative, rel)
            # Gradient also reaches candidates outside hard Top-K.
            anchor_l = F.smooth_l1_loss(out.soft_anchor, root)
            align_target = (root - out.anchor.detach()).clamp(-self.memory.max_align, self.memory.max_align)
            align_l = F.smooth_l1_loss(out.align, align_target)
            total = pose_l + rel_l + 2 * root_l + self.retrieval_weight * retrieval_l + self.anchor_weight * anchor_l + .2 * align_l
        return total, {name: float(value.detach()) for name, value in dict(
            loss=total, pose=pose_l, relative=rel_l, root=root_l,
            retrieval=retrieval_l, anchor=anchor_l, align=align_l).items()}
