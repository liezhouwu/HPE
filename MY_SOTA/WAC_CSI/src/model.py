from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

CSI_SHAPE = (3, 114, 10)
PARENTS = (-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 8, 13, 14, 15)
EDGES = tuple((p, j) for j, p in enumerate(PARENTS) if p >= 0)


def check_csi(x: Tensor) -> Tensor:
    if x.ndim != 4 or tuple(x.shape[1:]) != CSI_SHAPE:
        raise ValueError(f"expected CSI shape (B,3,114,10), got {tuple(x.shape)}")
    return x


class RadioFieldEncoder(nn.Module):
    """CSI-only encoder that retains link, frequency and amplitude structure."""

    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 48, (9, 3), padding=(4, 1)), nn.GELU(),
            nn.Conv2d(48, 96, (5, 3), padding=(2, 1)), nn.GELU(),
            nn.Conv2d(96, 128, (3, 3), padding=1), nn.GELU(),
        )
        self.link_proj = nn.Linear(128 + 4, dim)
        self.link_pos = nn.Parameter(torch.zeros(1, 3, dim))
        nn.init.trunc_normal_(self.link_pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=4, dim_feedforward=dim * 4,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.mixer = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.radio_proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))
        self.pose_proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))

    def forward(self, normalized: Tensor, raw: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        normalized, raw = check_csi(normalized), check_csi(raw)
        # Raw branch carries log-amplitude and link-ratio information. It is
        # finite-cleaned in the loader and is never normalized with E04 stats.
        raw_log = torch.log1p(raw.abs().clamp_min(0.0))
        link_mean = raw_log.mean(dim=(2, 3))
        link_std = raw_log.std(dim=(2, 3))
        link_min = raw_log.amin(dim=(2, 3))
        link_max = raw_log.amax(dim=(2, 3))
        stats = torch.cat([link_mean, link_std, link_min, link_max], dim=1).reshape(normalized.shape[0], 3, 4)
        link_input = torch.cat([normalized, raw_log], dim=1).reshape(-1, 2, 114, 10)
        h = self.conv(link_input).mean(dim=(2, 3)).reshape(normalized.shape[0], 3, 128)
        # Link-specific raw statistics are concatenated to each link token.
        tokens = self.link_proj(torch.cat([h, stats], dim=-1))
        tokens = self.norm(self.mixer(tokens + self.link_pos))
        pooled = tokens.mean(dim=1)
        return tokens, self.radio_proj(pooled), self.pose_proj(pooled), stats.reshape(normalized.shape[0], 12)


class RelativePoseDecoder(nn.Module):
    def __init__(self, dim: int = 128, hidden: int = 128) -> None:
        super().__init__()
        self.joint_embed = nn.Parameter(torch.zeros(1, 17, dim))
        nn.init.trunc_normal_(self.joint_embed, std=0.02)
        self.g1 = nn.Linear(dim, hidden)
        self.g2 = nn.Linear(hidden, hidden)
        self.direction = nn.Linear(hidden, 3)
        self.length = nn.Linear(hidden, 1)
        adjacency = torch.zeros(17, 17)
        for parent, child in EDGES:
            adjacency[child, parent] = 0.5
            adjacency[parent, child] = 0.5
        self.register_buffer("adjacency", adjacency, persistent=False)

    def forward(self, feature: Tensor) -> Tensor:
        h = F.gelu(self.g1(feature[:, None, :] + self.joint_embed))
        h = h + torch.einsum("ij,bjd->bid", self.adjacency, h)
        h = F.gelu(self.g2(h))
        direction = F.normalize(self.direction(h), dim=-1, eps=1e-6)
        length = F.softplus(self.length(h))
        joints = [feature.new_zeros(feature.shape[0], 3)]
        for joint in range(1, 17):
            parent = PARENTS[joint]
            joints.append(joints[parent] + direction[:, joint] * length[:, joint])
        return torch.stack(joints, dim=1)


class CoordinateAtlasHead(nn.Module):
    """Source-only sample memory atlas with leave-one-sequence-out masking."""

    def __init__(self, dim: int, memory_coords: Tensor, memory_descriptors: Tensor,
                 memory_sequence_ids: Tensor, top_k: int = 8,
                 temperature: float = 0.12, max_residual: float = 0.75) -> None:
        super().__init__()
        if memory_coords.ndim != 2 or memory_coords.shape[1] != 3 or memory_coords.shape[0] < 2:
            raise ValueError("memory_coords must have shape (N,3), N>=2")
        if memory_descriptors.ndim != 2 or memory_descriptors.shape != (memory_coords.shape[0], 12):
            raise ValueError("memory_descriptors must have shape (N,12)")
        if memory_sequence_ids.ndim != 1 or memory_sequence_ids.shape[0] != memory_coords.shape[0]:
            raise ValueError("memory_sequence_ids must have shape (N,)")
        self.register_buffer("memory_coords", memory_coords.float(), persistent=True)
        self.register_buffer("memory_descriptors", memory_descriptors.float(), persistent=True)
        self.register_buffer("memory_sequence_ids", memory_sequence_ids.long(), persistent=True)
        self.descriptor_proj = nn.Sequential(nn.Linear(12, 16), nn.GELU(), nn.LayerNorm(16))
        self.residual = nn.Sequential(
            nn.Linear(dim + 3, dim), nn.GELU(), nn.Linear(dim, 3),
        )
        self.top_k = int(max(1, min(top_k, memory_coords.shape[0])))
        self.temperature = float(temperature)
        self.max_residual = float(max_residual)

    def forward(self, feature: Tensor, descriptor: Tensor,
                exclude_sequence_ids: Optional[Tensor] = None) -> dict[str, Tensor]:
        query = F.normalize(self.descriptor_proj(descriptor), dim=-1, eps=1e-6)
        prototypes = F.normalize(self.descriptor_proj(self.memory_descriptors), dim=-1, eps=1e-6)
        scores = query.float() @ prototypes.t()
        valid = torch.ones_like(scores, dtype=torch.bool)
        if exclude_sequence_ids is not None:
            if exclude_sequence_ids.ndim != 1 or exclude_sequence_ids.shape[0] != scores.shape[0]:
                raise ValueError("exclude_sequence_ids must have shape (B,)")
            valid = self.memory_sequence_ids[None, :] != exclude_sequence_ids[:, None]
            scores = scores.masked_fill(~valid, float("-inf"))
        top_scores, top_idx = torch.topk(scores, k=self.top_k, dim=-1)
        top_weights = F.softmax(top_scores / self.temperature, dim=-1)
        top_centers = self.memory_coords[top_idx]
        anchor = (top_weights[..., None] * top_centers).sum(dim=1)
        delta = torch.tanh(self.residual(torch.cat([feature, anchor.to(feature.dtype)], dim=-1))) * self.max_residual
        root = anchor.to(delta.dtype) + delta
        confidence = top_weights[:, 0]
        entropy = -(top_weights * top_weights.clamp_min(1e-8).log()).sum(-1)
        return {
            "root": root, "anchor": anchor, "delta": delta, "scores": scores,
            "top_indices": top_idx, "top_weights": top_weights,
            "confidence": confidence, "entropy": entropy, "valid": valid,
        }


@dataclass
class WACOutput:
    pose: Tensor
    relative_pose: Tensor
    root: Tensor
    anchor: Tensor
    delta: Tensor
    scores: Tensor
    top_indices: Tensor
    top_weights: Tensor
    confidence: Tensor
    entropy: Tensor
    valid_memory: Tensor


class WACPoseModel(nn.Module):
    """Wireless Atlas Coordinate Pose, CSI-only cross-environment model."""

    def __init__(self, memory_coords: Tensor, memory_descriptors: Tensor,
                 memory_sequence_ids: Tensor,
                 dim: int = 128, top_k: int = 4,
                 temperature: float = 0.12, max_residual: float = 0.75,
                 cell_weight: float = 1.0, residual_weight: float = 1.0,
                 pose_weight: float = 1.0, relative_weight: float = 1.0,
                 bone_weight: float = 0.1, retrieval_weight: float = 0.1) -> None:
        super().__init__()
        self.encoder = RadioFieldEncoder(dim)
        self.relative_decoder = RelativePoseDecoder(dim)
        self.atlas = CoordinateAtlasHead(dim, memory_coords, memory_descriptors, memory_sequence_ids,
                                         top_k, temperature, max_residual)
        self.cell_weight = float(cell_weight)
        self.residual_weight = float(residual_weight)
        self.pose_weight = float(pose_weight)
        self.relative_weight = float(relative_weight)
        self.bone_weight = float(bone_weight)
        self.retrieval_weight = float(retrieval_weight)

    @classmethod
    def from_config(cls, config: dict, memory_coords: Tensor, memory_descriptors: Tensor,
                    memory_sequence_ids: Tensor) -> "WACPoseModel":
        m, w = config.get("model", {}), config.get("loss", {})
        return cls(
            memory_coords=memory_coords, memory_descriptors=memory_descriptors,
            memory_sequence_ids=memory_sequence_ids,
            dim=int(m.get("dim", 128)), top_k=int(m.get("top_k", 4)),
            temperature=float(m.get("temperature", 0.12)), max_residual=float(m.get("max_residual", 0.75)),
            cell_weight=float(w.get("cell_weight", 1.0)), residual_weight=float(w.get("residual_weight", 1.0)),
            pose_weight=float(w.get("pose_weight", 1.0)), relative_weight=float(w.get("relative_weight", 1.0)),
            bone_weight=float(w.get("bone_weight", 0.1)), retrieval_weight=float(w.get("retrieval_weight", 0.1)),
        )

    def forward(self, normalized: Tensor, raw: Tensor,
                exclude_sequence_ids: Optional[Tensor] = None) -> WACOutput:
        _, radio_feature, pose_feature, descriptor = self.encoder(normalized, raw)
        relative = self.relative_decoder(pose_feature)
        atlas = self.atlas(radio_feature, descriptor, exclude_sequence_ids)
        pose = atlas["root"][:, None] + relative
        return WACOutput(pose, relative, atlas["root"], atlas["anchor"], atlas["delta"],
                         atlas["scores"], atlas["top_indices"], atlas["top_weights"],
                         atlas["confidence"], atlas["entropy"], atlas["valid"])

    def memory_target(self, root: Tensor, valid_memory: Optional[Tensor] = None) -> Tensor:
        with torch.no_grad():
            distance = (root[:, None, :] - self.atlas.memory_coords[None, :, :]).square().sum(-1)
            if valid_memory is not None:
                distance = distance.masked_fill(~valid_memory, float("inf"))
            return distance.argmin(-1)

    @staticmethod
    def bone_loss(pred: Tensor, target: Tensor) -> Tensor:
        vals = [F.smooth_l1_loss((pred[:, c] - pred[:, p]).norm(dim=-1),
                                 (target[:, c] - target[:, p]).norm(dim=-1)) for p, c in EDGES]
        return torch.stack(vals).mean()

    def loss(self, output: WACOutput, target: Tensor) -> tuple[Tensor, dict[str, float]]:
        root_target = target[:, 0]
        relative_target = target - root_target[:, None]
        pose_loss = F.smooth_l1_loss(output.pose, target)
        relative_loss = F.smooth_l1_loss(output.relative_pose, relative_target)
        root_loss = F.smooth_l1_loss(output.root, root_target)
        residual_loss = F.smooth_l1_loss(output.delta, root_target - output.anchor.detach())
        memory_target = self.memory_target(root_target, output.valid_memory)
        retrieval_loss = F.cross_entropy(output.scores, memory_target)
        bone_loss = self.bone_loss(output.pose, target)
        # Penalize overconfident retrieval when the target lies far from the
        # selected source cell; this keeps OOD behavior bounded.
        anchor_error = F.smooth_l1_loss(output.anchor, root_target)
        total = (self.pose_weight * pose_loss + self.relative_weight * relative_loss
                 + 2.0 * root_loss + self.cell_weight * retrieval_loss
                 + self.residual_weight * residual_loss + self.bone_weight * bone_loss
                 + self.retrieval_weight * anchor_error)
        return total, {
            "loss": float(total.detach()), "pose_loss": float(pose_loss.detach()),
            "relative_loss": float(relative_loss.detach()), "root_loss": float(root_loss.detach()),
            "retrieval_loss": float(retrieval_loss.detach()), "residual_loss": float(residual_loss.detach()),
            "bone_loss": float(bone_loss.detach()), "anchor_error": float(anchor_error.detach()),
        }
