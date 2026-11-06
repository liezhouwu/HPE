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
        raise ValueError(f"expected CSI (B,3,114,10), got {tuple(x.shape)}")
    return x


class LinkEncoder(nn.Module):
    """Single-frame CSI encoder with explicit three-link tokens."""

    def __init__(self, dim: int = 128, heads: int = 4) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, (7, 3), padding=(3, 1)), nn.GELU(),
            nn.Conv2d(32, 64, (5, 3), padding=(2, 1)), nn.GELU(),
        )
        self.link_proj = nn.Linear(64 * 10, dim)
        self.link_pos = nn.Parameter(torch.zeros(1, 3, dim))
        nn.init.trunc_normal_(self.link_pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.mixer = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = check_csi(x)
        h = self.conv(x.reshape(x.shape[0] * 3, 1, 114, 10))
        h = h.mean(dim=2).reshape(x.shape[0], 3, -1)
        tokens = self.norm(self.mixer(self.link_proj(h) + self.link_pos))
        return tokens, tokens.mean(dim=1)


class RawAmplitudeEncoder(nn.Module):
    """Low-resolution raw-amplitude propagation feature for the root branch."""

    def __init__(self, out_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, (9, 3), padding=(4, 1)), nn.GELU(),
            nn.Conv2d(16, 24, (5, 3), padding=(2, 1)), nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 2)), nn.Flatten(),
            nn.Linear(24 * 4 * 2, out_dim), nn.LayerNorm(out_dim),
        )

    def forward(self, raw_csi: Tensor) -> Tensor:
        return self.net(check_csi(raw_csi))


class GraphRelativeDecoder(nn.Module):
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

    def _graph_mix(self, h: Tensor) -> Tensor:
        return h + torch.einsum("ij,bjd->bid", self.adjacency, h)

    def forward(self, feature: Tensor) -> Tensor:
        h = F.gelu(self.g1(feature[:, None, :] + self.joint_embed))
        h = self._graph_mix(h)
        h = F.gelu(self.g2(h))
        direction = F.normalize(self.direction(h), dim=-1, eps=1e-6)
        length = F.softplus(self.length(h))
        joints = [feature.new_zeros(feature.shape[0], 3)]
        for joint in range(1, 17):
            parent = PARENTS[joint]
            joints.append(joints[parent] + direction[:, joint] * length[:, joint])
        return torch.stack(joints, dim=1)


class RootMixtureHead(nn.Module):
    def __init__(self, feature_dim: int, calibration_dim: int = 12, components: int = 4) -> None:
        super().__init__()
        self.components = int(components)
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim + calibration_dim),
            nn.Linear(feature_dim + calibration_dim, 128), nn.GELU(),
        )
        self.means = nn.Linear(128, self.components * 3)
        self.logvar = nn.Linear(128, self.components * 3)
        self.logits = nn.Linear(128, self.components)

    def forward(self, feature: Tensor, calibration: Tensor):
        if calibration.ndim != 2 or calibration.shape[0] != feature.shape[0]:
            raise ValueError("calibration must have shape (B,calibration_dim)")
        h = self.net(torch.cat([feature, calibration], dim=-1))
        means = self.means(h).reshape(feature.shape[0], self.components, 3)
        logvar = self.logvar(h).reshape_as(means).clamp(-6.0, 3.0)
        weights = self.logits(h).softmax(dim=-1)
        root = (weights[..., None] * means).sum(dim=1)
        return root, means, logvar, weights


@dataclass
class CAGEOutput:
    pose: Tensor
    relative_pose: Tensor
    root: Tensor
    structure: Tensor
    link_tokens: Tensor
    mixture_means: Tensor
    mixture_logvar: Tensor
    mixture_weights: Tensor


class CAGEPoseModel(nn.Module):
    """Single-frame Calibration-Aware Geometric Environment-invariant Pose."""

    def __init__(self, dim: int = 128, calibration_dim: int = 12,
                 root_components: int = 4, root_weight: float = 2.0,
                 relative_weight: float = 1.0, bone_weight: float = 0.1,
                 consistency_weight: float = 0.2, absolute_consistency_weight: float = 0.2,
                 domain_weight: float = 0.05, risk_weight: float = 0.05) -> None:
        super().__init__()
        self.encoder = LinkEncoder(dim=dim)
        self.raw_encoder = RawAmplitudeEncoder(out_dim=32)
        self.relative_decoder = GraphRelativeDecoder(dim=dim)
        self.root_head = RootMixtureHead(dim + 32, calibration_dim, root_components)
        self.root_weight = float(root_weight)
        self.relative_weight = float(relative_weight)
        self.bone_weight = float(bone_weight)
        self.consistency_weight = float(consistency_weight)
        self.absolute_consistency_weight = float(absolute_consistency_weight)
        self.domain_weight = float(domain_weight)
        self.risk_weight = float(risk_weight)

    @classmethod
    def from_config(cls, config: dict) -> "CAGEPoseModel":
        m, w = config.get("model", {}), config.get("loss", {})
        return cls(
            dim=int(m.get("dim", 128)), calibration_dim=int(m.get("calibration_dim", 12)),
            root_components=int(m.get("root_components", 4)),
            root_weight=float(w.get("root_weight", 2.0)),
            relative_weight=float(w.get("relative_weight", 1.0)),
            bone_weight=float(w.get("bone_weight", 0.1)),
            consistency_weight=float(w.get("consistency_weight", 0.2)),
            absolute_consistency_weight=float(w.get("absolute_consistency_weight", 0.2)),
            domain_weight=float(w.get("domain_weight", 0.05)),
            risk_weight=float(w.get("risk_weight", 0.05)),
        )

    def forward(self, csi: Tensor, calibration: Tensor, raw_csi: Optional[Tensor] = None) -> CAGEOutput:
        tokens, structure = self.encoder(csi)
        relative = self.relative_decoder(structure)
        if raw_csi is None:
            raw_csi = csi.new_zeros(csi.shape)
        raw_feature = self.raw_encoder(raw_csi)
        root, means, logvar, weights = self.root_head(torch.cat([structure, raw_feature], dim=-1), calibration)
        return CAGEOutput(root[:, None] + relative, relative, root, structure,
                          tokens, means, logvar, weights)

    @staticmethod
    def bone_loss(pred: Tensor, target: Tensor) -> Tensor:
        vals = []
        for parent, child in EDGES:
            vals.append(F.smooth_l1_loss(
                (pred[:, child] - pred[:, parent]).norm(dim=-1),
                (target[:, child] - target[:, parent]).norm(dim=-1),
            ))
        return torch.stack(vals).mean()

    @staticmethod
    def coral_loss(features: Tensor, environments: Tensor) -> Tensor:
        groups = [features[environments == e] for e in environments.unique()]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            return features.new_zeros(())
        means = torch.stack([g.mean(0) for g in groups])
        loss = (means - means.mean(0)).square().mean()
        covs = []
        for g in groups:
            z = g - g.mean(0, keepdim=True)
            covs.append(z.t() @ z / max(1, len(g) - 1))
        for i in range(1, len(covs)):
            loss = loss + (covs[i] - covs[0]).square().mean()
        return loss

    def loss(self, output: CAGEOutput, target: Tensor, environments: Tensor,
             augmented: Optional[CAGEOutput] = None) -> tuple[Tensor, dict]:
        target_root = target[:, 0]
        target_relative = target - target_root[:, None]
        pose_loss = F.smooth_l1_loss(output.pose, target)
        root_loss = F.smooth_l1_loss(output.root, target_root)
        relative_loss = F.smooth_l1_loss(output.relative_pose, target_relative)
        bone_loss = self.bone_loss(output.pose, target)
        component_log_prob = -0.5 * (
            (target_root[:, None] - output.mixture_means).square()
            / output.mixture_logvar.exp()
            + output.mixture_logvar
            + torch.log(torch.tensor(2.0 * torch.pi, device=target.device))
        ).sum(-1)
        nll = -torch.logsumexp(
            torch.log(output.mixture_weights.clamp_min(1e-8)) + component_log_prob,
            dim=-1,
        ).mean()
        domain_loss = self.coral_loss(output.structure, environments)
        consistency = output.structure.new_zeros(())
        absolute_consistency = output.structure.new_zeros(())
        if augmented is not None:
            consistency = F.smooth_l1_loss(augmented.relative_pose, output.relative_pose.detach())
            absolute_consistency = F.smooth_l1_loss(augmented.root, output.root.detach())
        # Source-environment risk agreement. This is deliberately applied to
        # supervised source losses only; it never uses the held-out E04 split.
        env_risks = []
        for env_id in environments.unique():
            mask = environments == env_id
            if int(mask.sum()) > 1:
                env_risks.append(F.smooth_l1_loss(output.pose[mask], target[mask]))
        risk = torch.stack(env_risks).var(unbiased=False) if len(env_risks) > 1 else output.structure.new_zeros(())
        total = (pose_loss + self.root_weight * root_loss + self.relative_weight * relative_loss
                 + self.bone_weight * bone_loss + 0.1 * nll
                 + self.domain_weight * domain_loss + self.consistency_weight * consistency
                 + self.absolute_consistency_weight * absolute_consistency + self.risk_weight * risk)
        return total, {"loss": float(total.detach()), "pose_loss": float(pose_loss.detach()),
                       "root_loss": float(root_loss.detach()), "relative_loss": float(relative_loss.detach()),
                       "bone_loss": float(bone_loss.detach()), "root_nll": float(nll.detach()),
                       "domain_loss": float(domain_loss.detach()), "consistency_loss": float(consistency.detach()),
                       "absolute_consistency_loss": float(absolute_consistency.detach()),
                       "risk_loss": float(risk.detach())}
