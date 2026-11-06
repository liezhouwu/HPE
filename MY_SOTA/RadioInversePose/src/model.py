"""Learned radio forward model and per-frame analysis by synthesis."""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

OBS_SHAPE = (3, 3, 12, 5)
OBS_DIM = math.prod(OBS_SHAPE)


def observation(amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    if amplitude.ndim != 4 or tuple(amplitude.shape[1:]) != (3, 114, 10) or phase.shape != amplitude.shape:
        raise ValueError("expected amplitude and phase with shape (B,3,114,10)")
    x = torch.stack((torch.log1p(amplitude.abs()), phase.cos(), phase.sin()), dim=2)
    return F.adaptive_avg_pool2d(x.flatten(1, 2), (12, 5)).flatten(1)


class RadioInversePose(nn.Module):
    def __init__(self, obs_mean, obs_std, pose_mean, pose_basis, dim=256, nuisance_dim=8):
        super().__init__()
        self.register_buffer("obs_mean", obs_mean.reshape(1, OBS_DIM))
        self.register_buffer("obs_std", obs_std.reshape(1, OBS_DIM).clamp_min(0.05))
        self.register_buffer("pose_mean", pose_mean.reshape(1, 51))
        self.register_buffer("pose_basis", pose_basis.reshape(-1, 51))
        self.initializer = nn.Sequential(nn.Linear(OBS_DIM, dim), nn.LayerNorm(dim), nn.GELU(),
                                         nn.Linear(dim, dim), nn.GELU())
        self.pose_head = nn.Linear(dim, 51)
        self.nuisance_head = nn.Linear(dim, nuisance_dim)
        self.forward_model = nn.Sequential(nn.Linear(51 * 5 + nuisance_dim, dim), nn.GELU(),
                                           nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, OBS_DIM))

    def normalize(self, amplitude, phase):
        return ((observation(amplitude, phase) - self.obs_mean) / self.obs_std).clamp(-8, 8)

    def initialize(self, obs):
        h = self.initializer(obs)
        return self.pose_head(h) + self.pose_mean, self.nuisance_head(h)

    def render(self, pose, nuisance):
        # Pose coordinates in meters; sine/cosine features preserve absolute position.
        x = pose.reshape(-1, 51)
        features = torch.cat((x, torch.sin(x * math.pi), torch.cos(x * math.pi),
                              torch.sin(x * 2 * math.pi), torch.cos(x * 2 * math.pi)), dim=-1)
        return self.forward_model(torch.cat((features, nuisance), dim=-1))

    def prior(self, pose):
        centered = pose - self.pose_mean
        residual = centered - (centered @ self.pose_basis.T) @ self.pose_basis
        return residual.square().mean(dim=1)

    def objective(self, pose, nuisance, obs, initial, weights):
        error = F.smooth_l1_loss(self.render(pose, nuisance), obs, reduction="none").mean(dim=1)
        return (error + weights["pose_prior"] * self.prior(pose)
                + weights["nuisance"] * nuisance.square().mean(dim=1)
                + weights["trust"] * (pose - initial).square().mean(dim=1))

    def refine(self, obs, initial, nuisance, *, steps, learning_rate, weights, differentiable=False):
        if steps < 0 or learning_rate <= 0:
            raise ValueError("invalid optimization settings")
        # Each sample is optimized independently; there are no batch statistics.
        pose = initial if differentiable else initial.detach()
        z = nuisance if differentiable else nuisance.detach()
        with torch.enable_grad():
            pose = pose.requires_grad_(True)
            z = z.requires_grad_(True)
            best_pose, best_z = pose, z
            best_loss = self.objective(pose, z, obs, initial, weights)
            for _ in range(steps):
                loss = self.objective(pose, z, obs, initial, weights)
                gp, gz = torch.autograd.grad(loss.sum(), (pose, z), create_graph=differentiable)
                pose = pose - learning_rate * gp
                z = z - learning_rate * gz
                candidate = self.objective(pose, z, obs, initial, weights)
                better = torch.isfinite(candidate) & (candidate < best_loss)
                best_pose = torch.where(better[:, None], pose, best_pose)
                best_z = torch.where(better[:, None], z, best_z)
                best_loss = torch.where(better, candidate, best_loss)
        return best_pose, best_z, best_loss
