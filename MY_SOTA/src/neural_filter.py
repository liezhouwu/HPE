from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


CSI_SHAPE = (3, 114, 10)
CSI_CALIB_DIM = 12
CSI_PRIOR_DIM = 64
PARENTS = (-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 8, 13, 14, 15)


def _check_windows(windows: Tensor) -> Tensor:
    if windows.ndim == 6 and windows.shape[2] == 1:
        windows = windows[:, :, 0]
    if windows.ndim != 5 or tuple(windows.shape[2:]) != CSI_SHAPE:
        raise ValueError(
            f"expected CSI windows (B,T,3,114,10), got {tuple(windows.shape)}"
        )
    return windows


def csi_stats(x: Tensor) -> Tensor:
    """Low-dimensional CSI observation target used by the state-space decoder."""
    if x.ndim == 5:
        batch, steps = x.shape[:2]
        x = x.reshape(batch * steps, *CSI_SHAPE)
        restore = (batch, steps)
    else:
        restore = None
    if x.ndim != 4 or tuple(x.shape[1:]) != CSI_SHAPE:
        raise ValueError(f"expected CSI frame (B,3,114,10), got {tuple(x.shape)}")
    mean = x.mean(dim=(2, 3))
    std = x.std(dim=(2, 3), unbiased=False)
    bands = F.adaptive_avg_pool1d(
        x.permute(0, 1, 3, 2).reshape(x.shape[0] * 3 * 10, 114), 8
    ).reshape(x.shape[0], 3, 10, 8).mean(dim=2)
    stats = torch.cat([mean.reshape(x.shape[0], -1), std.reshape(x.shape[0], -1), bands.reshape(x.shape[0], -1)], dim=-1)
    if restore is not None:
        stats = stats.reshape(*restore, -1)
    return stats


def apply_csi_intervention(
    windows: Tensor,
    *,
    gain: float = 1.08,
    offset: float = 0.02,
    subcarrier_drop: float = 0.05,
) -> tuple[Tensor, Tensor]:
    """Apply a known CSI-only intervention and return its three-value code."""
    windows = _check_windows(windows)
    keep = (
        torch.rand(
            windows.shape[0], 1, 1, windows.shape[3], 1,
            device=windows.device, dtype=windows.dtype,
        )
        > float(subcarrier_drop)
    )
    intervened = windows * float(gain) + float(offset)
    intervened = intervened * keep
    code = windows.new_tensor([float(gain) - 1.0, float(offset), float(subcarrier_drop)])
    return intervened, code.expand(windows.shape[0], -1)


class CSIObservationEncoder(nn.Module):
    """Structured fast/slow observation encoder for CSI frames."""

    def __init__(self, fast_dim: int = 128, slow_dim: int = 64,
                 calibration_dim: int = CSI_CALIB_DIM) -> None:
        super().__init__()
        self.calibration_dim = int(calibration_dim)
        self.fast = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(7, 3), padding=(3, 1)),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=(5, 3), padding=(2, 1)),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((8, 4)),
            nn.Flatten(),
            nn.Linear(32 * 8 * 4, fast_dim),
            nn.LayerNorm(fast_dim),
        )
        self.slow = nn.Sequential(
            nn.Linear(30 + self.calibration_dim, slow_dim),
            nn.GELU(),
            nn.LayerNorm(slow_dim),
        )

    def forward(self, x: Tensor, calibration: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        if x.ndim == 5 and x.shape[1] == 1:
            x = x[:, 0]
        if x.ndim != 4 or tuple(x.shape[1:]) != CSI_SHAPE:
            raise ValueError(f"expected CSI frame (B,3,114,10), got {tuple(x.shape)}")
        stats = csi_stats(x)
        if calibration is None:
            calibration = stats.new_zeros(stats.shape[0], self.calibration_dim)
        if calibration.ndim != 2 or calibration.shape[0] != stats.shape[0] or calibration.shape[1] != self.calibration_dim:
            raise ValueError(
                f"expected calibration stats (B,{self.calibration_dim}), got {tuple(calibration.shape)}"
            )
        return self.fast(x), self.slow(torch.cat([stats, calibration], dim=-1))


class AmplitudeSpatialPrior(nn.Module):
    """CSI-only amplitude profile encoder for absolute-root localization.

    The normalized CSI stream is deliberately invariant to frame amplitude.
    This branch preserves the raw amplitude profile over links/subcarriers and
    supplies a spatial prior to the absolute root decoder.
    """

    def __init__(self, prior_dim: int = CSI_PRIOR_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=(9, 3), padding=(4, 1)),
            nn.GELU(),
            nn.Conv2d(24, 32, kernel_size=(5, 3), padding=(2, 1)),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((8, 4)),
            nn.Flatten(),
            nn.Linear(32 * 8 * 4, prior_dim),
            nn.LayerNorm(prior_dim),
        )

    def forward(self, raw_x: Tensor) -> Tensor:
        if raw_x.ndim == 5 and raw_x.shape[1] == 1:
            raw_x = raw_x[:, 0]
        if raw_x.ndim != 4 or tuple(raw_x.shape[1:]) != CSI_SHAPE:
            raise ValueError(f"expected raw CSI frame (B,3,114,10), got {tuple(raw_x.shape)}")
        return self.net(raw_x)


class DualTimeStateFilter(nn.Module):
    """Fast body dynamics plus a slow channel state with explicit carry support."""

    def __init__(
        self,
        fast_dim: int = 128,
        slow_dim: int = 64,
        body_dim: int = 128,
        channel_dim: int = 64,
        channel_alpha_init: float = -2.2,
    ) -> None:
        super().__init__()
        self.body_dim = int(body_dim)
        self.channel_dim = int(channel_dim)
        self.body_transition = nn.GRUCell(fast_dim, body_dim)
        self.channel_transition = nn.GRUCell(slow_dim, channel_dim)
        self.body_gate = nn.Sequential(
            nn.Linear(fast_dim + body_dim, body_dim),
            nn.Sigmoid(),
        )
        self.channel_alpha = nn.Parameter(torch.tensor(float(channel_alpha_init)))
        self.body_init = nn.Parameter(torch.zeros(1, body_dim))
        self.channel_init = nn.Parameter(torch.zeros(1, channel_dim))

    def forward(
        self,
        fast_observations: Tensor,
        slow_observations: Tensor,
        initial_state: Optional[tuple[Tensor, Tensor]] = None,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor]]:
        if fast_observations.ndim != 3 or slow_observations.ndim != 3:
            raise ValueError("state filter expects fast/slow observations shaped (B,T,D)")
        batch, steps = fast_observations.shape[:2]
        if initial_state is None:
            body = self.body_init.expand(batch, -1)
            channel = self.channel_init.expand(batch, -1)
        else:
            body, channel = initial_state
        body_states, channel_states = [], []
        alpha = torch.sigmoid(self.channel_alpha).clamp(0.02, 0.25)
        for step in range(steps):
            fast = fast_observations[:, step]
            slow = slow_observations[:, step]
            body_pred = self.body_transition(fast, body)
            body_gate = self.body_gate(torch.cat([fast, body_pred], dim=-1))
            body = body_gate * body_pred + (1.0 - body_gate) * body
            channel_pred = self.channel_transition(slow, channel)
            channel = (1.0 - alpha) * channel + alpha * channel_pred
            body_states.append(body)
            channel_states.append(channel)
        return (
            torch.stack(body_states, dim=1),
            torch.stack(channel_states, dim=1),
            (body, channel),
        )


class KinematicPoseDecoder(nn.Module):
    """Parent-chain decoder that predicts root and positive bone lengths."""

    def __init__(self, body_dim: int = 128, channel_dim: int = 64,
                 prior_dim: int = CSI_PRIOR_DIM, hidden_dim: int = 128) -> None:
        super().__init__()
        self.body_root_head = nn.Sequential(
            nn.LayerNorm(body_dim),
            nn.Linear(body_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.channel_offset_head = nn.Sequential(
            nn.LayerNorm(channel_dim + prior_dim),
            nn.Linear(channel_dim + prior_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.direction_head = nn.Sequential(
            nn.LayerNorm(body_dim),
            nn.Linear(body_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 17 * 3),
        )
        self.length_head = nn.Sequential(
            nn.LayerNorm(body_dim),
            nn.Linear(body_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 17),
        )
        self.register_buffer("parents", torch.tensor(PARENTS, dtype=torch.long), persistent=False)
        self.register_buffer(
            "edges",
            torch.tensor([(parent, child) for child, parent in enumerate(PARENTS) if parent >= 0],
                         dtype=torch.long),
            persistent=False,
        )

    def forward(self, body: Tensor, channel: Tensor,
                amplitude_prior: Optional[Tensor] = None,
                *, return_components: bool = False):
        if body.ndim != 3 or channel.ndim != 3:
            raise ValueError("decoder expects state sequences shaped (B,T,D)")
        body_root = self.body_root_head(body)
        if amplitude_prior is None:
            amplitude_prior = channel.new_zeros(*channel.shape[:2], self.channel_offset_head[0].normalized_shape[0] - channel.shape[-1])
        if amplitude_prior.shape[:2] != channel.shape[:2]:
            raise ValueError("amplitude prior must match body/channel sequence dimensions")
        channel_offset = self.channel_offset_head(torch.cat([channel, amplitude_prior], dim=-1))
        root = body_root + channel_offset
        directions = self.direction_head(body).reshape(body.shape[0], body.shape[1], 17, 3)
        directions = F.normalize(directions, dim=-1, eps=1e-6)
        lengths = F.softplus(self.length_head(body)).reshape(body.shape[0], body.shape[1], 17, 1)
        joints = [root]
        for joint in range(1, 17):
            parent = PARENTS[joint]
            joints.append(joints[parent] + directions[:, :, joint] * lengths[:, :, joint])
        pose = torch.stack(joints, dim=2)
        if return_components:
            return pose, body_root, channel_offset
        return pose

    def bone_length_loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        pred_edges = prediction[:, :, self.edges[:, 0]] - prediction[:, :, self.edges[:, 1]]
        target_edges = target[:, :, self.edges[:, 0]] - target[:, :, self.edges[:, 1]]
        return F.smooth_l1_loss(pred_edges.norm(dim=-1), target_edges.norm(dim=-1))


@dataclass
class FilterOutput:
    pose: Tensor
    pose_sequence: Tensor
    body_states: Tensor
    channel_states: Tensor
    predicted_stats: Tensor
    body_root: Tensor
    channel_offset: Tensor


class ICDTSPoseModel(nn.Module):
    """Intervention-Calibrated Dual-Time State-Space Pose model."""

    def __init__(
        self,
        fast_dim: int = 128,
        slow_dim: int = 64,
        body_dim: int = 128,
        channel_dim: int = 64,
        channel_alpha_init: float = -2.2,
        loss_weights: Optional[dict[str, float]] = None,
        calibration_dim: int = CSI_CALIB_DIM,
        prior_dim: int = CSI_PRIOR_DIM,
        use_amplitude_prior: bool = True,
    ) -> None:
        super().__init__()
        self.observation = CSIObservationEncoder(
            fast_dim=fast_dim, slow_dim=slow_dim, calibration_dim=calibration_dim
        )
        self.use_amplitude_prior = bool(use_amplitude_prior)
        self.amplitude_prior = AmplitudeSpatialPrior(prior_dim=prior_dim)
        self.filter = DualTimeStateFilter(
            fast_dim=fast_dim, slow_dim=slow_dim,
            body_dim=body_dim, channel_dim=channel_dim,
            channel_alpha_init=channel_alpha_init,
        )
        self.decoder = KinematicPoseDecoder(
            body_dim=body_dim, channel_dim=channel_dim, prior_dim=prior_dim,
        )
        self.observation_head = nn.Sequential(
            nn.LayerNorm(body_dim + channel_dim),
            nn.Linear(body_dim + channel_dim, 128),
            nn.GELU(),
            nn.Linear(128, 30),
        )
        self.intervention_delta_head = nn.Sequential(
            nn.LayerNorm(channel_dim),
            nn.Linear(channel_dim, 32),
            nn.GELU(),
            nn.Linear(32, 3),
        )
        self.loss_weights = {
            "pose_weight": 1.0,
            "velocity_weight": 0.25,
            "acceleration_weight": 0.10,
            "bone_weight": 0.10,
            "observation_weight": 0.10,
            "intervention_weight": 0.20,
            "root_decomposition_weight": 0.25,
            "root_weight": 4.0,
        }
        if loss_weights:
            self.loss_weights.update({key: float(value) for key, value in loss_weights.items()})

    @classmethod
    def from_config(cls, config: dict) -> "ICDTSPoseModel":
        observation = config.get("observation", {})
        state = config.get("state", {})
        losses = config.get("loss", {})
        return cls(
            fast_dim=int(observation.get("fast_dim", 128)),
            slow_dim=int(observation.get("slow_dim", 64)),
            body_dim=int(state.get("body_dim", 128)),
            channel_dim=int(state.get("channel_dim", 64)),
            channel_alpha_init=float(state.get("channel_alpha_init", -2.2)),
            calibration_dim=int(observation.get("calibration_dim", CSI_CALIB_DIM)),
            prior_dim=int(observation.get("prior_dim", CSI_PRIOR_DIM)),
            use_amplitude_prior=bool(observation.get("use_amplitude_prior", True)),
            loss_weights={
                "pose_weight": losses.get("pose_weight", 1.0),
                "velocity_weight": losses.get("velocity_weight", 0.25),
                "acceleration_weight": losses.get("acceleration_weight", 0.10),
                "bone_weight": losses.get("bone_length_weight", 0.10),
                "observation_weight": losses.get("observation_weight", 0.10),
                "intervention_weight": losses.get("intervention_weight", 0.20),
                "root_decomposition_weight": losses.get("root_decomposition_weight", 0.25),
                "root_weight": losses.get("root_weight", 4.0),
            },
        )

    def forward(
        self,
        windows: Tensor,
        *,
        calibration: Optional[Tensor] = None,
        raw_windows: Optional[Tensor] = None,
        initial_state: Optional[tuple[Tensor, Tensor]] = None,
        return_states: bool = False,
    ):
        windows = _check_windows(windows)
        batch, steps = windows.shape[:2]
        frames = windows.reshape(batch * steps, *CSI_SHAPE)
        flat_calibration = None
        if calibration is not None:
            if calibration.ndim != 3 or calibration.shape[:2] != (batch, steps):
                raise ValueError(
                    f"expected calibration stats (B,T,D), got {tuple(calibration.shape)}"
                )
            flat_calibration = calibration.reshape(batch * steps, -1)
        fast, slow = self.observation(frames, flat_calibration)
        fast = fast.reshape(batch, steps, -1)
        slow = slow.reshape(batch, steps, -1)
        body, channel, _ = self.filter(fast, slow, initial_state=initial_state)
        amplitude_prior = None
        if self.use_amplitude_prior:
            if raw_windows is None:
                raw_windows = windows.new_zeros(windows.shape)
            raw_windows = _check_windows(raw_windows)
            raw_frames = raw_windows.reshape(batch * steps, *CSI_SHAPE)
            amplitude_prior = self.amplitude_prior(raw_frames).reshape(batch, steps, -1)
        pose_sequence, body_root, channel_offset = self.decoder(
            body, channel, amplitude_prior, return_components=True
        )
        state_pair = torch.cat([body, channel], dim=-1)
        predicted_stats = self.observation_head(state_pair[:, :-1])
        if return_states:
            return FilterOutput(
                pose=pose_sequence[:, -1],
                pose_sequence=pose_sequence,
                body_states=body,
                channel_states=channel,
                predicted_stats=predicted_stats,
                body_root=body_root,
                channel_offset=channel_offset,
            )
        return pose_sequence[:, -1]

    def training_loss(
        self,
        windows: Tensor,
        targets: Tensor,
        *,
        calibration: Optional[Tensor] = None,
        raw_windows: Optional[Tensor] = None,
        intervention_windows: Optional[Tensor] = None,
        intervention_raw_windows: Optional[Tensor] = None,
        intervention_code: Optional[Tensor] = None,
        intervention_calibration: Optional[Tensor] = None,
        pose_weight: Optional[float] = None,
        velocity_weight: Optional[float] = None,
        acceleration_weight: Optional[float] = None,
        bone_weight: Optional[float] = None,
        observation_weight: Optional[float] = None,
        intervention_weight: Optional[float] = None,
        root_decomposition_weight: Optional[float] = None,
        root_weight: Optional[float] = None,
    ) -> tuple[Tensor, dict]:
        output = self.forward(windows, calibration=calibration, raw_windows=raw_windows, return_states=True)
        weights = self.loss_weights
        pose_weight = weights["pose_weight"] if pose_weight is None else float(pose_weight)
        velocity_weight = weights["velocity_weight"] if velocity_weight is None else float(velocity_weight)
        acceleration_weight = weights["acceleration_weight"] if acceleration_weight is None else float(acceleration_weight)
        bone_weight = weights["bone_weight"] if bone_weight is None else float(bone_weight)
        observation_weight = weights["observation_weight"] if observation_weight is None else float(observation_weight)
        intervention_weight = weights["intervention_weight"] if intervention_weight is None else float(intervention_weight)
        root_decomposition_weight = (
            weights["root_decomposition_weight"]
            if root_decomposition_weight is None else float(root_decomposition_weight)
        )
        root_weight = weights["root_weight"] if root_weight is None else float(root_weight)
        sequence_targets = targets.ndim == 4
        if targets.ndim == 3:
            if tuple(targets.shape[1:]) != (17, 3):
                raise ValueError("single-frame targets must be (B,17,3)")
            pose_loss = F.smooth_l1_loss(output.pose, targets)
        elif targets.ndim == 4 and targets.shape[1] == output.pose_sequence.shape[1]:
            pose_loss = F.smooth_l1_loss(output.pose_sequence, targets)
        else:
            raise ValueError("targets must be (B,17,3) or (B,T,17,3) matching the window")
        root_target = targets[:, :, 0] if sequence_targets else targets[:, 0]
        root_prediction = output.pose_sequence[:, :, 0] if sequence_targets else output.pose[:, 0]
        root_loss = F.smooth_l1_loss(root_prediction, root_target)
        if sequence_targets:
            target_root = targets[:, :, 0]
            target_anchor = target_root.mean(dim=1, keepdim=True)
            target_body_root = target_root - target_anchor
            root_decomposition_loss = (
                F.smooth_l1_loss(output.body_root, target_body_root)
                + F.smooth_l1_loss(
                    output.channel_offset,
                    target_anchor.expand_as(output.channel_offset),
                )
            )
        else:
            # With a single target frame only the sum is identifiable. Use the
            # last state and avoid broadcasting a one-frame target over history.
            root_decomposition_loss = F.smooth_l1_loss(
                output.body_root[:, -1], output.body_root[:, -1].detach()
            )
        velocity_loss = output.pose_sequence.new_zeros(())
        acceleration_loss = output.pose_sequence.new_zeros(())
        if sequence_targets and output.pose_sequence.shape[1] >= 2:
            pred_velocity = output.pose_sequence[:, 1:] - output.pose_sequence[:, :-1]
            true_velocity = targets[:, 1:] - targets[:, :-1]
            velocity_loss = F.smooth_l1_loss(pred_velocity, true_velocity)
        if sequence_targets and output.pose_sequence.shape[1] >= 3:
            pred_acc = output.pose_sequence[:, 2:] - 2 * output.pose_sequence[:, 1:-1] + output.pose_sequence[:, :-2]
            true_acc = targets[:, 2:] - 2 * targets[:, 1:-1] + targets[:, :-2]
            acceleration_loss = F.smooth_l1_loss(pred_acc, true_acc)
        if sequence_targets:
            bone_loss = self.decoder.bone_length_loss(output.pose_sequence, targets)
        else:
            bone_loss = self.decoder.bone_length_loss(
                output.pose[:, None], targets[:, None]
            )
        observation_target = csi_stats(windows[:, 1:])
        observation_loss = F.smooth_l1_loss(output.predicted_stats, observation_target)
        intervention_loss = windows.new_zeros(())
        if intervention_windows is not None:
            augmented = self.forward(
                intervention_windows,
                calibration=intervention_calibration,
                raw_windows=intervention_raw_windows,
                return_states=True,
            )
            body_consistency = F.smooth_l1_loss(augmented.body_states, output.body_states.detach())
            if intervention_code is None:
                intervention_code = windows.new_zeros(windows.shape[0], 3)
            channel_delta = augmented.channel_states[:, -1] - output.channel_states[:, -1].detach()
            predicted_code = self.intervention_delta_head(channel_delta)
            code_loss = F.smooth_l1_loss(predicted_code, intervention_code)
            intervention_loss = body_consistency + code_loss
        loss = (
            pose_weight * pose_loss
            + root_weight * root_loss
            + velocity_weight * velocity_loss
            + acceleration_weight * acceleration_loss
            + bone_weight * bone_loss
            + observation_weight * observation_loss
            + intervention_weight * intervention_loss
            + root_decomposition_weight * root_decomposition_loss
        )
        return loss, {
            "loss": float(loss.detach()),
            "pose_loss": float(pose_loss.detach()),
            "root_loss": float(root_loss.detach()),
            "velocity_loss": float(velocity_loss.detach()),
            "acceleration_loss": float(acceleration_loss.detach()),
            "bone_loss": float(bone_loss.detach()),
            "observation_loss": float(observation_loss.detach()),
            "intervention_loss": float(intervention_loss.detach()),
            "root_decomposition_loss": float(root_decomposition_loss.detach()),
        }


# Backward-compatible name for existing smoke/integration callers.
NeuralFilterPoseModel = ICDTSPoseModel

