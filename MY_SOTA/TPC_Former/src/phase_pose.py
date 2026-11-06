"""Phase-aware CSI views and a small TPC-Former prototype.

The module keeps phase as a paired real/imaginary representation until the
task heads. It is intentionally small: the first goal is to test whether the
phase views carry transferable signal, not to hide the question behind a
large backbone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


CSI_SHAPE = (3, 114, 10)
PARENTS = (-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 8, 13, 14, 15)
PhaseView = Literal["raw", "shared", "per_rx", "difference"]


def _finite_fill(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    fill = float(x[finite].mean()) if finite.any() else 0.0
    return np.nan_to_num(x, nan=fill, posinf=fill, neginf=fill)


def load_csi_phase_frame(frame_path: str | Path) -> np.ndarray:
    """Load CSIphase from a MM-Fi frame without treating it as amplitude."""
    path = Path(frame_path)
    npy_path = path.with_suffix(".phase.npy")
    if npy_path.is_file():
        phase = np.load(npy_path)
    else:
        import scipy.io as scio

        payload = scio.loadmat(path)
        if "CSIphase" not in payload:
            raise KeyError(f"CSIphase missing from {path}")
        phase = payload["CSIphase"]
    phase = _finite_fill(phase)
    if tuple(phase.shape) != CSI_SHAPE:
        raise ValueError(f"expected CSIphase {CSI_SHAPE}, got {tuple(phase.shape)}")
    return phase


def _unwrap_phase(phase: Tensor) -> Tensor:
    """Unwrap phase along subcarriers before fitting affine trends."""
    delta = torch.diff(phase, dim=2)
    wrapped_delta = torch.atan2(torch.sin(delta), torch.cos(delta))
    return torch.cat((phase[:, :, :1], phase[:, :, :1] + torch.cumsum(wrapped_delta, dim=2)), dim=2)


def fit_phase_calibration(phase: Tensor) -> dict[str, Tensor]:
    """Fit source statistics for the per-Rx phase trend.

    The returned slope/offset tensors are intended to be fitted on source
    training frames and reused for validation/test frames.
    """
    unwrapped = _unwrap_phase(phase)
    if unwrapped.ndim != 4 or tuple(unwrapped.shape[1:]) != CSI_SHAPE:
        raise ValueError(f"expected phase (B,{CSI_SHAPE}), got {tuple(unwrapped.shape)}")
    _, _, subcarriers, _ = unwrapped.shape
    k = torch.linspace(-1.0, 1.0, subcarriers, device=unwrapped.device, dtype=unwrapped.dtype)
    design = torch.stack((k, torch.ones_like(k)), dim=-1)
    pinv = torch.linalg.pinv(design)
    coeff = torch.einsum("ik,brkt->brit", pinv, unwrapped)
    return {"slope": coeff[:, :, 0].median(dim=0).values, "offset": coeff[:, :, 1].median(dim=0).values}


def _fit_linear_subcarrier(phase: Tensor, shared: bool, calibration: Optional[dict[str, Tensor]] = None) -> Tensor:
    """Remove a source-defined affine phase trend over subcarriers.

    ``phase`` is shaped (B, Rx, K, T). The fit is performed independently for
    each frame sample. With ``shared=True`` one affine trend is fitted across
    Rx; otherwise each Rx has its own trend.
    """
    if phase.ndim != 4 or phase.shape[1:] != CSI_SHAPE:
        raise ValueError(f"expected phase (B,{CSI_SHAPE}), got {tuple(phase.shape)}")
    _, rx, subcarriers, steps = phase.shape
    phase = _unwrap_phase(phase)
    k = torch.linspace(-1.0, 1.0, subcarriers, device=phase.device, dtype=phase.dtype)
    design = torch.stack([k, torch.ones_like(k)], dim=-1)  # (K, 2)
    if calibration is not None:
        slope = calibration["slope"].to(device=phase.device, dtype=phase.dtype)
        offset = calibration["offset"].to(device=phase.device, dtype=phase.dtype)
        if slope.shape != (rx, steps) or offset.shape != (rx, steps):
            raise ValueError(f"calibration must have shape ({rx},{steps}) per parameter")
        if shared:
            slope = slope.mean(dim=0, keepdim=True).expand(rx, -1)
            offset = offset.mean(dim=0, keepdim=True).expand(rx, -1)
        trend = slope[None, :, None, :] * k[None, None, :, None] + offset[None, :, None, :]
        return phase - trend
    pinv = torch.linalg.pinv(design)
    if shared:
        values = phase.mean(dim=1, keepdim=True)
        coeff = torch.einsum("ik,brkt->brit", pinv, values)
        trend = torch.einsum("ki,brit->brkt", design, coeff)
        return phase - trend
    coeff = torch.einsum("ik,brkt->brit", pinv, phase)
    trend = torch.einsum("ki,brit->brkt", design, coeff)
    return phase - trend


def phase_views(phase: Tensor, calibration: Optional[dict[str, Tensor]] = None) -> dict[str, Tensor]:
    """Return explicit phase views used by the controlled experiments."""
    if phase.ndim != 4 or tuple(phase.shape[1:]) != CSI_SHAPE:
        raise ValueError(f"expected phase (B,{CSI_SHAPE}), got {tuple(phase.shape)}")
    raw = torch.stack((torch.cos(phase), torch.sin(phase)), dim=2)
    shared = _fit_linear_subcarrier(phase, shared=True, calibration=calibration)
    per_rx = _fit_linear_subcarrier(phase, shared=False, calibration=calibration)
    shared_unit = torch.stack((torch.cos(shared), torch.sin(shared)), dim=2)
    per_rx_unit = torch.stack((torch.cos(per_rx), torch.sin(per_rx)), dim=2)
    # Differences are referenced to Rx 0; keep the pair on the unit circle.
    difference = phase[:, 1:] - phase[:, :1]
    difference_unit = torch.stack((torch.cos(difference), torch.sin(difference)), dim=2)
    shared_difference = shared[:, 1:] - shared[:, :1]
    per_rx_difference = per_rx[:, 1:] - per_rx[:, :1]
    return {
        "raw": raw,                 # (B, Rx, 2, K, T)
        "shared": shared_unit,
        "per_rx": per_rx_unit,
        "difference": difference_unit,  # (B, Rx-1, 2, K, T)
        "difference_shared": torch.stack((torch.cos(shared_difference), torch.sin(shared_difference)), dim=2),
        "difference_per_rx": torch.stack((torch.cos(per_rx_difference), torch.sin(per_rx_difference)), dim=2),
    }


def _same_pair_dropout(x: Tensor, p: float, training: bool) -> Tensor:
    """Drop phase pairs jointly so angle relations are not randomly changed."""
    if not training or p <= 0.0:
        return x
    if x.ndim != 5 or x.shape[2] != 2:
        raise ValueError("phase tensor must be (B,Rx,2,K,T)")
    keep = torch.rand(x.shape[0], x.shape[1], 1, x.shape[3], x.shape[4], device=x.device) >= p
    return x * keep.to(dtype=x.dtype)


class _StructuredEncoder(nn.Module):
    def __init__(self, in_channels: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(7, 3), padding=(3, 1)),
            nn.GELU(),
            nn.Conv2d(32, 48, kernel_size=(5, 3), padding=(2, 1)),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((8, 4)),
            nn.Flatten(),
            nn.Linear(48 * 8 * 4, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


@dataclass
class TPCOutput:
    relative_pose: Tensor
    root: Tensor
    absolute_pose: Tensor
    body_gate: Tensor
    root_gate: Tensor
    phase_quality: Tensor
    gauge_confidence: Tensor


class TPCFormer(nn.Module):
    """Task-Partitioned Complex Phase Fusion prototype.

    Inputs are one CSI frame at a time. The phase branch keeps cos/sin as a
    pair and uses shared dropout for both components. Body and root branches
    have separate gates and heads; root also sees raw amplitude statistics.
    """

    def __init__(self, hidden_dim: int = 128, phase_dropout: float = 0.10) -> None:
        super().__init__()
        self.phase_dropout = float(phase_dropout)
        self.body_amp = _StructuredEncoder(3, hidden_dim)
        self.body_phase = nn.ModuleList([_StructuredEncoder(6, hidden_dim) for _ in range(3)])
        self.root_amp = _StructuredEncoder(3, hidden_dim)
        self.root_phase = nn.ModuleList([_StructuredEncoder(4, hidden_dim) for _ in range(3)])
        self.body_gate = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 3))
        self.root_gate = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 3))
        self.root_confidence = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.relative_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 51)
        )
        self.length_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 17)
        )
        self.root_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2 + 14), nn.Linear(hidden_dim * 2 + 14, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3)
        )

    @staticmethod
    def _flatten_phase(view: Tensor) -> Tensor:
        # (B, Rx, 2, K, T) -> (B, 2*Rx, K, T)
        return view.flatten(1, 2)

    @staticmethod
    def _root_stats(raw_amplitude: Tensor) -> Tensor:
        mean = raw_amplitude.mean(dim=(2, 3))
        std = raw_amplitude.std(dim=(2, 3), unbiased=False)
        minimum = raw_amplitude.amin(dim=(2, 3))
        maximum = raw_amplitude.amax(dim=(2, 3))
        ratios = mean[:, 1:] / mean[:, :1].clamp_min(1e-4)
        return torch.cat((mean, std, minimum, maximum, ratios), dim=-1)

    @staticmethod
    def _phase_quality(amplitude: Tensor, phase: Tensor) -> Tensor:
        magnitude_quality = amplitude.abs().mean(dim=(1, 2, 3), keepdim=True).clamp(0.0, 1.0)
        delta = torch.diff(phase, dim=2)
        continuity = torch.cos(delta).mean(dim=(1, 2, 3), keepdim=True).clamp(0.0, 1.0)
        return (0.5 * magnitude_quality + 0.5 * continuity).reshape(amplitude.shape[0], 1)

    def forward(
        self,
        amplitude: Tensor,
        phase: Tensor,
        *,
        raw_amplitude: Optional[Tensor] = None,
        calibration: Optional[dict[str, Tensor]] = None,
    ) -> TPCOutput:
        if amplitude.ndim != 4 or tuple(amplitude.shape[1:]) != CSI_SHAPE:
            raise ValueError(f"expected amplitude (B,{CSI_SHAPE}), got {tuple(amplitude.shape)}")
        if phase.ndim != 4 or tuple(phase.shape[1:]) != CSI_SHAPE:
            raise ValueError(f"expected phase (B,{CSI_SHAPE}), got {tuple(phase.shape)}")
        if raw_amplitude is None:
            raw_amplitude = amplitude
        if raw_amplitude.ndim != 4 or tuple(raw_amplitude.shape[1:]) != CSI_SHAPE:
            raise ValueError(f"expected raw amplitude (B,{CSI_SHAPE}), got {tuple(raw_amplitude.shape)}")
        views = phase_views(phase, calibration=calibration)
        raw_phase = _same_pair_dropout(views["raw"], self.phase_dropout, self.training)
        body_phase_views = (raw_phase, views["shared"], views["per_rx"])
        root_phase_views = tuple(
            _same_pair_dropout(view, self.phase_dropout, self.training)
            for view in (views["difference"], views["difference_shared"], views["difference_per_rx"])
        )
        body_amp = self.body_amp(amplitude)
        body_candidates = torch.stack([encoder(self._flatten_phase(view)) for encoder, view in zip(self.body_phase, body_phase_views)], dim=1)
        body_weights = torch.softmax(self.body_gate(body_amp), dim=-1)
        body_phase = (body_candidates * body_weights.unsqueeze(-1)).sum(dim=1)
        body = body_amp + body_phase
        raw_directions = self.relative_head(body).reshape(amplitude.shape[0], 17, 3)
        directions = F.normalize(raw_directions, dim=-1, eps=1e-6)
        lengths = F.softplus(self.length_head(body)).unsqueeze(-1)
        joints = [directions[:, :1] * 0.0]
        for joint, parent in enumerate(PARENTS[1:], start=1):
            joints.append(joints[parent] + directions[:, joint:joint + 1] * lengths[:, joint:joint + 1])
        relative = torch.cat(joints, dim=1)

        root_candidates = torch.stack([encoder(self._flatten_phase(view)) for encoder, view in zip(self.root_phase, root_phase_views)], dim=1)
        root_weights = torch.softmax(self.root_gate(self.root_amp(amplitude)), dim=-1)
        root_phase = (root_candidates * root_weights.unsqueeze(-1)).sum(dim=1)
        root_amp = self.root_amp(amplitude)
        root_latent = root_amp + root_phase
        confidence = self.root_confidence(root_latent)
        root_raw = self.root_head(torch.cat((root_latent, body.detach(), self._root_stats(raw_amplitude)), dim=-1))
        quality = self._phase_quality(amplitude, phase)
        root = root_raw
        absolute = relative + root.unsqueeze(1)
        return TPCOutput(relative, root, absolute, body_weights, root_weights, quality, confidence)


__all__ = ["TPCFormer", "TPCOutput", "load_csi_phase_frame", "phase_views"]
