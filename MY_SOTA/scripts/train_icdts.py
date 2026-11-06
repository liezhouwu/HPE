from __future__ import annotations

import argparse
import json
import random
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mmfi_wifi.data import _load_csi_frame, _load_ground_truth
from mmfi_wifi.data_manifest import DataManifest
from mmfi_wifi.sequence_keys import SequenceKey
from MY_SOTA.src.neural_filter import ICDTSPoseModel, apply_csi_intervention


FRAME_COUNT = 297
CSI_SHAPE = (3, 114, 10)
CSI_CALIB_DIM = 12


def _load_raw_csi(frame_path: Path) -> np.ndarray:
    """Load the un-normalized CSI amplitude used for calibration statistics."""
    npy_path = frame_path.with_suffix(".npy")
    if npy_path.is_file():
        raw = np.load(npy_path)
    else:
        import scipy.io as scio
        raw = scio.loadmat(frame_path)["CSIamp"]
    raw = np.asarray(raw, dtype=np.float32)
    finite = np.isfinite(raw)
    fill = float(raw[finite].mean()) if finite.any() else 0.0
    return np.nan_to_num(raw, nan=fill, posinf=fill, neginf=fill)


def _csi_calibration(raw: np.ndarray) -> torch.Tensor:
    """Per-link raw amplitude statistics, kept outside frame-wise min-max CSI."""
    if tuple(raw.shape) != CSI_SHAPE:
        raise ValueError(f"expected raw CSI {CSI_SHAPE}, got {tuple(raw.shape)}")
    mean = raw.mean(axis=(1, 2))
    std = raw.std(axis=(1, 2))
    minimum = raw.min(axis=(1, 2))
    maximum = raw.max(axis=(1, 2))
    # Raw CSI amplitudes in MM-Fi are approximately in [0, 60]. Fixed scaling
    # keeps this side channel numerically comparable to the normalized CSI.
    stats = np.concatenate([mean / 60.0, std / 20.0,
                            minimum / 60.0, maximum / 60.0], axis=0)
    return torch.as_tensor(stats, dtype=torch.float32)


@lru_cache(maxsize=200000)
def _cached_csi_calibration(frame_path: str) -> torch.Tensor:
    return _csi_calibration(_load_raw_csi(Path(frame_path)))


@lru_cache(maxsize=200000)
def _cached_raw_csi(frame_path: str) -> torch.Tensor:
    # Fixed global scale retains amplitude profile while keeping the prior
    # branch in the same numerical range as normalized CSI.
    return torch.as_tensor(_load_raw_csi(Path(frame_path)) / 60.0, dtype=torch.float32)


def transform_calibration(calibration: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
    """Apply the known gain/offset part of a CSI intervention to raw stats."""
    if calibration.ndim != 3 or calibration.shape[-1] != CSI_CALIB_DIM:
        raise ValueError(f"expected calibration (B,T,{CSI_CALIB_DIM})")
    gain = 1.0 + code[:, 0:1, None]
    offset = code[:, 1:2, None]
    mean, std, minimum, maximum = calibration.split(3, dim=-1)
    # The dropped-subcarrier perturbation is stochastic and has a small effect
    # on these aggregate statistics; retaining the original moments is stable.
    return torch.cat([gain * mean + offset / 60.0,
                      gain * std,
                      gain * minimum + offset / 60.0,
                      gain * maximum + offset / 60.0], dim=-1)


def read_keys(path: Path) -> frozenset[SequenceKey]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return frozenset(SequenceKey(*item) for item in payload["selected_keys"])


class TemporalPoseDataset(Dataset):
    def __init__(self, dataset_root: Path, keys, offsets: tuple[int, ...]):
        self.dataset_root = dataset_root
        self.offsets = tuple(offsets)
        if tuple(self.offsets) != tuple(sorted(self.offsets)):
            raise ValueError("offsets must be sorted")
        self.items = [
            (key, anchor)
            for key in sorted(keys)
            for anchor in range(-min(self.offsets), FRAME_COUNT - max(self.offsets))
        ]
        if not self.items:
            raise ValueError("no temporal windows available for the provided keys")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        key, anchor = self.items[index]
        sequence_dir = self.dataset_root / key.scene / key.subject / key.action
        frames = []
        raw_frames = []
        calibration = []
        for offset in self.offsets:
            frame_path = sequence_dir / "wifi-csi" / f"frame{anchor + offset + 1:03d}.mat"
            frames.append(torch.as_tensor(_load_csi_frame(str(frame_path)), dtype=torch.float32))
            raw_frames.append(_cached_raw_csi(str(frame_path)))
            calibration.append(_cached_csi_calibration(str(frame_path)))
        gt = torch.as_tensor(
            _load_ground_truth(str(sequence_dir / "ground_truth.npy"))[
                [anchor + offset for offset in self.offsets]
            ].copy(),
            dtype=torch.float32,
        )
        return torch.stack(frames), torch.stack(raw_frames), torch.stack(calibration), gt


def mpjpe_mm(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).norm(dim=-1).mean() * 1000.0


def pa_mpjpe_mm(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Official PA-MPJPE: translation, rotation and scale Procrustes alignment."""
    pred_mean = pred.mean(dim=-2, keepdim=True)
    target_mean = target.mean(dim=-2, keepdim=True)
    pred0 = pred - pred_mean
    target0 = target - target_mean
    covariance = pred0.transpose(-2, -1) @ target0
    u, singular, vh = torch.linalg.svd(covariance)
    det = torch.det(u @ vh).sign().reshape(-1, 1, 1)
    correction = torch.eye(3, device=pred.device).expand(pred.shape[0], -1, -1).clone()
    correction[:, -1, -1] = det[:, 0, 0]
    rotation = u @ correction @ vh
    trace = (singular * torch.cat([
        torch.ones_like(singular[..., :2]), det.squeeze(-1)
    ], dim=-1)).sum(dim=-1, keepdim=True).unsqueeze(-1)
    scale = trace / pred0.square().sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    aligned = scale * (pred0 @ rotation) + target_mean
    return (aligned - target).norm(dim=-1).mean() * 1000.0


def collate(batch):
    windows, raw_windows, calibration, targets = zip(*batch)
    return torch.stack(windows), torch.stack(raw_windows), torch.stack(calibration), torch.stack(targets)


def evaluate(model, loader, device, max_batches: int = 0, amp_enabled: bool = False):
    model.eval()
    abs_sum = pa_sum = count = 0
    with torch.no_grad():
        for batch_index, (windows, raw_windows, calibration, targets) in enumerate(loader, start=1):
            windows = windows.to(device)
            raw_windows = raw_windows.to(device)
            calibration = calibration.to(device)
            targets = targets.to(device)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                output = model(windows, calibration=calibration, raw_windows=raw_windows, return_states=True)
            pred = output.pose
            target = targets[:, -1]
            abs_sum += float(mpjpe_mm(pred, target)) * windows.shape[0]
            pa_sum += float(pa_mpjpe_mm(pred, target)) * windows.shape[0]
            count += windows.shape[0]
            if max_batches and batch_index >= max_batches:
                break
    return {"absolute_mpjpe_mm": abs_sum / max(1, count), "pa_mpjpe_mm": pa_sum / max(1, count)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "MY_SOTA/configs/icdts_p3.yaml")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_cfg = config["experiment"]
    artifact_cfg = config["artifacts"]
    dataset_root = Path(run_cfg["dataset_root"])
    manifest_path = Path(run_cfg["manifest"])
    label_manifest_path = Path(run_cfg["label_manifest"])
    output_dir = Path(artifact_cfg["output_root"])
    if not dataset_root.is_absolute():
        dataset_root = (ROOT / dataset_root).resolve()
    if not manifest_path.is_absolute():
        manifest_path = (ROOT / manifest_path).resolve()
    if not label_manifest_path.is_absolute():
        label_manifest_path = (ROOT / label_manifest_path).resolve()
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    manifest = DataManifest.read_json(manifest_path)
    selected_keys = read_keys(label_manifest_path)
    if not selected_keys.issubset(set(manifest.internal_train_keys)):
        raise ValueError("label manifest contains keys outside strict internal_train_keys")
    offsets = tuple(config["experiment"].get("window_offsets", [-5, -4, -3, -2, -1, 0]))
    train_ds = TemporalPoseDataset(dataset_root, selected_keys, offsets)
    val_ds = TemporalPoseDataset(dataset_root, manifest.select_keys, offsets)
    test_ds = TemporalPoseDataset(dataset_root, manifest.test_keys, offsets)
    batch_size = int(run_cfg.get("batch_size", 8))
    num_workers = int(run_cfg.get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=collate)
    device_name = str(run_cfg.get("device", "cuda"))
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    model = ICDTSPoseModel.from_config(config).to(device)
    learning_rate = float(run_cfg.get("learning_rate", 3e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=float(run_cfg.get("weight_decay", 1e-4)))
    scheduler_name = str(run_cfg.get("scheduler", "cosine")).lower()
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, int(run_cfg.get("epochs", 25))),
            eta_min=float(run_cfg.get("min_learning_rate", 1e-5)),
        )
    elif scheduler_name == "none":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    else:
        raise ValueError("experiment.scheduler must be 'cosine' or 'none'")
    amp_enabled = bool(run_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(args.config.read_text(encoding="utf-8"), encoding="utf-8")
    epochs = int(run_cfg.get("epochs", 25))
    max_train_batches = int(run_cfg.get("max_train_batches", 0))
    max_val_batches = int(run_cfg.get("max_val_batches", 0))
    metrics = []
    best_metric = float("inf")
    best_state = None
    start_epoch = 0
    last_state_path = output_dir / str(artifact_cfg.get("last_state", "last_state.pth"))
    best_checkpoint_path = output_dir / str(artifact_cfg.get("checkpoint", "best_absolute.pth"))
    if bool(run_cfg.get("resume", True)) and last_state_path.is_file():
        state = torch.load(last_state_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        scaler.load_state_dict(state.get("scaler_state_dict", scaler.state_dict()))
        start_epoch = int(state["epoch"])
        best_metric = float(state.get("best_metric", float("inf")))
        metrics = list(state.get("metrics", []))
        print(f"[RESUME] epoch={start_epoch} best_absolute={best_metric:.4f}", flush=True)
    for epoch in range(start_epoch, epochs):
        model.train()
        running = 0.0
        batches = 0
        running_pose = 0.0
        running_root = 0.0
        optimizer_updated = False
        for batch_index, (windows, raw_windows, calibration, targets) in enumerate(train_loader, start=1):
            windows = windows.to(device)
            raw_windows = raw_windows.to(device)
            calibration = calibration.to(device)
            targets = targets.to(device)
            intervention, code = apply_csi_intervention(windows)
            intervention_raw = raw_windows * 1.08 + (0.02 / 60.0)
            intervention_calibration = transform_calibration(calibration, code)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                loss, log = model.training_loss(
                    windows, targets, calibration=calibration, raw_windows=raw_windows,
                    intervention_windows=intervention,
                    intervention_raw_windows=intervention_raw,
                    intervention_code=code,
                    intervention_calibration=intervention_calibration,
                )
            scaler.scale(loss).backward()
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # GradScaler can skip an optimizer step after an inf/nan gradient.
            # Do not advance the LR schedule for a skipped update.
            optimizer_updated = optimizer_updated or (not amp_enabled or scaler.get_scale() >= scale_before)
            running += float(loss.detach())
            running_pose += float(log.get("pose_loss", 0.0))
            running_root += float(log.get("root_loss", 0.0))
            batches += 1
            if max_train_batches and batch_index >= max_train_batches:
                break
        if optimizer_updated:
            scheduler.step()
        val_metrics = evaluate(model, val_loader, device, max_val_batches, amp_enabled)
        row = {"epoch": epoch + 1, "train_loss": running / max(1, batches),
               "train_pose_loss": running_pose / max(1, batches),
               "train_root_loss": running_root / max(1, batches),
               "learning_rate": optimizer.param_groups[0]["lr"], **val_metrics}
        metrics.append(row)
        if row["absolute_mpjpe_mm"] < best_metric:
            best_metric = row["absolute_mpjpe_mm"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save({
                "method": "IC-DTS",
                "status": "best_select_checkpoint",
                "epoch": epoch + 1,
                "config": config,
                "manifest": str(manifest_path),
                "label_manifest": str(label_manifest_path),
                "model_state_dict": best_state,
                "select_metrics": row,
            }, best_checkpoint_path)
        torch.save({
            "method": "IC-DTS",
            "status": "last_state",
            "epoch": epoch + 1,
            "config": config,
            "manifest": str(manifest_path),
            "label_manifest": str(label_manifest_path),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_metric": best_metric,
            "metrics": metrics,
        }, last_state_path)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    if best_checkpoint_path.is_file():
        best = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(best["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, int(run_cfg.get("max_test_batches", 0)), amp_enabled)
    final_report = {
        "method": "IC-DTS",
        "protocol": manifest.protocol,
        "split": manifest.split,
        "scope": manifest.scope,
        "seed": manifest.seed,
        "label_budget": run_cfg.get("label_budget"),
        "manifest_fingerprint": manifest.fingerprint(),
        "best_select_absolute_mpjpe_mm": best_metric,
        "select": min(metrics, key=lambda row: row["absolute_mpjpe_mm"]) if metrics else {},
        "test": test_metrics,
        "checkpoint": str(best_checkpoint_path),
        "test_used_for_selection": False,
    }
    (output_dir / str(artifact_cfg.get("final_report", "final_report.json"))).write_text(
        json.dumps(final_report, indent=2), encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

