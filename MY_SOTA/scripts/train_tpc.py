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
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mmfi_wifi.data import _load_csi_frame, _load_ground_truth
from mmfi_wifi.data_manifest import DataManifest
from mmfi_wifi.sequence_keys import SequenceKey
from MY_SOTA.src.phase_pose import CSI_SHAPE, TPCFormer, load_csi_phase_frame

FRAME_COUNT = 297


def _load_raw_amplitude(frame_path: Path) -> np.ndarray:
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


@lru_cache(maxsize=200000)
def _cached_raw_amplitude(path: str) -> torch.Tensor:
    return torch.as_tensor(_load_raw_amplitude(Path(path)), dtype=torch.float32)


@lru_cache(maxsize=200000)
def _cached_phase(path: str) -> torch.Tensor:
    return torch.as_tensor(load_csi_phase_frame(path), dtype=torch.float32)


def _read_keys(path: Path) -> frozenset[SequenceKey]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return frozenset(SequenceKey(*item) for item in payload["selected_keys"])


class FramePoseDataset(Dataset):
    def __init__(self, dataset_root: Path, keys: frozenset[SequenceKey]):
        self.dataset_root = dataset_root
        self.items = [(key, frame) for key in sorted(keys) for frame in range(FRAME_COUNT)]
        if not self.items:
            raise ValueError("no frames available")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        key, frame = self.items[index]
        sequence = self.dataset_root / key.scene / key.subject / key.action
        path = sequence / "wifi-csi" / f"frame{frame + 1:03d}.mat"
        amplitude = torch.as_tensor(_load_csi_frame(str(path)), dtype=torch.float32)
        raw_amplitude = _cached_raw_amplitude(str(path)) / 60.0
        phase = _cached_phase(str(path))
        target = torch.as_tensor(_load_ground_truth(str(sequence / "ground_truth.npy"))[frame].copy(), dtype=torch.float32)
        return amplitude, raw_amplitude, phase, target


def _mpjpe(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).norm(dim=-1).mean() * 1000.0


def _pa_mpjpe(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred0 = pred - pred.mean(dim=1, keepdim=True)
    target0 = target - target.mean(dim=1, keepdim=True)
    covariance = pred0.transpose(-2, -1) @ target0
    u, singular, vh = torch.linalg.svd(covariance)
    det = torch.det(u @ vh).sign().reshape(-1, 1, 1)
    correction = torch.eye(3, device=pred.device).expand(pred.shape[0], -1, -1).clone()
    correction[:, -1, -1] = det[:, 0, 0]
    rotation = u @ correction @ vh
    trace = (singular * torch.cat((torch.ones_like(singular[..., :2]), det.squeeze(-1)), dim=-1)).sum(dim=-1, keepdim=True).unsqueeze(-1)
    scale = trace / pred0.square().sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    aligned = scale * (pred0 @ rotation) + target.mean(dim=1, keepdim=True)
    return (aligned - target).norm(dim=-1).mean() * 1000.0


def _evaluate(model: TPCFormer, loader: DataLoader, device: torch.device, max_batches: int = 0) -> dict[str, float]:
    model.eval()
    sums = {"absolute_mpjpe_mm": 0.0, "relative_mpjpe_mm": 0.0, "pa_mpjpe_mm": 0.0, "root_mae_mm": 0.0}
    count = 0
    with torch.no_grad():
        for index, (amp, raw, phase, target) in enumerate(loader, start=1):
            amp, raw, phase, target = amp.to(device), raw.to(device), phase.to(device), target.to(device)
            out = model(amp, phase, raw_amplitude=raw)
            relative_target = target - target[:, :1]
            batch = amp.shape[0]
            sums["absolute_mpjpe_mm"] += float(_mpjpe(out.absolute_pose, target)) * batch
            sums["relative_mpjpe_mm"] += float(_mpjpe(out.relative_pose, relative_target)) * batch
            sums["pa_mpjpe_mm"] += float(_pa_mpjpe(out.absolute_pose, target)) * batch
            sums["root_mae_mm"] += float((out.root - target[:, 0]).norm(dim=-1).mean() * 1000.0) * batch
            count += batch
            if max_batches and index >= max_batches:
                break
    return {key: value / max(1, count) for key, value in sums.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "MY_SOTA/configs/tpc_p3.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    artifacts = config["artifacts"]
    seed = int(experiment.get("seed", 42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dataset_root = Path(experiment["dataset_root"]); manifest_path = Path(experiment["manifest"]); label_path = Path(experiment["label_manifest"])
    output = Path(artifacts["output_root"])
    if not dataset_root.is_absolute(): dataset_root = (ROOT / dataset_root).resolve()
    if not manifest_path.is_absolute(): manifest_path = (ROOT / manifest_path).resolve()
    if not label_path.is_absolute(): label_path = (ROOT / label_path).resolve()
    if not output.is_absolute(): output = (ROOT / output).resolve()
    manifest = DataManifest.read_json(manifest_path)
    train_keys = _read_keys(label_path)
    if not train_keys.issubset(set(manifest.internal_train_keys)):
        raise ValueError("label manifest contains keys outside internal_train_keys")
    train_loader = DataLoader(FramePoseDataset(dataset_root, train_keys), batch_size=int(experiment.get("batch_size", 64)), shuffle=True, num_workers=0)
    select_loader = DataLoader(FramePoseDataset(dataset_root, manifest.select_keys), batch_size=int(experiment.get("batch_size", 64)), shuffle=False, num_workers=0)
    test_loader = DataLoader(FramePoseDataset(dataset_root, manifest.test_keys), batch_size=int(experiment.get("batch_size", 64)), shuffle=False, num_workers=0)
    device_name = str(experiment.get("device", "cuda")); device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    model = TPCFormer(hidden_dim=int(config.get("model", {}).get("hidden_dim", 128)), phase_dropout=float(config.get("model", {}).get("phase_dropout", 0.1))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(experiment.get("learning_rate", 3e-4)), weight_decay=float(experiment.get("weight_decay", 1e-4)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(experiment.get("epochs", 25))))
    output.mkdir(parents=True, exist_ok=True); (output / "config.yaml").write_text(args.config.read_text(encoding="utf-8"), encoding="utf-8")
    best = float("inf"); metrics = []
    for epoch in range(int(experiment.get("epochs", 25))):
        model.train(); total = 0.0; batches = 0
        for batch_index, (amp, raw, phase, target) in enumerate(train_loader, start=1):
            amp, raw, phase, target = amp.to(device), raw.to(device), phase.to(device), target.to(device)
            out = model(amp, phase, raw_amplitude=raw)
            relative_target = target - target[:, :1]
            relative_loss = F.smooth_l1_loss(out.relative_pose, relative_target)
            root_loss = float(config.get("loss", {}).get("root_weight", 4.0)) * F.smooth_l1_loss(out.root, target[:, 0])
            # Keep the absolute objective from sending root-coordinate gradients
            # back through the body branch during the first factorized run.
            absolute_loss = F.smooth_l1_loss(out.relative_pose.detach() + out.root.unsqueeze(1), target)
            loss = relative_loss + root_loss + absolute_loss
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            total += float(loss.detach()); batches += 1
            if experiment.get("max_train_batches", 0) and batch_index >= int(experiment["max_train_batches"]): break
        scheduler.step(); select = _evaluate(model, select_loader, device, int(experiment.get("max_select_batches", 0)))
        row = {"epoch": epoch + 1, "train_loss": total / max(1, batches), **select}; metrics.append(row); print(json.dumps(row), flush=True)
        if select["absolute_mpjpe_mm"] < best:
            best = select["absolute_mpjpe_mm"]; torch.save({"method": "TPC-Former", "model_state_dict": model.state_dict(), "select": select, "config": config, "manifest_fingerprint": manifest.fingerprint()}, output / "best_absolute.pth")
    state = torch.load(output / "best_absolute.pth", map_location=device, weights_only=False); model.load_state_dict(state["model_state_dict"])
    test = _evaluate(model, test_loader, device, int(experiment.get("max_test_batches", 0)))
    (output / "final_report.json").write_text(json.dumps({"method": "TPC-Former", "protocol": manifest.protocol, "split": manifest.split, "scope": manifest.scope, "seed": seed, "select": min(metrics, key=lambda x: x["absolute_mpjpe_mm"]), "test": test, "test_used_for_selection": False}, indent=2), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
