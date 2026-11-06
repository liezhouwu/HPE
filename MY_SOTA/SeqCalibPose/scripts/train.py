from __future__ import annotations

import argparse
import json
import random
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mmfi_wifi.data import _load_csi_frame, _load_ground_truth
from mmfi_wifi.data_manifest import DataManifest
from mmfi_wifi.sequence_keys import SequenceKey
from MY_SOTA.TPC_Former.src.phase_pose import load_csi_phase_frame
from MY_SOTA.SeqCalibPose.src.model import SeqCalibPose, PARENTS

FRAME_COUNT = 297


def read_keys(path: Path) -> frozenset[SequenceKey]:
    return frozenset(SequenceKey(*x) for x in json.loads(path.read_text(encoding="utf-8-sig"))["selected_keys"])


def finite_raw(path: Path) -> np.ndarray:
    npy = path.with_suffix(".npy")
    if npy.is_file():
        x = np.load(npy)
    else:
        import scipy.io as scio
        x = scio.loadmat(path)["CSIamp"]
    x = np.asarray(x, dtype=np.float32)
    good = np.isfinite(x)
    fill = float(x[good].mean()) if good.any() else 0.0
    return np.nan_to_num(x, nan=fill, posinf=fill, neginf=fill)


@lru_cache(maxsize=8192)
def cached_raw(path: str) -> torch.Tensor:
    return torch.as_tensor(finite_raw(Path(path)) / 60.0, dtype=torch.float32)


@lru_cache(maxsize=8192)
def cached_phase(path: str) -> torch.Tensor:
    return torch.as_tensor(load_csi_phase_frame(path), dtype=torch.float32)


class FrameDataset(Dataset):
    def __init__(self, root: Path, keys: frozenset[SequenceKey]):
        self.root = root
        self.items = [(key, frame) for key in sorted(keys) for frame in range(FRAME_COUNT)]
        if not self.items:
            raise ValueError("empty dataset")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        key, frame = self.items[index]
        seq = self.root / key.scene / key.subject / key.action
        path = seq / "wifi-csi" / f"frame{frame + 1:03d}.mat"
        amp = torch.as_tensor(_load_csi_frame(str(path)), dtype=torch.float32)
        raw = cached_raw(str(path))
        phase = cached_phase(str(path))
        target = torch.as_tensor(_load_ground_truth(str(seq / "ground_truth.npy"))[frame].copy(), dtype=torch.float32)
        return amp, raw, phase, target


def mpjpe_mm(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).norm(dim=-1).mean() * 1000.0


def pa_mpjpe_mm(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pm, tm = pred.mean(1, keepdim=True), target.mean(1, keepdim=True)
    p, t = pred - pm, target - tm
    u, s, vh = torch.linalg.svd(p.transpose(-2, -1) @ t)
    det = torch.det(u @ vh).sign().reshape(-1, 1, 1)
    correction = torch.eye(3, device=pred.device).expand(pred.shape[0], -1, -1).clone()
    correction[:, -1, -1] = det[:, 0, 0]
    rotation = u @ correction @ vh
    trace = (s * torch.cat((torch.ones_like(s[..., :2]), det.squeeze(-1)), dim=-1)).sum(-1, keepdim=True).unsqueeze(-1)
    scale = trace / p.square().sum((-2, -1), keepdim=True).clamp_min(1e-8)
    aligned = scale * (p @ rotation) + tm
    return (aligned - target).norm(dim=-1).mean() * 1000.0


def bone_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    vals = []
    for j, p in enumerate(PARENTS):
        if p >= 0:
            vals.append(F.smooth_l1_loss((pred[:, j] - pred[:, p]).norm(dim=-1),
                                         (target[:, j] - target[:, p]).norm(dim=-1)))
    return torch.stack(vals).mean()


@torch.no_grad()
def evaluate(model, loader, device, max_batches=0):
    model.eval(); sums = {"absolute_mpjpe_mm": 0., "relative_mpjpe_mm": 0., "pa_mpjpe_mm": 0., "root_mae_mm": 0.}; n = 0
    for i, (amp, raw, phase, target) in enumerate(loader, 1):
        amp, raw, phase, target = amp.to(device), raw.to(device), phase.to(device), target.to(device)
        out = model(amp, raw, phase); rel_target = target - target[:, :1]; b = len(target)
        sums["absolute_mpjpe_mm"] += float(mpjpe_mm(out.pose, target)) * b
        sums["relative_mpjpe_mm"] += float(mpjpe_mm(out.relative, rel_target)) * b
        sums["pa_mpjpe_mm"] += float(pa_mpjpe_mm(out.pose, target)) * b
        sums["root_mae_mm"] += float((out.root - target[:, 0]).norm(dim=-1).mean() * 1000.) * b
        n += b
        if max_batches and i >= max_batches: break
    return {k: v / max(1, n) for k, v in sums.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="run one isolated train/select/test batch")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")); exp = cfg["experiment"]
    seed = int(exp.get("seed", 42)); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dataset_root = Path(exp["dataset_root"]); manifest_path = Path(exp["manifest"]); label_path = Path(exp["label_manifest"])
    if not dataset_root.is_absolute(): dataset_root = (ROOT / dataset_root).resolve()
    if not manifest_path.is_absolute(): manifest_path = (ROOT / manifest_path).resolve()
    if not label_path.is_absolute(): label_path = (ROOT / label_path).resolve()
    manifest = DataManifest.read_json(manifest_path); train_keys = read_keys(label_path)
    if not train_keys.issubset(set(manifest.internal_train_keys)): raise ValueError("labels outside train manifest")
    for field in ("protocol", "split", "scope", "seed"):
        if str(exp.get(field)) != str(getattr(manifest, field)): raise ValueError(f"manifest mismatch: {field}")
    train_ds, select_ds, test_ds = FrameDataset(dataset_root, train_keys), FrameDataset(dataset_root, manifest.select_keys), FrameDataset(dataset_root, manifest.test_keys)
    bs = int(exp.get("batch_size", 64)); workers = int(exp.get("num_workers", 4))
    loader_options = {"num_workers": workers, "pin_memory": torch.cuda.is_available()}
    if workers > 0:
        loader_options.update(prefetch_factor=int(exp.get("prefetch_factor", 2)), persistent_workers=True)
    loaders = [DataLoader(ds, bs, shuffle=i == 0, **loader_options) for i, ds in enumerate((train_ds, select_ds, test_ds))]
    device_name = str(exp.get("device", "cuda")); device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    model = SeqCalibPose(dim=int(cfg.get("model", {}).get("dim", 160)), root_residual_limit=float(cfg.get("model", {}).get("root_residual_limit", .45))).to(device)
    # Source-only coordinate prior; no E04 labels are read before final evaluation.
    roots = [np.load(dataset_root / k.scene / k.subject / k.action / "ground_truth.npy", mmap_mode="r")[:, 0] for k in sorted(train_keys)]
    model.root_prior.data.copy_(torch.as_tensor(np.concatenate(roots).mean(0), dtype=torch.float32, device=device))
    optim = torch.optim.AdamW(model.parameters(), lr=float(exp.get("learning_rate", 3e-4)), weight_decay=float(exp.get("weight_decay", 1e-4)))
    epochs = int(exp.get("epochs", 25)); sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max(1, epochs), eta_min=float(exp.get("min_learning_rate", 1e-5)))
    out_dir = Path(cfg["artifacts"]["output_root"]); out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir; out_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        out_dir = out_dir.parent / "_smoke"
        out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(args.config.read_text(encoding="utf-8"), encoding="utf-8")
    best = float("inf"); rows = []
    for epoch in range(1 if args.smoke else epochs):
        model.train(); total = 0.; batches = 0
        for batch_index, (amp, raw, phase, target) in enumerate(loaders[0], 1):
            amp, raw, phase, target = amp.to(device), raw.to(device), phase.to(device), target.to(device)
            out = model(amp, raw, phase); rel_target = target - target[:, :1]
            rel = F.smooth_l1_loss(out.relative, rel_target); root = F.smooth_l1_loss(out.root, target[:, 0]); absolute = F.smooth_l1_loss(out.pose, target); bones = bone_loss(out.pose, target)
            # Root receives the strongest signal; body receives relative and bone geometry.
            loss = float(cfg.get("loss", {}).get("relative_weight", 2.0)) * rel + float(cfg.get("loss", {}).get("root_weight", 8.0)) * root + float(cfg.get("loss", {}).get("absolute_weight", 1.0)) * absolute + float(cfg.get("loss", {}).get("bone_weight", .25)) * bones
            optim.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optim.step()
            total += float(loss.detach()); batches += 1
            if args.smoke or (exp.get("max_train_batches", 0) and batch_index >= int(exp["max_train_batches"])): break
        sched.step(); val = evaluate(model, loaders[1], device, 1 if args.smoke else int(exp.get("max_select_batches", 0)))
        row = {"epoch": epoch + 1, "train_loss": total / max(1, batches), **val}; rows.append(row); print(json.dumps(row), flush=True)
        if val["absolute_mpjpe_mm"] < best:
            best = val["absolute_mpjpe_mm"]
            torch.save({"method": "SeqCalibPose", "epoch": epoch + 1, "model_state_dict": model.state_dict(), "select": row, "root_prior": model.root_prior.detach().cpu().tolist(), "manifest_fingerprint": manifest.fingerprint(), "config": cfg}, out_dir / "best_absolute.pth")
    state = torch.load(out_dir / "best_absolute.pth", map_location=device, weights_only=False); model.load_state_dict(state["model_state_dict"]); test = evaluate(model, loaders[2], device, 1 if args.smoke else int(exp.get("max_test_batches", 0)))
    report = {"method": "SeqCalibPose", "protocol": manifest.protocol, "split": manifest.split, "scope": manifest.scope, "seed": seed, "label_budget": exp.get("label_budget"), "manifest_fingerprint": manifest.fingerprint(), "root_prior": state["root_prior"], "select": min(rows, key=lambda x: x["absolute_mpjpe_mm"]), "test": test, "test_used_for_selection": False, "test_independence": "exploratory: E04 label statistics and a smoke batch were inspected during method design", "smoke_run": bool(args.smoke)}
    (out_dir / "metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8"); (out_dir / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__": main()
