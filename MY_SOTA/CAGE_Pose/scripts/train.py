from __future__ import annotations

import argparse
import json
import random
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mmfi_wifi.data import _load_csi_frame, _load_ground_truth
from mmfi_wifi.data_manifest import DataManifest
from mmfi_wifi.sequence_keys import SequenceKey
from MY_SOTA.CAGE_Pose.src.model import CAGEPoseModel

FRAME_COUNT = 297
CSI_SHAPE = (3, 114, 10)


def read_keys(path: Path) -> frozenset[SequenceKey]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return frozenset(SequenceKey(*item) for item in payload["selected_keys"])


def raw_csi(path: Path) -> np.ndarray:
    npy = path.with_suffix(".npy")
    if npy.is_file():
        x = np.load(npy)
    else:
        import scipy.io as scio
        x = scio.loadmat(path)["CSIamp"]
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    fill = float(x[finite].mean()) if finite.any() else 0.0
    return np.nan_to_num(x, nan=fill, posinf=fill, neginf=fill)


def calibration(x: np.ndarray) -> torch.Tensor:
    mean = x.mean(axis=(1, 2)); std = x.std(axis=(1, 2))
    minimum = x.min(axis=(1, 2)); maximum = x.max(axis=(1, 2))
    return torch.as_tensor(np.concatenate([mean / 60.0, std / 20.0,
                                           minimum / 60.0, maximum / 60.0]), dtype=torch.float32)


@lru_cache(maxsize=200000)
def cached_raw(path: str) -> torch.Tensor:
    return torch.as_tensor(raw_csi(Path(path)) / 60.0, dtype=torch.float32)


@lru_cache(maxsize=200000)
def cached_calibration(path: str) -> torch.Tensor:
    return calibration(raw_csi(Path(path)))


class SingleFramePoseDataset(Dataset):
    def __init__(self, dataset_root: Path, keys, env_to_id: dict[str, int]):
        self.dataset_root = dataset_root
        self.items = []
        for key in sorted(keys):
            seq = self.dataset_root / key.scene / key.subject / key.action
            gt = np.load(seq / "ground_truth.npy", mmap_mode="r")
            if gt.ndim != 3 or tuple(gt.shape[1:]) != (17, 3):
                raise ValueError(f"invalid ground truth shape for {key}: {gt.shape}")
            for frame in range(len(gt)):
                frame_path = seq / "wifi-csi" / f"frame{frame + 1:03d}.mat"
                if not frame_path.is_file() and not frame_path.with_suffix(".npy").is_file():
                    raise FileNotFoundError(frame_path)
                self.items.append((key, frame))
        self.env_to_id = env_to_id

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        key, frame = self.items[index]
        seq = self.dataset_root / key.scene / key.subject / key.action
        frame_path = seq / "wifi-csi" / f"frame{frame + 1:03d}.mat"
        norm = torch.as_tensor(_load_csi_frame(str(frame_path)), dtype=torch.float32)
        raw = cached_raw(str(frame_path))
        cal = cached_calibration(str(frame_path))
        gt = torch.as_tensor(_load_ground_truth(str(seq / "ground_truth.npy"))[frame].copy(), dtype=torch.float32)
        return norm, raw, cal, gt, torch.tensor(self.env_to_id[key.scene], dtype=torch.long)


def collate(batch):
    a, b, c, d, e = zip(*batch)
    return torch.stack(a), torch.stack(b), torch.stack(c), torch.stack(d), torch.stack(e)


def augment_csi(x: torch.Tensor) -> torch.Tensor:
    gain = 1.0 + torch.empty(x.shape[0], 1, 1, 1, device=x.device).uniform_(-0.08, 0.08)
    offset = torch.empty(x.shape[0], 1, 1, 1, device=x.device).uniform_(-0.03, 0.03)
    keep = torch.rand(x.shape[0], 1, x.shape[2], 1, device=x.device) > 0.05
    return (x * gain + offset) * keep


def mpjpe_mm(pred, target):
    return (pred - target).norm(dim=-1).mean() * 1000.0


def pa_mpjpe_mm(pred, target):
    pm, tm = pred.mean(-2, keepdim=True), target.mean(-2, keepdim=True)
    p, t = pred - pm, target - tm
    u, s, vh = torch.linalg.svd(p.transpose(-2, -1) @ t)
    det = torch.det(u @ vh).sign().reshape(-1, 1, 1)
    c = torch.eye(3, device=pred.device).expand(pred.shape[0], -1, -1).clone()
    c[:, -1, -1] = det[:, 0, 0]
    r = u @ c @ vh
    trace = (s * torch.cat([torch.ones_like(s[..., :2]), det.squeeze(-1)], -1)).sum(-1, keepdim=True).unsqueeze(-1)
    scale = trace / p.square().sum((-2, -1), keepdim=True).clamp_min(1e-8)
    aligned = scale * (p @ r) + tm
    return (aligned - target).norm(dim=-1).mean() * 1000.0


@torch.no_grad()
def evaluate(model, loader, device, max_batches=0):
    model.eval(); abs_sum = pa_sum = count = 0
    for i, (x, raw, cal, target, env) in enumerate(loader, 1):
        out = model(x.to(device), cal.to(device), raw_csi=raw.to(device)); target = target.to(device)
        abs_sum += float(mpjpe_mm(out.pose, target)) * len(target)
        pa_sum += float(pa_mpjpe_mm(out.pose, target)) * len(target)
        count += len(target)
        if max_batches and i >= max_batches: break
    return {"absolute_mpjpe_mm": abs_sum / max(1, count), "pa_mpjpe_mm": pa_sum / max(1, count)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")); exp = cfg["experiment"]
    random.seed(int(exp.get("seed", 42))); np.random.seed(int(exp.get("seed", 42))); torch.manual_seed(int(exp.get("seed", 42)))
    dataset_root = Path(exp["dataset_root"]); manifest_path = Path(exp["manifest"]); label_path = Path(exp["label_manifest"])
    if not dataset_root.is_absolute(): dataset_root = (ROOT / dataset_root).resolve()
    if not manifest_path.is_absolute(): manifest_path = (ROOT / manifest_path).resolve()
    if not label_path.is_absolute(): label_path = (ROOT / label_path).resolve()
    manifest = DataManifest.read_json(manifest_path); train_keys = read_keys(label_path)
    if not train_keys.issubset(set(manifest.internal_train_keys)):
        raise ValueError("label manifest contains keys outside manifest.internal_train_keys")
    for field in ("protocol", "split", "scope", "seed"):
        if str(exp.get(field)) != str(getattr(manifest, field)):
            raise ValueError(f"config/manifest mismatch for {field}")
    all_scenes = {k.scene for k in (set(manifest.internal_train_keys) | set(manifest.select_keys) | set(manifest.test_keys))}
    env_to_id = {scene: i for i, scene in enumerate(sorted(all_scenes))}
    train_ds = SingleFramePoseDataset(dataset_root, train_keys, env_to_id)
    select_ds = SingleFramePoseDataset(dataset_root, manifest.select_keys, env_to_id)
    test_ds = SingleFramePoseDataset(dataset_root, manifest.test_keys, env_to_id)
    bs = int(exp.get("batch_size", 64)); workers = int(exp.get("num_workers", 0))
    train_loader = DataLoader(train_ds, bs, shuffle=True, num_workers=workers, collate_fn=collate)
    select_loader = DataLoader(select_ds, bs, shuffle=False, num_workers=workers, collate_fn=collate)
    test_loader = DataLoader(test_ds, bs, shuffle=False, num_workers=workers, collate_fn=collate)
    device = torch.device(exp.get("device", "cuda") if exp.get("device", "cuda") == "cpu" or torch.cuda.is_available() else "cpu")
    model = CAGEPoseModel.from_config(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=float(exp.get("learning_rate", 3e-4)), weight_decay=float(exp.get("weight_decay", 1e-4)))
    epochs = int(exp.get("epochs", 25))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, epochs), eta_min=float(exp.get("min_learning_rate", 1e-5)))
    amp = bool(exp.get("amp", True)) and device.type == "cuda"; scaler = torch.amp.GradScaler("cuda", enabled=amp)
    out_dir = Path(cfg["artifacts"]["output_root"]); out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir; out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(args.config.read_text(encoding="utf-8"), encoding="utf-8")
    best = float("inf"); rows = []; start_epoch = 0
    last_state_path = out_dir / cfg["artifacts"].get("last_state", "last_state.pth")
    best_path = out_dir / cfg["artifacts"].get("checkpoint", "best_absolute.pth")
    if bool(exp.get("resume", True)) and last_state_path.is_file():
        state = torch.load(last_state_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optim.load_state_dict(state["optimizer_state_dict"])
        sched.load_state_dict(state["scheduler_state_dict"])
        scaler.load_state_dict(state.get("scaler_state_dict", scaler.state_dict()))
        start_epoch = int(state["epoch"]); best = float(state.get("best_metric", best)); rows = list(state.get("metrics", []))
        print(f"[CAGE][RESUME] epoch={start_epoch} best_select_abs={best:.2f}mm", flush=True)
    print(f"[CAGE][START] device={device} amp={amp} train={len(train_ds)} select={len(select_ds)} test={len(test_ds)} epochs={epochs} batch={bs}", flush=True)
    for epoch in range(start_epoch, epochs):
        model.train(); total = count = 0; parts = {"pose_loss": 0., "root_loss": 0., "relative_loss": 0., "bone_loss": 0., "root_nll": 0., "domain_loss": 0., "consistency_loss": 0.}
        optimizer_updated = False
        epoch_start = time.perf_counter()
        for i, (x, raw, cal, target, env) in enumerate(train_loader, 1):
            x, raw, cal, target, env = x.to(device), raw.to(device), cal.to(device), target.to(device), env.to(device)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                out = model(x, cal, raw_csi=raw); aug = model(augment_csi(x), cal, raw_csi=raw)
                loss, logs = model.loss(out, target, env, aug)
            scaler.scale(loss).backward(); scale_before = scaler.get_scale()
            scaler.step(optim); scaler.update()
            optimizer_updated = optimizer_updated or (not amp or scaler.get_scale() >= scale_before)
            total += float(loss.detach()); count += 1
            for k in parts: parts[k] += logs[k]
            interval = int(exp.get("log_interval", 50))
            if interval > 0 and (i == 1 or i % interval == 0):
                print(f"[CAGE][epoch {epoch + 1}/{epochs}][batch {i}/{len(train_loader)}] loss={float(loss.detach()):.4f} root={logs['root_loss']:.4f} rel={logs['relative_loss']:.4f}", flush=True)
            if exp.get("max_train_batches", 0) and i >= int(exp["max_train_batches"]): break
        if optimizer_updated:
            sched.step()
        val = evaluate(model, select_loader, device, int(exp.get("max_select_batches", 0)))
        row = {"epoch": epoch + 1, "train_loss": total / max(1, count), "learning_rate": optim.param_groups[0]["lr"], **{f"train_{k}": v / max(1, count) for k, v in parts.items()}, **val}; rows.append(row)
        print(f"[CAGE][EPOCH {epoch + 1}/{epochs}] train={row['train_loss']:.4f} pose={row['train_pose_loss']:.4f} root={row['train_root_loss']:.4f} rel={row['train_relative_loss']:.4f} bone={row['train_bone_loss']:.4f} nll={row['train_root_nll']:.4f} domain={row['train_domain_loss']:.4f} cons={row['train_consistency_loss']:.4f} abs={row['absolute_mpjpe_mm']:.2f}mm pa={row['pa_mpjpe_mm']:.2f}mm lr={row['learning_rate']:.3g} time={time.perf_counter() - epoch_start:.1f}s", flush=True)
        if val["absolute_mpjpe_mm"] < best:
            best = val["absolute_mpjpe_mm"]
            torch.save({"method": "CAGE-Pose", "epoch": epoch + 1, "model_state_dict": model.state_dict(), "select_metrics": row, "config": cfg}, best_path)
        torch.save({"method": "CAGE-Pose", "epoch": epoch + 1, "model_state_dict": model.state_dict(), "optimizer_state_dict": optim.state_dict(), "scheduler_state_dict": sched.state_dict(), "scaler_state_dict": scaler.state_dict(), "best_metric": best, "metrics": rows, "config": cfg}, last_state_path)
    best_state = torch.load(best_path, map_location=device, weights_only=False); model.load_state_dict(best_state["model_state_dict"])
    test = evaluate(model, test_loader, device, int(exp.get("max_test_batches", 0)))
    report = {"method": "CAGE-Pose", "protocol": manifest.protocol, "split": manifest.split, "scope": manifest.scope, "seed": manifest.seed, "label_budget": exp.get("label_budget"), "manifest_fingerprint": manifest.fingerprint(), "select": min(rows, key=lambda r: r["absolute_mpjpe_mm"]), "test": test, "test_used_for_selection": False}
    (out_dir / "metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8"); (out_dir / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[CAGE][DONE] best_select_abs={report['select']['absolute_mpjpe_mm']:.2f}mm test_abs={test['absolute_mpjpe_mm']:.2f}mm test_pa={test['pa_mpjpe_mm']:.2f}mm output={out_dir}", flush=True)


if __name__ == "__main__": main()
