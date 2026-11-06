from __future__ import annotations

import argparse
import hashlib
import json
import os
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
from MY_SOTA.WAC_CSI.src.model import CSI_SHAPE, WACPoseModel


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
    x = np.nan_to_num(x, nan=fill, posinf=fill, neginf=fill)
    if tuple(x.shape) != CSI_SHAPE:
        raise ValueError(f"expected raw CSI {CSI_SHAPE}, got {x.shape} at {path}")
    return x


@lru_cache(maxsize=200000)
def cached_raw(path: str) -> torch.Tensor:
    # Keep absolute amplitude information in a stable numerical range.
    return torch.as_tensor(raw_csi(Path(path)) / 60.0, dtype=torch.float32)


def frame_paths(seq: Path) -> list[Path]:
    mat = sorted((seq / "wifi-csi").glob("frame*.mat"))
    if mat:
        return mat
    return sorted((seq / "wifi-csi").glob("frame*.npy"))


def radio_descriptor(x: np.ndarray) -> np.ndarray:
    y = np.log1p(np.abs(x / 60.0).astype(np.float32))
    return np.concatenate([
        y.mean(axis=(1, 2)), y.std(axis=(1, 2)),
        y.min(axis=(1, 2)), y.max(axis=(1, 2)),
    ], axis=0).astype(np.float32)


class SingleFrameDataset(Dataset):
    def __init__(self, dataset_root: Path, keys, sequence_to_id: dict[str, int],
                 use_gt_length: bool) -> None:
        self.dataset_root = dataset_root
        self.items: list[tuple[SequenceKey, int, Path]] = []
        for key in sorted(keys):
            seq = dataset_root / key.scene / key.subject / key.action
            paths = frame_paths(seq)
            if not paths:
                raise FileNotFoundError(f"no CSI frames found in {seq / 'wifi-csi'}")
            if use_gt_length:
                gt = np.load(seq / "ground_truth.npy", mmap_mode="r")
                if gt.ndim != 3 or tuple(gt.shape[1:]) != (17, 3):
                    raise ValueError(f"invalid ground truth shape for {key}: {gt.shape}")
                if len(paths) != len(gt):
                    raise ValueError(f"CSI/GT length mismatch for {key}: {len(paths)} vs {len(gt)}")
            for frame, path in enumerate(paths):
                self.items.append((key, frame, path))
        self.sequence_to_id = sequence_to_id

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index):
        key, frame, path = self.items[index]
        seq = self.dataset_root / key.scene / key.subject / key.action
        normalized = torch.as_tensor(_load_csi_frame(str(path)), dtype=torch.float32)
        raw = cached_raw(str(path))
        gt = torch.as_tensor(_load_ground_truth(str(seq / "ground_truth.npy"))[frame].copy(), dtype=torch.float32)
        sequence_id = self.sequence_to_id[str(key)]
        return normalized, raw, gt, torch.tensor(sequence_id, dtype=torch.long)


def collate(batch):
    a, b, c, d = zip(*batch)
    return torch.stack(a), torch.stack(b), torch.stack(c), torch.stack(d)


def memory_spec(dataset_root: Path, train_keys, sequence_to_id: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    roots = []
    descriptors = []
    sequence_ids = []
    for key in sorted(train_keys):
        seq = dataset_root / key.scene / key.subject / key.action
        gt = np.load(seq / "ground_truth.npy", mmap_mode="r")
        roots.append(np.asarray(gt[:, 0], dtype=np.float32))
        paths = frame_paths(seq)
        descriptors.append(np.stack([radio_descriptor(raw_csi(path)) for path in paths], axis=0))
        sequence_ids.append(np.full(len(paths), sequence_to_id[str(key)], dtype=np.int64))
    if not roots:
        raise ValueError("cannot build memory atlas from empty source training keys")
    roots = np.concatenate(roots, axis=0); descriptors = np.concatenate(descriptors, axis=0)
    sequence_ids = np.concatenate(sequence_ids, axis=0)
    if len(roots) < 2:
        raise ValueError("source memory atlas needs at least two frames")
    spec = {
        "source_only": True, "memory_type": "sample_level", "memory_count": int(len(roots)),
        "descriptor_dim": int(descriptors.shape[1]), "sequence_count": int(len(set(sequence_ids.tolist()))),
        "sequence_fingerprint": fingerprint(sorted(map(str, train_keys))),
    }
    return torch.as_tensor(roots), torch.as_tensor(descriptors), torch.as_tensor(sequence_ids), spec


def fingerprint(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def mpjpe_mm(pred, target):
    return (pred - target).norm(dim=-1).mean() * 1000.0


def pa_mpjpe_mm(pred, target):
    pred, target = pred.float(), target.float()
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
def evaluate(model, loader, device, max_batches: int = 0) -> dict:
    model.eval()
    sums = {"abs": 0.0, "pa": 0.0, "root": 0.0, "anchor": 0.0, "cell": 0.0, "conf": 0.0, "entropy": 0.0, "n": 0}
    for i, (x, raw, target, _) in enumerate(loader, 1):
        out = model(x.to(device), raw.to(device)); target = target.to(device)
        n = len(target)
        sums["abs"] += float(mpjpe_mm(out.pose, target)) * n
        sums["pa"] += float(pa_mpjpe_mm(out.pose, target)) * n
        sums["root"] += float((out.root - target[:, 0]).norm(dim=-1).mean() * 1000.0) * n
        sums["anchor"] += float((out.anchor - target[:, 0]).norm(dim=-1).mean() * 1000.0) * n
        memory_target = model.memory_target(target[:, 0], out.valid_memory)
        sums["cell"] += float((out.top_indices[:, 0] == memory_target).float().mean()) * n
        sums["conf"] += float(out.confidence.mean()) * n
        sums["entropy"] += float(out.entropy.mean()) * n
        sums["n"] += n
        if max_batches and i >= max_batches:
            break
    n = max(1, sums["n"])
    return {"absolute_mpjpe_mm": sums["abs"] / n, "pa_mpjpe_mm": sums["pa"] / n,
            "root_mae_mm": sums["root"] / n, "anchor_error_mm": sums["anchor"] / n,
            "memory_hit_at_1": sums["cell"] / n,
            "retrieval_confidence": sums["conf"] / n, "retrieval_entropy": sums["entropy"] / n,
            "frames": int(sums["n"])}


def atomic_torch_save(payload: dict, path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")); exp = cfg["experiment"]
    seed = int(exp.get("seed", 42)); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dataset_root = Path(exp["dataset_root"]); manifest_path = Path(exp["manifest"]); label_path = Path(exp["label_manifest"])
    if not dataset_root.is_absolute(): dataset_root = (ROOT / dataset_root).resolve()
    if not manifest_path.is_absolute(): manifest_path = (ROOT / manifest_path).resolve()
    if not label_path.is_absolute(): label_path = (ROOT / label_path).resolve()
    manifest = DataManifest.read_json(manifest_path); train_keys = read_keys(label_path)
    if not train_keys.issubset(set(manifest.internal_train_keys)):
        raise ValueError("label manifest contains keys outside internal_train_keys")
    for field in ("protocol", "split", "scope", "seed"):
        if str(exp.get(field)) != str(getattr(manifest, field)):
            raise ValueError(f"config/manifest mismatch for {field}")
    train_set, select_set, test_set = set(train_keys), set(manifest.select_keys), set(manifest.test_keys)
    if train_set & (select_set | test_set) or select_set & test_set:
        raise ValueError("train/select/test sequence sets must be disjoint")
    all_keys = train_set | select_set | test_set
    sequence_to_id = {str(key): i for i, key in enumerate(sorted(all_keys))}
    train_ds = SingleFrameDataset(dataset_root, train_set, sequence_to_id, True)
    select_ds = SingleFrameDataset(dataset_root, select_set, sequence_to_id, True)
    test_ds = SingleFrameDataset(dataset_root, test_set, sequence_to_id, True)
    memory_coords, memory_descriptors, memory_sequence_ids, atlas_meta = memory_spec(dataset_root, train_set, sequence_to_id)
    atlas_meta["descriptor_sha256"] = fingerprint(memory_descriptors.tolist())
    atlas_meta["train_key_fingerprint"] = fingerprint(sorted(map(str, train_set)))
    atlas_meta["manifest_fingerprint"] = manifest.fingerprint(); atlas_meta["fingerprint"] = fingerprint(atlas_meta)
    config_fingerprint = fingerprint(cfg)
    bs = int(exp.get("batch_size", 64)); workers = int(exp.get("num_workers", 0))
    train_loader = DataLoader(train_ds, bs, shuffle=True, num_workers=workers, collate_fn=collate)
    select_loader = DataLoader(select_ds, bs, shuffle=False, num_workers=workers, collate_fn=collate)
    test_loader = DataLoader(test_ds, bs, shuffle=False, num_workers=workers, collate_fn=collate)
    device = torch.device(exp.get("device", "cuda") if exp.get("device", "cuda") == "cpu" or torch.cuda.is_available() else "cpu")
    model = WACPoseModel.from_config(cfg, memory_coords, memory_descriptors, memory_sequence_ids).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(exp.get("learning_rate", 3e-4)), weight_decay=float(exp.get("weight_decay", 1e-4)))
    epochs = int(exp.get("epochs", 25))
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, epochs), eta_min=float(exp.get("min_learning_rate", 1e-5)))
    amp = bool(exp.get("amp", True)) and device.type == "cuda"; scaler = torch.amp.GradScaler("cuda", enabled=amp)
    out_dir = Path(cfg["artifacts"]["output_root"]); out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir; out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(args.config.read_text(encoding="utf-8"), encoding="utf-8")
    (out_dir / "atlas.json").write_text(json.dumps({**atlas_meta, "memory_coords_m": memory_coords.tolist(), "memory_sequence_ids": memory_sequence_ids.tolist(), "memory_descriptors": memory_descriptors.tolist()}, indent=2), encoding="utf-8")
    last_path = out_dir / cfg["artifacts"].get("last_state", "last_state.pth"); best_path = out_dir / cfg["artifacts"].get("checkpoint", "best_absolute.pth")
    best, rows, start = float("inf"), [], 0
    if bool(exp.get("resume", True)) and last_path.is_file():
        state = torch.load(last_path, map_location=device, weights_only=False)
        if (state.get("method") != "WAC-CSI-Memory" or state.get("atlas_fingerprint") != atlas_meta["fingerprint"]
                or state.get("manifest_fingerprint") != manifest.fingerprint()
                or state.get("config_fingerprint") != config_fingerprint):
            raise RuntimeError("checkpoint does not match WAC-CSI or source-only atlas; use a fresh output_root")
        model.load_state_dict(state["model_state_dict"]); optimizer.load_state_dict(state["optimizer_state_dict"]); scheduler.load_state_dict(state["scheduler_state_dict"]); scaler.load_state_dict(state.get("scaler_state_dict", scaler.state_dict()))
        start, best, rows = int(state["epoch"]), float(state["best_metric"]), list(state.get("metrics", []))
        print(f"[WAC-MEMORY][RESUME] epoch={start} best_select_abs={best:.2f}mm", flush=True)
    print(f"[WAC-MEMORY][START] device={device} amp={amp} train={len(train_ds)} select={len(select_ds)} test={len(test_ds)} memory={len(memory_coords)} topk={model.atlas.top_k} leave_one_sequence_out=True", flush=True)
    for epoch in range(start, epochs):
        model.train(); totals = {}; count = 0; optimizer_updated = False; begin = time.perf_counter()
        for i, (x, raw, target, sequence_ids) in enumerate(train_loader, 1):
            x, raw, target, sequence_ids = x.to(device), raw.to(device), target.to(device), sequence_ids.to(device); optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                output = model(x, raw, exclude_sequence_ids=sequence_ids); loss, logs = model.loss(output, target)
            scaler.scale(loss).backward(); scale_before = scaler.get_scale(); scaler.step(optimizer); scaler.update()
            optimizer_updated = optimizer_updated or (not amp or scaler.get_scale() >= scale_before)
            count += 1
            for k, v in logs.items(): totals[k] = totals.get(k, 0.0) + v
            interval = int(exp.get("log_interval", 50))
            if interval > 0 and (i == 1 or i % interval == 0):
                print(f"[WAC-MEMORY][epoch {epoch + 1}/{epochs}][batch {i}/{len(train_loader)}] loss={logs['loss']:.4f} retrieval={logs['retrieval_loss']:.4f} residual={logs['residual_loss']:.4f} anchor={logs['anchor_error']:.4f}", flush=True)
            if exp.get("max_train_batches", 0) and i >= int(exp["max_train_batches"]): break
        if optimizer_updated:
            scheduler.step()
        val = evaluate(model, select_loader, device, int(exp.get("max_select_batches", 0)))
        row = {"epoch": epoch + 1, "learning_rate": optimizer.param_groups[0]["lr"], **{f"train_{k}": v / max(1, count) for k, v in totals.items()}, **val}; rows.append(row)
        print(f"[WAC-MEMORY][EPOCH {epoch + 1}/{epochs}] loss={row['train_loss']:.4f} retrieval={row['train_retrieval_loss']:.4f} residual={row['train_residual_loss']:.4f} abs={row['absolute_mpjpe_mm']:.2f}mm pa={row['pa_mpjpe_mm']:.2f}mm root={row['root_mae_mm']:.2f}mm anchor={row['anchor_error_mm']:.2f}mm hit1={row['memory_hit_at_1']:.3f} conf={row['retrieval_confidence']:.3f} lr={row['learning_rate']:.3g} time={time.perf_counter() - begin:.1f}s", flush=True)
        if val["absolute_mpjpe_mm"] < best:
            best = val["absolute_mpjpe_mm"]; atomic_torch_save({"method": "WAC-CSI-Memory", "epoch": epoch + 1, "model_state_dict": model.state_dict(), "select_metrics": row, "atlas_fingerprint": atlas_meta["fingerprint"], "manifest_fingerprint": manifest.fingerprint(), "config_fingerprint": config_fingerprint, "config": cfg}, best_path)
        atomic_torch_save({"method": "WAC-CSI-Memory", "epoch": epoch + 1, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict(), "best_metric": best, "metrics": rows, "atlas_fingerprint": atlas_meta["fingerprint"], "manifest_fingerprint": manifest.fingerprint(), "config_fingerprint": config_fingerprint, "config": cfg}, last_path)
    best_state = torch.load(best_path, map_location=device, weights_only=False)
    if best_state.get("method") != "WAC-CSI-Memory" or best_state.get("atlas_fingerprint") != atlas_meta["fingerprint"]:
        raise RuntimeError("selected checkpoint failed WAC-CSI atlas validation")
    model.load_state_dict(best_state["model_state_dict"]); test = evaluate(model, test_loader, device, int(exp.get("max_test_batches", 0)))
    if not rows:
        raise RuntimeError("no training metrics available for checkpoint selection")
    report = {"method": "WAC-CSI-Memory", "protocol": manifest.protocol, "split": manifest.split, "scope": manifest.scope, "seed": manifest.seed, "label_budget": exp.get("label_budget"), "manifest_fingerprint": manifest.fingerprint(), "atlas_fingerprint": atlas_meta["fingerprint"], "atlas_source_only": True, "memory_type": "sample_level", "leave_one_sequence_out_training": True, "select": min(rows, key=lambda r: r["absolute_mpjpe_mm"]), "test": test, "test_used_for_selection": False, "test_labels_used_for_training_or_selection": False, "target_shift_correction": False}
    (out_dir / "metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8"); (out_dir / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[WAC-MEMORY][DONE] best_select_abs={report['select']['absolute_mpjpe_mm']:.2f}mm test_abs={test['absolute_mpjpe_mm']:.2f}mm test_pa={test['pa_mpjpe_mm']:.2f}mm output={out_dir}", flush=True)


if __name__ == "__main__":
    main()
