from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mmfi_wifi.data_manifest import DataManifest
from mmfi_wifi.metrics import evaluate_pose_official, mpjpe_mm, pa_mpjpe_mm
from scripts.metafi_ssl.train_supervised import load_audited_artifacts
from WIFIJEPA.src.data import WiFiJEPAData, _load_ground_truth, csi_preprocessing
from WIFIJEPA.src.paths import audit_paths, repo_path
from MY_SOTA.RePos_343_Baseline.model import RePos, stage1_loss, stage2_loss


class PoseFrameDataset(Dataset):
    def __init__(self, dataset_root: str, keys, *, protocol: str, split: str, scope: str, manifest_fingerprint: str, normalize: bool):
        self.base = WiFiJEPAData(
            dataset_root, keys, protocol=protocol, split=split, scope=scope,
            manifest_fingerprint=manifest_fingerprint, normalize=normalize,
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        record = self.base._records[index]
        x = self.base[index].anchor
        gt_path = self.base.dataset_root / record.sequence_key.scene / record.sequence_key.subject / record.sequence_key.action / "ground_truth.npy"
        y = torch.as_tensor(_load_ground_truth(str(gt_path))[record.frame_index].copy(), dtype=torch.float32)
        return x, y


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model: RePos, loader: DataLoader, device: torch.device, *, stage: int, use_amp: bool):
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model.forward_relative(x) if stage == 1 else model.forward_absolute(x)
            if stage == 1:
                pred = pred + y[:, :1]  # only for common absolute-space diagnostics
            predictions.append(pred.float().cpu().numpy())
            targets.append(y.float().cpu().numpy())
    pred = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets, axis=0)
    absolute, pa = evaluate_pose_official(pred, target)
    pelvis = mpjpe_mm(pred, target, already_root_relative=False)
    relative = mpjpe_mm(pred, target, already_root_relative=False)
    return {"absolute_mpjpe_mm": absolute, "pelvis_mpjpe_mm": pelvis, "pa_mpjpe_mm": pa, "predictions": pred, "targets": target}


def run(config: dict):
    exp = config["experiment"]
    model_cfg = config["model"]
    ft = config["finetune"]
    seed = int(exp["seed"])
    seed_all(seed)
    device_name = str(exp.get("device", "cuda"))
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    normalize = not bool(config["input"].get("disable_normalization", False))
    audit = audit_paths(config)
    manifest, labels, _ = load_audited_artifacts(
        audit["manifest"], audit["label_manifest"], audit["leakage_audit"], audit["axis_stats"],
        protocol=exp["protocol"], split=exp["split"], label_budget=exp["label_budget"],
    )
    dataset_root = repo_path(exp["dataset_root"])
    train_ds = PoseFrameDataset(str(dataset_root), labels.selected_keys, protocol=manifest.protocol, split=manifest.split, scope=manifest.scope, manifest_fingerprint=manifest.fingerprint(), normalize=normalize)
    select_ds = PoseFrameDataset(str(dataset_root), manifest.select_keys, protocol=manifest.protocol, split=manifest.split, scope=manifest.scope, manifest_fingerprint=manifest.fingerprint(), normalize=normalize)
    test_ds = PoseFrameDataset(str(dataset_root), manifest.test_keys, protocol=manifest.protocol, split=manifest.split, scope=manifest.scope, manifest_fingerprint=manifest.fingerprint(), normalize=normalize)
    batch_size = int(ft.get("batch_size", 64))
    eval_batch_size = int(ft.get("eval_batch_size", 128))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=device.type == "cuda")
    select_loader = DataLoader(select_ds, batch_size=eval_batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    model = RePos(
        embed_dim=int(model_cfg["embed_dim"]), depth=int(model_cfg["depth"]), num_heads=int(model_cfg["num_heads"]),
        ffn_dim=int(model_cfg["ffn_dim"]), n_time=int(config["input"]["time"]), n_links=int(config["input"]["links"]),
        n_subcarriers=int(config["input"]["subcarriers"]), query_depth=int(model_cfg.get("query_depth", 1)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        n_angle=int(model_cfg.get("n_angle", 90)),
        phase_mode=str(model_cfg.get("phase_mode", "paper")),
    ).to(device)
    pretrain_path = exp.get("pretrain_checkpoint")
    pretrain_info = None
    if pretrain_path:
        pretrain_path = repo_path(pretrain_path)
        checkpoint = torch.load(pretrain_path, map_location="cpu", weights_only=True)
        state = checkpoint.get("encoder_state_dict")
        if not isinstance(state, dict):
            raise ValueError(f"JEPA checkpoint lacks encoder_state_dict: {pretrain_path}")
        model.structure.encoder.load_state_dict(state, strict=True)
        pretrain_info = {
            "path": str(pretrain_path.resolve()),
            "method": checkpoint.get("method"),
            "model_variant": checkpoint.get("model_variant"),
            "manifest_fingerprint": checkpoint.get("manifest_fingerprint"),
            "seed": checkpoint.get("seed"),
        }
        print(f"[init] loaded JEPA encoder: {pretrain_path}", flush=True)
    aspn_pretrain_path = exp.get("aspn_pretrain_checkpoint")
    if aspn_pretrain_path:
        aspn_pretrain_path = repo_path(aspn_pretrain_path)
        aspn_checkpoint = torch.load(aspn_pretrain_path, map_location="cpu", weights_only=True)
        aspn_state = aspn_checkpoint.get("aspn_state_dict")
        if not isinstance(aspn_state, dict):
            raise ValueError(f"ASPN checkpoint lacks aspn_state_dict: {aspn_pretrain_path}")
        model.aspn.load_state_dict(aspn_state, strict=True)
        print(f"[init] loaded ASPN pretraining: {aspn_pretrain_path}", flush=True)
    aspn_phase_path = exp.get("aspn_phase_pretrain_checkpoint")
    if aspn_phase_path:
        aspn_phase_path = repo_path(aspn_phase_path)
        phase_checkpoint = torch.load(aspn_phase_path, map_location="cpu", weights_only=True)
        phase_state = phase_checkpoint.get("phase_net_state_dict")
        if not isinstance(phase_state, dict):
            raise ValueError(f"geometric ASPN checkpoint lacks phase_net_state_dict: {aspn_phase_path}")
        load_mode = str(exp.get("aspn_phase_pretrain_load", "first"))
        if load_mode == "first":
            shallow = {key: value for key, value in phase_state.items() if key.startswith("0.")}
            model.aspn.phase_net.load_state_dict(shallow, strict=False)
            print("[init] loaded only ASPN phase stem layer 0", flush=True)
        elif load_mode == "all":
            model.aspn.phase_net.load_state_dict(phase_state, strict=True)
        else:
            raise ValueError("aspn_phase_pretrain_load must be first or all")
        print(f"[init] loaded ASPN geometric phase pretraining: {aspn_phase_path}", flush=True)
    root_aware_path = exp.get("aspn_root_aware_checkpoint")
    if root_aware_path:
        root_aware_path = repo_path(root_aware_path)
        root_checkpoint = torch.load(root_aware_path, map_location="cpu", weights_only=True)
        root_state = root_checkpoint.get("aspn_state_dict")
        if not isinstance(root_state, dict):
            raise ValueError(f"root-aware checkpoint lacks aspn_state_dict: {root_aware_path}")
        root_load = str(exp.get("aspn_root_aware_load", "all"))
        if root_load == "phase":
            phase_state = {key[len("phase_net."):]: value for key, value in root_state.items() if key.startswith("phase_net.")}
            model.aspn.phase_net.load_state_dict(phase_state, strict=True)
            print("[init] loaded only root-aware ASPN phase_net", flush=True)
        elif root_load == "all":
            model.aspn.load_state_dict(root_state, strict=True)
        else:
            raise ValueError("aspn_root_aware_load must be phase or all")
        print(f"[init] loaded root-aware ASPN pretraining: {root_aware_path}", flush=True)
    use_amp = bool(ft.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    output = repo_path(exp["output_root"]) / exp["protocol"] / exp["split"] / f"seed{seed}" / exp["label_budget"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    metrics = {"stage1": [], "stage2": [], "data": {"train_frames": len(train_ds), "select_frames": len(select_ds), "test_frames": len(test_ds)}, "pretrain": pretrain_info}

    # Stage 1: relative structure only; no root/absolute loss reaches structure.
    optimizer = torch.optim.AdamW(model.structure.parameters(), lr=float(ft.get("stage1_lr", 3e-4)), weight_decay=float(ft.get("weight_decay", 1e-4)))
    best = float("inf")
    stage1_epochs = int(ft.get("stage1_epochs", 8))
    with csi_preprocessing(normalize):
        for epoch in range(stage1_epochs):
            model.train(); model.aspn.eval(); total = 0.0; batches = 0; started = time.perf_counter()
            for x, y in train_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = model.forward_relative(x)
                    loss, log = stage1_loss(pred, y, limb_weight=float(ft.get("limb_weight", 0.1)))
                scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.structure.parameters(), 1.0); scaler.step(optimizer); scaler.update()
                total += float(loss.detach()); batches += 1
            sel = evaluate(model, select_loader, device, stage=1, use_amp=use_amp)
            row = {"epoch": epoch + 1, "train_loss": total / max(1, batches), "select_absolute_mpjpe_mm": sel["absolute_mpjpe_mm"], "select_pelvis_mpjpe_mm": sel["pelvis_mpjpe_mm"], "select_pa_mpjpe_mm": sel["pa_mpjpe_mm"], "seconds": time.perf_counter() - started}
            metrics["stage1"].append(row)
            print(f"[stage1 {epoch + 1:02d}/{stage1_epochs:02d}] loss={row['train_loss']:.5f} select_abs={row['select_absolute_mpjpe_mm']:.1f}mm", flush=True)
            if sel["pelvis_mpjpe_mm"] < best:
                best = sel["pelvis_mpjpe_mm"]
                torch.save({"stage": 1, "model_state_dict": model.state_dict(), "metrics": row, "config": config}, output / "stage1_best.pth")
        best_ckpt = torch.load(output / "stage1_best.pth", map_location=device, weights_only=True)
        model.load_state_dict(best_ckpt["model_state_dict"], strict=True)

        # Stage 2: freeze structure, learn only ASPN absolute root branch.
        for p in model.structure.parameters():
            p.requires_grad_(False)
        model.structure.eval()
        optimizer = torch.optim.AdamW(model.aspn.parameters(), lr=float(ft.get("stage2_lr", 3e-4)), weight_decay=float(ft.get("weight_decay", 1e-4)))
        best = float("inf")
        stage2_epochs = int(ft.get("stage2_epochs", 10))
        for epoch in range(stage2_epochs):
            model.train(); model.structure.eval(); total = 0.0; batches = 0; started = time.perf_counter()
            for x, y in train_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    rel = model.forward_relative(x)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    root = model.forward_root(x)
                    loss, log = stage2_loss(root, rel, y, root_weight=float(ft.get("root_weight", 1.0)))
                scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.aspn.parameters(), 1.0); scaler.step(optimizer); scaler.update()
                total += float(loss.detach()); batches += 1
            sel = evaluate(model, select_loader, device, stage=2, use_amp=use_amp)
            row = {"epoch": epoch + 1, "train_loss": total / max(1, batches), "select_absolute_mpjpe_mm": sel["absolute_mpjpe_mm"], "select_pelvis_mpjpe_mm": sel["pelvis_mpjpe_mm"], "select_pa_mpjpe_mm": sel["pa_mpjpe_mm"], "seconds": time.perf_counter() - started}
            metrics["stage2"].append(row)
            print(f"[stage2 {epoch + 1:02d}/{stage2_epochs:02d}] loss={row['train_loss']:.5f} select_abs={row['select_absolute_mpjpe_mm']:.1f}mm", flush=True)
            if sel["absolute_mpjpe_mm"] < best:
                best = sel["absolute_mpjpe_mm"]
                torch.save({"stage": 2, "model_state_dict": model.state_dict(), "metrics": row, "config": config}, output / "stage2_best.pth")
        best_ckpt = torch.load(output / "stage2_best.pth", map_location=device, weights_only=True)
        model.load_state_dict(best_ckpt["model_state_dict"], strict=True)
        test = evaluate(model, test_loader, device, stage=2, use_amp=use_amp)
    np.savez_compressed(output / "test_outputs_absolute.npz", predictions=test["predictions"].astype(np.float32), targets=test["targets"].astype(np.float32))
    final = {"method": "repos_cross_scene", "protocol": manifest.protocol, "split": manifest.split, "seed": seed, "label_budget": exp["label_budget"], "stage2_best": best_ckpt.get("metrics"), "test_metrics": {k: float(test[k]) for k in ("absolute_mpjpe_mm", "pelvis_mpjpe_mm", "pa_mpjpe_mm")}, "metrics": metrics}
    (output / "metrics.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "done.txt").write_text("complete\n", encoding="utf-8")
    print(f"[DONE] output={output} test_abs={test['absolute_mpjpe_mm']:.2f}mm test_pelvis={test['pelvis_mpjpe_mm']:.2f}mm test_pa={test['pa_mpjpe_mm']:.2f}mm", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="MY_SOTA/RePos_343_Baseline/config.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8-sig"))
    run(config)


if __name__ == "__main__":
    main()
