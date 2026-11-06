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
from MY_SOTA.RePos.model import RePos, stage1_loss, stage2_loss


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


def augment_csi(x: torch.Tensor, *, temporal_jitter: int = 1, noise_std: float = 0.01) -> torch.Tensor:
    """Paper augmentation: small temporal jitter plus amplitude Gaussian noise."""
    if temporal_jitter:
        shifts = torch.randint(-temporal_jitter, temporal_jitter + 1, (x.shape[0],), device=x.device)
        x = torch.stack([torch.roll(sample, int(shift), dims=-1) for sample, shift in zip(x, shifts)], dim=0)
    if noise_std:
        x = x + torch.randn_like(x) * noise_std
    return x


def cosine_warmup(optimizer, *, epochs: int, warmup_epochs: int):
    warmup_epochs = min(max(0, warmup_epochs), max(1, epochs - 1))
    def schedule(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return float(0.5 * (1.0 + np.cos(np.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


class RunLogger:
    def __init__(self, path: Path):
        self.handle = path.open("w", encoding="utf-8", newline="\n")

    def write(self, message: str):
        print(message, flush=True)
        self.handle.write(message + "\n")
        self.handle.flush()

    def close(self):
        self.handle.close()


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


def run(config: dict, eval_checkpoint: str | None = None):
    exp = config["experiment"]
    model_cfg = config["model"]
    ft = config["finetune"]
    seed = int(exp["seed"])
    seed_all(seed)
    device_name = str(exp.get("device", "cuda"))
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    use_amp = bool(ft.get("amp", True)) and device.type == "cuda"
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
    data_cfg = config.get("dataset", {})
    observed = {
        "train_sequences": len(labels.selected_keys),
        "select_sequences": len(manifest.select_keys),
        "test_sequences": len(manifest.test_keys),
        "train_frames": len(train_ds),
        "select_frames": len(select_ds),
        "test_frames": len(test_ds),
    }
    for key, actual in observed.items():
        expected = data_cfg.get(key)
        if expected is not None and int(expected) != actual:
            raise ValueError(f"dataset.{key}={expected} does not match audited data ({actual})")
    output = repo_path(exp["output_root"]) / exp["protocol"] / exp["split"] / f"seed{seed}" / exp["label_budget"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    logger = RunLogger(output / ("evaluation.log" if eval_checkpoint else "training.log"))
    log = logger.write
    model = RePos(
        embed_dim=int(model_cfg["embed_dim"]), depth=int(model_cfg["depth"]), num_heads=int(model_cfg["num_heads"]),
        ffn_dim=int(model_cfg["ffn_dim"]), n_time=int(config["input"]["time"]), n_links=int(config["input"]["links"]),
        n_subcarriers=int(config["input"]["subcarriers"]), query_depth=int(model_cfg.get("query_depth", 1)),
        sga_depth=int(model_cfg.get("sga_depth", 4)), gcn_layers=int(model_cfg.get("gcn_layers", 3)),
        joint_attention_layers=int(model_cfg.get("joint_attention_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        n_angle=int(model_cfg.get("n_angle", 90)),
        phase_mode=str(model_cfg.get("phase_mode", "paper")),
        delay_mode=str(model_cfg.get("delay_mode", "legacy_truncate")),
    ).to(device)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    log("[run] method=RePos protocol=%s split=%s seed=%d label_budget=%s" % (exp["protocol"], exp["split"], seed, exp["label_budget"]))
    log("[run] device=%s amp=%s normalize=%s" % (device, use_amp, normalize))
    log("[data] train=%d seq/%d frames select=%d seq/%d frames test=%d seq/%d frames" % (observed["train_sequences"], observed["train_frames"], observed["select_sequences"], observed["select_frames"], observed["test_sequences"], observed["test_frames"]))
    log("[model] parameters=%d trainable=%d embed_dim=%d depth=%d heads=%d" % (total_params, trainable_params, model_cfg["embed_dim"], model_cfg["depth"], model_cfg["num_heads"]))
    log("[config] batch=%d eval_batch=%d stage1_epochs=%d stage2_epochs=%d stage1_lr=%.2e stage2_lr=%.2e weight_decay=%.2e" % (batch_size, eval_batch_size, int(ft.get("stage1_epochs", 25)), int(ft.get("stage2_epochs", 15)), float(ft.get("stage1_lr", 1e-4)), float(ft.get("stage2_lr", 3e-4)), float(ft.get("weight_decay", 2e-2))))
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
        log(f"[init] loaded JEPA encoder: {pretrain_path}")
    aspn_pretrain_path = exp.get("aspn_pretrain_checkpoint")
    if aspn_pretrain_path:
        aspn_pretrain_path = repo_path(aspn_pretrain_path)
        aspn_checkpoint = torch.load(aspn_pretrain_path, map_location="cpu", weights_only=True)
        aspn_state = aspn_checkpoint.get("aspn_state_dict")
        if not isinstance(aspn_state, dict):
            raise ValueError(f"ASPN checkpoint lacks aspn_state_dict: {aspn_pretrain_path}")
        model.aspn.load_state_dict(aspn_state, strict=True)
        log(f"[init] loaded ASPN pretraining: {aspn_pretrain_path}")
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
            log("[init] loaded only ASPN phase stem layer 0")
        elif load_mode == "all":
            model.aspn.phase_net.load_state_dict(phase_state, strict=True)
        else:
            raise ValueError("aspn_phase_pretrain_load must be first or all")
        log(f"[init] loaded ASPN geometric phase pretraining: {aspn_phase_path}")
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
            log("[init] loaded only root-aware ASPN phase_net")
        elif root_load == "all":
            model.aspn.load_state_dict(root_state, strict=True)
        else:
            raise ValueError("aspn_root_aware_load must be phase or all")
        log(f"[init] loaded root-aware ASPN pretraining: {root_aware_path}")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    metrics = {"stage1": [], "stage2": [], "data": observed, "pretrain": pretrain_info}

    if eval_checkpoint:
        checkpoint_path = repo_path(eval_checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        checkpoint_mode = checkpoint.get("config", {}).get("model", {}).get("delay_mode", "legacy_truncate")
        if checkpoint_mode != model.aspn.delay_mode:
            raise ValueError(f"checkpoint ASPN delay_mode={checkpoint_mode} does not match config delay_mode={model.aspn.delay_mode}")
        state = checkpoint.get("model_state_dict")
        if not isinstance(state, dict):
            raise ValueError(f"checkpoint lacks model_state_dict: {checkpoint_path}")
        model.load_state_dict(state, strict=True)
        test = evaluate(model, test_loader, device, stage=2, use_amp=use_amp)
        report = {
            "method": "repos_cross_scene",
            "evaluation_only": True,
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_stage": checkpoint.get("stage"),
            "protocol": manifest.protocol,
            "split": manifest.split,
            "seed": seed,
            "label_budget": exp["label_budget"],
            "data": observed,
            "test_metrics": {key: float(test[key]) for key in ("absolute_mpjpe_mm", "pelvis_mpjpe_mm", "pa_mpjpe_mm")},
        }
        report_path = output / "evaluation_from_checkpoint.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"[EVAL DONE] checkpoint={checkpoint_path} test_abs={test['absolute_mpjpe_mm']:.2f}mm test_rootrel={test['pelvis_mpjpe_mm']:.2f}mm test_pa={test['pa_mpjpe_mm']:.2f}mm report={report_path}")
        logger.close()
        return output

    # Stage 1: relative structure only; no root/absolute loss reaches structure.
    optimizer = torch.optim.AdamW(model.structure.parameters(), lr=float(ft.get("stage1_lr", 3e-4)), weight_decay=float(ft.get("weight_decay", 1e-4)))
    stage1_epochs = int(ft.get("stage1_epochs", 100))
    scheduler = cosine_warmup(optimizer, epochs=stage1_epochs, warmup_epochs=int(ft.get("warmup_epochs", 5)))
    best = float("inf")
    stale_epochs = 0
    patience = int(ft.get("early_stop_patience", 20))
    with csi_preprocessing(normalize):
        for epoch in range(stage1_epochs):
            model.train(); model.aspn.eval(); totals = {"loss": 0.0, "l1": 0.0, "l2": 0.0, "limb": 0.0}; batches = 0; started = time.perf_counter()
            for x, y in train_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                x = augment_csi(x, temporal_jitter=int(ft.get("temporal_jitter", 1)), noise_std=float(ft.get("noise_std", 0.01)))
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = model.forward_relative(x)
                    loss, loss_log = stage1_loss(pred, y, limb_weight=float(ft.get("limb_weight", 0.1)))
                scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.structure.parameters(), 1.0); scaler.step(optimizer); scaler.update()
                for key in totals:
                    totals[key] += loss_log.get(key, loss.detach())
                batches += 1
                if batches % int(ft.get("log_every_batches", 100)) == 0:
                    log(f"[stage1 {epoch + 1:02d}/{stage1_epochs:02d}] batch={batches:04d}/{len(train_loader):04d} loss={float(totals['loss'] / batches):.5f}")
            sel = evaluate(model, select_loader, device, stage=1, use_amp=use_amp)
            row = {"epoch": epoch + 1, "train_loss": float(totals["loss"] / max(1, batches)), "train_l1": float(totals["l1"] / max(1, batches)), "train_l2": float(totals["l2"] / max(1, batches)), "train_limb": float(totals["limb"] / max(1, batches)), "select_absolute_mpjpe_mm": float(sel["absolute_mpjpe_mm"]), "select_pelvis_mpjpe_mm": float(sel["pelvis_mpjpe_mm"]), "select_pa_mpjpe_mm": float(sel["pa_mpjpe_mm"]), "seconds": float(time.perf_counter() - started), "lr": float(optimizer.param_groups[0]["lr"]), "batches": int(batches)}
            metrics["stage1"].append(row)
            improved = sel["pelvis_mpjpe_mm"] < best
            log(f"[stage1 {epoch + 1:02d}/{stage1_epochs:02d}] loss={row['train_loss']:.5f} l1={row['train_l1']:.5f} l2={row['train_l2']:.5f} limb={row['train_limb']:.5f} lr={row['lr']:.2e} select_abs={row['select_absolute_mpjpe_mm']:.1f}mm select_rootrel={row['select_pelvis_mpjpe_mm']:.1f}mm select_pa={row['select_pa_mpjpe_mm']:.1f}mm time={row['seconds']:.1f}s%s" % (" [BEST]" if improved else ""))
            if improved:
                best = sel["pelvis_mpjpe_mm"]
                stale_epochs = 0
                torch.save({"stage": 1, "model_state_dict": model.state_dict(), "metrics": row, "config": config}, output / "stage1_best.pth")
                log(f"[stage1] checkpoint=stage1_best.pth metric=select_rootrel value={best:.2f}mm")
            else:
                stale_epochs += 1
            scheduler.step()
            if stale_epochs >= patience:
                log(f"[stage1] early_stop patience={patience}")
                break
        # These checkpoints are produced by this script and include config/metrics
        # metadata; trusted local loading keeps PyTorch 2.6 from rejecting NumPy
        # scalar metadata from older runs.
        best_ckpt = torch.load(output / "stage1_best.pth", map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"], strict=True)

        # Stage 2: freeze structure, learn only ASPN absolute root branch.
        for p in model.structure.parameters():
            p.requires_grad_(False)
        model.structure.eval()
        optimizer = torch.optim.AdamW(model.aspn.parameters(), lr=float(ft.get("stage2_lr", 3e-4)), weight_decay=float(ft.get("weight_decay", 1e-4)))
        stage2_epochs = int(ft.get("stage2_epochs", 50))
        scheduler = cosine_warmup(optimizer, epochs=stage2_epochs, warmup_epochs=int(ft.get("warmup_epochs", 5)))
        best = float("inf")
        stale_epochs = 0
        for epoch in range(stage2_epochs):
            model.train(); model.structure.eval(); totals = {"loss": 0.0, "absolute_l1": 0.0, "absolute_l2": 0.0, "root_l1": 0.0}; batches = 0; started = time.perf_counter()
            for x, y in train_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                x = augment_csi(x, temporal_jitter=int(ft.get("temporal_jitter", 1)), noise_std=float(ft.get("noise_std", 0.01)))
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    rel = model.forward_relative(x)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    root = model.forward_root(x)
                    loss, loss_log = stage2_loss(root, rel, y, root_weight=float(ft.get("root_weight", 1.0)))
                scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.aspn.parameters(), 1.0); scaler.step(optimizer); scaler.update()
                for key in totals:
                    totals[key] += loss_log.get(key, loss.detach())
                batches += 1
                if batches % int(ft.get("log_every_batches", 100)) == 0:
                    log(f"[stage2 {epoch + 1:02d}/{stage2_epochs:02d}] batch={batches:04d}/{len(train_loader):04d} loss={float(totals['loss'] / batches):.5f}")
            sel = evaluate(model, select_loader, device, stage=2, use_amp=use_amp)
            row = {"epoch": epoch + 1, "train_loss": float(totals["loss"] / max(1, batches)), "train_absolute_l1": float(totals["absolute_l1"] / max(1, batches)), "train_absolute_l2": float(totals["absolute_l2"] / max(1, batches)), "train_root_l1": float(totals["root_l1"] / max(1, batches)), "select_absolute_mpjpe_mm": float(sel["absolute_mpjpe_mm"]), "select_pelvis_mpjpe_mm": float(sel["pelvis_mpjpe_mm"]), "select_pa_mpjpe_mm": float(sel["pa_mpjpe_mm"]), "seconds": float(time.perf_counter() - started), "lr": float(optimizer.param_groups[0]["lr"]), "batches": int(batches)}
            metrics["stage2"].append(row)
            improved = sel["absolute_mpjpe_mm"] < best
            log(f"[stage2 {epoch + 1:02d}/{stage2_epochs:02d}] loss={row['train_loss']:.5f} abs_l1={row['train_absolute_l1']:.5f} abs_l2={row['train_absolute_l2']:.5f} root_l1={row['train_root_l1']:.5f} lr={row['lr']:.2e} select_abs={row['select_absolute_mpjpe_mm']:.1f}mm select_rootrel={row['select_pelvis_mpjpe_mm']:.1f}mm select_pa={row['select_pa_mpjpe_mm']:.1f}mm time={row['seconds']:.1f}s%s" % (" [BEST]" if improved else ""))
            if improved:
                best = sel["absolute_mpjpe_mm"]
                stale_epochs = 0
                torch.save({"stage": 2, "model_state_dict": model.state_dict(), "metrics": row, "config": config}, output / "stage2_best.pth")
                log(f"[stage2] checkpoint=stage2_best.pth metric=select_abs value={best:.2f}mm")
            else:
                stale_epochs += 1
            scheduler.step()
            if stale_epochs >= patience:
                log(f"[stage2] early_stop patience={patience}")
                break
        best_ckpt = torch.load(output / "stage2_best.pth", map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"], strict=True)
        test = evaluate(model, test_loader, device, stage=2, use_amp=use_amp)
    np.savez_compressed(output / "test_outputs_absolute.npz", predictions=test["predictions"].astype(np.float32), targets=test["targets"].astype(np.float32))
    final = {"method": "repos_cross_scene", "protocol": manifest.protocol, "split": manifest.split, "seed": seed, "label_budget": exp["label_budget"], "stage2_best": best_ckpt.get("metrics"), "test_metrics": {k: float(test[k]) for k in ("absolute_mpjpe_mm", "pelvis_mpjpe_mm", "pa_mpjpe_mm")}, "metrics": metrics}
    (output / "metrics.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "done.txt").write_text("complete\n", encoding="utf-8")
    log(f"[DONE] output={output} test_abs={test['absolute_mpjpe_mm']:.2f}mm test_rootrel={test['pelvis_mpjpe_mm']:.2f}mm test_pa={test['pa_mpjpe_mm']:.2f}mm")
    logger.close()
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="MY_SOTA/RePos/config.yaml")
    parser.add_argument("--eval-checkpoint", default=None, help="只评估已有 stage2 checkpoint，不重新训练")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8-sig"))
    run(config, eval_checkpoint=args.eval_checkpoint)


if __name__ == "__main__":
    main()

