from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mmfi_wifi.data_manifest import DataManifest
from scripts.metafi_ssl.train_supervised import load_audited_artifacts
from WIFIJEPA.src.data import WiFiJEPAData, WiFiJEPAPoseData
from pose_ssl.metafi.pretrain_subset import make_small_pretrain_manifest, selection_summary
from WIFIJEPA.src.model import StructuredViTMAE, WiFiJEPA
from pose_ssl.model import PoseHead
from mmfi_wifi.engine import validate_pose_layout_batch
from WIFIJEPA.src.paths import (
    MAE_ENCODER_CONTRACT,
    audit_paths,
    pretrain_output,
    repo_path,
)


def cosine_momentum(step: int, total_steps: int, start: float, end: float) -> float:
    if total_steps <= 1:
        return end
    progress = min(step / (total_steps - 1), 1.0)
    return end - (end - start) * (math.cos(math.pi * progress) + 1.0) / 2.0


def collate_anchors(samples):
    return torch.stack([sample.anchor for sample in samples])


def collate_pose_anchors(samples):
    anchors = torch.stack([sample.anchor for sample, _ in samples])
    poses = torch.zeros(len(samples), 17, 3, dtype=torch.float32)
    labeled = torch.zeros(len(samples), dtype=torch.bool)
    for index, (_, pose) in enumerate(samples):
        if pose is not None:
            poses[index] = pose
            labeled[index] = True
    return anchors, poses, labeled


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def run_mae_split(config: dict) -> None:
    run_cfg = config["experiment"]
    pretrain_cfg = {**config.get("pretrain", {}), **config.get("mae", {})}
    encoder_contract = str(pretrain_cfg.get("encoder_contract", MAE_ENCODER_CONTRACT))
    if encoder_contract != MAE_ENCODER_CONTRACT:
        raise ValueError(
            f"mae.encoder_contract must be {MAE_ENCODER_CONTRACT!r}; "
            f"got {encoder_contract!r}"
        )
    model_cfg = dict(config["model"])
    input_cfg = config["input"]
    epochs = int(pretrain_cfg.get("epochs", 10))
    seed = int(run_cfg["seed"])
    torch.manual_seed(seed)
    output = pretrain_output(config)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    manifest_path = audit_paths(config)["manifest"]
    full_manifest = DataManifest.read_json(manifest_path)
    manifest = make_small_pretrain_manifest(full_manifest, fraction=float(pretrain_cfg.get("sequence_fraction", 1.0)), seed=seed)
    manifest_path_out = output / "pretrain_manifest.json"
    manifest.write_json(manifest_path_out)
    signature = {"method": "vit_mae_structured", "encoder_arch": "structured_vit", "encoder_contract": encoder_contract, "manifest_fingerprint": manifest.fingerprint(), "input": input_cfg, "model": model_cfg, "mae": pretrain_cfg}
    final_path = output / "encoder.pth"
    if final_path.is_file():
        completed = torch.load(final_path, map_location="cpu", weights_only=True)
        if completed.get("status") == "complete":
            if completed.get("signature") != signature:
                raise ValueError(f"Completed MAE checkpoint does not match this run: {final_path}")
            print(f"[SKIP] split={manifest.split} already complete: {final_path}", flush=True)
            return
    dataset = WiFiJEPAData(str((ROOT / run_cfg["dataset_root"]).resolve()) if not Path(run_cfg["dataset_root"]).is_absolute() else run_cfg["dataset_root"], manifest.pretrain_keys, protocol=manifest.protocol, split=manifest.split, scope=manifest.scope, manifest_fingerprint=manifest.fingerprint(), normalize=not bool(input_cfg.get("disable_normalization", False)))
    loader = DataLoader(dataset, batch_size=int(pretrain_cfg.get("batch_size", 32)), shuffle=True, num_workers=int(run_cfg.get("num_workers", 0)), drop_last=True, collate_fn=collate_anchors, generator=torch.Generator().manual_seed(seed))
    encoder_cfg = {"embed_dim": model_cfg["embed_dim"], "depth": model_cfg["depth"], "num_heads": model_cfg["num_heads"], "ffn_dim": model_cfg["ffn_dim"], "n_time": input_cfg["time"], "n_links": input_cfg["links"], "mask_ratio": float(pretrain_cfg.get("mask_ratio", 0.75)), "decoder_dim": int(pretrain_cfg.get("decoder_dim", 512)), "decoder_depth": int(pretrain_cfg.get("decoder_depth", 2)), "decoder_heads": int(pretrain_cfg.get("decoder_heads", 8)), "n_subcarriers": int(input_cfg.get("subcarriers", 114))}
    device_name = str(run_cfg["device"])
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    model = StructuredViTMAE(**encoder_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(pretrain_cfg.get("learning_rate", 2.4e-4)), weight_decay=float(pretrain_cfg.get("weight_decay", 0.01)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=float(pretrain_cfg.get("lr_min", 1e-5)))
    use_amp = bool(pretrain_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    log_every = max(1, int(pretrain_cfg.get("log_every_batches", 20)))
    print(
        f"[StructuredViT-MAE] split={manifest.split} protocol={manifest.protocol} "
        f"device={device} amp={use_amp} sequences={len(manifest.pretrain_keys)} "
        f"frames={len(dataset)} batches/epoch={len(loader)} epochs={epochs} "
        f"output={output}",
        flush=True,
    )
    metrics = []
    for epoch in range(epochs):
        epoch_started = time.perf_counter()
        model.train()
        total_loss = 0.0
        batches = 0
        for batch_index, batch in enumerate(loader, start=1):
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss, log = model(batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_value = float(loss.detach())
            total_loss += loss_value
            batches += 1
            if batch_index == 1 or batch_index % log_every == 0:
                print(
                    f"[MAE E{epoch + 1:03d}/{epochs:03d} "
                    f"B{batch_index:05d}/{len(loader)}] "
                    f"loss={loss_value:.5f} mask={log['mask_ratio']:.3f}",
                    flush=True,
                )
        if not batches:
            raise RuntimeError("MAE pretraining produced no batches")
        epoch_loss = total_loss / batches
        metrics.append({
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        scheduler.step()
        print(
            f"[MAE epoch {epoch + 1:03d}/{epochs:03d}] "
            f"loss={epoch_loss:.5f} lr={metrics[-1]['learning_rate']:.3g} "
            f"elapsed={time.perf_counter() - epoch_started:.1f}s",
            flush=True,
        )
    torch.save({"method": "vit_mae_structured", "encoder_arch": "structured_vit", "encoder_contract": encoder_contract, "status": "complete", "seed": seed, "manifest": str(manifest_path_out.resolve()), "manifest_fingerprint": manifest.fingerprint(), "signature": signature, "config": config, "encoder_state_dict": model.export_encoder_state_dict(), "metrics": metrics}, final_path)
    (output / "pretrain_metrics.yaml").write_text(yaml.safe_dump(metrics, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"[DONE] StructuredViT-MAE encoder checkpoint: {final_path}", flush=True)


def run_split(config: dict) -> None:
        run_cfg = config["experiment"]
        mode = str(run_cfg.get("mode", "jepa"))
        if mode == "mae":
            run_mae_split(config)
            return
        if mode not in {"jepa", "pose_jepa"}:
            raise ValueError("experiment.mode must be 'jepa', 'pose_jepa', or 'mae'")
        pretrain_cfg = config["pretrain"]
        pose_cfg = config.get("pose_pretrain", {})
        pose_supervision = mode == "pose_jepa"
        if pose_supervision and not bool(pretrain_cfg.get("include_labeled_sequences", False)):
            raise ValueError("pose_jepa requires pretrain.include_labeled_sequences: true")
        pose_weight = float(pose_cfg.get("loss_weight", 10.0))
        if pose_supervision and (not math.isfinite(pose_weight) or pose_weight <= 0):
            raise ValueError("pose_pretrain.loss_weight must be finite and > 0")
        log_config = {
            "experiment": {
                key: run_cfg.get(key)
                for key in ("protocol", "scope", "split", "seed", "dataset_root", "device", "num_workers")
            },
            "input": config["input"],
            "model": config["model"],
            "mask": config["mask"],
            "pretrain": pretrain_cfg,
        }
        print("\n[WiFi-JEPA pretrain] starting with config:", flush=True)
        print(yaml.safe_dump(log_config, allow_unicode=True, sort_keys=False).rstrip(), flush=True)
        model_cfg = config["model"]
        mask_cfg = config["mask"]
        epochs = int(pretrain_cfg["epochs"])
        fraction = float(pretrain_cfg["sequence_fraction"])
        output = pretrain_output(config)
        output.mkdir(parents=True, exist_ok=True)
        # Save the exact pretraining configuration, including masked_links,
        # next to the checkpoint for auditability without loading encoder.pth.
        (output / "config.yaml").write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

        seed = int(run_cfg["seed"])
        torch.manual_seed(seed)
        device_name = str(run_cfg["device"])
        device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
        manifest_path = audit_paths(config)["manifest"]
        full_manifest = DataManifest.read_json(manifest_path)
        include_labeled = bool(pretrain_cfg.get("include_labeled_sequences", False))
        labels = None
        if include_labeled:
            artifacts = audit_paths(config)
            _, labels, _ = load_audited_artifacts(
                artifacts["manifest"], artifacts["label_manifest"],
                artifacts["leakage_audit"], artifacts["axis_stats"],
                protocol=run_cfg["protocol"], split=run_cfg["split"],
                label_budget=run_cfg["label_budget"],
            )
        required_keys = labels.selected_keys if include_labeled else ()
        manifest = make_small_pretrain_manifest(
            full_manifest, fraction=fraction, seed=seed, required_keys=required_keys
        )
        manifest_path_out = output / "pretrain_manifest.json"
        manifest.write_json(manifest_path_out)
        signature = {
            "protocol": manifest.protocol,
            "split": manifest.split,
            "scope": manifest.scope,
            "seed": seed,
            "manifest_fingerprint": manifest.fingerprint(),
            "input": config["input"],
            "model": model_cfg,
            "mask": mask_cfg,
            "pretrain": pretrain_cfg,
        }
        if pose_supervision:
            signature["mode"] = mode
            signature["pose_pretrain"] = pose_cfg
        if labels is not None:
            signature["label_manifest_fingerprint"] = labels.fingerprint
        final_path = output / "encoder.pth"
        if final_path.is_file():
            completed = torch.load(final_path, map_location="cpu", weights_only=True)
            if completed.get("status") == "complete":
                saved_signature = completed.get("signature")
                if saved_signature is not None:
                    matches = saved_signature == signature
                else:
                    # Legacy complete checkpoints predate signatures; compare only
                    # pretraining inputs, not unrelated fine-tuning settings.
                    saved_config = completed.get("config", {})
                    saved_experiment = saved_config.get("experiment", {})
                    matches = (
                        completed.get("manifest_fingerprint") == manifest.fingerprint()
                        and all(saved_experiment.get(key) == run_cfg.get(key)
                                for key in ("protocol", "scope", "split", "seed"))
                        and all(saved_config.get(key) == config.get(key)
                                for key in ("input", "model", "mask", "pretrain"))
                    )
                if not matches:
                    raise ValueError(f"Completed checkpoint does not match this run: {final_path}")
                print(f"[SKIP] split={manifest.split} already complete: {final_path}", flush=True)
                return

        summary = selection_summary(manifest)
        dataset_cls = WiFiJEPAPoseData if pose_supervision else WiFiJEPAData
        dataset_kwargs = {"pose_keys": labels.selected_keys} if pose_supervision else {}
        dataset = dataset_cls(
            str((ROOT / run_cfg["dataset_root"]).resolve())
            if not Path(run_cfg["dataset_root"]).is_absolute()
            else run_cfg["dataset_root"],
            manifest.pretrain_keys,
            protocol=manifest.protocol,
            split=manifest.split,
            scope=manifest.scope,
            manifest_fingerprint=manifest.fingerprint(),
            normalize=not bool(config["input"].get("disable_normalization", False)),
            **dataset_kwargs,
        )
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            dataset,
            batch_size=int(pretrain_cfg["batch_size"]),
            shuffle=True,
            num_workers=int(run_cfg.get("num_workers", 0)),
            drop_last=True,
            collate_fn=collate_pose_anchors if pose_supervision else collate_anchors,
            generator=generator,
        )
        model = WiFiJEPA(
            **model_cfg,
            n_time=config["input"]["time"],
            n_links=config["input"]["links"],
            n_masked_links=mask_cfg["masked_links"],
        ).to(device)
        pose_head = None
        if pose_supervision:
            # Keep the JEPA initialization and random mask stream matched to the control.
            with torch.random.fork_rng():
                torch.manual_seed(seed + 1)
                pose_head = PoseHead(model_cfg["embed_dim"], dropout_p=0.0).to(device)
        trainable = [p for p in model.parameters() if p.requires_grad]
        if pose_head is not None:
            trainable.extend(pose_head.parameters())
        optimizer = torch.optim.AdamW(
            trainable,
            lr=float(pretrain_cfg["learning_rate"]),
            weight_decay=float(pretrain_cfg["weight_decay"]),
        )
        batches_per_epoch = len(loader)
        total_steps = max(1, epochs * batches_per_epoch)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=float(pretrain_cfg["lr_min"])
        )
        use_amp = bool(pretrain_cfg.get("amp", True)) and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        step = 0
        start_epoch = 0
        metrics = []
        latest_path = output / "latest.pth"
        if latest_path.is_file():
            latest = torch.load(latest_path, map_location=device, weights_only=True)
            if latest.get("signature") != signature:
                raise ValueError(f"?? checkpoint ????????: {latest_path}")
            model.load_state_dict(latest["model_state_dict"])
            if pose_head is not None:
                if "pose_head_state_dict" not in latest:
                    raise ValueError(f"Pose-JEPA resume checkpoint has no pose head: {latest_path}")
                pose_head.load_state_dict(latest["pose_head_state_dict"], strict=True)
            optimizer.load_state_dict(latest["optimizer_state_dict"])
            scheduler.load_state_dict(latest["scheduler_state_dict"])
            scaler.load_state_dict(latest["scaler_state_dict"])
            start_epoch = int(latest["next_epoch"])
            step = int(latest["step"])
            metrics = latest["metrics"]
            generator.set_state(latest["loader_rng_state"].cpu())
            torch.set_rng_state(latest["torch_rng_state"].cpu())
            if device.type == "cuda" and latest.get("cuda_rng_states") is not None:
                torch.cuda.set_rng_state_all([state.cpu() for state in latest["cuda_rng_states"]])
            if start_epoch < 0 or start_epoch > epochs or len(metrics) != start_epoch:
                raise ValueError(f"?? checkpoint epoch/metrics ???: {latest_path}")
            print(f"[RESUME] split={manifest.split} ? epoch {start_epoch + 1}/{epochs} ??", flush=True)
        cuda_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "not used"
        print(
            f"[WiFi-JEPA pretrain] split={manifest.split} protocol={manifest.protocol} "
            f"scope={manifest.scope} seed={seed} device={device} ({cuda_name}) "
            f"amp={use_amp} sequences={summary['pretrain_sequences']} frames={len(dataset)} "
            f"batches/epoch={batches_per_epoch} fraction={fraction} output={output}",
            flush=True,
        )

        for epoch in range(start_epoch, epochs):
            epoch_started = time.perf_counter()
            model.train()
            if pose_head is not None:
                pose_head.train()
            total_loss = 0.0
            total_jepa_loss = 0.0
            total_pose_loss = 0.0
            supervised_frames = 0
            batches = 0
            for batch in loader:
                if pose_supervision:
                    batch, pose_targets, labeled = batch
                    batch = batch.to(device)
                    pose_targets = pose_targets.to(device)
                    labeled = labeled.to(device)
                else:
                    batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    if pose_head is not None:
                        jepa_loss, _, context = model(batch, return_context=True)
                        pose_count = int(labeled.sum())
                        if pose_count:
                            pose_target = pose_targets[labeled]
                            validate_pose_layout_batch(pose_target)
                        if pose_count >= 2:
                            pose_prediction = pose_head(context[labeled])
                            pose_loss = F.mse_loss(pose_prediction, pose_target)
                            supervised_frames += pose_count
                        else:
                            pose_loss = jepa_loss.new_zeros(())
                        loss = jepa_loss + pose_weight * pose_loss
                    else:
                        jepa_loss, _ = model(batch)
                        pose_loss = jepa_loss.new_zeros(())
                        loss = jepa_loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                step += 1
                momentum = cosine_momentum(
                    step - 1,
                    total_steps,
                    float(pretrain_cfg["ema_start"]),
                    float(pretrain_cfg["ema_end"]),
                )
                model.update_target(momentum)
                total_loss += float(loss.detach())
                total_jepa_loss += float(jepa_loss.detach())
                total_pose_loss += float(pose_loss.detach())
                batches += 1
            if not batches:
                raise RuntimeError("??????? batch")
            if pose_supervision and not supervised_frames:
                raise RuntimeError("Pose-JEPA epoch contained no usable labeled frames")
            row = {
                "epoch": epoch + 1,
                "loss": total_loss / batches,
                "jepa_loss": total_jepa_loss / batches,
                "pose_loss": total_pose_loss / batches,
                "supervised_frames": supervised_frames,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "optimizer_steps": step,
            }
            metrics.append(row)
            pose_metrics = (
                f" jepa={row['jepa_loss']:.5f} pose={row['pose_loss']:.5f} "
                f"labeled={supervised_frames}"
                if pose_supervision else ""
            )
            print(
                f"[epoch {epoch + 1:03d}/{epochs:03d}] loss={row['loss']:.5f}{pose_metrics} "
                f"lr={row['learning_rate']:.3g} steps={step}/{total_steps} "
                f"elapsed={time.perf_counter() - epoch_started:.1f}s",
                flush=True,
            )
            scheduler.step()
            latest = {
                "signature": signature,
                "next_epoch": epoch + 1,
                "step": step,
                "model_state_dict": model.state_dict(),
                "pose_head_state_dict": pose_head.state_dict() if pose_head is not None else None,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "metrics": metrics,
                "loader_rng_state": generator.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            }
            temp_latest = latest_path.with_suffix(".tmp")
            torch.save(latest, temp_latest)
            os.replace(temp_latest, latest_path)

        checkpoint = {
            "method": "wifi_jepa_amp_pose" if pose_supervision else "wifi_jepa_amp",
            "encoder_arch": "structured_vit",
            "status": "complete",
            "seed": seed,
            "epochs": epochs,
            "optimizer_steps": step,
            "pretrain_sequences": summary["pretrain_sequences"],
            "pretrain_frames": len(dataset),
            "manifest": str((output / "pretrain_manifest.json").resolve()),
            "manifest_fingerprint": manifest.fingerprint(),
            "signature": signature,
            "config": config,
            "pose_pretrain": pose_cfg if pose_supervision else None,
            "label_manifest_fingerprint": labels.fingerprint if labels else None,
            "encoder_state_dict": model.export_encoder_state_dict(),
            "metrics": metrics,
        }
        torch.save(checkpoint, output / "encoder.pth")
        (output / "pretrain_metrics.yaml").write_text(
            yaml.safe_dump(metrics, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        print(f"[DONE] encoder checkpoint: {output / 'encoder.pth'}", flush=True)

def main() -> None:
    parser = argparse.ArgumentParser(description="WiFi-JEPA MM-Fi")
    parser.add_argument("--config", default="WIFIJEPA/configs/matched_small.yaml")
    parser.add_argument(
        "--split", action="append", dest="splits",
        help="??????????????????? split",
    )
    args = parser.parse_args()
    config = load_config(Path(args.config))
    splits = args.splits or [config["experiment"]["split"]]
    if len(splits) != len(set(splits)):
        parser.error("--split ????")
    for split in splits:
        split_config = {**config, "experiment": {**config["experiment"], "split": split}}
        run_split(split_config)



if __name__ == "__main__":
    main()
