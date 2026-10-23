from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pose_ssl.metafi.pretrain_subset import make_small_pretrain_manifest, selection_summary
from mmfi_wifi.data_manifest import DataManifest
from WIFIJEPA.scripts.pretrain import collate_anchors, cosine_momentum
from WIFIJEPA.src.data import WiFiJEPAData, csi_preprocessing
from WIFIJEPA.src.paths import audit_paths, pretrain_output
from WIFIJEPA.cross_sence.models import LinkFactorizedJEPA, TemporalLinkJEPA, TemporalWindowDataset


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8-sig"))


def _model_and_dataset(config: dict, manifest: DataManifest):
    run = config["experiment"]
    pretrain = config["pretrain"]
    inputs = config["input"]
    model_cfg = dict(config["model"])
    dataset_root = Path(run["dataset_root"])
    if not dataset_root.is_absolute():
        dataset_root = (ROOT / dataset_root).resolve()
    normalize = not bool(inputs.get("disable_normalization", False))
    common = {
        **model_cfg,
        "n_time": int(inputs["time"]),
        "n_links": int(inputs["links"]),
    }

    variant = str(pretrain.get("variant", "factorized_link_shared"))
    if variant == "factorized_link_shared":
        model = LinkFactorizedJEPA(
            **common,
            n_masked_links=int(config["mask"]["masked_links"]),
            residual_loss_weight=float(pretrain.get("residual_loss_weight", 1.0)),
            shared_std_floor=float(pretrain.get("shared_std_floor", 0.25)),
        )
        dataset = WiFiJEPAData(
            str(dataset_root), manifest.pretrain_keys,
            protocol=manifest.protocol, split=manifest.split, scope=manifest.scope,
            manifest_fingerprint=manifest.fingerprint(), normalize=normalize,
        )
        loader = DataLoader(
            dataset, batch_size=int(pretrain["batch_size"]), shuffle=True,
            num_workers=int(run.get("num_workers", 0)), drop_last=True,
            collate_fn=collate_anchors,
            generator=torch.Generator().manual_seed(int(run["seed"])),
        )
    elif variant == "temporal_link_joint":
        context_frames = int(pretrain.get("context_frames", 4))
        future_frames = int(pretrain.get("future_frames", 2))
        window_frames = context_frames + future_frames
        model = TemporalLinkJEPA(
            **common,
            window_frames=window_frames,
            future_frames=future_frames,
            masked_links=int(config["mask"]["masked_links"]),
            mask_mode=str(config["mask"].get("mode", "joint")),
        )
        dataset = TemporalWindowDataset(
            str(dataset_root), manifest.pretrain_keys,
            context_frames=context_frames,
            future_frames=future_frames,
            protocol=manifest.protocol,
            split=manifest.split,
            scope=manifest.scope,
            manifest_fingerprint=manifest.fingerprint(),
        )
        loader = DataLoader(
            dataset, batch_size=int(pretrain["batch_size"]), shuffle=True,
            num_workers=int(run.get("num_workers", 0)), drop_last=True,
            generator=torch.Generator().manual_seed(int(run["seed"])),
        )
    else:
        raise ValueError(
            "pretrain.variant must be factorized_link_shared or temporal_link_joint"
        )
    return model, dataset, loader, variant


def _signature(config: dict, manifest: DataManifest, variant: str) -> dict:
    return {
        "model_variant": variant,
        "manifest_fingerprint": manifest.fingerprint(),
        "input": config["input"],
        "model": config["model"],
        "mask": config["mask"],
        "pretrain": config["pretrain"],
    }


def run(config: dict) -> Path:
    run_cfg = config["experiment"]
    pretrain_cfg = config["pretrain"]
    if str(run_cfg.get("mode", "jepa")) != "jepa":
        raise ValueError("cross_sence pretraining configs must set experiment.mode: jepa")
    if int(run_cfg.get("num_workers", 0)) and bool(config["input"].get("disable_normalization", False)):
        raise ValueError("disable_normalization requires num_workers: 0 in this isolated runner")

    audit = audit_paths(config)
    full_manifest = DataManifest.read_json(audit["manifest"])
    manifest = make_small_pretrain_manifest(
        full_manifest,
        fraction=float(pretrain_cfg.get("sequence_fraction", 0.2)),
        seed=int(run_cfg["seed"]),
    )
    output = pretrain_output(config)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "pretrain_manifest.json"
    manifest.write_json(manifest_path)
    signature = _signature(config, manifest, str(pretrain_cfg.get("variant")))
    (output / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    seed = int(run_cfg["seed"])
    torch.manual_seed(seed)
    device_name = str(run_cfg["device"])
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    model, dataset, loader, variant = _model_and_dataset(config, manifest)
    model = model.to(device)
    if bool(pretrain_cfg.get("match_frame_compute_budget", False)) and variant == "temporal_link_joint":
        batch_size = int(pretrain_cfg["batch_size"])
        window_frames = int(pretrain_cfg["context_frames"]) + int(pretrain_cfg["future_frames"])
        steps_per_epoch = min(
            len(loader), max(1, math.ceil(len(dataset) / (batch_size * window_frames)))
        )
    else:
        steps_per_epoch = len(loader)
    if not steps_per_epoch:
        raise RuntimeError("pretraining loader has no full batches; lower batch_size")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(pretrain_cfg["learning_rate"]),
        weight_decay=float(pretrain_cfg.get("weight_decay", 0.01)),
    )
    epochs = int(pretrain_cfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=float(pretrain_cfg.get("lr_min", 1e-5))
    )
    use_amp = bool(pretrain_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    latest_path = output / "latest.pth"
    final_path = output / "encoder.pth"
    start_epoch = 0
    step = 0
    metrics: list[dict] = []
    generator = loader.generator

    if final_path.is_file():
        completed = torch.load(final_path, map_location="cpu", weights_only=True)
        if completed.get("status") == "complete":
            if completed.get("signature") != signature:
                raise ValueError(f"Completed checkpoint does not match config: {final_path}")
            print(f"[SKIP] already complete: {final_path}", flush=True)
            return final_path

    if latest_path.is_file():
        latest = torch.load(latest_path, map_location=device, weights_only=True)
        if latest.get("signature") != signature:
            raise ValueError(f"Resume checkpoint does not match config: {latest_path}")
        model.load_state_dict(latest["model_state_dict"], strict=True)
        optimizer.load_state_dict(latest["optimizer_state_dict"])
        scheduler.load_state_dict(latest["scheduler_state_dict"])
        scaler.load_state_dict(latest["scaler_state_dict"])
        generator.set_state(latest["loader_rng_state"].cpu())
        torch.set_rng_state(latest["torch_rng_state"].cpu())
        if device.type == "cuda" and latest.get("cuda_rng_states") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in latest["cuda_rng_states"]])
        start_epoch = int(latest["next_epoch"])
        step = int(latest["step"])
        metrics = latest["metrics"]
        if not 0 <= start_epoch <= epochs or len(metrics) != start_epoch:
            raise ValueError(f"Invalid resume epoch in {latest_path}")

    summary = selection_summary(manifest)
    print(
        f"[cross_sence JEPA] variant={variant} protocol={manifest.protocol} "
        f"split={manifest.split} seed={seed} sequences={summary['pretrain_sequences']} "
        f"samples={len(dataset)} batches/epoch={steps_per_epoch}/{len(loader)} "
        f"amp={use_amp} output={output}",
        flush=True,
    )

    normalize = not bool(config["input"].get("disable_normalization", False))
    with csi_preprocessing(normalize):
        for epoch in range(start_epoch, epochs):
            model.train()
            started = time.perf_counter()
            total_loss = total_shared = total_residual = 0.0
            total_masked = batches = 0
            for batch in loader:
                batch = batch.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss, log = model(batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                step += 1
                momentum = cosine_momentum(
                    step - 1, max(1, epochs * steps_per_epoch),
                    float(pretrain_cfg.get("ema_start", 0.996)),
                    float(pretrain_cfg.get("ema_end", 1.0)),
                )
                model.update_target(momentum)
                total_loss += float(loss.detach())
                total_shared += float(log.get("shared_loss", 0.0))
                total_residual += float(log.get("residual_loss", 0.0))
                total_masked += int(log["masked_tokens"])
                batches += 1
                if batches >= steps_per_epoch:
                    break
            scheduler.step()
            row = {
                "epoch": epoch + 1,
                "loss": total_loss / batches,
                "masked_tokens_per_batch": total_masked / batches,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "optimizer_steps": step,
            }
            if variant == "factorized_link_shared":
                row["shared_loss"] = total_shared / batches
                row["residual_loss"] = total_residual / batches
            metrics.append(row)
            print(
                f"[epoch {epoch + 1:03d}/{epochs:03d}] loss={row['loss']:.5f} "
                f"masked={row['masked_tokens_per_batch']:.1f} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
            if variant == "factorized_link_shared":
                print(
                    f"  shared={row['shared_loss']:.5f} residual={row['residual_loss']:.5f}",
                    flush=True,
                )
            latest = {
                "signature": signature,
                "next_epoch": epoch + 1,
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "metrics": metrics,
                "loader_rng_state": generator.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            }
            temp_path = latest_path.with_suffix(".tmp")
            torch.save(latest, temp_path)
            os.replace(temp_path, latest_path)

    checkpoint = {
        "method": "wifi_jepa_amp",
        "encoder_arch": "structured_vit",
        "model_variant": model.model_variant,
        "status": "complete",
        "seed": seed,
        "epochs": epochs,
        "optimizer_steps": step,
        "pretrain_sequences": summary["pretrain_sequences"],
        "pretrain_frames": len(dataset),
        "manifest": str(manifest_path.resolve()),
        "manifest_fingerprint": manifest.fingerprint(),
        "signature": signature,
        "config": config,
        "encoder_state_dict": model.export_encoder_state_dict(),
        "metrics": metrics,
    }
    if variant == "factorized_link_shared":
        checkpoint["shared_projection_state_dict"] = model.export_shared_projection_state_dict()
    torch.save(checkpoint, final_path)
    (output / "pretrain_metrics.yaml").write_text(
        yaml.safe_dump(metrics, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    print(f"[DONE] encoder checkpoint: {final_path}", flush=True)
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimental cross-scene WiFi-JEPA pretraining")
    parser.add_argument(
        "--config", default="WIFIJEPA/cross_sence/configs/factorized_link_shared.yaml"
    )
    args = parser.parse_args()
    run(load_config(Path(args.config)))


if __name__ == "__main__":
    main()
