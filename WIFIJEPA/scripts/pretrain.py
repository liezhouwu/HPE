from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mmfi_wifi.data_manifest import DataManifest
from WIFIJEPA.src.data import WiFiJEPAData
from pose_ssl.metafi.pretrain_subset import make_small_pretrain_manifest, selection_summary
from WIFIJEPA.src.model import WiFiJEPA
from WIFIJEPA.src.paths import audit_paths, pretrain_output, repo_path


def cosine_momentum(step: int, total_steps: int, start: float, end: float) -> float:
    if total_steps <= 1:
        return end
    progress = min(step / (total_steps - 1), 1.0)
    return end - (end - start) * (math.cos(math.pi * progress) + 1.0) / 2.0


def collate_anchors(samples):
    return torch.stack([sample.anchor for sample in samples])


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="WiFi-JEPA MM-Fi pre-training")
    parser.add_argument("--config", default="WIFIJEPA/configs/matched_small.yaml")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    run_cfg = config["experiment"]
    pretrain_cfg = config["pretrain"]
    model_cfg = config["model"]
    mask_cfg = config["mask"]
    epochs = int(pretrain_cfg["epochs"])
    fraction = float(pretrain_cfg["sequence_fraction"])
    output = pretrain_output(config)
    output.mkdir(parents=True, exist_ok=True)

    seed = int(run_cfg["seed"])
    torch.manual_seed(seed)
    device_name = str(run_cfg["device"])
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    manifest_path = audit_paths(config)["manifest"]
    full_manifest = DataManifest.read_json(manifest_path)
    manifest = make_small_pretrain_manifest(full_manifest, fraction=fraction, seed=seed)
    manifest.write_json(output / "pretrain_manifest.json")

    summary = selection_summary(manifest)
    dataset = WiFiJEPAData(
        str((ROOT / run_cfg["dataset_root"]).resolve())
        if not Path(run_cfg["dataset_root"]).is_absolute()
        else run_cfg["dataset_root"],
        manifest.pretrain_keys,
        protocol=manifest.protocol,
        split=manifest.split,
        scope=manifest.scope,
        manifest_fingerprint=manifest.fingerprint(),
        normalize=not bool(config["input"].get("disable_normalization", False)),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(pretrain_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(run_cfg.get("num_workers", 0)),
        drop_last=True,
        collate_fn=collate_anchors,
        generator=generator,
    )
    model = WiFiJEPA(
        **model_cfg,
        n_time=config["input"]["time"],
        n_links=config["input"]["links"],
        n_masked_links=mask_cfg["masked_links"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
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
    metrics = []
    print(
        f"[WiFi-JEPA] sequences={summary['pretrain_sequences']} "
        f"frames={len(dataset)} fraction={fraction} epochs={epochs}",
        flush=True,
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        batches = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss, _ = model(batch)
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
            batches += 1
        if not batches:
            raise RuntimeError("pre-training produced no batches; increase the pre-train set or reduce batch_size")
        row = {
            "epoch": epoch + 1,
            "loss": total_loss / batches,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "optimizer_steps": step,
        }
        metrics.append(row)
        print(f"E{epoch + 1:03d} loss={row['loss']:.5f} steps={step}", flush=True)
        scheduler.step()

    checkpoint = {
        "method": "wifi_jepa_amp",
        "encoder_arch": "structured_vit",
        "status": "complete",
        "seed": seed,
        "epochs": epochs,
        "optimizer_steps": step,
        "pretrain_sequences": summary["pretrain_sequences"],
        "pretrain_frames": len(dataset),
        "manifest": str((output / "pretrain_manifest.json").resolve()),
        "manifest_fingerprint": manifest.fingerprint(),
        "config": config,
        "encoder_state_dict": model.export_encoder_state_dict(),
        "metrics": metrics,
    }
    torch.save(checkpoint, output / "encoder.pth")
    (output / "pretrain_metrics.yaml").write_text(
        yaml.safe_dump(metrics, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    print(f"[DONE] encoder checkpoint: {output / 'encoder.pth'}", flush=True)


if __name__ == "__main__":
    main()
