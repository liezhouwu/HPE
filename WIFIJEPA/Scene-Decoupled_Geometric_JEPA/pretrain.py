from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader, Dataset
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mmfi_wifi.data_manifest import DataManifest
from mmfi_wifi.provenance import build_run_provenance, write_run_provenance
from pose_ssl.metafi.pretrain_subset import make_small_pretrain_manifest, selection_summary
from WIFIJEPA.src.data import WiFiJEPAData, csi_preprocessing
from WIFIJEPA.src.paths import audit_paths, pretrain_output
from WIFIJEPA.cross_sence.pretrain_experiments import load_config
from model import SceneDecoupledJEPA, build_encoder_kwargs


class SceneFrameDataset(Dataset):
    def __init__(self, dataset_root: str, sequence_keys, *, protocol: str, split: str, scope: str, manifest_fingerprint: str, normalize: bool):
        self.base = WiFiJEPAData(
            dataset_root, sequence_keys, protocol=protocol, split=split, scope=scope,
            manifest_fingerprint=manifest_fingerprint, normalize=normalize,
        )
        scenes = sorted({record.sequence_key.scene for record in self.base._records})
        self.scene_to_id = {scene: index for index, scene in enumerate(scenes)}
        if len(self.scene_to_id) < 2:
            raise ValueError("scene-decoupled pretraining needs at least two source scenes")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        record = self.base._records[index]
        sample = self.base[index]
        return sample.anchor, self.scene_to_id[record.sequence_key.scene]


def collate(samples):
    x = torch.stack([sample[0] for sample in samples])
    scene = torch.tensor([sample[1] for sample in samples], dtype=torch.long)
    return x, scene


def cosine_momentum(step: int, total_steps: int, start: float, end: float) -> float:
    if total_steps <= 1:
        return end
    progress = min(step / (total_steps - 1), 1.0)
    return end - (end - start) * (math.cos(math.pi * progress) + 1.0) / 2.0


def signature(config: dict, manifest: DataManifest) -> dict:
    return {
        "method": "scene_decoupled_jepa",
        "model_variant": SceneDecoupledJEPA.model_variant,
        "manifest_fingerprint": manifest.fingerprint(),
        "input": config["input"],
        "model": config["model"],
        "mask": config["mask"],
        "pretrain": config["pretrain"],
    }


def run(config: dict, source_config: Path) -> Path:
    run_cfg = config["experiment"]
    pretrain_cfg = config["pretrain"]
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
    sig = signature(config, manifest)
    (output / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    provenance = build_run_provenance(
        result_dir=output,
        stage="pretrain",
        config=config,
        execution={
            "config_fingerprint": hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest(),
            "data_manifest_fingerprint": manifest.fingerprint(),
        },
        context={"source_config": str(source_config.resolve())},
        artifacts={"pretrain_manifest.json": str(manifest_path)},
        identity={
            "method": "scene_decoupled_jepa",
            "encoder_arch": "scene_decoupled_graph",
            "protocol": manifest.protocol,
            "split": manifest.split,
            "data_scope": manifest.scope,
            "pretrain_seed": int(run_cfg["seed"]),
            "label_budget": str(run_cfg.get("label_budget", "none")),
            "model_variant": SceneDecoupledJEPA.model_variant,
            "manifest_fingerprint": manifest.fingerprint(),
        },
    )
    write_run_provenance(output, provenance)

    seed = int(run_cfg["seed"])
    torch.manual_seed(seed)
    device_name = str(run_cfg["device"])
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    normalize = not bool(config["input"].get("disable_normalization", False))
    dataset_root = Path(run_cfg["dataset_root"])
    if not dataset_root.is_absolute():
        dataset_root = (ROOT / dataset_root).resolve()
    dataset = SceneFrameDataset(
        str(dataset_root), manifest.pretrain_keys,
        protocol=manifest.protocol, split=manifest.split, scope=manifest.scope,
        manifest_fingerprint=manifest.fingerprint(), normalize=normalize,
    )
    loader = DataLoader(
        dataset, batch_size=int(pretrain_cfg["batch_size"]), shuffle=True, drop_last=True,
        num_workers=int(run_cfg.get("num_workers", 0)), collate_fn=collate,
        generator=torch.Generator().manual_seed(seed),
    )
    model = SceneDecoupledJEPA(
        **build_encoder_kwargs(config["model"], config["input"]),
        scene_count=len(dataset.scene_to_id),
        predictor_depth=int(config["model"].get("predictor_depth", 1)),
        lambda_body=float(pretrain_cfg.get("lambda_body", 1.0)),
        lambda_adv=float(pretrain_cfg.get("lambda_adv", 0.05)),
        lambda_layout=float(pretrain_cfg.get("lambda_layout", 0.10)),
        lambda_orth=float(pretrain_cfg.get("lambda_orth", 0.01)),
        mask_links=int(pretrain_cfg.get("mask_links", config["mask"].get("masked_links", 1))),
    ).to(device)
    if not len(loader):
        raise RuntimeError("pretraining loader has no full batches")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(pretrain_cfg["learning_rate"]), weight_decay=float(pretrain_cfg.get("weight_decay", 0.01)),
    )
    epochs = int(pretrain_cfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=float(pretrain_cfg.get("lr_min", 1e-5))
    )
    use_amp = bool(pretrain_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    final_path = output / "encoder.pth"
    latest_path = output / "latest.pth"
    if final_path.is_file():
        completed = torch.load(final_path, map_location="cpu", weights_only=True)
        if completed.get("status") == "complete":
            if completed.get("signature") != sig:
                raise ValueError(f"completed checkpoint does not match config: {final_path}")
            print(f"[SKIP] already complete: {final_path}", flush=True)
            return final_path

    start_epoch = 0
    step = 0
    metrics = []
    if latest_path.is_file():
        latest = torch.load(latest_path, map_location=device, weights_only=True)
        if latest.get("signature") != sig:
            raise ValueError(f"resume checkpoint does not match config: {latest_path}")
        model.load_state_dict(latest["model_state_dict"])
        optimizer.load_state_dict(latest["optimizer_state_dict"])
        scheduler.load_state_dict(latest["scheduler_state_dict"])
        scaler.load_state_dict(latest["scaler_state_dict"])
        start_epoch = int(latest["next_epoch"])
        step = int(latest["step"])
        metrics = latest["metrics"]
        torch.set_rng_state(latest["torch_rng_state"].cpu())

    print(
        f"[scene-decoupled pretrain] device={device} scenes={dataset.scene_to_id} "
        f"sequences={selection_summary(manifest)['pretrain_sequences']} samples={len(dataset)} output={output}",
        flush=True,
    )
    with csi_preprocessing(normalize):
        for epoch in range(start_epoch, epochs):
            model.train()
            started = time.perf_counter()
            totals = {"loss": 0.0, "masked_loss": 0.0, "body_consistency": 0.0, "body_scene_acc": 0.0, "layout_scene_acc": 0.0, "orth_loss": 0.0, "masked_tokens": 0}
            batches = 0
            for x, scene in loader:
                x, scene = x.to(device, non_blocking=True), scene.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss, log = model(x, scene)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                step += 1
                model.update_target(cosine_momentum(step - 1, max(1, epochs * len(loader)), float(pretrain_cfg.get("ema_start", 0.996)), float(pretrain_cfg.get("ema_end", 1.0))))
                for key in totals:
                    totals[key] += float(log[key])
                batches += 1
            scheduler.step()
            row = {key: value / batches for key, value in totals.items()}
            row.update({"epoch": epoch + 1, "learning_rate": optimizer.param_groups[0]["lr"], "optimizer_steps": step})
            metrics.append(row)
            print(
                f"[epoch {epoch + 1:03d}/{epochs:03d}] loss={row['loss']:.5f} "
                f"masked={row['masked_loss']:.5f} body_acc={row['body_scene_acc']:.3f} "
                f"layout_acc={row['layout_scene_acc']:.3f} elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
            latest = {
                "signature": sig, "next_epoch": epoch + 1, "step": step,
                "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict(),
                "metrics": metrics, "torch_rng_state": torch.get_rng_state(),
            }
            temp = latest_path.with_suffix(".tmp")
            torch.save(latest, temp)
            os.replace(temp, latest_path)

    checkpoint = {
        "method": "scene_decoupled_jepa",
        "encoder_arch": "scene_decoupled_graph",
        "model_variant": SceneDecoupledJEPA.model_variant,
        "status": "complete",
        "seed": seed,
        "epochs": epochs,
        "optimizer_steps": step,
        "pretrain_sequences": selection_summary(manifest)["pretrain_sequences"],
        "pretrain_frames": len(dataset),
        "manifest": str(manifest_path.resolve()),
        "manifest_fingerprint": manifest.fingerprint(),
        "signature": sig,
        "config": config,
        "encoder_config": build_encoder_kwargs(config["model"], config["input"]),
        "scene_to_id": dataset.scene_to_id,
        "encoder_state_dict": model.export_encoder_state_dict(),
        "metrics": metrics,
        "run_id": provenance.get("run_id"),
        "run_label": provenance.get("run_label"),
        "provenance_file": "run_provenance.json",
    }
    torch.save(checkpoint, final_path)
    (output / "pretrain_metrics.yaml").write_text(yaml.safe_dump(metrics, sort_keys=False), encoding="utf-8")
    print(f"[DONE] encoder checkpoint: {final_path}", flush=True)
    return final_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="WIFIJEPA/Scene-Decoupled_Geometric_JEPA/config.yaml")
    args = parser.parse_args()
    path = Path(args.config)
    run(load_config(path), path)


if __name__ == "__main__":
    main()
