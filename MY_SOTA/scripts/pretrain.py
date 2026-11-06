from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mmfi_wifi.data_manifest import DataManifest
from WIFIJEPA.src.data import WiFiJEPAData
from MY_SOTA.src.manifest_adapter import records_to_environment_ids, validate_source_only_records
from MY_SOTA.src.model import EILJEPA


def collate(samples):
    anchors = torch.stack([sample.anchor for sample in samples])
    environment_ids = records_to_environment_ids(samples)
    return anchors, environment_ids


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain EIL-JEPA on an audited MM-Fi manifest.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", required=True, help="Audited or derived pretrain_manifest.json")
    parser.add_argument("--config", default=str(ROOT / "MY_SOTA/configs/eil_jepa_p3.yaml"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_cfg, model_cfg, loss_cfg = config["experiment"], config["model"], config["loss"]
    torch.manual_seed(int(run_cfg["seed"]))

    manifest = DataManifest.read_json(args.manifest)
    validate_source_only_records([type("R", (), {"sequence_key": key})() for key in manifest.pretrain_keys])
    dataset = WiFiJEPAData(
        args.dataset_root,
        manifest.pretrain_keys,
        protocol=manifest.protocol,
        split=manifest.split,
        scope=manifest.scope,
        manifest_fingerprint=manifest.fingerprint(),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    device_name = str(run_cfg.get("device", "cuda"))
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    model_loss_cfg = {
        key: loss_cfg[key]
        for key in ("invariant_weight", "environment_weight", "temporal_weight", "grl_scale")
        if key in loss_cfg
    }
    model = EILJEPA(**model_cfg, **model_loss_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
    epochs = int(args.epochs or run_cfg["pretrain_epochs"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")

    metrics = []
    for epoch in range(epochs):
        model.train()
        started = time.perf_counter()
        totals = {"loss": 0.0, "jepa_loss": 0.0, "invariance_loss": 0.0, "environment_loss": 0.0}
        batches = 0
        for anchors, environment_ids in loader:
            anchors = anchors.to(device)
            environment_ids = environment_ids.to(device)
            optimizer.zero_grad(set_to_none=True)
            result = model(anchors, environment_ids=environment_ids)
            result.loss.backward()
            optimizer.step()
            progress = epoch / max(1, epochs - 1)
            momentum = float(loss_cfg["target_momentum_start"]) + progress * (
                float(loss_cfg["target_momentum_end"]) - float(loss_cfg["target_momentum_start"])
            )
            model.update_target(momentum)
            batches += 1
            for key in totals:
                totals[key] += float(getattr(result, key).detach())
        if batches == 0:
            raise RuntimeError("no complete batches; reduce batch size or check the manifest")
        row = {"epoch": epoch + 1, **{key: value / batches for key, value in totals.items()},
               "seconds": time.perf_counter() - started}
        metrics.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    checkpoint = {
        "status": "complete",
        "method": "EIL-JEPA",
        "config": config,
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_fingerprint": manifest.fingerprint(),
        "encoder_state_dict": model.export_encoder_state_dict(),
        "metrics": metrics,
    }
    torch.save(checkpoint, output_dir / "encoder.pth")
    (output_dir / "pretrain_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[DONE] {output_dir / 'encoder.pth'}")


if __name__ == "__main__":
    main()

