from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
import yaml
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mmfi_wifi.engine import setup_cuda, train_one_experiment
from mmfi_wifi.run_identity import RunIdentity, prepare_run_directory
from scripts.metafi_ssl.train_supervised import (
    load_audited_artifacts,
    pretraining_manifest_for_finetune,
)
from WIFIJEPA.src.data import csi_preprocessing
from WIFIJEPA.src.model import StructuredPoseModel
from WIFIJEPA.src.paths import (
    audit_paths,
    finetune_output,
    pretrain_checkpoint,
    pretrain_manifest,
    repo_path,
)


def sha256(value: object) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode()).hexdigest()


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def config_path(value: str) -> str:
    path = Path(value)
    return str((ROOT / path).resolve()) if not path.is_absolute() else str(path)


def build_engine_config(config: dict, protocol: str, split: str, seed: int, epochs: int) -> dict:
    baseline_path = Path(config.get("baseline_config", "configs/baseline_config.yaml"))
    if not baseline_path.is_absolute():
        baseline_path = ROOT / baseline_path
    engine = yaml.safe_load(baseline_path.read_text(encoding="utf-8"))
    engine.update({
        "preload_packed_csi": False,
        "preload_in_workers": False,
        "protocol": protocol,
        "split_to_use": split,
        "init_rand_seed": seed,
        "target_space": "absolute",
        "num_epochs": epochs,
        "fine_tune_strategy": "matched",
        "modality": "wifi-csi",
        "data_unit": "frame",
        "amp_dtype": "fp16",
        "use_compile": False,
    })
    finetune = config.get("finetune", {})
    for key in (
        "optimizer", "learning_rate", "weight_decay", "sgd_momentum", "scheduler",
        "lr_warmup_epochs", "lr_min", "lr_milestones", "lr_gamma", "dropout_p",
        "selection_metric", "early_stopping_patience", "use_epoch_patience",
    ):
        if key in finetune:
            engine[key] = finetune[key]
    return engine


def main() -> None:
    parser = argparse.ArgumentParser(description="WiFi-JEPA MM-Fi fine-tuning")
    parser.add_argument("--config", default="WIFIJEPA/configs/matched_small.yaml")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    run_cfg = config["experiment"]
    finetune = config.get("finetune", {})
    epochs = int(finetune.get("epochs", 25))
    mode = str(run_cfg.get("mode", "supervised"))
    if mode not in {"supervised", "jepa"}:
        raise ValueError("experiment.mode must be supervised or jepa")

    artifacts = audit_paths(config)
    manifest, labels, _ = load_audited_artifacts(
        artifacts["manifest"],
        artifacts["label_manifest"],
        artifacts["leakage_audit"],
        artifacts["axis_stats"],
        protocol=run_cfg["protocol"],
        split=run_cfg["split"],
        label_budget=run_cfg["label_budget"],
    )
    checkpoint_path = pretrain_checkpoint(config) if mode == "jepa" else None
    pretrain_manifest_path = pretrain_manifest(config) if mode == "jepa" else None
    pretrain_boundary_manifest = pretraining_manifest_for_finetune(
        manifest, Path(pretrain_manifest_path) if pretrain_manifest_path else None
    )

    pretrain = None
    method = "sup_structured"
    pretrain_cfg = None
    if mode == "jepa":
        if not checkpoint_path or not pretrain_manifest_path:
            raise ValueError("jepa mode requires a pre-training checkpoint and manifest")
        pretrain = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if pretrain["manifest_fingerprint"] != pretrain_boundary_manifest.fingerprint():
            raise ValueError("pre-training manifest does not match the downstream experiment boundary")
        method = "wifi_jepa_amp"
        pretrain_cfg = pretrain["config"]

    source_config = pretrain_cfg or config
    model_cfg = dict(source_config["model"])
    input_cfg = source_config["input"]
    encoder_cfg = {
        "embed_dim": model_cfg["embed_dim"],
        "depth": model_cfg["depth"],
        "num_heads": model_cfg["num_heads"],
        "ffn_dim": model_cfg["ffn_dim"],
        "n_time": input_cfg["time"],
        "n_links": input_cfg["links"],
    }
    engine_config = build_engine_config(
        config,
        run_cfg["protocol"],
        run_cfg["split"],
        int(run_cfg["seed"]),
        epochs,
    )
    engine_config["wifi_jepa"] = {
        "method": method,
        "model": model_cfg,
        "input": input_cfg,
        "pretrain_checkpoint": str(Path(checkpoint_path).resolve()) if checkpoint_path else None,
    }
    identity = RunIdentity(
        method=method,
        encoder_arch="structured_vit",
        protocol=run_cfg["protocol"],
        split=run_cfg["split"],
        data_scope=manifest.scope,
        pretrain_seed=int(pretrain.get("seed", run_cfg["seed"])) if pretrain else int(run_cfg["seed"]),
        finetune_seed=int(run_cfg["seed"]),
        label_budget=run_cfg["label_budget"],
        loss_name=str(finetune.get("loss_name", "mse")),
        fine_tune_strategy="matched",
        config_fingerprint=sha256(engine_config),
        manifest_fingerprint=manifest.fingerprint(),
        label_manifest_fingerprint=labels.fingerprint,
    )
    result_dir = finetune_output(config)
    prepare_run_directory(result_dir, identity, resume=False)

    def factory(*, dropout_p: float, target_space: str) -> nn.Module:
        model = StructuredPoseModel(dropout_p=dropout_p, **encoder_cfg)
        if pretrain is not None:
            model.encoder.load_state_dict(pretrain["encoder_state_dict"], strict=True)
        return model

    device_name = str(run_cfg["device"])
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    setup_cuda(device)
    normalize = not bool(input_cfg.get("disable_normalization", False))
    workers = int(run_cfg.get("num_workers", 0)) if normalize else 0
    with csi_preprocessing(normalize):
        train_one_experiment(
            config_path(run_cfg["dataset_root"]),
            engine_config,
            str(result_dir),
            device,
            num_workers=workers,
            use_amp=bool(finetune.get("amp", True)) and not bool(finetune.get("no_amp", False)),
            val_every=int(finetune.get("val_every", 5)),
            model_factory=factory,
            data_manifest=manifest,
            label_manifest=labels,
            run_identity=identity,
            experiment_context={
                "source_config": str(Path(args.config).resolve()),
                "mode": mode,
                "pretrain_checkpoint": str(checkpoint_path) if checkpoint_path else None,
            },
            experiment_artifacts={
                "data_manifest.json": str(artifacts["manifest"]),
                "leakage_audit.json": str(artifacts["leakage_audit"]),
                "label_stats.json": str(artifacts["axis_stats"]),
                **({"pretrain_manifest.json": str(pretrain_manifest_path)}
                   if pretrain_manifest_path else {}),
            },
        )


if __name__ == "__main__":
    main()
