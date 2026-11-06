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
from scripts.metafi_ssl.train_supervised import load_audited_artifacts, pretraining_manifest_for_finetune
from WIFIJEPA.src.data import csi_preprocessing
from WIFIJEPA.src.paths import audit_paths, pretrain_checkpoint, pretrain_manifest, pretrain_output, repo_path
from model import SceneDecoupledPoseModel, build_encoder_kwargs


class AbsoluteWithRelativeAuxLoss(nn.Module):
    def __init__(self, relative_weight: float):
        super().__init__()
        self.relative_weight = float(relative_weight)
        self.mse = nn.MSELoss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.mse(prediction, target)
        if self.relative_weight:
            loss = loss + self.relative_weight * self.mse(
                prediction - prediction[:, :1], target - target[:, :1]
            )
        return loss


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8-sig"))


def full_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def build_engine_config(config: dict) -> dict:
    baseline = repo_path(config["experiment"].get("baseline_config", "configs/baseline_config.yaml"))
    engine = yaml.safe_load(baseline.read_text(encoding="utf-8-sig"))
    run = config["experiment"]
    ft = config["finetune"]
    engine.update({
        "preload_packed_csi": False,
        "preload_in_workers": False,
        "protocol": run["protocol"],
        "split_to_use": run["split"],
        "init_rand_seed": int(run["seed"]),
        "target_space": "absolute",
        "pose_layout": "mmfi_3d_h36m_root0_v1",
        "num_epochs": int(ft["epochs"]),
        "fine_tune_strategy": "matched",
        "modality": "wifi-csi",
        "data_unit": "frame",
        "amp_dtype": "fp16",
        "use_compile": False,
        "optimizer": ft.get("optimizer", "adamw"),
        "learning_rate": float(ft.get("learning_rate", 3e-4)),
        "weight_decay": float(ft.get("weight_decay", 0.0)),
        "scheduler": ft.get("scheduler", "cosine"),
        "lr_min": float(ft.get("lr_min", 1e-6)),
        "dropout_p": float(ft.get("dropout_p", 0.0)),
        "selection_metric": ft.get("selection_metric", "mpjpe"),
        "early_stopping_patience": int(ft.get("early_stopping_patience", 10)),
        "use_epoch_patience": bool(ft.get("use_epoch_patience", False)),
        "train_loader": {"batch_size": int(ft.get("batch_size", 32))},
        "validation_loader": {"batch_size": int(ft.get("validation_batch_size", 32))},
    })
    return engine


def run(config: dict, source_config: Path) -> Path:
    run_cfg = config["experiment"]
    ft = config["finetune"]
    artifacts = audit_paths(config)
    manifest, labels, _ = load_audited_artifacts(
        artifacts["manifest"], artifacts["label_manifest"], artifacts["leakage_audit"], artifacts["axis_stats"],
        protocol=run_cfg["protocol"], split=run_cfg["split"], label_budget=run_cfg["label_budget"],
    )
    checkpoint_path = Path(run_cfg.get("pretrain_checkpoint", "")) if run_cfg.get("pretrain_checkpoint") else pretrain_checkpoint(config)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (ROOT / checkpoint_path).resolve()
    manifest_path = Path(run_cfg.get("pretrain_manifest", "")) if run_cfg.get("pretrain_manifest") else pretrain_manifest(config)
    if not manifest_path.is_absolute():
        manifest_path = (ROOT / manifest_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"pretraining checkpoint not found: {checkpoint_path}; run pretrain.py first")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("method") != "scene_decoupled_jepa" or checkpoint.get("encoder_arch") != "scene_decoupled_graph":
        raise ValueError("checkpoint is not a Scene-Decoupled Geometric JEPA encoder")
    pretrain_boundary = pretraining_manifest_for_finetune(manifest, manifest_path)
    if checkpoint.get("manifest_fingerprint") != pretrain_boundary.fingerprint():
        raise ValueError("pretraining checkpoint manifest does not match downstream boundary")

    model_cfg = dict(config["model"])
    input_cfg = config["input"]
    encoder_kwargs = build_encoder_kwargs(model_cfg, input_cfg)
    query_cfg = {
        "n_joints": int(ft.get("n_joints", 17)),
        "query_depth": int(ft.get("query_depth", 2)),
        "query_heads": int(ft.get("query_heads", model_cfg["num_heads"])),
    }
    engine_config = build_engine_config(config)
    engine_config["pose_head"] = "scene_decoupled_joint_query"
    engine_config["encoder_readout"] = "body_layout_joint_query"
    engine_config["wifi_jepa"] = {
        "method": "scene_decoupled_jepa",
        "encoder_arch": "scene_decoupled_graph",
        "model_variant": checkpoint.get("model_variant"),
        "pretrain_checkpoint": str(checkpoint_path.resolve()),
        "pretrain_manifest": str(manifest_path.resolve()),
        "model": model_cfg,
        "input": input_cfg,
    }
    loss_name = str(ft.get("loss_name", "mse"))
    identity_config = dict(engine_config)
    identity_config["scene_decoupled"] = config
    identity = RunIdentity(
        method="scene_decoupled_jepa",
        encoder_arch="scene_decoupled_graph",
        protocol=manifest.protocol,
        split=manifest.split,
        data_scope=manifest.scope,
        pretrain_seed=int(checkpoint.get("seed", run_cfg["seed"])),
        finetune_seed=int(run_cfg["seed"]),
        label_budget=run_cfg["label_budget"],
        loss_name=loss_name,
        fine_tune_strategy="matched",
        config_fingerprint=full_sha256(identity_config),
        manifest_fingerprint=manifest.fingerprint(),
        label_manifest_fingerprint=labels.fingerprint,
    )
    root = repo_path(run_cfg.get("finetune_root", "WIFIJEPA/Scene-Decoupled_Geometric_JEPA/new_result"))
    result_dir = root / "scene_decoupled_jepa" / manifest.protocol / manifest.split / f"seed{run_cfg['seed']}" / run_cfg["label_budget"] / (full_sha256(identity.to_dict())[:10] + "-root0v1")
    resume = (result_dir / "last_state.pth").is_file()
    prepare_run_directory(result_dir, identity, resume=resume)

    def factory(*, dropout_p: float, target_space: str):
        model = SceneDecoupledPoseModel(
            dropout_p=dropout_p,
            **query_cfg,
            **encoder_kwargs,
        )
        model.encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
        return model

    device_name = str(run_cfg["device"])
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    setup_cuda(device)
    normalize = not bool(input_cfg.get("disable_normalization", False))
    workers = int(run_cfg.get("num_workers", 0)) if normalize else 0
    criterion = AbsoluteWithRelativeAuxLoss(float(ft.get("relative_aux_weight", 0.5)))
    print(f"[scene-decoupled finetune] device={device} output={result_dir}", flush=True)
    with csi_preprocessing(normalize):
        train_one_experiment(
            str(repo_path(run_cfg["dataset_root"])), engine_config, str(result_dir), device,
            num_workers=workers,
            use_amp=bool(ft.get("amp", True)),
            val_every=int(ft.get("val_every", 5)),
            resume=resume,
            model_factory=factory,
            criterion=criterion,
            data_manifest=manifest,
            label_manifest=labels,
            run_identity=identity,
            experiment_context={
                "source_config": str(source_config.resolve()),
                "pretrain_checkpoint": str(checkpoint_path.resolve()),
                "pretrain_manifest": str(manifest_path.resolve()),
            },
            experiment_artifacts={
                "data_manifest.json": str(artifacts["manifest"]),
                "leakage_audit.json": str(artifacts["leakage_audit"]),
                "label_stats.json": str(artifacts["axis_stats"]),
                "pretrain_manifest.json": str(manifest_path),
            },
        )
    return result_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="WIFIJEPA/Scene-Decoupled_Geometric_JEPA/config.yaml")
    args = parser.parse_args()
    run(load_config(Path(args.config)), Path(args.config))


if __name__ == "__main__":
    main()
