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
from WIFIJEPA.src.model import JointQueryPoseModel, StructuredPoseModel
from WIFIJEPA.src.paths import (
    MAE_ENCODER_CONTRACT,
    audit_paths,
    finetune_output,
    pretrain_checkpoint,
    pretrain_manifest,
    repo_path,
)
from WIFIJEPA.cross_sence.models import SharedStructuredPoseModel


class AbsoluteWithRelativeAuxLoss(nn.Module):
    """Keep absolute coordinates primary while softly supervising joint-0 pose."""

    def __init__(self, relative_weight: float):
        super().__init__()
        if relative_weight < 0:
            raise ValueError("relative_aux_weight must be >= 0")
        self.relative_weight = float(relative_weight)
        self.mse = nn.MSELoss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        absolute = self.mse(prediction, target)
        if self.relative_weight == 0:
            return absolute
        pred_relative = prediction - prediction[:, :1, :]
        target_relative = target - target[:, :1, :]
        return absolute + self.relative_weight * self.mse(pred_relative, target_relative)


def sha256(value: object) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode()).hexdigest()


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def config_path(value: str) -> str:
    path = Path(value)
    return str((ROOT / path).resolve()) if not path.is_absolute() else str(path)


def build_wifi_jepa_config(
    method: str,
    model_cfg: dict,
    input_cfg: dict,
    source_config: dict,
    mode: str,
    checkpoint_path: Path | None,
) -> dict:
    result = {
        "method": method,
        "model": model_cfg,
        "input": input_cfg,
        # Keep the pretraining mask in the saved finetune config so runs
        # with different masked_links values remain distinguishable.
        "mask": dict(source_config.get("mask", {})) if mode in {"jepa", "pose_jepa"} else {
            "mode": None,
            "masked_links": None,
        },
        "pretrain": dict(source_config.get("pretrain", {})) if mode in {"jepa", "pose_jepa", "mae"} else None,
        "pretrain_checkpoint": str(Path(checkpoint_path).resolve()) if checkpoint_path else None,
        "mae": dict(source_config.get("mae", {})) if mode == "mae" else None,
    }
    variant = source_config.get("pretrain", {}).get("variant") if mode in {"jepa", "pose_jepa"} else None
    if variant:
        result["model_variant"] = variant
    if mode == "pose_jepa":
        result["pose_pretrain"] = dict(source_config.get("pose_pretrain", {}))
    return result


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
        "pose_layout": "mmfi_3d_h36m_root0_v1",
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


def run_split(config: dict, config_file: str) -> None:
        run_cfg = config["experiment"]
        finetune = config.get("finetune", {})
        epochs = int(finetune.get("epochs", 25))
        mode = str(run_cfg.get("mode", "supervised"))
        if mode not in {"supervised", "jepa", "pose_jepa", "mae"}:
            raise ValueError("experiment.mode must be one of: supervised, jepa, pose_jepa, mae")

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
        checkpoint_path = pretrain_checkpoint(config) if mode in {"jepa", "pose_jepa", "mae"} else None
        pretrain_manifest_path = pretrain_manifest(config) if mode in {"jepa", "pose_jepa", "mae"} else None
        pretrain_boundary_manifest = pretraining_manifest_for_finetune(
            manifest, Path(pretrain_manifest_path) if pretrain_manifest_path else None
        )

        pretrain = None
        method = "sup_structured"
        pretrain_cfg = None
        if mode in {"jepa", "pose_jepa", "mae"}:
            if not checkpoint_path or not pretrain_manifest_path:
                raise ValueError(f"{mode} requires experiment.pretrain_checkpoint and experiment.pretrain_manifest")
            pretrain = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if pretrain.get("manifest_fingerprint") != pretrain_boundary_manifest.fingerprint():
                raise ValueError("pretrain checkpoint manifest does not match the fine-tuning boundary")
            saved_pretrain_config = pretrain.get("config", {})
            if bool(saved_pretrain_config.get("pretrain", {}).get("include_labeled_sequences", False)):
                if pretrain.get("label_manifest_fingerprint") != labels.fingerprint:
                    raise ValueError("pretrain checkpoint label manifest does not match fine-tuning labels")
            if mode == "pose_jepa":
                if not labels.selected_keys.issubset(pretrain_boundary_manifest.pretrain_keys):
                    raise ValueError("Pose-JEPA pretraining manifest omits few-shot labeled sequences")
                if pretrain.get("label_manifest_fingerprint") != labels.fingerprint:
                    raise ValueError("Pose-JEPA checkpoint was trained with a different label manifest")
                if pretrain.get("pose_pretrain") != config.get("pose_pretrain", {}):
                    raise ValueError("Pose-JEPA checkpoint loss settings do not match this config")
            expected_method = {
                "jepa": "wifi_jepa_amp",
                "pose_jepa": "wifi_jepa_amp_pose",
                "mae": "vit_mae_structured",
            }[mode]
            if pretrain.get("method") != expected_method or pretrain.get("encoder_arch") != "structured_vit":
                raise ValueError(f"{mode} checkpoint must be a {expected_method} StructuredViT checkpoint")
            if mode == "mae" and pretrain.get("encoder_contract") != MAE_ENCODER_CONTRACT:
                raise ValueError(
                    "MAE checkpoint does not train CLS through reconstruction; "
                    f"expected encoder_contract={MAE_ENCODER_CONTRACT!r}"
                )
            method = expected_method
            pretrain_cfg = pretrain.get("config", {})
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
        engine_config["wifi_jepa"] = build_wifi_jepa_config(
            method,
            model_cfg,
            input_cfg,
            source_config,
            mode,
            checkpoint_path,
        )
        relative_aux_weight = float(finetune.get("relative_aux_weight", 0.0))
        engine_config["relative_aux_weight"] = relative_aux_weight
        pose_head = str(finetune.get("pose_head", "mlp"))
        encoder_readout = str(finetune.get("encoder_readout", "cls"))
        if pose_head not in {"mlp", "joint_query"}:
            raise ValueError("finetune.pose_head must be 'mlp' or 'joint_query'")
        if encoder_readout not in {"cls", "shared_mean"}:
            raise ValueError("finetune.encoder_readout must be 'cls' or 'shared_mean'")
        if encoder_readout == "shared_mean" and pose_head != "mlp":
            raise ValueError("shared_mean readout currently supports pose_head: mlp only")
        factorized = bool(pretrain and pretrain.get("model_variant") == "factorized_link_shared_v1")
        if factorized and encoder_readout == "shared_mean" and "shared_projection_state_dict" not in pretrain:
            raise ValueError("factorized checkpoint has no shared_projection_state_dict")
        engine_config["pose_head"] = pose_head
        engine_config["encoder_readout"] = encoder_readout
        query_cfg = {
            "n_joints": int(finetune.get("n_joints", 17)),
            "query_depth": int(finetune.get("query_depth", 2)),
            "query_heads": int(finetune.get("query_heads", model_cfg["num_heads"])),
        }
        engine_config["query_decoder"] = query_cfg
        identity = RunIdentity(
            method=method,
            encoder_arch=(
                "structured_vit_joint_query" if pose_head == "joint_query" else
                "structured_vit_shared_mean" if encoder_readout == "shared_mean" else
                "structured_vit"
            ),
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
        result_dir = finetune_output(config).parent / (finetune_output(config).name + "-root0v1")
        done_path = result_dir / "done.txt"
        final_report_path = result_dir / "final_report.json"
        if done_path.is_file() and final_report_path.is_file():
            try:
                done = json.loads(done_path.read_text(encoding="utf-8"))
                report = json.loads(final_report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                done = report = {}
            if (
                done.get("status") == "complete"
                and done.get("final_report") == "final_report.json"
                and report.get("primary_checkpoint_role") == "absolute"
                and "absolute" in report.get("roles", {})
            ):
                print(f"[SKIP] split={run_cfg['split']} ????????: {result_dir}", flush=True)
                return
        resume = (result_dir / "last_state.pth").is_file()
        prepare_run_directory(result_dir, identity, resume=resume)
        print(
            f"[{'RESUME' if resume else 'START'}] finetune split={run_cfg['split']} "
            f"protocol={run_cfg['protocol']} seed={run_cfg['seed']} mode={mode} output={result_dir}",
            flush=True,
        )

        def factory(*, dropout_p: float, target_space: str) -> nn.Module:
            if encoder_readout == "shared_mean":
                model = SharedStructuredPoseModel(dropout_p=dropout_p, **encoder_cfg)
            else:
                model_cls = JointQueryPoseModel if pose_head == "joint_query" else StructuredPoseModel
                model = model_cls(dropout_p=dropout_p, **query_cfg, **encoder_cfg) if pose_head == "joint_query" else model_cls(
                    dropout_p=dropout_p, **encoder_cfg
                )
            if pretrain is not None:
                model.encoder.load_state_dict(pretrain["encoder_state_dict"], strict=True)
                if encoder_readout == "shared_mean" and "shared_projection_state_dict" in pretrain:
                    model.shared_projection.load_state_dict(
                        pretrain["shared_projection_state_dict"], strict=True
                    )
            return model

        device_name = str(run_cfg["device"])
        device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
        setup_cuda(device)
        normalize = not bool(input_cfg.get("disable_normalization", False))
        workers = int(run_cfg.get("num_workers", 0)) if normalize else 0
        with csi_preprocessing(normalize):
            criterion = (
                AbsoluteWithRelativeAuxLoss(relative_aux_weight)
                if relative_aux_weight > 0 else None
            )
            train_one_experiment(
                config_path(run_cfg["dataset_root"]),
                engine_config,
                str(result_dir),
                device,
                num_workers=workers,
                use_amp=bool(finetune.get("amp", True)) and not bool(finetune.get("no_amp", False)),
                val_every=int(finetune.get("val_every", 5)),
                resume=resume,
                model_factory=factory,
                criterion=criterion,
                data_manifest=manifest,
                label_manifest=labels,
                run_identity=identity,
                experiment_context={
                    "source_config": str(Path(config_file).resolve()),
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
        run_split(split_config, args.config)



if __name__ == "__main__":
    main()
