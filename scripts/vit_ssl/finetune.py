"""Audited 4shot fine-tuning for the ViT-csi-small backbone.

Two matched arms share this script:

* ``--pretrain-checkpoint <vit-mae latest.pth>`` -> **MAE-ViT** (encoder initialised
  from ViT-MAE pretraining);
* omitting it -> **Sup-ViT** (random initialisation control).

Runs use the engine's identity-bound MetaFi contract (audited manifest, label
manifest, ``run_identity.json``, transactional finalisation).  Outputs live in
``result_metafi_ssl/vit/runs/...``; the MetaFi method directories and the
small-sample workflow are untouched.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch
import yaml
from torch import Tensor, nn

_BASE = Path(__file__).resolve().parents[2]
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))

from mmfi_wifi.engine import setup_cuda, train_one_experiment
from mmfi_wifi.run_identity import RunIdentity, prepare_run_directory
from pose_ssl.model import ViTCSIEncoder, build_pose_model
from scripts.metafi_ssl.train_supervised import (
    load_audited_artifacts,
    pretraining_manifest_for_finetune,
)


ENCODER_ARCH = "vit_csi_small"
MAE_METHOD = "mae_vit"
SUP_METHOD = "sup_vit"
MATCHED_STRATEGY = "matched"
DEFAULT_BASELINE_CONFIG = "configs/baseline_config.yaml"


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune ViT-csi-small (MAE-ViT / Sup-ViT)")
    parser.add_argument("dataset_root")
    parser.add_argument("config_file")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--leakage-audit", default=None)
    parser.add_argument("--label-manifest", default=None)
    parser.add_argument("--axis-stats", default=None)
    parser.add_argument(
        "--audit-dir",
        default=None,
        help="审计目录（例如 .../audit/b2s）；给出时自动推导上面四条路径",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--label-budget", required=True)
    parser.add_argument("--pretrain-checkpoint", default=None)
    parser.add_argument("--pretrain-manifest", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--val-every", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args(argv)


def _load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"unable to read config file: {source}") from error
    if not isinstance(payload, dict):
        raise ValueError("config root must be a mapping")
    return payload


def _section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    section = config.get(name, {})
    if not isinstance(section, Mapping):
        raise ValueError(f"{name} config must be a mapping")
    return dict(section)


def _full_sha256(value: object) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


_AUDIT_ARTIFACTS = (
    ("manifest", "data_manifest.json"),
    ("leakage_audit", "leakage_audit.json"),
    ("label_manifest", "fewshot_manifest.json"),
    ("axis_stats", "label_stats.json"),
)


def _resolve_artifact_paths(args: argparse.Namespace) -> dict[str, str]:
    """Return the four audited artifact paths, derived from ``--audit-dir`` when given."""

    explicit = {name: getattr(args, name) for name, _ in _AUDIT_ARTIFACTS}
    if args.audit_dir:
        provided = sorted(name for name, value in explicit.items() if value)
        if provided:
            raise ValueError(
                "--audit-dir cannot be combined with explicit artifact paths: " + ", ".join(provided)
            )
        directory = Path(args.audit_dir)
        derived = {name: str(directory / filename) for name, filename in _AUDIT_ARTIFACTS}
        missing = sorted(path for path in derived.values() if not Path(path).is_file())
        if missing:
            raise FileNotFoundError(f"audit directory is incomplete: {missing}")
        return derived
    missing_names = sorted(name for name, value in explicit.items() if not value)
    if missing_names:
        raise ValueError(
            "audited artifact paths are required (or pass --audit-dir): " + ", ".join(missing_names)
        )
    return {name: str(value) for name, value in explicit.items()}


def _patch_size(config: Mapping[str, Any]) -> tuple[int, int]:
    """Return the configured ViT patch size (must match the pretraining run)."""

    section = config.get("vit_mae", {})
    if not isinstance(section, Mapping):
        raise ValueError("vit_mae config must be a mapping")
    patch_h = int(section.get("patch_h", 6))
    patch_w = int(section.get("patch_w", 5))
    if patch_h < 1 or patch_w < 1 or 114 % patch_h or 10 % patch_w:
        raise ValueError(f"invalid ViT patch size {patch_h}x{patch_w}")
    return patch_h, patch_w


def _validate_encoder_geometry(encoder_state: Mapping[str, Any], patch_size: tuple[int, int]) -> None:
    """Fail closed when the pretrained token grid does not match the configured patch size."""

    patch_h, patch_w = patch_size
    expected_tokens = (114 // patch_h) * (10 // patch_w) + 1  # + CLS
    for name, value in encoder_state.items():
        if name.endswith("pos_embed"):
            tokens = getattr(value, "shape", ())[1] if getattr(value, "ndim", 0) == 3 else None
            if tokens != expected_tokens:
                raise ValueError(
                    f"pretrained encoder token grid {None if tokens is None else tokens - 1} does not match "
                    f"the configured patch size {patch_h}x{patch_w} -> {expected_tokens - 1} tokens; "
                    "the fine-tuning config must use the same vit_mae.patch_h/patch_w as the pretraining"
                )


def _with_pretrain_provenance(
    engine_config: Mapping[str, Any], pretrain_identity: RunIdentity | None
) -> dict[str, Any]:
    """Fold the pretraining provenance into the engine config.

    The config is both what the engine stores as ``config.yaml`` and what the run
    identity hashes: without these keys two runs differing only in which encoder
    they started from would share one identity and one result directory.
    """

    config = dict(engine_config)
    if pretrain_identity is not None:
        config["pretrain_config_fingerprint"] = pretrain_identity.config_fingerprint
        config["pretrain_identity_fingerprint"] = _full_sha256(pretrain_identity.to_dict())
    return config


def _finetune_identity(
    *,
    manifest: Any,
    labels: Any,
    engine_config: Mapping[str, Any],
    protocol: str,
    split: str,
    label_budget: str,
    loss_name: str,
    seed: int,
    pretrain_seed: int,
    pretrain_identity: RunIdentity | None,
) -> RunIdentity:
    """Bind the downstream manifest, label budget and pretraining provenance."""

    return RunIdentity(
        method=MAE_METHOD if pretrain_identity is not None else SUP_METHOD,
        encoder_arch=ENCODER_ARCH,
        protocol=protocol,
        split=split,
        data_scope=manifest.scope,
        pretrain_seed=pretrain_seed,
        finetune_seed=seed,
        label_budget=label_budget,
        loss_name=loss_name,
        fine_tune_strategy=MATCHED_STRATEGY,
        config_fingerprint=_full_sha256(engine_config),
        manifest_fingerprint=manifest.fingerprint(),
        label_manifest_fingerprint=labels.fingerprint,
    )


def _load_encoder_state(
    checkpoint_path: str | Path,
    checkpoint_manifest: Any,
) -> tuple[dict[str, Tensor], RunIdentity]:
    """Load and validate the ViT-MAE pretraining checkpoint against its data boundary."""

    source = Path(checkpoint_path)
    try:
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as error:  # torch exposes several backend-specific errors.
        raise ValueError(f"unable to load ViT-MAE checkpoint: {source}") from error
    if not isinstance(checkpoint, Mapping):
        raise ValueError("ViT-MAE checkpoint must be a mapping")
    if checkpoint.get("checkpoint_kind") != "metafi_ssl_encoder":
        raise ValueError("ViT-MAE checkpoint kind mismatch")
    if checkpoint.get("status") != "complete":
        raise ValueError("ViT-MAE checkpoint is not complete")
    if checkpoint.get("encoder_arch") != ENCODER_ARCH:
        raise ValueError("ViT-MAE checkpoint encoder_arch mismatch")
    try:
        identity = RunIdentity.from_dict(checkpoint["identity"])
        bound_manifest_fingerprint = str(checkpoint["identity"]["manifest_fingerprint"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("ViT-MAE checkpoint identity is invalid") from error
    if identity.encoder_arch != ENCODER_ARCH or identity.method != MAE_METHOD:
        raise ValueError("ViT-MAE checkpoint identity mismatch")
    if checkpoint.get("identity_json") != identity.canonical_json():
        raise ValueError("ViT-MAE checkpoint identity_json is not canonical")
    expected = {
        "protocol": checkpoint_manifest.protocol,
        "split": checkpoint_manifest.split,
        "data_scope": checkpoint_manifest.scope,
        "manifest_fingerprint": checkpoint_manifest.fingerprint(),
    }
    for field, expected_value in expected.items():
        actual = getattr(identity, field)
        if actual != expected_value:
            raise ValueError(
                f"ViT-MAE checkpoint {field} mismatch: expected {expected_value!r}, got {actual!r}"
            )
    if bound_manifest_fingerprint != expected["manifest_fingerprint"]:
        raise ValueError("ViT-MAE checkpoint manifest_fingerprint mismatch")
    state = checkpoint.get("encoder_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("ViT-MAE checkpoint lacks a non-empty encoder_state_dict")
    return {name: value for name, value in state.items()}, identity


def _engine_config(
    config: Mapping[str, Any],
    *,
    protocol: str,
    split: str,
    seed: int,
    epochs: int,
    finetune: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge the baseline (official splits + engine keys) with the ViT fine-tune section."""

    baseline_path = str(config.get("baseline_config", DEFAULT_BASELINE_CONFIG))
    baseline = _load_config(_BASE / baseline_path if not Path(baseline_path).is_absolute() else baseline_path)
    engine = dict(baseline)
    engine.update(
        {
            "protocol": protocol,
            "split_to_use": split,
            "init_rand_seed": seed,
            "target_space": "absolute",
            "num_epochs": epochs,
            "fine_tune_strategy": MATCHED_STRATEGY,
            "modality": "wifi-csi",
            "data_unit": "frame",
            "amp_dtype": "fp16",
        }
    )
    for key in (
        "optimizer",
        "learning_rate",
        "weight_decay",
        "sgd_momentum",
        "scheduler",
        "lr_warmup_epochs",
        "lr_min",
        "lr_milestones",
        "lr_gamma",
        "dropout_p",
        "selection_metric",
        "early_stopping_patience",
        "use_epoch_patience",
    ):
        if key in finetune:
            engine[key] = finetune[key]
    return engine


def run_finetune(args: argparse.Namespace) -> dict[str, Any]:
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("epochs must be a positive integer")
    config = _load_config(args.config_file)
    finetune = _section(config, "finetune")
    epochs = int(args.epochs if args.epochs is not None else finetune.get("epochs", 25))
    loss_name = str(finetune.get("loss_name", "mse"))
    engine_config = _engine_config(
        config, protocol=args.protocol, split=args.split, seed=args.seed, epochs=epochs, finetune=finetune
    )
    num_workers = int(args.num_workers if args.num_workers is not None else finetune.get("num_workers", 0))
    val_every = int(args.val_every if args.val_every is not None else finetune.get("val_every", 5))
    if num_workers < 0 or val_every < 1:
        raise ValueError("num_workers must be >= 0 and val_every must be >= 1")

    paths = _resolve_artifact_paths(args)
    manifest, labels, axis_stats = load_audited_artifacts(
        paths["manifest"],
        paths["label_manifest"],
        paths["leakage_audit"],
        paths["axis_stats"],
        protocol=args.protocol,
        split=args.split,
        label_budget=args.label_budget,
    )
    checkpoint_manifest = pretraining_manifest_for_finetune(
        manifest, Path(args.pretrain_manifest) if args.pretrain_manifest else None
    )

    encoder_state: dict[str, Tensor] | None = None
    pretrain_identity: RunIdentity | None = None
    pretrain_seed = args.seed
    patch_size = _patch_size(config)
    if args.pretrain_checkpoint:
        encoder_state, pretrain_identity = _load_encoder_state(
            args.pretrain_checkpoint, checkpoint_manifest
        )
        _validate_encoder_geometry(encoder_state, patch_size)
        pretrain_seed = pretrain_identity.pretrain_seed
    engine_config = _with_pretrain_provenance(engine_config, pretrain_identity)

    identity = _finetune_identity(
        manifest=manifest,
        labels=labels,
        engine_config=engine_config,
        protocol=args.protocol,
        split=args.split,
        label_budget=args.label_budget,
        loss_name=loss_name,
        seed=args.seed,
        pretrain_seed=pretrain_seed,
        pretrain_identity=pretrain_identity,
    )
    result_dir = Path(args.output_dir)
    prepare_run_directory(result_dir, identity, args.resume)

    def factory(*, dropout_p: float, target_space: str) -> nn.Module:
        model = build_pose_model("vit_csi_small", dropout_p=dropout_p, target_space=target_space)
        if patch_size != (6, 5):
            # The backbone token grid follows vit_mae.patch_h/patch_w; the head only
            # depends on the embedding width, so replacing the encoder keeps it valid.
            model.encoder = ViTCSIEncoder(patch_h=patch_size[0], patch_w=patch_size[1])
        if encoder_state is not None:
            try:
                model.encoder.load_state_dict(encoder_state, strict=True)
            except RuntimeError as error:
                raise ValueError("ViT-MAE encoder_state_dict is incompatible with vit_csi_small") from error
        return model

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    setup_cuda(device)
    return train_one_experiment(
        args.dataset_root,
        engine_config,
        str(result_dir),
        device,
        num_workers=num_workers,
        use_amp=not args.no_amp,
        val_every=val_every,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        model_factory=factory,
        data_manifest=manifest,
        label_manifest=labels,
        run_identity=identity,
        resume=args.resume,
        experiment_context={
            "source_config": str(Path(args.config_file).resolve()),
            "arguments": vars(args),
            "pretrain_checkpoint": str(Path(args.pretrain_checkpoint).resolve()) if args.pretrain_checkpoint else None,
            "pretrain_identity": pretrain_identity.to_dict() if pretrain_identity is not None else None,
        },
        experiment_artifacts={
            "data_manifest.json": paths["manifest"],
            "leakage_audit.json": paths["leakage_audit"],
            "label_stats.json": paths["axis_stats"],
            **({"pretrain_manifest.json": args.pretrain_manifest} if args.pretrain_manifest else {}),
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        summary = run_finetune(args)
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"[ERROR] {error}", flush=True)
        return 1
    print(
        f"[ViT 微调完成] method={args.pretrain_checkpoint and MAE_METHOD or SUP_METHOD} | "
        f"best MPJPE {summary['best_mpjpe_mm']:.1f}mm | "
        f"test MPJPE {summary['test_mpjpe_mm']:.1f}mm | "
        f"test PA-MPJPE {summary['test_pampjpe_mm']:.1f}mm",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
