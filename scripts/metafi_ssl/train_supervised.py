"""Matched supervised MetaFi-R34 training entry point.

This module constructs the exact same full random MetaFi pose model for the
supervised control and SSL fine-tuning.  Fine-tuning overwrites *only* the
encoder after construction, so a shared seed guarantees decoder parity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
import yaml

_BASE = Path(__file__).resolve().parents[2]
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))

from mmfi_wifi.experiment_config import file_sha256
from mmfi_wifi.data_manifest import DataManifest, LeakageAudit, audit_manifest
from mmfi_wifi.engine import preflight_train_one_experiment, setup_cuda, train_one_experiment
from mmfi_wifi.metafi_decoder import AxisStats
from mmfi_wifi.metafi_pose_model import MetaFiPoseModel
from mmfi_wifi.run_identity import RunIdentity, assert_new_result_root, prepare_run_directory
from pose_ssl.metafi.label_budget import LabelManifest
from pose_ssl.metafi.label_stats import fingerprint_keys
from pose_ssl.metafi.fine_tune_strategy import (
    MATCHED_STRATEGY,
    TRANSFER_STRATEGY,
    SUP_DIFFERENTIAL_LR_STRATEGY,
    STRATEGY_NAMES,
)
from pose_ssl.loss import build_ablation_pose_loss


ENCODER_ARCH = "metafi_r34"
_MATCHED_STRATEGY = MATCHED_STRATEGY
PRETRAIN_CHECKPOINT_SCHEMA_VERSION = 1
PRETRAIN_CHECKPOINT_KIND = "metafi_ssl_encoder"
# Shared lifecycle contract for the later SSL pretraining writer:
# resumable epoch-boundary states must use ``"in_progress"``; only a
# finalized, fine-tune-exportable encoder checkpoint may use ``"complete"``.
# This fine-tuning reader deliberately accepts exactly the latter.
PRETRAIN_CHECKPOINT_STATUS_COMPLETE = "complete"
SUPPORTED_PRETRAIN_METHODS = frozenset({"simclr", "moco", "swav", "relpos", "mfm", "mae"})


def _full_sha256(value: object) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _set_construction_seed(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed 必须是非负 int")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pretraining_manifest_for_finetune(
    downstream_manifest: DataManifest,
    pretrain_manifest_path: Path | None,
) -> DataManifest:
    """Resolve the manifest that defined a checkpoint's pretraining data.

    Small-sample SSL intentionally replaces only ``pretrain_keys`` with a
    strict subset. Fine-tuning still uses the full manifest's labels, select,
    and test partitions, but checkpoint provenance must be checked against the
    subset manifest actually consumed by pretraining.
    """

    if pretrain_manifest_path is None:
        return downstream_manifest

    pretrain_manifest = DataManifest.read_json(pretrain_manifest_path)
    boundary_fields = (
        "protocol",
        "split",
        "seed",
        "scope",
        "official_train_keys",
        "internal_train_keys",
        "select_keys",
        "test_keys",
    )
    mismatches = [
        field
        for field in boundary_fields
        if getattr(pretrain_manifest, field) != getattr(downstream_manifest, field)
    ]
    if mismatches:
        raise ValueError(
            "pretraining manifest and downstream manifest have different fixed data boundaries: "
            + ", ".join(mismatches)
        )
    if not pretrain_manifest.pretrain_keys.issubset(downstream_manifest.internal_train_keys):
        raise ValueError("pretraining manifest pretrain_keys are outside the downstream internal train set")
    return pretrain_manifest


def _validated_pretraining_identity(
    checkpoint: Mapping[str, Any],
    expected_identity: RunIdentity,
) -> RunIdentity:
    """Read exactly one versioned, canonical encoder-checkpoint contract.

    Pretraining checkpoints are produced later in the plan.  This reader is
    intentionally strict now so a fine-tune cannot silently claim provenance
    that the serialized checkpoint did not establish.
    """

    if checkpoint.get("schema_version") != PRETRAIN_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("不支持或缺少预训练 checkpoint schema_version")
    if checkpoint.get("checkpoint_kind") != PRETRAIN_CHECKPOINT_KIND:
        raise ValueError("预训练 checkpoint checkpoint_kind 不匹配")
    if checkpoint.get("status") != PRETRAIN_CHECKPOINT_STATUS_COMPLETE:
        raise ValueError(
            "预训练 checkpoint status 必须是 complete；"
            "仅最终导出的 encoder 可用于 fine-tune"
        )
    if checkpoint.get("encoder_arch") != ENCODER_ARCH:
        raise ValueError("预训练 checkpoint encoder_arch 不匹配")

    raw_identity = checkpoint.get("identity")
    if not isinstance(raw_identity, Mapping):
        raise ValueError("预训练 checkpoint 缺少完整 identity")
    try:
        identity = RunIdentity.from_dict(raw_identity)
    except (TypeError, ValueError) as error:
        raise ValueError("预训练 checkpoint identity 无效") from error

    identity_json = checkpoint.get("identity_json")
    if not isinstance(identity_json, str) or identity_json != identity.canonical_json():
        raise ValueError("预训练 checkpoint identity_json 不是规范序列化")
    if identity.encoder_arch != checkpoint["encoder_arch"]:
        raise ValueError("预训练 checkpoint encoder_arch 与 identity 不一致")
    if identity.method not in SUPPORTED_PRETRAIN_METHODS:
        raise ValueError("预训练 checkpoint method 不受支持")

    expected = {
        "encoder_arch": expected_identity.encoder_arch,
        "protocol": expected_identity.protocol,
        "split": expected_identity.split,
        "data_scope": expected_identity.data_scope,
        "manifest_fingerprint": expected_identity.manifest_fingerprint,
    }
    for field, expected_value in expected.items():
        actual = getattr(identity, field)
        if actual != expected_value:
            raise ValueError(
                f"预训练 checkpoint {field} 不匹配: "
                f"expected {expected_value!r}, got {actual!r}"
            )
    return identity


def load_pretrained_encoder(
    model: MetaFiPoseModel,
    checkpoint: Mapping[str, Any],
    expected_identity: RunIdentity,
) -> RunIdentity:
    """Fail closed, load only the validated encoder, and return provenance."""

    if not isinstance(model, MetaFiPoseModel):
        raise TypeError("model 必须是 MetaFiPoseModel")
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint 必须是 Mapping")

    identity = _validated_pretraining_identity(checkpoint, expected_identity)
    state = checkpoint.get("encoder_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("预训练 checkpoint 缺少非空 encoder_state_dict")
    try:
        model.encoder.load_state_dict(dict(state), strict=True)
    except (RuntimeError, TypeError) as error:
        raise ValueError("预训练 encoder_state_dict 与 metafi_r34 不兼容") from error
    return identity


def build_matched_model(
    axis_stats: AxisStats,
    encoder_checkpoint: Path | None,
    seed: int,
) -> MetaFiPoseModel:
    """Build a seeded full model, then optionally overwrite only its encoder."""

    if not isinstance(axis_stats, AxisStats):
        raise TypeError("axis_stats 必须是 AxisStats")
    _set_construction_seed(seed)
    model = MetaFiPoseModel(target_space="absolute", axis_stats=axis_stats)
    # benchmark 未重置 ResNet/姿态头参数；监督对照沿用同一默认初始化。

    if encoder_checkpoint is not None:
        path = Path(encoder_checkpoint)
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError) as error:
            raise ValueError(f"无法读取 encoder checkpoint: {path}") from error
        if not isinstance(checkpoint, Mapping):
            raise ValueError("encoder checkpoint 必须是 Mapping")
        # Construction callers that require identity validation call
        # ``load_pretrained_encoder`` themselves.  This direct loader supports
        # deterministic parity tests and requires only the exported encoder.
        state = checkpoint.get("encoder_state_dict")
        if not isinstance(state, Mapping) or not state:
            raise ValueError("encoder checkpoint 缺少非空 encoder_state_dict")
        try:
            model.encoder.load_state_dict(dict(state), strict=True)
        except (RuntimeError, TypeError) as error:
            raise ValueError("encoder checkpoint 与 metafi_r34 不兼容") from error
    return model


def _read_json(path: str | Path, name: str) -> Mapping[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取 {name}: {source}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} 根节点必须是对象")
    return payload


def _load_label_manifest(path: str | Path) -> LabelManifest:
    payload = _read_json(path, "label manifest")
    try:
        selected = payload["selected_keys"]
        mode = payload["mode"]
    except KeyError as error:
        raise ValueError("label manifest 缺少字段") from error
    if not isinstance(selected, list):
        raise ValueError("label manifest selected_keys 必须是列表")
    from mmfi_wifi.sequence_keys import SequenceKey

    try:
        keys = {
            SequenceKey(scene=item[0], subject=item[1], action=item[2])
            for item in selected
        }
    except (IndexError, TypeError, ValueError) as error:
        raise ValueError("label manifest selected_keys 无效") from error
    manifest = LabelManifest.create(mode, keys)
    if payload.get("fingerprint") != manifest.fingerprint:
        raise ValueError("label manifest fingerprint 不匹配")
    return manifest


def _load_axis_stats(path: str | Path, label_manifest: LabelManifest) -> AxisStats:
    payload = _read_json(path, "axis stats")
    if payload.get("label_manifest_fingerprint") != label_manifest.fingerprint:
        raise ValueError("axis stats label_manifest_fingerprint 不匹配")
    if payload.get("source_fingerprint") != fingerprint_keys(label_manifest.selected_keys):
        raise ValueError("axis stats source_fingerprint 不匹配")
    try:
        return AxisStats(
            mean=tuple(payload["mean"]),
            std=tuple(payload["std"]),
            source_fingerprint=payload["source_fingerprint"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("axis stats 无效") from error


def _load_passing_audit(path: str | Path, manifest: DataManifest) -> LeakageAudit:
    payload = _read_json(path, "leakage audit")
    if payload.get("passed") is not True:
        raise ValueError("leakage audit 未通过")
    if payload.get("data_manifest_fingerprint") != manifest.fingerprint():
        raise ValueError("leakage audit manifest fingerprint 不匹配")
    audit = audit_manifest(manifest)
    if not audit.passed:
        raise ValueError("data manifest 边界审计失败: " + "; ".join(audit.violations))
    return audit


def load_audited_artifacts(
    manifest_path: str | Path,
    label_manifest_path: str | Path,
    leakage_audit_path: str | Path,
    axis_stats_path: str | Path,
    *,
    protocol: str,
    split: str,
    label_budget: str,
) -> tuple[DataManifest, LabelManifest, AxisStats]:
    """Load and cross-check immutable audit artifacts before model allocation."""

    manifest = DataManifest.read_json(manifest_path)
    if manifest.protocol != protocol:
        raise ValueError("manifest protocol 不匹配")
    if manifest.split != split:
        raise ValueError("manifest split 不匹配")
    labels = _load_label_manifest(label_manifest_path)
    if labels.mode != label_budget:
        raise ValueError("label manifest mode 与 label_budget 不匹配")
    audit = audit_manifest(manifest, labels.selected_keys)
    if not audit.passed:
        raise ValueError("label manifest 数据边界审计失败: " + "; ".join(audit.violations))
    _load_passing_audit(leakage_audit_path, manifest)
    axis_stats = _load_axis_stats(axis_stats_path, labels)
    return manifest, labels, axis_stats


def _load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        config = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"无法读取 config: {source}") from error
    if not isinstance(config, dict):
        raise ValueError("config 根节点必须是对象")
    return config


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a configured random-init MetaFi supervised control")
    parser.add_argument("dataset_root")
    parser.add_argument("config_file")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--label-manifest", required=True)
    parser.add_argument("--leakage-audit", required=True)
    parser.add_argument("--axis-stats", required=True)
    parser.add_argument("--label-budget", required=True)
    parser.add_argument("--protocol", required=True, choices=("protocol1", "protocol2", "protocol3"))
    parser.add_argument("--split", required=True, choices=("random_split", "cross_subject_split", "cross_scene_split"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--loss-name", choices=("mse", "mse_bone"), default=None)
    parser.add_argument("--bone-set", choices=("legacy15",), default=None)
    parser.add_argument("--strategy", required=True, choices=(MATCHED_STRATEGY, SUP_DIFFERENTIAL_LR_STRATEGY))
    parser.add_argument("--optimizer", choices=("adamw", "sgd"), default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--sgd-momentum", type=float, default=None)
    parser.add_argument("--scheduler", choices=("cosine", "multistep", "constant"), default=None)
    parser.add_argument("--lr-warmup-epochs", type=int, default=None)
    parser.add_argument("--lr-min", type=float, default=None)
    parser.add_argument("--lr-milestones", type=int, nargs="*", default=None)
    parser.add_argument("--lr-gamma", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args(argv)


def _build_criterion(
    config: Mapping[str, Any],
    *,
    requested_loss_name: str | None = None,
    requested_bone_set: str | None = None,
) -> tuple[str, torch.nn.Module]:
    configured_loss_name = config.get("loss_name", "mse")
    if configured_loss_name not in {"mse", "mse_bone"}:
        raise ValueError(f"unsupported supervised loss_name: {configured_loss_name!r}")
    if requested_loss_name is not None and requested_loss_name != configured_loss_name:
        raise ValueError(
            "--loss-name must match merged config loss_name: "
            f"expected {configured_loss_name!r}, got {requested_loss_name!r}"
        )
    loss_name = configured_loss_name
    if loss_name == "mse":
        if requested_bone_set is not None:
            raise ValueError("--bone-set is only valid with --loss-name mse_bone")
        return loss_name, torch.nn.MSELoss()

    ablation = config.get("ablation")
    if not isinstance(ablation, Mapping) or ablation.get("name") != "bone_loss":
        raise ValueError("mse_bone requires the bone_loss ablation config")
    bone_set = ablation.get("bone_set")
    if bone_set != "legacy15":
        raise ValueError("mse_bone requires explicit bone_set=legacy15")
    if requested_bone_set is not None and requested_bone_set != bone_set:
        raise ValueError(
            "--bone-set must match merged config ablation.bone_set: "
            f"expected {bone_set!r}, got {requested_bone_set!r}"
        )
    return loss_name, build_ablation_pose_loss(ablation)


def _identity_for_run(
    *,
    method: str,
    manifest: DataManifest,
    config: Mapping[str, Any],
    pretrain_seed: int,
    finetune_seed: int,
    label_budget: str,
    label_manifest_fingerprint: str,
    fine_tune_strategy: str,
    loss_name: str = "mse",
) -> RunIdentity:
    return RunIdentity(
        method=method,
        encoder_arch=ENCODER_ARCH,
        protocol=manifest.protocol,
        split=manifest.split,
        data_scope=manifest.scope,
        pretrain_seed=pretrain_seed,
        finetune_seed=finetune_seed,
        label_budget=label_budget,
        loss_name=loss_name,
        fine_tune_strategy=fine_tune_strategy,
        config_fingerprint=_full_sha256(config),
        manifest_fingerprint=manifest.fingerprint(),
        label_manifest_fingerprint=label_manifest_fingerprint,
    )


def run_matched(
    args: argparse.Namespace,
    *,
    encoder_checkpoint: Path | None,
    method: str,
    pretrain_manifest: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    allowed_strategies = (
        {_MATCHED_STRATEGY, SUP_DIFFERENTIAL_LR_STRATEGY}
        if encoder_checkpoint is None
        else {_MATCHED_STRATEGY, TRANSFER_STRATEGY}
    )
    if args.strategy not in allowed_strategies:
        run_kind = "supervised training" if encoder_checkpoint is None else "SSL fine-tuning"
        raise ValueError(f"strategy {args.strategy!r} is invalid for {run_kind}")
    result_dir = Path(args.output_dir)
    assert_new_result_root(result_dir)
    manifest, labels, axis_stats = load_audited_artifacts(
        args.manifest,
        args.label_manifest,
        args.leakage_audit,
        args.axis_stats,
        protocol=args.protocol,
        split=args.split,
        label_budget=args.label_budget,
    )
    checkpoint_manifest = pretraining_manifest_for_finetune(manifest, pretrain_manifest)
    config = _load_config(args.config_file)
    loss_name, criterion = _build_criterion(
        config,
        requested_loss_name=getattr(args, "loss_name", None),
        requested_bone_set=getattr(args, "bone_set", None),
    )
    config.update({
        "protocol": args.protocol,
        "split_to_use": args.split,
        "init_rand_seed": args.seed,
        "target_space": "absolute",
        "selection_metric": "mpjpe",
    })
    if args.epochs is not None:
        config["num_epochs"] = args.epochs
    for field in (
        "optimizer", "learning_rate", "weight_decay", "sgd_momentum",
        "scheduler", "lr_warmup_epochs", "lr_min", "lr_milestones",
        "lr_gamma", "strategy_params",
    ):
        value = getattr(args, field, None)
        if value is not None:
            config[field] = value
    config["fine_tune_strategy"] = args.strategy
    if args.num_workers < 0:
        raise ValueError("num_workers 必须 >= 0")

    raw_checkpoint: Mapping[str, Any] | None = None
    expected_boundary = RunIdentity(
        method="sup",
        encoder_arch=ENCODER_ARCH,
        protocol=checkpoint_manifest.protocol,
        split=checkpoint_manifest.split,
        data_scope=checkpoint_manifest.scope,
        pretrain_seed=0,
        finetune_seed=args.seed,
        label_budget=args.label_budget,
        loss_name=loss_name,
        fine_tune_strategy=args.strategy,
        config_fingerprint=_full_sha256(config),
        manifest_fingerprint=checkpoint_manifest.fingerprint(),
    )
    actual_pretrain_identity: RunIdentity | None = None
    if encoder_checkpoint is not None:
        try:
            raw_checkpoint = torch.load(encoder_checkpoint, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError) as error:
            raise ValueError(f"无法读取 encoder checkpoint: {encoder_checkpoint}") from error
        if not isinstance(raw_checkpoint, Mapping):
            raise ValueError("encoder checkpoint 必须是 Mapping")
        # A throwaway model is used solely to validate strict loading before any
        # result directory exists; the actual factory reconstructs from seed.
        probe = build_matched_model(axis_stats, None, args.seed)
        actual_pretrain_identity = load_pretrained_encoder(probe, raw_checkpoint, expected_boundary)
        config["pretrain_config_fingerprint"] = actual_pretrain_identity.config_fingerprint
        config["pretrain_identity_fingerprint"] = _full_sha256(actual_pretrain_identity.to_dict())

    identity = _identity_for_run(
        method=actual_pretrain_identity.method if actual_pretrain_identity is not None else method,
        manifest=manifest,
        config=config,
        pretrain_seed=(actual_pretrain_identity.pretrain_seed if actual_pretrain_identity is not None else args.seed),
        finetune_seed=args.seed,
        label_budget=args.label_budget,
        label_manifest_fingerprint=labels.fingerprint,
        fine_tune_strategy=args.strategy,
        loss_name=loss_name,
    )

    # Validate every fallible config/dataset/manifest selection before committing
    # the identity-bound output directory.  The engine repeats this read-only
    # setup when it starts training, but failed preflight leaves no run artifact.
    preflight_train_one_experiment(
        args.dataset_root,
        config,
        num_workers=args.num_workers,
        data_manifest=manifest,
        label_manifest=labels,
        run_identity=identity,
    )
    prepare_run_directory(result_dir, identity, resume=resume)

    def factory(*, dropout_p: float, target_space: str) -> MetaFiPoseModel:
        if target_space != "absolute":
            raise ValueError("matched MetaFi 实验必须使用 absolute target_space")
        model = build_matched_model(axis_stats, None, args.seed)
        if raw_checkpoint is not None:
            load_pretrained_encoder(model, raw_checkpoint, expected_boundary)
        model.decoder.dropout_p = dropout_p
        return model

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    setup_cuda(device)
    return train_one_experiment(
        args.dataset_root,
        config,
        str(result_dir),
        device,
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
        val_every=args.val_every,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        model_factory=factory,
        criterion=criterion,
        data_manifest=manifest,
        label_manifest=labels,
        run_identity=identity,
        resume=resume,
        experiment_context={
            "source_config": getattr(args, "experiment_source", None),
            "arguments": vars(args),
            "pretrain_checkpoint": str(encoder_checkpoint.resolve()) if encoder_checkpoint is not None else None,
            "pretrain_checkpoint_sha256": file_sha256(encoder_checkpoint) if encoder_checkpoint is not None else None,
            "pretrain_identity": actual_pretrain_identity.to_dict() if actual_pretrain_identity else None,
            "pretrain_experiment_config": (
                json.loads(encoder_checkpoint.with_name("experiment_config.json").read_text(encoding="utf-8"))
                if encoder_checkpoint is not None and encoder_checkpoint.with_name("experiment_config.json").is_file()
                else None
            ),
        },
        experiment_artifacts={
            "data_manifest.json": args.manifest, "leakage_audit.json": args.leakage_audit,
            "label_stats.json": args.axis_stats,
            **({"pretrain_manifest.json": pretrain_manifest} if pretrain_manifest is not None else {}),
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run_matched(args, encoder_checkpoint=None, method="sup")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
