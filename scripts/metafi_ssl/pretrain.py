"""Leakage-audited, resumable MetaFi SSL pretraining runner.

This runner deliberately owns only pretraining execution. It consumes the
immutable manifest/audit produced by ``audit_data.py`` and writes below the
new ``result_metafi_ssl`` tree; the legacy ``result`` tree is never a target.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader

_BASE = Path(__file__).resolve().parents[2]
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))

from mmfi_wifi.experiment_config import write_experiment_config
from mmfi_wifi.data_manifest import DataManifest, audit_manifest
from mmfi_wifi.metafi_encoder import MetaFiEncoder
from mmfi_wifi.run_identity import RunIdentity, assert_new_result_root
from scripts.metafi_ssl.profile_hardware import (
    HardwareProfile,
    load_hardware_profile,
    find_safe_micro_batch,
    validate_profile_identity,
    write_hardware_profile,
)
from pose_ssl.metafi.base import MetaFiSSLMethod, SSLStepOutput
from pose_ssl.metafi.factory import build_metafi_ssl_method
from pose_ssl.metafi.pretrain_checkpoint import (
    CHECKPOINT_KIND,
    CHECKPOINT_SCHEMA_VERSION,
    STATUS_COMPLETE,
    STATUS_IN_PROGRESS,
    build_checkpoint_payload,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint_atomic,
    validate_checkpoint,
)
from pose_ssl.metafi.pretrain_data import MetaFiPretrainDataset, PretrainBatch, collate_pretrain_samples


class PretrainRunError(RuntimeError):
    """Raised for a failed pretraining run that must not be marked complete."""


MethodFactory = Callable[[], MetaFiSSLMethod]


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run audited MetaFi SSL pretraining")
    parser.add_argument("dataset_root")
    parser.add_argument("config_file")
    parser.add_argument("--method", required=True, choices=("simclr", "moco", "swav", "relpos", "mfm", "mae"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--micro-batch", type=int, default=None)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-cuda-profile", action="store_true")
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


def resolve_pretrain_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten an optional YAML ``pretrain`` section for direct CLI use."""

    resolved = dict(config)
    section = config.get("pretrain")
    if isinstance(section, Mapping):
        resolved.update(section)
    return resolved


def _read_json(path: str | Path, name: str) -> Mapping[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read {name}: {source}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return payload


def load_audited_manifest(manifest_path: str | Path, audit_path: str | Path | None = None) -> DataManifest:
    """Load a manifest and require its independent leakage audit to pass."""

    manifest = DataManifest.read_json(manifest_path)
    audit_source = Path(audit_path) if audit_path is not None else Path(manifest_path).with_name("leakage_audit.json")
    audit_payload = _read_json(audit_source, "leakage audit")
    if audit_payload.get("passed") is not True:
        raise ValueError("leakage audit did not pass")
    if audit_payload.get("data_manifest_fingerprint") != manifest.fingerprint():
        raise ValueError("leakage audit manifest fingerprint mismatch")
    audit = audit_manifest(manifest)
    if not audit.passed:
        raise ValueError("manifest audit failed: " + "; ".join(audit.violations))
    if manifest.scope == "strict" and not set(manifest.pretrain_keys).issubset(manifest.internal_train_keys):
        raise ValueError("strict manifest pretrain boundary is invalid")
    return manifest


def _canonical_fingerprint(value: object) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("pretraining config must be JSON-serializable and finite") from error
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _legacy_pretrain_config(config: Mapping[str, Any]) -> dict:
    """提取预训练相关的配置参数，排除微调策略相关参数。

    这确保了不同的微调策略（matched、transfer、sup-differential-lr）
    可以复用相同的预训练 checkpoint，只要预训练参数相同。
    """
    pretrain_relevant = {
        "pretrain": config.get("pretrain", {}),
        "methods": config.get("methods", {}),
        # 排除 supervised、finetune、small_sample 等微调相关配置
        # 保留数据集配置（影响预训练数据）
        "dataset_root": config.get("dataset_root"),
        "baseline_config": config.get("baseline_config"),
        "protocol": config.get("protocol"),
        "scope": config.get("scope"),
        "seed": config.get("seed"),
        "official_alignment": config.get("official_alignment"),
    }
    return pretrain_relevant


def _extract_pretrain_config(config: Mapping[str, Any], method_name: str) -> dict:
    """One effective training spec, shared by cache names and checkpoint identity."""
    resolved = resolve_pretrain_config(config)
    defaults = {
        "epochs": resolved.get("num_epochs", 1),
        "optimizer": "adamw", "pretrain_lr": resolved.get("lr", 1e-4),
        "weight_decay": 1e-4, "sgd_momentum": 0.9,
        "scheduler": "cosine", "scheduler_t_max": 100, "lr_min": 0.0,
        "lr_milestones": [], "lr_gamma": 0.1,
        "micro_batch": resolved.get("batch_size", 2), "gradient_accumulation": 1,
        "max_batches": None, "num_workers": 0, "amp": True, "profile_cuda": False,
    }
    training = {key: resolved.get(key, default) for key, default in defaults.items()}
    dataset_root = Path(str(config.get("dataset_root", ".")))
    if not dataset_root.is_absolute():
        dataset_root = _BASE / dataset_root
    return {"dataset_root": str(dataset_root.resolve()), "pretrain": training, "method": method_name,
            "method_config": dict(_method_config(resolved, method_name))}


def make_pretrain_identity(
    manifest: DataManifest,
    method_name: str,
    seed: int,
    config_fingerprint: str,
) -> RunIdentity:
    """Construct the pretraining identity used by schema-v1 checkpoints."""

    return RunIdentity(
        method=method_name,
        encoder_arch="metafi_r34",
        protocol=manifest.protocol,
        split=manifest.split,
        data_scope=manifest.scope,
        pretrain_seed=seed,
        finetune_seed=0,
        label_budget="none",
        loss_name="ssl",
        fine_tune_strategy="pretrain",
        config_fingerprint=config_fingerprint,
        manifest_fingerprint=manifest.fingerprint(),
    )


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _prepare_directory(path: Path, identity: RunIdentity, resume: bool) -> None:
    assert_new_result_root(path)
    identity_path = path / "run_identity.json"
    if resume:
        if not path.is_dir():
            raise FileNotFoundError(f"resume directory does not exist: {path}")
        try:
            raw = identity_path.read_text(encoding="utf-8")
            saved = RunIdentity.from_dict(json.loads(raw))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError("resume run_identity.json is invalid") from error
        if raw != saved.canonical_json() or saved != identity:
            raise ValueError("resume RunIdentity mismatch")
        if not (path / "latest.pth").is_file():
            raise FileNotFoundError("resume directory has no latest.pth")
        return
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"output path is not a directory: {path}")
        if any(path.iterdir()):
            raise FileExistsError(f"output directory is not empty: {path}")
    else:
        path.mkdir(parents=True, exist_ok=False)
    _atomic_text(identity_path, identity.canonical_json())


def _set_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_batch(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, PretrainBatch):
        return PretrainBatch(
            anchor=batch.anchor.to(device),
            neighbors=tuple(value.to(device) for value in batch.neighbors),
            neighbor_offsets=batch.neighbor_offsets,
            sequence_keys=batch.sequence_keys,
            frame_indices=batch.frame_indices.to(device),
            neighbor_available=tuple(value.to(device) for value in batch.neighbor_available),
        )
    if hasattr(batch, "anchor") and isinstance(batch.anchor, Tensor):
        moved = copy.copy(batch)
        moved.anchor = batch.anchor.to(device)
        return moved
    if isinstance(batch, Tensor):
        return batch.to(device)
    raise TypeError("pretraining collate output must contain an anchor tensor")


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _make_scaler(device: torch.device, enabled: bool):
    return torch.amp.GradScaler(
        "cuda", enabled=enabled and device.type == "cuda"
    )


def build_pretrain_optimizer(parameters, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    """Build the configured SSL pretraining optimizer."""

    name = str(config.get("optimizer", "adamw")).lower()
    learning_rate = float(config.get("pretrain_lr", config.get("lr", 1e-4)))
    weight_decay = float(config.get("weight_decay", 1e-4))
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=float(config.get("sgd_momentum", 0.9)),
            weight_decay=weight_decay,
        )
    raise ValueError("pretraining optimizer must be adamw or sgd")


class MultiStepFloorLR(torch.optim.lr_scheduler.MultiStepLR):
    """MultiStepLR with an absolute learning-rate floor."""

    def __init__(self, optimizer, milestones, gamma=0.1, eta_min=0.0, last_epoch=-1):
        self.eta_min = float(eta_min)
        if self.eta_min < 0.0:
            raise ValueError("lr_min must be non-negative")
        super().__init__(optimizer, milestones, gamma=gamma, last_epoch=last_epoch)

    def get_lr(self):
        return [max(self.eta_min, lr) for lr in super().get_lr()]


def build_pretrain_scheduler(optimizer: torch.optim.Optimizer, config: Mapping[str, Any]):
    """Build the configured SSL pretraining scheduler."""

    name = str(config.get("scheduler", "cosine")).lower()
    lr_min = float(config.get("lr_min", 0.0))
    if lr_min < 0.0:
        raise ValueError("lr_min must be non-negative")
    if name == "cosine":
        t_max = int(config.get("scheduler_t_max", 100))
        if t_max < 1:
            raise ValueError("scheduler_t_max must be a positive integer")
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=lr_min
        )
    if name == "multistep":
        return MultiStepFloorLR(
            optimizer,
            milestones=list(config.get("lr_milestones", ())),
            gamma=float(config.get("lr_gamma", 0.1)),
            eta_min=lr_min,
        )
    if name == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _epoch: 1.0)
    raise ValueError("pretraining scheduler must be cosine, multistep, or constant")


def _optimizer_step(
    scaler: Any,
    optimizer: torch.optim.Optimizer,
) -> bool:
    """Apply one scaled step and report whether GradScaler skipped it."""

    old_scale = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    return float(scaler.get_scale()) >= old_scale


def _is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


def _method_config(config: Mapping[str, Any], method_name: str) -> Mapping[str, Any]:
    methods = config.get("methods", config.get("ssl_methods", {}))
    if methods and not isinstance(methods, Mapping):
        raise ValueError("methods config must be a mapping")
    selected = methods.get(method_name, config.get(method_name, {})) if isinstance(methods, Mapping) else {}
    if selected is None:
        return {}
    if not isinstance(selected, Mapping):
        raise ValueError(f"config for method {method_name!r} must be a mapping")
    return selected


def _build_default_factory(method_name: str, config: Mapping[str, Any], seed: int) -> MethodFactory:
    method_config = dict(_method_config(config, method_name))
    method_config.setdefault("augmentation_seed", seed)

    def build() -> MetaFiSSLMethod:
        return build_metafi_ssl_method(method_name, MetaFiEncoder(), method_config)

    return build



def _resolve_collate_fn(collate_fn: Callable | None) -> Callable:
    """MetaFiPretrainDataset 的 PretrainSample 必须使用项目专用拼接函数。"""
    return collate_pretrain_samples if collate_fn is None else collate_fn


def _make_loader(dataset: Any, *, micro_batch: int, generator: torch.Generator, collate_fn: Callable | None, num_workers: int) -> DataLoader:
    if len(dataset) < micro_batch:
        raise ValueError("pretraining dataset is smaller than micro_batch")
    return DataLoader(
        dataset,
        batch_size=micro_batch,
        shuffle=True,
        drop_last=True,
        generator=generator,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=False,
    )


def _truncate_csv(path: Path, next_epoch: int) -> None:
    """Reconcile CSV history with the last durably committed checkpoint.

    The runner commits the CSV row before publishing ``latest.pth``.  A
    failure between those two operations therefore leaves one or more rows
    that are not represented by the checkpoint.  Those rows are removed on
    resume.  Rows belonging to committed epochs are validated rather than
    silently repaired: a committed epoch must occur exactly once and the
    committed prefix must be complete.
    """

    if not isinstance(next_epoch, int) or isinstance(next_epoch, bool) or next_epoch < 0:
        raise ValueError("next_epoch must be a non-negative integer")
    if not path.exists():
        if next_epoch:
            raise ValueError("metrics.csv is missing committed epoch history")
        return
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        if next_epoch:
            raise ValueError("metrics.csv is empty despite committed epochs")
        return
    expected_header = ["epoch", "loss", "batches"]
    if rows[0] != expected_header:
        raise ValueError("metrics.csv has an invalid header")

    committed_rows: dict[int, list[str]] = {}
    for row in rows[1:]:
        if not row:
            continue
        if len(row) != len(expected_header):
            raise ValueError("metrics.csv contains a malformed row")
        try:
            epoch = int(row[0])
        except (TypeError, ValueError) as error:
            raise ValueError("metrics.csv contains an invalid epoch") from error
        if epoch < 0:
            raise ValueError("metrics.csv contains a negative epoch")
        if epoch < next_epoch:
            if epoch in committed_rows:
                raise ValueError(f"metrics.csv contains duplicate committed epoch {epoch}")
            committed_rows[epoch] = row

    expected_epochs = set(range(next_epoch))
    if set(committed_rows) != expected_epochs:
        missing = sorted(expected_epochs.difference(committed_rows))
        raise ValueError(f"metrics.csv is missing committed epochs: {missing}")

    kept = [expected_header] + [committed_rows[epoch] for epoch in range(next_epoch)]
    # Rewrite even when there is no trailing row.  This canonicalizes order
    # and gives every resume the same exact one-row-per-epoch representation.
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerows(kept)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _append_metrics(path: Path, epoch: int, metrics: Mapping[str, float], batches: int) -> None:
    """Atomically publish one complete metrics row.

    Rewriting the small epoch log to a temporary file avoids exposing a
    partially written CSV row if the process stops during the write.  The
    checkpoint is published only after this replacement succeeds.
    """

    header = ["epoch", "loss", "batches"]
    rows: list[list[object]] = [header]
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            existing = list(csv.reader(handle))
        if not existing or existing[0] != header:
            raise ValueError("metrics.csv has an invalid header")
        rows.extend(existing[1:])
    if any(row and row[0] == str(epoch) for row in rows[1:]):
        raise ValueError(f"metrics.csv already contains epoch {epoch}")
    rows.append([epoch, repr(float(metrics.get("loss", float("nan")))), batches])
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _resolve_execution_parameters(
    *,
    requested_micro_batch: int,
    requested_gradient_accumulation: int,
    resolved_micro_batch: int,
    max_batches: int | None,
    use_amp: bool,
    device: torch.device,
    num_workers: int,
    profile_cuda: bool,
) -> dict[str, Any]:
    """Return the immutable execution settings bound to a pretraining run."""

    if resolved_micro_batch < 2:
        raise ValueError("resolved micro_batch must be >= 2")
    target_effective_batch = requested_micro_batch * requested_gradient_accumulation
    if target_effective_batch % resolved_micro_batch != 0:
        raise ValueError(
            "resolved micro_batch must exactly divide the target effective batch "
            f"size (target={target_effective_batch}, "
            f"resolved={resolved_micro_batch}); refusing to alter the "
            "optimization batch size"
        )
    resolved_gradient_accumulation = target_effective_batch // resolved_micro_batch
    if resolved_gradient_accumulation < requested_gradient_accumulation:
        raise ValueError(
            "resolved micro_batch cannot preserve the target effective batch "
            "without reducing gradient accumulation"
        )
    if resolved_micro_batch * resolved_gradient_accumulation != target_effective_batch:
        raise ValueError("resolved execution parameters do not preserve effective batch")
    amp_mode = "cuda_fp16" if use_amp and device.type == "cuda" else "disabled"
    return {
        "requested_micro_batch": requested_micro_batch,
        "requested_gradient_accumulation": requested_gradient_accumulation,
        "micro_batch": resolved_micro_batch,
        "gradient_accumulation": resolved_gradient_accumulation,
        "effective_batch_size": resolved_micro_batch * resolved_gradient_accumulation,
        "max_batches": max_batches,
        "use_amp": bool(use_amp),
        "amp_mode": amp_mode,
        "num_workers": num_workers,
        "profile_cuda": bool(profile_cuda),
        "device_type": device.type,
    }


def _validate_execution_parameters(
    checkpoint: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    actual = checkpoint.get("execution_parameters")
    if not isinstance(actual, Mapping) or dict(actual) != dict(expected):
        raise PretrainRunError("checkpoint execution parameters mismatch")


def _validate_method(method: Any) -> None:
    if not isinstance(method, nn.Module):
        raise TypeError("method_factory must return an nn.Module")
    if not callable(getattr(method, "export_encoder_state_dict", None)):
        raise TypeError("pretraining method must expose export_encoder_state_dict()")


def _infer_profile_input_shape(dataset: Any, collate_fn: Callable | None) -> tuple[int, ...]:
    """Infer one per-example anchor shape without starting a training epoch."""

    sample = dataset[0]
    batch = collate_fn([sample]) if collate_fn is not None else sample
    anchor = getattr(batch, "anchor", batch)
    if isinstance(anchor, Tensor):
        shape = tuple(int(value) for value in anchor.shape)
        if len(shape) > 1 and shape[0] == 1:
            shape = shape[1:]
        if shape:
            return shape
    raise PretrainRunError("unable to infer a non-empty profile input shape")


def _cuda_profile_identity(
    device: torch.device,
    *,
    method_name: str,
    config: Mapping[str, Any],
    input_shape: tuple[int, ...],
    use_amp: bool,
) -> dict[str, Any]:
    try:
        device_name = str(torch.cuda.get_device_name(device))
        total_memory_bytes = int(torch.cuda.get_device_properties(device).total_memory)
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise PretrainRunError("unable to read CUDA device identity for hardware profile") from error
    if not device_name or total_memory_bytes < 1:
        raise PretrainRunError("CUDA device identity is incomplete")
    method_config = _method_config(config, method_name)
    view_policy = str(method_config.get("view_policy", config.get("view_policy", "single")))
    return {
        "device_name": device_name,
        "total_memory_bytes": total_memory_bytes,
        "method": method_name,
        "view_policy": view_policy,
        "encoder_arch": "metafi_r34",
        "input_shape": input_shape,
        "amp_dtype": "float16" if use_amp else "float32",
        "config_fingerprint": _canonical_fingerprint(
            {"method": method_name, "config": _extract_pretrain_config(config, method_name)}
        ),
    }


def _hardware_profile_path(repository_root: Path, method_name: str) -> Path:
    return repository_root / "result_metafi_ssl" / "hardware_profiles" / f"{method_name}.json"


def _repository_root_from_output_dir(output_dir: Path) -> Path:
    """Resolve the repository root from an output path in the new result tree."""

    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a pathlib.Path")
    try:
        assert_new_result_root(output_dir)
    except (OSError, TypeError, ValueError) as error:
        raise PretrainRunError(
            "output_dir is not inside the canonical result_metafi_ssl tree"
        ) from error
    result_roots: list[Path] = []
    current = output_dir
    while True:
        if current.name.casefold() == "result_metafi_ssl":
            result_roots.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
    if len(result_roots) != 1:
        raise PretrainRunError(
            "output_dir must contain exactly one result_metafi_ssl ancestor"
        )
    result_root = result_roots[0]
    try:
        assert_new_result_root(result_root)
    except (OSError, TypeError, ValueError) as error:
        raise PretrainRunError(
            "output_dir is not inside the canonical result_metafi_ssl tree"
        ) from error
    repository_root = result_root.parent
    if repository_root.name.casefold() in {"result", "result_metafi_ssl"}:
        raise PretrainRunError("profile repository root cannot be a result directory")
    return repository_root


def _load_or_create_cuda_profile(
    *,
    repository_root: Path,
    method_name: str,
    target_effective_batch: int,
    requested_micro_batch: int,
    gradient_accumulation: int,
    config_fingerprint: str | None,
    device_name: str | None,
    total_memory_bytes: int | None,
    encoder_arch: str | None,
    input_shape: Sequence[int] | None,
    amp_dtype: str | None,
    view_policy: str | None,
    profile_factory: Callable[[], object] | None = None,
    require_existing: bool = False,
) -> HardwareProfile | int:
    """Load an identity-matched profile or create it exactly once."""

    path = _hardware_profile_path(repository_root, method_name)
    if path.exists():
        if (
            config_fingerprint is None
            or device_name is None
            or total_memory_bytes is None
            or encoder_arch is None
            or input_shape is None
            or amp_dtype is None
            or view_policy is None
        ):
            raise PretrainRunError("CUDA profile identity is incomplete")
        profile = load_hardware_profile(path)
        validate_profile_identity(
            profile,
            device_name=device_name,
            total_memory_bytes=total_memory_bytes,
            encoder_arch=encoder_arch,
            input_shape=input_shape,
            amp_dtype=amp_dtype,
            method=method_name,
            view_policy=view_policy,
            config_fingerprint=config_fingerprint,
        )
        if profile.micro_batch > requested_micro_batch:
            raise PretrainRunError("saved hardware profile exceeds the requested micro-batch")
        if profile.micro_batch * profile.gradient_accumulation != target_effective_batch:
            raise PretrainRunError("saved hardware profile changes the target effective batch")
        return profile
    if require_existing:
        raise PretrainRunError(f"resume requires existing hardware profile: {path}")
    if profile_factory is None:
        raise PretrainRunError("CUDA profile creation requires a profile factory")
    factory = profile_factory
    result = factory()
    if isinstance(result, int):
        return result
    if not isinstance(result, HardwareProfile):
        raise PretrainRunError("CUDA profiler returned an invalid hardware profile")
    if (
        config_fingerprint is not None
        and device_name is not None
        and total_memory_bytes is not None
        and encoder_arch is not None
        and input_shape is not None
        and amp_dtype is not None
        and view_policy is not None
    ):
        validate_profile_identity(
            result,
            device_name=device_name,
            total_memory_bytes=total_memory_bytes,
            encoder_arch=encoder_arch,
            input_shape=input_shape,
            amp_dtype=amp_dtype,
            method=method_name,
            view_policy=view_policy,
            config_fingerprint=config_fingerprint,
        )
    elif result.method != method_name:
        raise PretrainRunError("CUDA profiler returned a profile for another method")
    if result.micro_batch * result.gradient_accumulation != target_effective_batch:
        raise PretrainRunError("CUDA profiler changed the target effective batch")
    write_hardware_profile(result, repository_root, filename=f"{method_name}.json")
    return result


def _profile_cuda_pre_run(
    *,
    repository_root: Path,
    method_name: str,
    target_effective_batch: int,
    requested_micro_batch: int,
    gradient_accumulation: int,
    config_fingerprint: str,
    device_name: str,
    total_memory_bytes: int,
    encoder_arch: str,
    input_shape: Sequence[int],
    amp_dtype: str,
    view_policy: str,
    factory: MethodFactory,
    dataset: Any,
    collate_fn: Callable | None,
    device: torch.device,
    num_workers: int,
    use_amp: bool,
    seed: int,
    config: Mapping[str, Any] | None = None,
) -> HardwareProfile:
    """Run a bounded three-step CUDA probe before epoch zero."""

    config = {} if config is None else config
    candidates = [
        candidate
        for candidate in range(requested_micro_batch, 1, -1)
        if target_effective_batch % candidate == 0
    ]

    def probe(candidate: int, *, steps: int) -> bool:
        _set_seed(seed)
        generator = torch.Generator(device="cpu").manual_seed(seed + 17)
        loader = _make_loader(
            dataset,
            micro_batch=candidate,
            generator=generator,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
        method = factory().to(device)
        _validate_method(method)
        optimizer = build_pretrain_optimizer(method.parameters(), config)
        scaler = _make_scaler(device, use_amp)
        iterator = iter(loader)
        try:
            for _ in range(steps):
                batch = _move_batch(next(iterator), device)
                optimizer.zero_grad(set_to_none=True)
                with _autocast_context(device, use_amp):
                    result = method(batch)
                if not isinstance(result, SSLStepOutput) or not torch.isfinite(result.loss):
                    raise PretrainRunError("pre-run profile produced a non-finite loss")
                scaler.scale(result.loss / gradient_accumulation).backward()
            return True
        finally:
            del method, optimizer, scaler, loader, iterator

    def clear_cache() -> None:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

    def reset_peak() -> None:
        with torch.cuda.device(device):
            torch.cuda.reset_peak_memory_stats(device)

    def read_peak() -> int:
        with torch.cuda.device(device):
            return max(
                int(torch.cuda.max_memory_allocated(device)),
                int(torch.cuda.max_memory_reserved(device)),
            )

    return find_safe_micro_batch(
        candidates=candidates,
        target_effective_batch=target_effective_batch,
        probe=probe,
        device_name=device_name,
        total_memory_bytes=total_memory_bytes,
        method=method_name,
        view_policy=view_policy,
        encoder_arch=encoder_arch,
        input_shape=input_shape,
        amp_dtype=amp_dtype,
        config_fingerprint=config_fingerprint,
        probe_steps=3,
        clear_cache=clear_cache,
        reset_peak=reset_peak,
        read_peak=read_peak,
    )


def _run_profile(
    factory: MethodFactory,
    dataset: Any,
    collate_fn: Callable | None,
    device: torch.device,
    micro_batch: int,
    gradient_accumulation: int,
    num_workers: int,
    use_amp: bool,
    seed: int,
    method_name: str = "simclr",
    config: Mapping[str, Any] | None = None,
) -> HardwareProfile:
    """Compatibility wrapper for the real identity-bound CUDA profiler."""

    if device.type != "cuda":
        raise PretrainRunError("CUDA profile requested for a non-CUDA device")
    config = {} if config is None else config
    input_shape = _infer_profile_input_shape(dataset, collate_fn)
    identity = _cuda_profile_identity(
        device,
        method_name=method_name,
        config=config,
        input_shape=input_shape,
        use_amp=use_amp,
    )
    return _profile_cuda_pre_run(
        repository_root=Path("."),
        method_name=method_name,
        target_effective_batch=micro_batch * gradient_accumulation,
        requested_micro_batch=micro_batch,
        gradient_accumulation=gradient_accumulation,
        config_fingerprint=identity["config_fingerprint"],
        device_name=identity["device_name"],
        total_memory_bytes=identity["total_memory_bytes"],
        encoder_arch=identity["encoder_arch"],
        input_shape=identity["input_shape"],
        amp_dtype=identity["amp_dtype"],
        view_policy=identity["view_policy"],
        factory=factory,
        dataset=dataset,
        collate_fn=collate_fn,
        device=device,
        num_workers=num_workers,
        use_amp=use_amp,
        seed=seed,
        config=config,
    )


def _restore_training_state(
    checkpoint: Mapping[str, Any],
    method: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    train_generator: torch.Generator,
    execution_parameters: Mapping[str, Any],
) -> int:
    try:
        _validate_execution_parameters(checkpoint, execution_parameters)
        method.load_state_dict(checkpoint["method_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        restore_rng_state(checkpoint, train_generator)
    except PretrainRunError:
        raise
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise PretrainRunError("checkpoint training state cannot be restored") from error
    return int(checkpoint["next_epoch"])


def run_pretraining(
    *,
    dataset_root: str | Path,
    config: Mapping[str, Any],
    method_name: str,
    manifest_path: str | Path,
    output_dir: str | Path,
    device: str | torch.device,
    epochs: int | None = None,
    max_batches: int | None = None,
    micro_batch: int | None = None,
    gradient_accumulation: int = 1,
    resume: bool = False,
    seed: int = 42,
    num_workers: int = 0,
    use_amp: bool = True,
    dataset: Any | None = None,
    collate_fn: Callable | None = None,
    method_factory: MethodFactory | None = None,
    audit_path: str | Path | None = None,
    profile_cuda: bool = True,
) -> Path:
    """Run or resume one audited pretraining job and return its run directory."""

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    source_config = copy.deepcopy(dict(config))
    config = resolve_pretrain_config(config)
    config["dataset_root"] = str(Path(dataset_root).resolve())
    if method_name not in {"simclr", "moco", "swav", "relpos", "mfm", "mae"}:
        raise ValueError(f"unsupported MetaFi SSL method: {method_name!r}")
    if not isinstance(gradient_accumulation, int) or isinstance(gradient_accumulation, bool) or gradient_accumulation < 1:
        raise ValueError("gradient_accumulation must be a positive integer")
    if epochs is None:
        epochs = int(config.get("epochs", config.get("num_epochs", 1)))
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if max_batches is not None and (not isinstance(max_batches, int) or isinstance(max_batches, bool) or max_batches < 1):
        raise ValueError("max_batches must be a positive integer")
    if micro_batch is None:
        micro_batch = int(config.get("micro_batch", config.get("batch_size", 2)))
    if not isinstance(micro_batch, int) or isinstance(micro_batch, bool) or micro_batch < 2:
        raise ValueError("micro_batch must be an integer >= 2")
    if not isinstance(num_workers, int) or isinstance(num_workers, bool) or num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    _set_seed(seed)
    manifest = load_audited_manifest(manifest_path, audit_path)
    if dataset is None:
        dataset = MetaFiPretrainDataset(
            str(dataset_root),
            manifest.pretrain_keys,
            protocol=manifest.protocol,
            split=manifest.split,
            scope=manifest.scope,
            manifest_fingerprint=manifest.fingerprint(),
        )
    # 默认 DataLoader 无法处理 PretrainSample 数据类；统一使用专用 collate。
    collate_fn = _resolve_collate_fn(collate_fn)
    if method_factory is None:
        method_factory = _build_default_factory(method_name, config, seed)

    log_every_batches = int(config.get("log_every_batches", 20))
    if log_every_batches < 1:
        raise ValueError("log_every_batches must be a positive integer")
    print(
        f"[预训练] method={method_name} | sequences={len(manifest.pretrain_keys)} | "
        f"frames={len(dataset):,} | micro_batch={micro_batch} | epochs={epochs}",
        flush=True,
    )

    if device is None:
        raise ValueError("device is required")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but CUDA is unavailable")
    output = Path(output_dir)
    try:
        assert_new_result_root(output)
    except (OSError, TypeError, ValueError) as error:
        raise PretrainRunError(
            "output_dir is not inside the canonical result_metafi_ssl tree"
        ) from error
    profiled_micro_batch = micro_batch
    if target_device.type == "cuda" and (profile_cuda or resume):
        repository_root = _repository_root_from_output_dir(output)
        profile_path = _hardware_profile_path(repository_root, method_name)
        expected_identity: dict[str, Any] = {}
        if profile_path.exists():
            input_shape = _infer_profile_input_shape(dataset, collate_fn)
            expected_identity = _cuda_profile_identity(
                target_device,
                method_name=method_name,
                config=config,
                input_shape=input_shape,
                use_amp=use_amp,
            )
        profile = _load_or_create_cuda_profile(
            repository_root=repository_root,
            method_name=method_name,
            target_effective_batch=micro_batch * gradient_accumulation,
            requested_micro_batch=micro_batch,
            gradient_accumulation=gradient_accumulation,
            config_fingerprint=expected_identity.get("config_fingerprint"),
            device_name=expected_identity.get("device_name"),
            total_memory_bytes=expected_identity.get("total_memory_bytes"),
            encoder_arch=expected_identity.get("encoder_arch"),
            input_shape=expected_identity.get("input_shape"),
            amp_dtype=expected_identity.get("amp_dtype"),
            view_policy=expected_identity.get("view_policy"),
            profile_factory=lambda: _run_profile(
                method_factory,
                dataset,
                collate_fn,
                target_device,
                micro_batch,
                gradient_accumulation,
                num_workers,
                use_amp,
                seed,
                method_name,
                config,
            ),
            require_existing=resume,
        )
        profiled_micro_batch = (
            profile.micro_batch if isinstance(profile, HardwareProfile) else profile
        )
        if resume and not profile_cuda:
            raise PretrainRunError(
                "profile_cuda=False cannot resume a CUDA run after profile validation"
            )
    execution_parameters = _resolve_execution_parameters(
        requested_micro_batch=micro_batch,
        requested_gradient_accumulation=gradient_accumulation,
        resolved_micro_batch=profiled_micro_batch,
        max_batches=max_batches,
        use_amp=use_amp,
        device=target_device,
        num_workers=num_workers,
        profile_cuda=profile_cuda,
    )
    # 只使用预训练相关的配置参数计算指纹，使不同微调策略可复用预训练
    effective_config = {
        "method": method_name,
        "config": _extract_pretrain_config({**config, "pretrain": {**config.get("pretrain", {}), "epochs": epochs,
            "micro_batch": micro_batch, "gradient_accumulation": gradient_accumulation,
            "max_batches": max_batches, "num_workers": num_workers,
            "amp": use_amp, "profile_cuda": profile_cuda}}, method_name),
        "execution_parameters": execution_parameters,
    }
    identity = make_pretrain_identity(
        manifest,
        method_name,
        seed,
        _canonical_fingerprint(effective_config),
    )
    if resume:
        saved = RunIdentity.from_dict(json.loads((output / "run_identity.json").read_text(encoding="utf-8")))
        legacy_fingerprint = _canonical_fingerprint({
            "method": method_name, "config": _legacy_pretrain_config(source_config),
            "execution_parameters": execution_parameters,
        })
        legacy_identity = make_pretrain_identity(manifest, method_name, seed, legacy_fingerprint)
        if source_config.get("pretrain") and saved == legacy_identity:
            identity = saved
    assert_new_result_root(output)
    if not resume and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    micro_batch = int(execution_parameters["micro_batch"])
    gradient_accumulation = int(execution_parameters["gradient_accumulation"])
    _set_seed(seed)
    method = method_factory().to(target_device)
    _validate_method(method)
    optimizer = build_pretrain_optimizer(method.parameters(), config)
    scheduler = build_pretrain_scheduler(optimizer, config)
    scaler = _make_scaler(target_device, use_amp)
    train_generator = torch.Generator(device="cpu").manual_seed(seed + 17)
    def record_config():
        resolved_config = {**dict(config), "epochs": epochs, "pretrain": effective_config["config"]["pretrain"]}
        if not (output / "config.yaml").is_file():
            _atomic_text(output / "config.yaml", yaml.safe_dump(resolved_config, allow_unicode=True, sort_keys=False))
        write_experiment_config(
            output, stage="pretrain", config=resolved_config,
            execution={**execution_parameters, "epochs": epochs, "seed": seed,
                       "device": str(target_device), "dataset_root": str(Path(dataset_root).resolve()),
                       "optimizer": optimizer.state_dict()["param_groups"], "scheduler": scheduler.state_dict()},
            context={"source_config": source_config, "identity": identity.to_dict(),
                     "pretrain_spec": effective_config["config"]},
            artifacts={"data_manifest.json": manifest_path,
                       "leakage_audit.json": audit_path or Path(manifest_path).with_name("leakage_audit.json")},
        )

    start_epoch = 0
    if resume:
        _prepare_directory(output, identity, True)
        checkpoint = load_checkpoint(output / "latest.pth", identity)
        start_epoch = _restore_training_state(
            checkpoint,
            method,
            optimizer,
            scheduler,
            scaler,
            train_generator,
            execution_parameters,
        )
        _truncate_csv(output / "metrics.csv", start_epoch)
        record_config()
        if start_epoch >= epochs:
            if checkpoint["status"] != STATUS_COMPLETE:
                checkpoint = dict(checkpoint)
                checkpoint["status"] = STATUS_COMPLETE
                validate_checkpoint(checkpoint, identity, expected_status=STATUS_COMPLETE)
                save_checkpoint_atomic(output / "latest.pth", checkpoint)
            _atomic_text(
            output / "done.txt",
            json.dumps(
                {
                    "schema_version": 1,
                    "status": STATUS_COMPLETE,
                    "checkpoint": "latest.pth",
                    "identity": identity.to_dict(),
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ),
        )
            return output
    else:
        _prepare_directory(output, identity, False)
        record_config()

    csv_path = output / "metrics.csv"
    try:
        for epoch in range(start_epoch, epochs):
            method.train()
            loader = _make_loader(
                dataset,
                micro_batch=micro_batch,
                generator=train_generator,
                collate_fn=collate_fn,
                num_workers=num_workers,
            )
            optimizer.zero_grad(set_to_none=True)
            loss_total = 0.0
            batch_count = 0
            pending = 0
            optimizer_steps = 0
            print(f"[预训练] E{epoch:03d} batches={len(loader)}", flush=True)
            try:
                for batch in loader:
                    if max_batches is not None and batch_count >= max_batches:
                        break
                    batch_count += 1
                    moved = _move_batch(batch, target_device)
                    with _autocast_context(target_device, use_amp):
                        result = method(moved)
                    if not isinstance(result, SSLStepOutput):
                        raise PretrainRunError("SSL method must return SSLStepOutput")
                    loss = result.loss
                    if not isinstance(loss, Tensor) or loss.ndim != 0 or not torch.isfinite(loss).item():
                        raise PretrainRunError("pretraining loss must be finite")
                    scaled_loss = loss / gradient_accumulation
                    scaler.scale(scaled_loss).backward()
                    pending += 1
                    loss_total += float(loss.detach())
                    if batch_count == 1 or batch_count % log_every_batches == 0:
                        print(
                            f"  E{epoch:03d} B{batch_count:05d}/{len(loader)} "
                            f"loss={loss.detach().float().item():.5f}",
                            flush=True,
                        )
                    if pending == gradient_accumulation:
                        optimizer_steps += int(_optimizer_step(scaler, optimizer))
                        optimizer.zero_grad(set_to_none=True)
                        pending = 0
                if batch_count == 0:
                    raise PretrainRunError("pretraining epoch produced no batches")
                if pending:
                    optimizer_steps += int(_optimizer_step(scaler, optimizer))
                    optimizer.zero_grad(set_to_none=True)
            except RuntimeError as error:
                if _is_cuda_oom(error):
                    raise PretrainRunError(
                        "CUDA OOM after training began; run stopped without changing execution identity"
                    ) from error
                raise
            if optimizer_steps > 0:
                scheduler.step()
            metrics = {"loss": loss_total / batch_count}
            in_progress = build_checkpoint_payload(
                identity=identity,
                model=method,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                next_epoch=epoch + 1,
            status=STATUS_IN_PROGRESS,
            train_generator=train_generator,
            execution_parameters=execution_parameters,
        )
            _append_metrics(csv_path, epoch, metrics, batch_count)
            save_checkpoint_atomic(output / "latest.pth", in_progress)

        final_checkpoint = build_checkpoint_payload(
            identity=identity,
            model=method,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            next_epoch=epochs,
            status=STATUS_COMPLETE,
            train_generator=train_generator,
            execution_parameters=execution_parameters,
        )
        validate_checkpoint(final_checkpoint, identity, expected_status=STATUS_COMPLETE)
        save_checkpoint_atomic(output / "latest.pth", final_checkpoint)
        validated = load_checkpoint(output / "latest.pth", identity, expected_status=STATUS_COMPLETE)
        validate_checkpoint(validated, identity, expected_status=STATUS_COMPLETE)
        _atomic_text(
            output / "done.txt",
            json.dumps(
                {
                    "schema_version": 1,
                    "status": STATUS_COMPLETE,
                    "checkpoint": "latest.pth",
                    "identity": identity.to_dict(),
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ),
        )
        return output
    except PretrainRunError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise PretrainRunError(str(error)) from error
    finally:
        del method, optimizer, scheduler, scaler
        if target_device.type == "cuda":
            torch.cuda.empty_cache()


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        config = resolve_pretrain_config(_load_config(args.config_file))
        run_pretraining(
            dataset_root=args.dataset_root,
            config=config,
            method_name=args.method,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            device=args.device,
            epochs=args.epochs,
            max_batches=args.max_batches,
            micro_batch=args.micro_batch,
            gradient_accumulation=args.gradient_accumulation,
            resume=args.resume,
            seed=args.seed,
            num_workers=args.num_workers,
            use_amp=not args.no_amp,
            profile_cuda=not args.no_cuda_profile,
        )
        return 0
    except (PretrainRunError, OSError, TypeError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
