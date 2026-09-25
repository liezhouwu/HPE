"""Audited ViT-MAE pretraining for the ViT-csi-small backbone.

This is the ViT line's own entry point: it neither imports nor modifies the MetaFi
runner state.  It reuses the shared building blocks -- audited manifest loading,
the CSI-only pretraining dataset, the schema-v1 checkpoint machinery and the
runner's optimizer/scheduler/loop helpers -- so a ViT-MAE run stays leakage-audited,
resumable and self-describing.  Outputs land in the dedicated subtree
``result_metafi_ssl/vit/pretrain/<tag>/`` so the canonical result root guard holds
while the MetaFi method directories stay untouched.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any

import torch
import yaml
from torch import Tensor, nn

_BASE = Path(__file__).resolve().parents[2]
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))

from mmfi_wifi.data_manifest import DataManifest, audit_manifest
from mmfi_wifi.experiment_config import write_experiment_config
from mmfi_wifi.run_identity import RunIdentity
from pose_ssl.metafi.pretrain_subset import _validate_fraction, make_small_pretrain_manifest, selection_summary
from pose_ssl.metafi.pretrain_checkpoint import (
    STATUS_COMPLETE,
    STATUS_IN_PROGRESS,
    build_checkpoint_payload,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint_atomic,
    validate_checkpoint,
)
from pose_ssl.metafi.pretrain_data import MetaFiPretrainDataset
from pose_ssl.vit_ssl.mae import ViTMAEMethod
from scripts.metafi_ssl.pretrain import (
    PretrainRunError,
    _append_metrics,
    _atomic_text,
    _autocast_context,
    _canonical_fingerprint,
    _is_cuda_oom,
    _make_loader,
    _make_scaler,
    _move_batch,
    _optimizer_step,
    _prepare_directory,
    _resolve_collate_fn,
    _set_seed,
    _truncate_csv,
    _validate_execution_parameters,
    build_pretrain_optimizer,
    build_pretrain_scheduler,
    load_audited_manifest,
    resolve_pretrain_config,
)


METHOD_NAME = "mae_vit"
ENCODER_ARCH = "vit_csi_small"
CHECKPOINT_FILENAME = "latest.pth"
METRICS_FILENAME = "pretrain_metrics.csv"
DONE_FILENAME = "done.txt"

# Config fields the pretraining actually consumes, i.e. the ones allowed to define
# its identity.  ``epochs``/``num_epochs``, ``pretrain_lr``/``lr`` and
# ``micro_batch``/``batch_size`` are both listed because the runners accept either
# spelling and read the same fallback chain.
_PRETRAIN_SPEC_KEYS = (
    "dataset_root",
    "pretrain_sequence_fraction",
    "epochs",
    "num_epochs",
    "vit_mae",
    "optimizer",
    "pretrain_lr",
    "lr",
    "weight_decay",
    "sgd_momentum",
    "scheduler",
    "scheduler_t_max",
    "lr_milestones",
    "lr_gamma",
    "lr_min",
    "micro_batch",
    "batch_size",
    "gradient_accumulation",
    "num_workers",
    "max_batches",
    "log_every_batches",
    "amp",
)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run audited ViT-MAE pretraining")
    parser.add_argument("dataset_root")
    parser.add_argument("config_file")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--audit", default=None, help="默认取 manifest 同目录的 leakage_audit.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument(
        "--sequence-fraction",
        type=float,
        default=None,
        help="预训练只用该比例的无标注 sequence（<1 时启用小样本子集）；"
             "缺省读配置 small_sample.pretrain_sequence_fraction，再否则 1.0",
    )
    parser.add_argument(
        "--manifest-dir",
        default=None,
        help="子集清单/审计的落盘目录；缺省为 run 目录的兄弟目录 <variant>/pretrain_manifests/<tag>",
    )
    parser.add_argument("--micro-batch", type=int, default=None)
    parser.add_argument("--gradient-accumulation", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
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


def _vit_mae_config(config: Mapping[str, Any]) -> dict[str, Any]:
    section = config.get("vit_mae", {})
    if not isinstance(section, Mapping):
        raise ValueError("vit_mae config must be a mapping")
    allowed = {"mask_ratio", "decoder_dim", "decoder_depth", "decoder_heads", "patch_h", "patch_w"}
    unknown = set(section).difference(allowed)
    if unknown:
        raise ValueError(f"unknown vit_mae config fields: {sorted(unknown)!r}")
    return dict(section)


def _resolve_sequence_fraction(config: Mapping[str, Any], requested: float | None) -> float:
    """Resolve the unlabeled-sequence fraction: CLI > ``small_sample`` > 1.0."""

    if requested is not None:
        return _validate_fraction(requested)
    section = config.get("small_sample", {})
    if section is None:
        return 1.0
    if not isinstance(section, Mapping):
        raise ValueError("small_sample config must be a mapping")
    if "pretrain_sequence_fraction" not in section:
        return 1.0
    return _validate_fraction(section["pretrain_sequence_fraction"])


def _subset_manifest(manifest: DataManifest, fraction: float, seed: int) -> DataManifest:
    """Return the audited strict subset used by small-sample pretraining."""

    subset = make_small_pretrain_manifest(manifest, fraction=fraction, seed=seed)
    audit = audit_manifest(subset)
    if not audit.passed:
        raise PretrainRunError(
            "small-sample pretraining data audit failed: " + "; ".join(audit.violations)
        )
    return subset


def _default_manifest_dir(output_dir: Path) -> Path:
    """Default location of the subset manifest: a sibling of the run directory.

    The runner requires an empty output directory, so the manifest can never live
    inside it.
    """

    output_dir = Path(output_dir)
    return output_dir.parent / "pretrain_manifests" / output_dir.name


def _reusable_subset(
    manifest_path: Path,
    audit_path: Path,
    base: DataManifest,
    fraction: float,
) -> None:
    """Accept a recorded subset only while it still describes this run's boundary.

    The recorded subset is what the encoder actually saw and is never re-drawn: the
    S3 draw consumes the RNG in scene order, so re-drawing can legitimately return a
    different -- equally valid -- sample and would then look like a mismatch.  Only
    the properties that would make reuse *wrong* are re-checked, fail-closed.
    """

    stored = DataManifest.read_json(manifest_path)
    try:
        payload = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"existing small-sample audit is unreadable: {audit_path}") from error
    selection = payload.get("selection") if isinstance(payload, Mapping) else None
    reasons: list[str] = []
    if (stored.protocol, stored.split, stored.scope, stored.seed) != (
        base.protocol,
        base.split,
        base.scope,
        base.seed,
    ):
        reasons.append("数据边界不同")
    if not isinstance(selection, Mapping) or float(selection.get("fraction", -1.0)) != float(fraction):
        reasons.append(f"fraction 不是本轮的 {float(fraction)}")
    if payload.get("data_manifest_fingerprint") != stored.fingerprint():
        reasons.append("审计记录的指纹与清单不一致")
    if not set(stored.pretrain_keys).issubset(base.internal_train_keys):
        reasons.append("子集超出本轮训练边界")
    audit = audit_manifest(stored)
    if not audit.passed:
        reasons.append("泄漏审计未通过: " + "; ".join(audit.violations))
    if reasons:
        raise ValueError(
            f"existing small-sample manifest does not match this run: {manifest_path}（"
            + "；".join(reasons)
            + "）"
        )


def _small_pretrain_artifacts(
    *,
    manifest: DataManifest,
    fraction: float,
    seed: int,
    directory: Path,
) -> tuple[Path, Path]:
    """Persist the subset manifest and its audit, reusing an existing pair.

    The reuse branch is read-only: rewriting the files would invalidate the
    content hashes recorded in ``experiment_config.json``.
    """

    directory = Path(directory)
    manifest_path = directory / "pretrain_manifest.json"
    audit_path = directory / "pretrain_audit.json"
    if manifest_path.is_file() and audit_path.is_file():
        _reusable_subset(manifest_path, audit_path, manifest, fraction)
        return manifest_path, audit_path

    subset = _subset_manifest(manifest, fraction, seed)
    audit = audit_manifest(subset)
    directory.mkdir(parents=True, exist_ok=True)
    subset.write_json(manifest_path)
    _atomic_text(
        audit_path,
        json.dumps(
            {
                "passed": True,
                "data_manifest_fingerprint": subset.fingerprint(),
                "counts": audit.counts,
                "selection": {"fraction": float(fraction), **selection_summary(subset)},
            },
            ensure_ascii=False,
            indent=2,
        ),
    )
    summary = selection_summary(subset)
    print(
        f"[ViT-MAE] 小样本子集 sequences={summary['pretrain_sequences']} | fraction={fraction} | "
        f"per-action {summary['min_sequences_per_action']}~{summary['max_sequences_per_action']}",
        flush=True,
    )
    return manifest_path, audit_path


def _pretrain_spec(resolved: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only the settings the pretraining itself consumes.

    HAS to stay in sync with what ``run_pretraining`` and the optimizer/scheduler
    builders read: everything else (``label_budget``, ``finetune``, ``all_splits``,
    ``single_split``, ``device`` …) is downstream-only, and folding it into the
    identity would make one 20% encoder unusable for another label budget even
    though the pretraining is unchanged -- the same rule the MetaFi line applies
    in ``_extract_pretrain_config``.
    """

    return {key: resolved[key] for key in _PRETRAIN_SPEC_KEYS if key in resolved}


def _pretrain_identity(
    manifest: DataManifest,
    resolved: Mapping[str, Any],
    execution_parameters: Mapping[str, Any],
    seed: int,
) -> RunIdentity:
    """Bind the resolved config, the data boundary and execution settings to one identity."""

    config_fingerprint = _canonical_fingerprint(
        {
            "method": METHOD_NAME,
            "config": _pretrain_spec(resolved),
            "execution_parameters": dict(execution_parameters),
        }
    )
    return make_identity(manifest, METHOD_NAME, seed, config_fingerprint)


def make_identity(
    manifest: Any,
    method_name: str,
    seed: int,
    config_fingerprint: str,
) -> RunIdentity:
    """Construct the ViT pretraining identity used by schema-v1 checkpoints."""

    return RunIdentity(
        method=method_name,
        encoder_arch=ENCODER_ARCH,
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


def _identity_without_fingerprint(identity: RunIdentity) -> dict[str, Any]:
    """The independently checkable half of an identity: everything but the config hash."""

    payload = identity.to_dict()
    payload.pop("config_fingerprint")
    return payload


def _stored_resolved(output_dir: Path) -> dict[str, Any] | None:
    """The resolved config an existing run recorded, when that record is readable."""

    try:
        payload = json.loads(
            (Path(output_dir) / "experiment_config.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return None
    config = payload.get("config") if isinstance(payload, Mapping) else None
    return dict(config) if isinstance(config, Mapping) else None


def _resumable_identity(
    output_dir: Path,
    identity: RunIdentity,
    resolved: Mapping[str, Any],
) -> RunIdentity:
    """Adopt the identity of an existing run when only the fingerprint scope differs.

    Runs created before the fingerprint was narrowed hashed the whole config file, so
    editing a downstream-only field changed their identity although the pretraining is
    unchanged.  The stored identity is adopted only when every other identity field
    matches this run's data boundary *and* the pretraining settings recorded next to it
    equal the ones now in effect; any real change stays fail-closed and surfaces as the
    usual ``resume RunIdentity mismatch``.
    """

    output_dir = Path(output_dir)
    try:
        raw = (output_dir / "run_identity.json").read_text(encoding="utf-8")
        saved = RunIdentity.from_dict(json.loads(raw))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return identity
    if saved == identity:
        return identity
    if _identity_without_fingerprint(saved) != _identity_without_fingerprint(identity):
        return identity
    stored = _stored_resolved(output_dir)
    if stored is None or _pretrain_spec(stored) != _pretrain_spec(resolved):
        return identity
    print(
        f"[ViT-MAE] 该目录的 identity 按旧口径（整份 config）生成，"
        f"预训练相关配置一致，按原 identity 复用: {output_dir}",
        flush=True,
    )
    return saved


def _resolve_execution_parameters(
    *,
    micro_batch: int,
    gradient_accumulation: int,
    max_batches: int | None,
    use_amp: bool,
    device: torch.device,
    num_workers: int,
    sequence_fraction: float,
) -> dict[str, Any]:
    """Return the execution settings bound to this ViT pretraining run."""

    if micro_batch < 2:
        raise ValueError("micro_batch must be an integer >= 2")
    if gradient_accumulation < 1:
        raise ValueError("gradient_accumulation must be a positive integer")
    return {
        "sequence_fraction": float(sequence_fraction),
        "requested_micro_batch": micro_batch,
        "requested_gradient_accumulation": gradient_accumulation,
        "micro_batch": micro_batch,
        "gradient_accumulation": gradient_accumulation,
        "effective_batch_size": micro_batch * gradient_accumulation,
        "max_batches": max_batches,
        "use_amp": bool(use_amp),
        "amp_mode": "cuda_fp16" if use_amp and device.type == "cuda" else "disabled",
        "num_workers": num_workers,
        "device_type": device.type,
    }


def _restore_training_state(
    checkpoint: Mapping[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    train_generator: torch.Generator,
    execution_parameters: Mapping[str, Any],
) -> int:
    try:
        _validate_execution_parameters(checkpoint, execution_parameters)
        model.load_state_dict(checkpoint["method_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        restore_rng_state(checkpoint, train_generator)
    except PretrainRunError:
        raise
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise PretrainRunError("checkpoint training state cannot be restored") from error
    return int(checkpoint["next_epoch"])


def _write_done(directory: Path, identity: RunIdentity) -> None:
    _atomic_text(
        directory / DONE_FILENAME,
        json.dumps(
            {
                "schema_version": 1,
                "status": STATUS_COMPLETE,
                "checkpoint": CHECKPOINT_FILENAME,
                "identity": identity.to_dict(),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ),
    )


def run_pretraining(
    *,
    dataset_root: str,
    config: Mapping[str, Any],
    manifest_path: str | Path,
    output_dir: Path,
    device: str,
    audit_path: str | Path | None = None,
    seed: int = 42,
    epochs: int | None = None,
    max_batches: int | None = None,
    micro_batch: int | None = None,
    gradient_accumulation: int | None = None,
    num_workers: int | None = None,
    use_amp: bool = True,
    resume: bool = False,
    sequence_fraction: float | None = None,
    manifest_dir: str | Path | None = None,
) -> Path:
    """Run or resume one audited ViT-MAE pretraining job and return its run directory.

    With ``sequence_fraction < 1`` the run trains on an audited stratified subset of
    the unlabeled training sequences (small-sample mode); the subset manifest and its
    audit are persisted next to the run so fine-tuning can bind the checkpoint to the
    data it actually saw.
    """

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a pathlib.Path")
    source_config = {key: value for key, value in config.items() if key != "pretrain"}
    resolved = resolve_pretrain_config(config)
    resolved["dataset_root"] = str(Path(dataset_root).resolve())

    if epochs is None:
        epochs = int(resolved.get("epochs", resolved.get("num_epochs", 1)))
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if max_batches is not None and (not isinstance(max_batches, int) or isinstance(max_batches, bool) or max_batches < 1):
        raise ValueError("max_batches must be a positive integer")
    if micro_batch is None:
        micro_batch = int(resolved.get("micro_batch", resolved.get("batch_size", 2)))
    if not isinstance(micro_batch, int) or isinstance(micro_batch, bool):
        raise ValueError("micro_batch must be an integer")
    if gradient_accumulation is None:
        gradient_accumulation = int(resolved.get("gradient_accumulation", 1))
    if not isinstance(gradient_accumulation, int) or isinstance(gradient_accumulation, bool) or gradient_accumulation < 1:
        raise ValueError("gradient_accumulation must be a positive integer")
    if num_workers is None:
        num_workers = int(resolved.get("num_workers", 0))
    if not isinstance(num_workers, int) or isinstance(num_workers, bool) or num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    _set_seed(seed)
    fraction = _resolve_sequence_fraction(config, sequence_fraction)
    resolved["pretrain_sequence_fraction"] = fraction
    base_manifest = load_audited_manifest(manifest_path, audit_path)
    if fraction < 1.0:
        # The subset is sampled with the audit manifest's own seed: the finetuning
        # side compares `seed` as a fixed boundary field, so a mismatch here would
        # only surface after the whole pretraining had been paid for.
        if seed != base_manifest.seed:
            raise ValueError(
                "sequence_fraction < 1 samples the subset with the audited manifest's seed; "
                f"got seed={seed} but the manifest was audited with seed={base_manifest.seed}"
            )
        directory = (
            Path(manifest_dir) if manifest_dir is not None else _default_manifest_dir(output_dir)
        )
        manifest_path, audit_path = _small_pretrain_artifacts(
            manifest=base_manifest,
            fraction=fraction,
            seed=base_manifest.seed,
            directory=directory,
        )
        manifest = load_audited_manifest(manifest_path, audit_path)
    else:
        manifest = base_manifest
    dataset = MetaFiPretrainDataset(
        str(Path(dataset_root).resolve()),
        manifest.pretrain_keys,
        neighbor_offsets=(),
        protocol=manifest.protocol,
        split=manifest.split,
        scope=manifest.scope,
        manifest_fingerprint=manifest.fingerprint(),
    )
    requested_device = torch.device(device)
    target_device = (
        requested_device
        if requested_device.type == "cpu" or torch.cuda.is_available()
        else torch.device("cpu")
    )
    use_amp = bool(use_amp) and target_device.type == "cuda"
    execution_parameters = _resolve_execution_parameters(
        micro_batch=micro_batch,
        gradient_accumulation=gradient_accumulation,
        max_batches=max_batches,
        use_amp=use_amp,
        device=target_device,
        num_workers=num_workers,
        sequence_fraction=fraction,
    )
    identity = _pretrain_identity(manifest, resolved, execution_parameters, seed)

    model = ViTMAEMethod(**_vit_mae_config(resolved)).to(target_device)
    optimizer = build_pretrain_optimizer(model.parameters(), resolved)
    scheduler = build_pretrain_scheduler(optimizer, resolved)
    scaler = _make_scaler(target_device, use_amp)
    train_generator = torch.Generator(device="cpu").manual_seed(seed + 17)
    collate_fn = _resolve_collate_fn(None)
    output = Path(output_dir)
    log_every_batches = int(resolved.get("log_every_batches", 20))
    if log_every_batches < 1:
        raise ValueError("log_every_batches must be a positive integer")

    print(
        f"[ViT-MAE] sequences={len(manifest.pretrain_keys)} | frames={len(dataset):,} | "
        f"micro_batch={micro_batch} x accum={gradient_accumulation} | epochs={epochs} | "
        f"fraction={fraction}",
        flush=True,
    )

    def record_config() -> None:
        write_experiment_config(
            output,
            stage="vit_pretrain",
            config=resolved,
            execution={**execution_parameters, "epochs": epochs, "seed": seed, "device": str(target_device)},
            context={"source_config": source_config, "identity": identity.to_dict()},
            artifacts={
                "data_manifest.json": manifest_path,
                "leakage_audit.json": audit_path or Path(manifest_path).with_name("leakage_audit.json"),
            },
        )

    start_epoch = 0
    if resume:
        identity = _resumable_identity(output, identity, resolved)
        _prepare_directory(output, identity, True)
        checkpoint = load_checkpoint(output / CHECKPOINT_FILENAME, identity)
        start_epoch = _restore_training_state(
            checkpoint, model, optimizer, scheduler, scaler, train_generator, execution_parameters
        )
        _truncate_csv(output / METRICS_FILENAME, start_epoch)
        record_config()
        if start_epoch >= epochs:
            if checkpoint["status"] != STATUS_COMPLETE:
                completed = dict(checkpoint)
                completed["status"] = STATUS_COMPLETE
                validate_checkpoint(completed, identity, expected_status=STATUS_COMPLETE)
                save_checkpoint_atomic(output / CHECKPOINT_FILENAME, completed)
            _write_done(output, identity)
            return output
    else:
        _prepare_directory(output, identity, False)
        record_config()

    csv_path = output / METRICS_FILENAME
    try:
        for epoch in range(start_epoch, epochs):
            model.train()
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
            print(f"[ViT-MAE] E{epoch:03d} batches={len(loader)}", flush=True)
            try:
                for batch in loader:
                    if max_batches is not None and batch_count >= max_batches:
                        break
                    batch_count += 1
                    moved = _move_batch(batch, target_device)
                    with _autocast_context(target_device, use_amp):
                        loss, log = model(moved.anchor)
                    if not isinstance(loss, Tensor) or loss.ndim != 0 or not torch.isfinite(loss).item():
                        raise PretrainRunError("pretraining loss must be finite")
                    scaler.scale(loss / gradient_accumulation).backward()
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
                    raise PretrainRunError("CUDA OOM after training began; run stopped") from error
                raise
            if optimizer_steps > 0:
                scheduler.step()
            metrics = {"loss": loss_total / batch_count}
            if isinstance(log, Mapping) and "n_keep" in log:
                metrics["n_keep"] = float(log["n_keep"])
            in_progress = build_checkpoint_payload(
                identity=identity,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                next_epoch=epoch + 1,
                status=STATUS_IN_PROGRESS,
                train_generator=train_generator,
                execution_parameters=execution_parameters,
            )
            _append_metrics(csv_path, epoch, metrics, batch_count)
            save_checkpoint_atomic(output / CHECKPOINT_FILENAME, in_progress)

        final_checkpoint = build_checkpoint_payload(
            identity=identity,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            next_epoch=epochs,
            status=STATUS_COMPLETE,
            train_generator=train_generator,
            execution_parameters=execution_parameters,
        )
        validate_checkpoint(final_checkpoint, identity, expected_status=STATUS_COMPLETE)
        save_checkpoint_atomic(output / CHECKPOINT_FILENAME, final_checkpoint)
        validated = load_checkpoint(output / CHECKPOINT_FILENAME, identity, expected_status=STATUS_COMPLETE)
        validate_checkpoint(validated, identity, expected_status=STATUS_COMPLETE)
        _write_done(output, identity)
        return output
    except PretrainRunError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise PretrainRunError(str(error)) from error
    finally:
        del model, optimizer, scheduler, scaler
        if target_device.type == "cuda":
            torch.cuda.empty_cache()


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
    except SystemExit as error:
        return int(error.code or 0)
    try:
        config = _load_config(args.config_file)
        run_pretraining(
            dataset_root=args.dataset_root,
            config=config,
            manifest_path=args.manifest,
            audit_path=args.audit,
            output_dir=Path(args.output_dir),
            device=args.device,
            seed=args.seed,
            epochs=args.epochs,
            max_batches=args.max_batches,
            micro_batch=args.micro_batch,
            gradient_accumulation=args.gradient_accumulation,
            num_workers=args.num_workers,
            use_amp=not args.no_amp,
            resume=args.resume,
            sequence_fraction=args.sequence_fraction,
            manifest_dir=args.manifest_dir,
        )
    except (OSError, TypeError, ValueError, PretrainRunError) as error:
        print(f"[ERROR] {error}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
