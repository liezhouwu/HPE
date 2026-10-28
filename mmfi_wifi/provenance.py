"""Immutable, human-readable provenance records for experiment outputs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

PROVENANCE_FILENAME = "run_provenance.json"
PROVENANCE_SCHEMA_VERSION = 1


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _identity_payload(identity: Any) -> tuple[dict[str, Any], str | None]:
    if identity is None:
        return {}, None
    payload = identity.to_dict() if hasattr(identity, "to_dict") else dict(identity)
    fingerprint = getattr(identity, "fingerprint", None)
    if not fingerprint:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return dict(payload), fingerprint


def _source_checkpoint_info(context: Mapping[str, Any] | None) -> dict[str, Any] | None:
    context = context or {}
    raw = context.get("pretrain_checkpoint")
    if not raw:
        return None
    path = Path(str(raw))
    info: dict[str, Any] = {"path": str(path.resolve())}
    if path.is_file():
        info["sha256"] = _sha256_file(path)
        try:
            import torch

            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(checkpoint, Mapping):
                for key in (
                    "method", "encoder_arch", "model_variant", "seed",
                    "epochs", "optimizer_steps", "manifest_fingerprint",
                ):
                    if key in checkpoint:
                        info[key] = checkpoint[key]
                config = checkpoint.get("config")
                if isinstance(config, Mapping):
                    pretrain = config.get("pretrain")
                    if isinstance(pretrain, Mapping):
                        for key in ("variant", "objective_version", "sequence_fraction", "epochs"):
                            if key in pretrain:
                                info[f"pretrain_{key}"] = pretrain[key]
        except Exception:
            # Provenance must remain useful even when a legacy checkpoint is not loadable.
            pass
    return info


def build_run_provenance(
    *,
    result_dir: str | os.PathLike[str],
    stage: str,
    config: Mapping[str, Any] | None = None,
    execution: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, str] | None = None,
    identity: Any = None,
) -> dict[str, Any]:
    result_path = Path(result_dir).resolve()
    identity_data, identity_fingerprint = _identity_payload(identity)
    config = dict(config or {})
    execution = dict(execution or {})
    context = dict(context or {})
    artifact_records: dict[str, Any] = {}
    for name, source in (artifacts or {}).items():
        source_path = Path(source)
        record: dict[str, Any] = {
            "source": str(source_path.resolve()),
            "local": str((result_path / name).resolve()),
        }
        if source_path.is_file():
            record["sha256"] = _sha256_file(source_path)
        artifact_records[name] = record

    wifi = config.get("wifi_jepa") if isinstance(config.get("wifi_jepa"), Mapping) else {}
    # Fine-tune snapshots nest this under wifi_jepa; the isolated JEPA
    # pretrain runner keeps input/model/pretrain at the top level.
    if not wifi and isinstance(config.get("pretrain"), Mapping):
        wifi = {
            "method": "wifi_jepa_amp",
            "model_variant": config.get("pretrain", {}).get("variant"),
            "pretrain": config.get("pretrain", {}),
        }
    pretrain = wifi.get("pretrain") if isinstance(wifi, Mapping) else {}
    if not isinstance(pretrain, Mapping):
        pretrain = {}

    method = identity_data.get("method", wifi.get("method") if isinstance(wifi, Mapping) else None)
    encoder_arch = identity_data.get("encoder_arch")
    protocol = identity_data.get("protocol", execution.get("protocol"))
    split = identity_data.get("split", execution.get("split"))
    seed = identity_data.get(
        "finetune_seed",
        identity_data.get("pretrain_seed", execution.get("init_rand_seed")),
    )
    label_budget = identity_data.get("label_budget")
    source_checkpoint = _source_checkpoint_info(context)
    source_config = context.get("source_config")

    run_label_parts = [str(value) for value in (method, encoder_arch, protocol, split) if value]
    if seed is not None:
        run_label_parts.append(f"seed{seed}")
    if label_budget:
        run_label_parts.append(str(label_budget))

    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "run_id": identity_fingerprint or result_path.name,
        "run_label": "/".join(run_label_parts),
        "stage": stage,
        "result_dir": str(result_path),
        "identity": identity_data,
        "identity_fingerprint": identity_fingerprint,
        "source": {
            "config": str(Path(source_config).resolve()) if source_config else None,
            "pretrain_checkpoint": source_checkpoint,
        },
        "experiment": {
            "method": method,
            "encoder_arch": encoder_arch,
            "protocol": protocol,
            "split": split,
            "label_budget": label_budget,
            "pretrain_seed": identity_data.get("pretrain_seed"),
            "finetune_seed": identity_data.get("finetune_seed"),
            "encoder_readout": config.get("encoder_readout"),
            "model_variant": wifi.get("model_variant") if isinstance(wifi, Mapping) else None,
            "pretrain_variant": pretrain.get("variant"),
            "objective_version": pretrain.get("objective_version"),
        },
        "fingerprints": {
            "config": execution.get("config_fingerprint"),
            "execution": execution.get("execution_fingerprint"),
            "manifest": execution.get("data_manifest_fingerprint", identity_data.get("manifest_fingerprint")),
            "label_manifest": execution.get("label_manifest_fingerprint", identity_data.get("label_manifest_fingerprint")),
        },
        "artifacts": artifact_records,
    }


def write_run_provenance(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> Path:
    target = Path(path) / PROVENANCE_FILENAME
    if target.is_file():
        return target
    temporary = target.with_suffix(".json.tmp")
    target.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(target)
    return target


def load_run_provenance(path: str | os.PathLike[str]) -> dict[str, Any] | None:
    target = Path(path) / PROVENANCE_FILENAME
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None
