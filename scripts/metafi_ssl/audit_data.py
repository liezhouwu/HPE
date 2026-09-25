"""Fail-closed manifest, label-budget, and axis-statistics audit CLI."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys
import uuid
from typing import Any, Sequence

import yaml

_BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BASE))

from mmfi_wifi.data_manifest import (  # noqa: E402
    DataManifest,
    LeakageAudit,
    audit_manifest,
    build_data_manifest,
)
from pose_ssl.metafi.label_budget import (  # noqa: E402
    LabelManifest,
    sample_kshot,
    sample_subject_budget,
)
from pose_ssl.metafi.label_stats import compute_axis_stats  # noqa: E402


_ARTIFACT_NAMES = (
    "data_manifest.json",
    "leakage_audit.json",
    "fewshot_manifest.json",
    "label_stats.json",
)
_KSHOT_PATTERN = re.compile(r"^(1|2|4|8|16)shot$")
_SUBJECT_PATTERN = re.compile(r"^(\d+)subjects?$")


class AuditFailure(RuntimeError):
    """Raised when a data boundary is unsafe or an output contract is invalid."""


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审计 MetaFi SSL 数据边界并生成不可变 JSON 工件")
    parser.add_argument("dataset_root", type=str)
    parser.add_argument("config_file", type=str)
    parser.add_argument("--protocol", required=True, choices=("protocol1", "protocol2", "protocol3"))
    parser.add_argument(
        "--split",
        required=True,
        choices=("random_split", "cross_subject_split", "cross_scene_split"),
    )
    parser.add_argument("--scope", required=True, choices=("strict", "transductive"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--label-budget", required=True, dest="label_budget")
    parser.add_argument("--output-dir", required=True, type=str)
    return parser.parse_args(argv)


def _load_config(path: str) -> dict[str, Any]:
    source = Path(path)
    try:
        loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as error:
        raise AuditFailure(f"无法读取 config_file: {source}") from error
    except yaml.YAMLError as error:
        raise AuditFailure(f"config_file YAML 无法解析: {source}") from error
    if not isinstance(loaded, dict):
        raise AuditFailure("config_file 根节点必须是对象")
    return loaded


def _is_legacy_result_path(output_dir: Path) -> bool:
    legacy_root = (_BASE / "result").resolve()
    candidate = output_dir.resolve()
    try:
        candidate.relative_to(legacy_root)
    except ValueError:
        return False
    return True


def _select_labels(manifest: DataManifest, label_budget: str) -> LabelManifest:
    kshot = _KSHOT_PATTERN.fullmatch(label_budget)
    if kshot:
        selected = sample_kshot(
            manifest.internal_train_keys,
            int(kshot.group(1)),
            manifest.seed,
            manifest.split,
        )
        return LabelManifest.create(label_budget, selected)

    subjects = _SUBJECT_PATTERN.fullmatch(label_budget)
    if subjects:
        selected = sample_subject_budget(
            manifest.internal_train_keys,
            int(subjects.group(1)),
            manifest.seed,
        )
        return LabelManifest.create(label_budget, selected)

    raise AuditFailure(
        "label-budget must be 1shot, 2shot, 4shot, 8shot, 16shot, or Nsubjects (for example 4subjects)"
    )


def _audit_payload(audit: LeakageAudit, manifest: DataManifest) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "passed": audit.passed,
        "counts": audit.counts,
        "violations": list(audit.violations),
        "data_manifest_fingerprint": manifest.fingerprint(),
    }


def _label_stats_payload(stats: Any, label_manifest: LabelManifest) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "mean": list(stats.mean),
        "std": list(stats.std),
        "source_fingerprint": stats.source_fingerprint,
        "label_manifest_fingerprint": label_manifest.fingerprint,
        "labeled_sequences": len(label_manifest.selected_keys),
    }


def _atomic_write_artifacts(output_dir: Path, payloads: dict[str, dict[str, Any]]) -> None:
    """Publish all JSON artifacts by atomically renaming a fully staged directory."""

    if set(payloads) != set(_ARTIFACT_NAMES):
        raise AuditFailure("工件集合不完整")
    if output_dir.exists():
        raise AuditFailure(f"output-dir 已存在，拒绝覆盖: {output_dir}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.tmp-{uuid.uuid4().hex}"
    try:
        staging.mkdir()
        for name in _ARTIFACT_NAMES:
            destination = staging / name
            serialized = json.dumps(
                payloads[name],
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ) + "\n"
            with destination.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(staging, output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def run_audit(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    if _is_legacy_result_path(output_dir):
        raise AuditFailure(f"禁止写入 legacy result 根目录: {_BASE / 'result'}")

    config = _load_config(args.config_file)
    manifest = build_data_manifest(
        args.dataset_root,
        config,
        args.protocol,
        args.split,
        args.seed,
        args.scope,
    )

    boundary_audit = audit_manifest(manifest)
    if not boundary_audit.passed:
        raise AuditFailure("数据边界审计失败: " + "; ".join(boundary_audit.violations))

    label_manifest = _select_labels(manifest, args.label_budget)
    audit = audit_manifest(manifest, label_manifest.selected_keys)
    if not audit.passed:
        raise AuditFailure("标签边界审计失败: " + "; ".join(audit.violations))

    stats = compute_axis_stats(args.dataset_root, label_manifest.selected_keys)


    payloads = {
        "data_manifest.json": {**manifest.to_dict(), "fingerprint": manifest.fingerprint()},
        "leakage_audit.json": _audit_payload(audit, manifest),
        "fewshot_manifest.json": label_manifest.to_dict(),
        "label_stats.json": _label_stats_payload(stats, label_manifest),
    }
    _atomic_write_artifacts(output_dir, payloads)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        run_audit(args)
    except AuditFailure as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2
    print("[PASS] data manifest, leakage audit, label manifest, and label statistics written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
