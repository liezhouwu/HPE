"""Fail-closed binding between a few-shot manifest and its RunIdentity."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

from pose_ssl.metafi.label_budget import (
    LABEL_MANIFEST_ALGORITHM_VERSION,
    LABEL_MANIFEST_SCHEMA_VERSION,
    LabelManifest,
)
from .sequence_keys import SequenceKey


BOUND_LABEL_MANIFEST_FIELDS = frozenset({
    "schema_version",
    "algorithm_version",
    "mode",
    "selected_keys",
    "fingerprint",
    "run_identity",
    "run_identity_fingerprint",
})


def _identity_dict(identity: Any) -> dict[str, Any]:
    payload = identity.to_dict()
    if not isinstance(payload, Mapping):
        raise TypeError("identity.to_dict() 必须返回对象")
    return dict(payload)


def _identity_fingerprint(identity: Any, identity_payload: Mapping[str, Any]) -> str:
    value = getattr(identity, "fingerprint", None)
    if isinstance(value, str):
        return value
    canonical = getattr(identity, "canonical_json", None)
    if callable(canonical):
        return hashlib.sha256(canonical().encode("utf-8")).hexdigest()
    encoded = json.dumps(
        dict(identity_payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    )
    return hashlib.sha256((encoded + "\n").encode("utf-8")).hexdigest()


def bound_label_manifest_payload(label_manifest: LabelManifest, identity: Any) -> dict[str, Any]:
    if not isinstance(label_manifest, LabelManifest):
        raise TypeError("label_manifest 必须是 LabelManifest")
    identity_payload = _identity_dict(identity)
    expected_label_fingerprint = identity_payload.get("label_manifest_fingerprint")
    if not isinstance(expected_label_fingerprint, str):
        raise ValueError("Task7 RunIdentity 缺少 label_manifest_fingerprint")
    if expected_label_fingerprint != label_manifest.fingerprint:
        raise ValueError("label_manifest_fingerprint 与 RunIdentity 不一致")
    return {
        **label_manifest.to_dict(),
        "run_identity": identity_payload,
        "run_identity_fingerprint": _identity_fingerprint(identity, identity_payload),
    }


def load_bound_label_manifest(
    path: str | Path,
    *,
    expected_identity: Any,
    expected_fingerprint: str | None = None,
) -> LabelManifest:
    destination = Path(path)
    try:
        payload = json.loads(destination.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("fewshot_manifest.json 无法读取或 JSON 非法") from error
    if not isinstance(payload, Mapping):
        raise ValueError("fewshot_manifest.json 必须是对象")
    actual_fields = set(payload)
    if actual_fields != set(BOUND_LABEL_MANIFEST_FIELDS):
        missing = sorted(BOUND_LABEL_MANIFEST_FIELDS - actual_fields)
        unknown = sorted(actual_fields - BOUND_LABEL_MANIFEST_FIELDS)
        raise ValueError(
            "fewshot_manifest.json schema 字段不匹配: "
            f"missing={missing}, unknown={unknown}"
        )
    if payload["schema_version"] != LABEL_MANIFEST_SCHEMA_VERSION:
        raise ValueError("fewshot_manifest.json schema_version 不匹配")
    if payload["algorithm_version"] != LABEL_MANIFEST_ALGORITHM_VERSION:
        raise ValueError("fewshot_manifest.json algorithm_version 不匹配")

    identity_payload = _identity_dict(expected_identity)
    expected_identity_fingerprint = _identity_fingerprint(expected_identity, identity_payload)
    if payload["run_identity"] != identity_payload:
        raise ValueError("fewshot_manifest.json run_identity 与 RunIdentity 不一致")
    if payload["run_identity_fingerprint"] != expected_identity_fingerprint:
        raise ValueError("fewshot_manifest.json run_identity_fingerprint 不匹配")
    expected_label_fingerprint = identity_payload.get("label_manifest_fingerprint")
    if not isinstance(expected_label_fingerprint, str):
        raise ValueError("Task7 RunIdentity 缺少 label_manifest_fingerprint")
    if payload["mode"] != identity_payload.get("label_budget"):
        raise ValueError("fewshot_manifest.json mode 与 RunIdentity.label_budget 不匹配")

    selected = payload["selected_keys"]
    if not isinstance(selected, list):
        raise ValueError("fewshot_manifest.json selected_keys 必须是 list")
    triplets: list[tuple[str, str, str]] = []
    for item in selected:
        if not (
            isinstance(item, list)
            and len(item) == 3
            and all(isinstance(part, str) for part in item)
        ):
            raise ValueError("fewshot_manifest.json selected_keys 每项必须是三个字符串 triplet 的 list")
        triplets.append((item[0], item[1], item[2]))
    if len(set(triplets)) != len(triplets):
        raise ValueError("fewshot_manifest.json selected_keys 不得重复")
    try:
        manifest = LabelManifest.create(payload["mode"], {SequenceKey(*triplet) for triplet in triplets})
    except (TypeError, ValueError) as error:
        raise ValueError("fewshot_manifest.json 内容无效") from error
    if manifest.canonical_payload()["selected_keys"] != selected:
        raise ValueError("fewshot_manifest.json selected_keys 不是规范序列化")
    if payload["fingerprint"] != manifest.fingerprint:
        raise ValueError("fewshot_manifest.json fingerprint 重算不匹配")
    if expected_fingerprint is not None and manifest.fingerprint != expected_fingerprint:
        raise ValueError("fewshot_manifest.json fingerprint 与执行合同不匹配")
    if expected_label_fingerprint != manifest.fingerprint:
        raise ValueError("fewshot_manifest.json fingerprint 与 RunIdentity 不匹配")
    return manifest


def write_bound_label_manifest(path: str | Path, label_manifest: LabelManifest, identity: Any) -> None:
    payload = bound_label_manifest_payload(label_manifest, identity)
    Path(path).write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
