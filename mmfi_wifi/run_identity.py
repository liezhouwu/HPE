"""Schema-versioned identity and safe result-directory preparation for MetaFi SSL.

This module is deliberately independent of training CLIs.  New MetaFi SSL
runs must be rooted below ``result_metafi_ssl`` and may only resume when their
logical experiment identity and committed ``last_state.pth`` are both valid.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, ClassVar

import torch


RUN_IDENTITY_FILENAME = "run_identity.json"
LAST_STATE_FILENAME = "last_state.pth"
RUN_IDENTITY_SCHEMA_VERSION = 1
NEW_PIPELINE_LAST_STATE_SCHEMA_VERSION = 3
_RESULT_ROOT_NAME = "result_metafi_ssl"
_LEGACY_RESULT_ROOT_NAME = "result"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_IDENTITY_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_WINDOWS_RESERVED_SEGMENT_RE = re.compile(
    r"^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?$", re.IGNORECASE
)


def _require_nonempty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} 必须是 str")
    if not value:
        raise ValueError(f"{field_name} 必须是非空字符串")
    return value


def _require_safe_identity_segment(value: object, field_name: str) -> str:
    """Validate a portable, single path-segment experiment identifier."""
    value = _require_nonempty_string(value, field_name)
    if (
        ".." in value
        or value.endswith((".", " "))
        or _WINDOWS_RESERVED_SEGMENT_RE.fullmatch(value)
        or not _SAFE_IDENTITY_SEGMENT_RE.fullmatch(value)
    ):
        raise ValueError(
            f"{field_name} 必须是安全路径段 "
            "([A-Za-z0-9][A-Za-z0-9._-]*，不得包含 ..，且必须是跨平台规范段)"
        )
    return value


def _require_seed(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field_name} 必须是 int")
    if value < 0:
        raise ValueError(f"{field_name} 必须 >= 0")
    return value


def _require_sha256(value: object, field_name: str) -> str:
    value = _require_nonempty_string(value, field_name)
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field_name} 必须是小写 64 位 SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class RunIdentity:
    """The complete immutable identity of one new MetaFi SSL run.

    Filesystem paths and machine-specific execution settings are intentionally
    absent: a run identity describes the experiment, not where it happens to
    execute.  ``schema_version`` is emitted in serialization so future formats
    cannot be silently mistaken for this one.
    """

    method: str
    encoder_arch: str
    protocol: str
    split: str
    data_scope: str
    pretrain_seed: int
    finetune_seed: int
    label_budget: str
    loss_name: str
    fine_tune_strategy: str
    config_fingerprint: str
    manifest_fingerprint: str
    label_manifest_fingerprint: str | None = None

    schema_version: ClassVar[int] = RUN_IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "method",
            "encoder_arch",
            "protocol",
            "split",
            "label_budget",
            "loss_name",
            "fine_tune_strategy",
        ):
            _require_safe_identity_segment(getattr(self, field_name), field_name)
        if self.data_scope not in {"strict", "transductive"}:
            raise ValueError("data_scope 必须是 strict 或 transductive")
        _require_seed(self.pretrain_seed, "pretrain_seed")
        _require_seed(self.finetune_seed, "finetune_seed")
        _require_sha256(self.config_fingerprint, "config_fingerprint")
        _require_sha256(self.manifest_fingerprint, "manifest_fingerprint")
        if self.label_manifest_fingerprint is not None:
            _require_sha256(self.label_manifest_fingerprint, "label_manifest_fingerprint")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "method": self.method,
            "encoder_arch": self.encoder_arch,
            "protocol": self.protocol,
            "split": self.split,
            "data_scope": self.data_scope,
            "pretrain_seed": self.pretrain_seed,
            "finetune_seed": self.finetune_seed,
            "label_budget": self.label_budget,
            "loss_name": self.loss_name,
            "fine_tune_strategy": self.fine_tune_strategy,
            "config_fingerprint": self.config_fingerprint,
            "manifest_fingerprint": self.manifest_fingerprint,
        }
        if self.label_manifest_fingerprint is not None:
            payload["label_manifest_fingerprint"] = self.label_manifest_fingerprint
        return payload

    @classmethod
    def from_dict(cls, payload: object) -> "RunIdentity":
        if not isinstance(payload, Mapping):
            raise ValueError("run_identity.json 必须是对象")
        base_fields = {
            "schema_version", "method", "encoder_arch", "protocol", "split",
            "data_scope", "pretrain_seed", "finetune_seed", "label_budget",
            "loss_name", "fine_tune_strategy", "config_fingerprint",
            "manifest_fingerprint",
        }
        optional_fields = {"label_manifest_fingerprint"}
        actual_fields = set(payload)
        if actual_fields not in (base_fields, base_fields | optional_fields):
            missing = sorted(base_fields - actual_fields)
            unexpected = sorted(actual_fields - base_fields - optional_fields)
            raise ValueError(
                "run_identity.json 字段不匹配: "
                f"missing={missing}, unexpected={unexpected}"
            )
        if payload["schema_version"] != cls.schema_version:
            raise ValueError(
                "不支持的 run identity schema_version: "
                f"{payload['schema_version']}"
            )
        return cls(
            method=payload["method"],
            encoder_arch=payload["encoder_arch"],
            protocol=payload["protocol"],
            split=payload["split"],
            data_scope=payload["data_scope"],
            pretrain_seed=payload["pretrain_seed"],
            finetune_seed=payload["finetune_seed"],
            label_budget=payload["label_budget"],
            loss_name=payload["loss_name"],
            fine_tune_strategy=payload["fine_tune_strategy"],
            config_fingerprint=payload["config_fingerprint"],
            manifest_fingerprint=payload["manifest_fingerprint"],
            label_manifest_fingerprint=payload.get("label_manifest_fingerprint"),
        )

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ) + "\n"

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _is_link_or_junction(path: Path) -> bool:
    """Return whether an existing path can redirect filesystem resolution."""
    is_junction = getattr(path, "is_junction", None)
    return path.is_symlink() or (callable(is_junction) and is_junction())


def _result_root_candidate(path: Path) -> Path:
    """Return the single lexical ``result_metafi_ssl`` ancestor of ``path``."""
    if any(part == ".." for part in path.parts):
        raise ValueError(
            "新 MetaFi SSL 结果必须写入 result_metafi_ssl 子树，"
            "禁止使用 .. 路径遍历"
        )

    candidate = path
    matches = 0
    selected: Path | None = None
    while True:
        if candidate.name.casefold() == _RESULT_ROOT_NAME:
            matches += 1
            selected = candidate
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent

    if matches != 1 or selected is None:
        raise ValueError(
            "新 MetaFi SSL 结果必须写入 result_metafi_ssl 子树，"
            "禁止写入 legacy result/ 根目录"
        )
    return selected


def assert_new_result_root(path: Path) -> None:
    """Require a canonical, non-redirecting ``result_metafi_ssl`` result subtree."""
    if not isinstance(path, Path):
        raise TypeError("path 必须是 pathlib.Path")

    root_candidate = _result_root_candidate(path)
    # ``resolve(strict=False)`` resolves every existing ancestor (including
    # Windows symlinks/junctions) while retaining missing terminal components.
    # This makes containment fail closed before ``prepare_run_directory`` creates
    # anything.
    try:
        resolved_root = root_candidate.resolve(strict=False)
        resolved_target = path.resolve(strict=False)
    except OSError as error:
        raise ValueError("无法规范解析 result_metafi_ssl 结果路径") from error

    if _is_link_or_junction(root_candidate) or resolved_root.name.casefold() != _RESULT_ROOT_NAME:
        raise ValueError(
            "新 MetaFi SSL 结果必须写入 canonical result_metafi_ssl 子树，"
            "result_metafi_ssl 根目录不得重定向到外部位置"
        )
    try:
        resolved_target.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(
            "新 MetaFi SSL 结果必须写入 result_metafi_ssl 子树，"
            "禁止路径遍历或链接逃逸"
        ) from error


def _atomic_write_text(path: Path, text: str) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _load_exact_identity(path: Path) -> RunIdentity:
    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取有效的 {RUN_IDENTITY_FILENAME}") from error
    identity = RunIdentity.from_dict(payload)
    if raw != identity.canonical_json():
        raise ValueError(f"{RUN_IDENTITY_FILENAME} 不是规范序列化")
    return identity


def _assert_loadable_last_state(path: Path, identity: RunIdentity | None = None) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"无法续训，缺少 {path}")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:  # torch exposes several backend-specific errors.
        raise ValueError(f"无法续训，{LAST_STATE_FILENAME} 无法加载") from error
    if not isinstance(state, Mapping):
        raise ValueError(f"无法续训，{LAST_STATE_FILENAME} 必须是字典")
    next_epoch = state.get("next_epoch")
    if not isinstance(next_epoch, int) or isinstance(next_epoch, bool) or next_epoch < 0:
        raise ValueError(f"无法续训，{LAST_STATE_FILENAME} 缺少有效 next_epoch")
    if identity is None:
        return

    if state.get("schema_version") != NEW_PIPELINE_LAST_STATE_SCHEMA_VERSION:
        raise ValueError(
            f"无法续训，{LAST_STATE_FILENAME} schema_version 必须是 "
            f"{NEW_PIPELINE_LAST_STATE_SCHEMA_VERSION}"
        )
    raw_identity = state.get("run_identity")
    saved_identity = RunIdentity.from_dict(raw_identity)
    if saved_identity != identity:
        raise ValueError(f"无法续训，{LAST_STATE_FILENAME} RunIdentity 不一致")
    expected_json = identity.canonical_json()
    if state.get("run_identity_json") != expected_json:
        raise ValueError(f"无法续训，{LAST_STATE_FILENAME} run_identity_json 不规范或不一致")
    strategy_state = state.get("strategy_state")
    if not isinstance(strategy_state, Mapping):
        raise ValueError(f"无法续训，{LAST_STATE_FILENAME} 缺少有效 strategy_state")
    required = {
        "model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
        "scaler_state_dict", "best_records", "python_rng_state", "numpy_rng_state",
        "torch_rng_state", "train_generator_state",
    }
    missing = sorted(name for name in required if name not in state)
    if missing:
        raise ValueError(
            f"无法续训，{LAST_STATE_FILENAME} 缺少 schema-v3 字段: {missing}"
        )


def prepare_run_directory(path: Path, identity: RunIdentity, resume: bool) -> None:
    """Prepare one identity-bound run directory with fail-closed resume checks.

    Fresh runs accept only an absent or empty directory and atomically create
    ``run_identity.json``.  Resume never mutates identity metadata: it requires
    canonical exact identity equality and a loadable committed ``last_state``.
    """
    if not isinstance(path, Path):
        raise TypeError("path 必须是 pathlib.Path")
    if not isinstance(identity, RunIdentity):
        raise TypeError("identity 必须是 RunIdentity")
    if not isinstance(resume, bool):
        raise TypeError("resume 必须是 bool")
    assert_new_result_root(path)

    identity_path = path / RUN_IDENTITY_FILENAME
    state_path = path / LAST_STATE_FILENAME
    if resume:
        if not path.is_dir():
            raise FileNotFoundError(f"无法续训，结果目录不存在: {path}")
        saved_identity = _load_exact_identity(identity_path)
        if saved_identity != identity:
            raise ValueError("RunIdentity 与已有 run_identity.json 不一致，拒绝续训")
        _assert_loadable_last_state(state_path, identity)
        return

    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"结果路径不是目录: {path}")
        if any(path.iterdir()):
            raise FileExistsError(f"结果目录非空，拒绝混写: {path}")
    else:
        path.mkdir(parents=True, exist_ok=False)
    _atomic_write_text(identity_path, identity.canonical_json())
