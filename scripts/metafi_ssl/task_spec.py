"""MetaFi SSL 任务描述与结果路径。

任务 ID 保持稳定，结果统一写入 result_metafi_ssl，方便批量实验和复盘。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, ClassVar


RESULT_ROOT_NAME = "result_metafi_ssl"
LEGACY_RESULT_ROOT_NAME = "result"
TASK_ID_HASH_LENGTH = 12

ALLOWED_STAGES = frozenset({0, 1, 2, 3, 4})
ALLOWED_RUN_KINDS = frozenset(
    {"pretrain", "supervised", "finetune_matched", "finetune_transfer", "ablations"}
)
ALLOWED_METHODS = frozenset({"sup", "simclr", "moco", "swav", "relpos", "mfm", "mae"})
ALLOWED_PROTOCOLS = frozenset({"protocol1", "protocol2", "protocol3"})
ALLOWED_SPLITS = frozenset({"random_split", "cross_subject_split", "cross_scene_split"})
ALLOWED_SCOPES = frozenset({"strict", "transductive"})
ALLOWED_LABEL_BUDGETS = frozenset({"1shot", "2shot", "4shot", "8shot", "16shot", "100%"})
ALLOWED_LOSSES = frozenset({"mse", "mse_bone"})
ALLOWED_STRATEGIES = frozenset({"matched", "transfer", "sup_differential"})

_SAFE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_CONFIG_SEGMENT_RE = re.compile(r"^[^<>:\"/\\|?*\x00-\x1f]+$")
_DEVICE_RE = re.compile(r"^(?:cpu|cuda(?::[0-9]+)?)$")
_TASK_ID_RE = re.compile(r"^[A-Za-z0-9._%+-]+$")

_SCRIPT_BY_RUN_KIND = {
    "pretrain": "pretrain.py",
    "supervised": "train_supervised.py",
    "finetune_matched": "finetune.py",
    "finetune_transfer": "finetune.py",
    "ablations": "finetune.py",
}
_RUNNER_STRATEGY_BY_TASK_STRATEGY = {
    "matched": "matched",
    "transfer": "transfer",
    "sup_differential": "sup-differential-lr",
}


class ExperimentTaskError(ValueError):
    """Raised when a task is not a safe, supported experiment specification."""


@dataclass(frozen=True, slots=True)
class RuntimeInputs:
    """Concrete paths and runtime settings required by a MetaFi runner."""

    dataset_root: Path
    config_file: Path
    manifest: Path
    output_root: Path
    device: str
    label_manifest: Path | None = None
    leakage_audit: Path | None = None
    axis_stats: Path | None = None
    encoder_checkpoint: Path | None = None
    runner_script: Path | None = None

    def __post_init__(self) -> None:
        required_paths = {
            "dataset_root": self.dataset_root,
            "config_file": self.config_file,
            "manifest": self.manifest,
            "output_root": self.output_root,
        }
        for field_name, value in required_paths.items():
            if not isinstance(value, Path):
                raise TypeError(f"{field_name} must be a pathlib.Path")
        optional_paths = {
            "label_manifest": self.label_manifest,
            "leakage_audit": self.leakage_audit,
            "axis_stats": self.axis_stats,
            "encoder_checkpoint": self.encoder_checkpoint,
            "runner_script": self.runner_script,
        }
        for field_name, value in optional_paths.items():
            if value is not None and not isinstance(value, Path):
                raise TypeError(f"{field_name} must be a pathlib.Path or None")
        if not isinstance(self.device, str) or not self.device:
            raise TypeError("device must be a non-empty str")


def _require_int(value: object, field_name: str, *, minimum: int = 0) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an int")
    if value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}")
    return value


def _require_choice(value: object, field_name: str, choices: frozenset[str]) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a str")
    if value not in choices:
        raise ValueError(f"{field_name} is unsupported: {value!r}")
    return value


def _require_device(value: object) -> str:
    if not isinstance(value, str) or not _DEVICE_RE.fullmatch(value):
        raise ValueError("device must be cpu, cuda, or cuda:<index>")
    return value


def _require_task_id(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or ".." in value
        or not _TASK_ID_RE.fullmatch(value)
    ):
        raise ExperimentTaskError(f"{field_name} must be a safe task ID")
    return value


def _normalise_dependencies(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("dependencies must be a non-string sequence of task IDs")
    dependencies = tuple(value)
    if len(set(dependencies)) != len(dependencies):
        raise ValueError("dependencies must not contain duplicates")
    for index, dependency in enumerate(dependencies):
        _require_task_id(dependency, f"dependencies[{index}]")
    return tuple(sorted(dependencies))


def _require_safe_segment(value: object, field_name: str) -> str:
    """任务目录名只需避免空值、路径分隔符和父目录跳转。"""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a str")
    if not value or value in {".", ".."} or not _SAFE_SEGMENT_RE.fullmatch(value):
        raise ExperimentTaskError(f"{field_name} must be a safe single path segment")
    return value


def _validate_config_path(path: Path, index: int) -> Path:
    """配置层允许普通相对路径，但不接受父目录跳转。"""
    if not isinstance(path, Path):
        raise TypeError(f"config_paths[{index}] must be a pathlib.Path")
    if not path.parts or any(part == ".." for part in path.parts):
        raise ExperimentTaskError(f"config_paths[{index}] must be a file path")
    return path


def _config_path_sort_key(path: Path) -> tuple[int, str]:
    # Keep the historical canonical order for ordinary layers, but force every
    # explicit ablation layer after them because config_loader merges in order.
    is_ablation = int("ablations" in path.parts)
    return is_ablation, path.as_posix()


def _normalise_config_paths(value: object) -> tuple[Path, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("config_paths must be a non-empty sequence of pathlib.Path")
    paths = tuple(_validate_config_path(path, index) for index, path in enumerate(value))
    if not paths:
        raise ValueError("config_paths must not be empty")
    ordered = tuple(sorted(paths, key=_config_path_sort_key))
    if len(set(ordered)) != len(ordered):
        raise ValueError("config_paths must not contain duplicates")
    return ordered


def _canonical_config_path(path: Path) -> str:
    return path.as_posix()


def _legacy_result_in_path(path: Path) -> bool:
    try:
        parts = path.resolve(strict=False).parts
    except OSError as error:
        raise ExperimentTaskError("unable to resolve output path") from error
    return any(part.casefold() == LEGACY_RESULT_ROOT_NAME for part in parts)


def _assert_new_result_root(path: Path) -> None:
    """保证任务结果落在 result_metafi_ssl 子目录中。"""
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    if any(part == ".." for part in path.parts):
        raise ExperimentTaskError("output path must not contain path traversal")
    roots = [item for item in (path, *path.parents) if item.name.casefold() == RESULT_ROOT_NAME]
    if len(roots) != 1:
        raise ExperimentTaskError("output must be below result_metafi_ssl")
    try:
        path.resolve(strict=False).relative_to(roots[0].resolve(strict=False))
    except (OSError, ValueError) as error:
        raise ExperimentTaskError("output must remain inside result_metafi_ssl") from error


@dataclass(frozen=True, slots=True)
class ExperimentTask:
    """The complete immutable identity of one orchestrated experiment task."""

    stage: int
    run_kind: str
    method: str
    protocol: str
    split: str
    scope: str
    pretrain_seed: int
    finetune_seed: int
    label_budget: str
    loss_name: str
    strategy: str
    config_paths: tuple[Path, ...]
    dependencies: tuple[str, ...] = ()
    track: str = "main"
    device: str = "cuda"

    schema_version: ClassVar[int] = 1

    def __post_init__(self) -> None:
        stage = _require_int(self.stage, "stage")
        if stage not in ALLOWED_STAGES:
            raise ValueError(f"stage is unsupported: {stage!r}")
        _require_safe_segment(
            _require_choice(self.run_kind, "run_kind", ALLOWED_RUN_KINDS), "run_kind"
        )
        _require_safe_segment(
            _require_choice(self.method, "method", ALLOWED_METHODS), "method"
        )
        _require_safe_segment(
            _require_choice(self.protocol, "protocol", ALLOWED_PROTOCOLS), "protocol"
        )
        _require_safe_segment(
            _require_choice(self.split, "split", ALLOWED_SPLITS), "split"
        )
        _require_safe_segment(
            _require_choice(self.scope, "scope", ALLOWED_SCOPES), "scope"
        )
        _require_int(self.pretrain_seed, "pretrain_seed")
        _require_int(self.finetune_seed, "finetune_seed")
        label_budget = _require_choice(
            self.label_budget, "label_budget", ALLOWED_LABEL_BUDGETS
        )
        if label_budget != "100%":
            _require_safe_segment(label_budget, "label_budget")
        _require_safe_segment(
            _require_choice(self.loss_name, "loss_name", ALLOWED_LOSSES), "loss_name"
        )
        _require_safe_segment(
            _require_choice(self.strategy, "strategy", ALLOWED_STRATEGIES), "strategy"
        )
        config_paths = _normalise_config_paths(self.config_paths)
        object.__setattr__(self, "config_paths", config_paths)
        dependencies = _normalise_dependencies(self.dependencies)
        object.__setattr__(self, "dependencies", dependencies)
        _require_safe_segment(self.track, "track")
        _require_device(self.device)

        if self.run_kind == "pretrain" and self.method == "sup":
            raise ValueError("pretrain tasks cannot use the sup method")
        if self.run_kind == "supervised" and self.method != "sup":
            raise ValueError("supervised tasks must use the sup method")
        if self.run_kind == "supervised" and self.strategy not in {"matched", "sup_differential"}:
            raise ExperimentTaskError(
                "supervised tasks only support matched or sup_differential strategy"
            )
        if self.run_kind == "finetune_matched" and self.strategy != "matched":
            raise ValueError("finetune_matched tasks must use matched strategy")
        if self.run_kind == "finetune_transfer" and self.strategy != "transfer":
            raise ValueError("finetune_transfer tasks must use transfer strategy")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible canonical payload."""
        return {
            "config_paths": [_canonical_config_path(path) for path in self.config_paths],
            "dependencies": list(self.dependencies),
            "device": self.device,
            "finetune_seed": self.finetune_seed,
            "label_budget": self.label_budget,
            "loss_name": self.loss_name,
            "method": self.method,
            "pretrain_seed": self.pretrain_seed,
            "protocol": self.protocol,
            "run_kind": self.run_kind,
            "schema_version": self.schema_version,
            "scope": self.scope,
            "split": self.split,
            "stage": self.stage,
            "strategy": self.strategy,
            "track": self.track,
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

    def to_json(self) -> str:
        return self.canonical_json()

    @classmethod
    def from_dict(cls, payload: object) -> "ExperimentTask":
        if not isinstance(payload, Mapping):
            raise TypeError("ExperimentTask JSON payload must be an object")
        required = {
            "schema_version",
            "stage",
            "run_kind",
            "method",
            "protocol",
            "split",
            "scope",
            "pretrain_seed",
            "finetune_seed",
            "label_budget",
            "loss_name",
            "strategy",
            "config_paths",
        }
        optional = {"dependencies", "track", "device"}
        actual = set(payload)
        expected = required | optional
        if not required.issubset(actual) or actual - expected:
            missing = sorted(required - actual)
            extra = sorted(actual - expected)
            raise ValueError(f"task payload fields mismatch; missing={missing}, extra={extra}")
        if payload["schema_version"] != cls.schema_version:
            raise ValueError("unsupported ExperimentTask schema_version")
        raw_config_paths = payload["config_paths"]
        if isinstance(raw_config_paths, (str, bytes)) or not isinstance(
            raw_config_paths, Sequence
        ):
            raise TypeError("config_paths must be a non-string sequence of str or pathlib.Path")
        config_paths: list[Path] = []
        for index, raw_path in enumerate(raw_config_paths):
            if isinstance(raw_path, Path):
                config_paths.append(raw_path)
            elif isinstance(raw_path, str):
                config_paths.append(Path(raw_path))
            else:
                raise TypeError(
                    f"config_paths[{index}] must be a str or pathlib.Path"
                )
        return cls(
            stage=payload["stage"],
            run_kind=payload["run_kind"],
            method=payload["method"],
            protocol=payload["protocol"],
            split=payload["split"],
            scope=payload["scope"],
            pretrain_seed=payload["pretrain_seed"],
            finetune_seed=payload["finetune_seed"],
            label_budget=payload["label_budget"],
            loss_name=payload["loss_name"],
            strategy=payload["strategy"],
            config_paths=tuple(config_paths),
            dependencies=payload.get("dependencies", ()),
            track=payload.get("track", "main"),
            device=payload.get("device", "cuda"),
        )

    @classmethod
    def from_json(cls, encoded: str) -> "ExperimentTask":
        if not isinstance(encoded, str):
            raise TypeError("encoded task JSON must be a str")
        try:
            payload = json.loads(encoded)
        except json.JSONDecodeError as error:
            raise ValueError("invalid ExperimentTask JSON") from error
        task = cls.from_dict(payload)
        if encoded != task.canonical_json():
            raise ValueError("ExperimentTask JSON is not canonical")
        return task

    def task_id(self) -> str:
        """Return a readable, deterministic ID with a short full-identity hash."""
        digest = hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
        short_hash = digest[:TASK_ID_HASH_LENGTH]
        return (
            f"stage{self.stage}-{self.run_kind}-{self.method}-{self.protocol}-"
            f"{self.split}-{self.scope}-{self.label_budget}-seed{self.finetune_seed}-"
            f"{self.strategy}-{short_hash}"
        )

    def _result_root(self, root: Path) -> Path:
        if not isinstance(root, Path):
            raise TypeError("root must be a pathlib.Path")
        if any(part == ".." for part in root.parts):
            raise ExperimentTaskError("root must not contain path traversal")
        result_root = root if root.name.casefold() == RESULT_ROOT_NAME else root / RESULT_ROOT_NAME
        if _legacy_result_in_path(result_root):
            raise ExperimentTaskError(
                "new MetaFi SSL output cannot equal or be nested below legacy result"
            )
        return result_root

    def output_dir(self, root: Path) -> Path:
        """Return the identity-bound output path below a canonical new-result root."""
        result_root = self._result_root(root)
        path_parts = [result_root, self.scope, self.run_kind]
        if self.track != "main":
            path_parts.append(self.track)
        path_parts.extend(
            (
                self.protocol,
                self.split,
                self.method,
                self.label_budget,
                f"seed{self.finetune_seed}",
                self.strategy,
                f"task-{self.task_id().rsplit('-', 1)[-1]}",
            )
        )
        target = Path(*path_parts)
        if _legacy_result_in_path(target):
            raise ExperimentTaskError("output path is nested below legacy result")
        try:
            _assert_new_result_root(target)
        except (TypeError, ValueError) as error:
            raise ExperimentTaskError(str(error)) from error
        return target

    def to_command(
        self,
        python_executable: Path,
        *,
        runtime_inputs: RuntimeInputs | None = None,
    ) -> list[str]:
        """Build a shell-free argument vector accepted by the selected runner."""
        if not isinstance(python_executable, Path):
            raise TypeError("python_executable must be a pathlib.Path")
        if python_executable == Path(".") or any(
            part == ".." for part in python_executable.parts
        ):
            raise ExperimentTaskError("python_executable must not contain path traversal")
        if runtime_inputs is None:
            raise ExperimentTaskError(
                "runtime_inputs are required to build an executable runner command"
            )
        if not isinstance(runtime_inputs, RuntimeInputs):
            raise TypeError("runtime_inputs must be a RuntimeInputs instance")
        if runtime_inputs.device != self.device:
            raise ExperimentTaskError(
                "runtime_inputs.device must exactly match ExperimentTask.device"
            )

        required_fields = ["dataset_root", "config_file", "manifest"]
        if self.run_kind != "pretrain":
            required_fields.extend(["label_manifest", "leakage_audit", "axis_stats"])
        if self.run_kind != "pretrain" and self.run_kind != "supervised":
            required_fields.append("encoder_checkpoint")
        missing = [
            field_name
            for field_name in required_fields
            if getattr(runtime_inputs, field_name) is None
        ]
        if missing:
            raise ExperimentTaskError(
                "runtime_inputs missing required field(s): " + ", ".join(missing)
            )

        script = runtime_inputs.runner_script or Path(__file__).resolve().with_name(
            _SCRIPT_BY_RUN_KIND[self.run_kind]
        )
        output_dir = self.output_dir(runtime_inputs.output_root)
        command = [
            str(python_executable),
            str(script),
            str(runtime_inputs.dataset_root),
            str(runtime_inputs.config_file),
        ]
        if self.run_kind == "pretrain":
            command.extend(
                (
                    "--method",
                    self.method,
                    "--manifest",
                    str(runtime_inputs.manifest),
                    "--output-dir",
                    str(output_dir),
                    "--device",
                    runtime_inputs.device,
                    "--seed",
                    str(self.pretrain_seed),
                )
            )
        else:
            command.extend(
                (
                    "--manifest",
                    str(runtime_inputs.manifest),
                    "--label-manifest",
                    str(runtime_inputs.label_manifest),
                    "--leakage-audit",
                    str(runtime_inputs.leakage_audit),
                    "--axis-stats",
                    str(runtime_inputs.axis_stats),
                    "--label-budget",
                    self.label_budget,
                    "--protocol",
                    self.protocol,
                    "--split",
                    self.split,
                    "--seed",
                    str(self.finetune_seed),
                    "--output-dir",
                    str(output_dir),
                    "--strategy",
                    _RUNNER_STRATEGY_BY_TASK_STRATEGY[self.strategy],
                    "--device",
                    runtime_inputs.device,
                )
            )
            if self.run_kind == "supervised":
                command.extend(("--loss-name", self.loss_name))
                if self.loss_name == "mse_bone":
                    command.extend(("--bone-set", "legacy15"))
            else:
                command.extend(
                    ("--encoder-checkpoint", str(runtime_inputs.encoder_checkpoint))
                )
        return command


__all__ = [
    "ALLOWED_LABEL_BUDGETS",
    "ALLOWED_LOSSES",
    "ALLOWED_METHODS",
    "ALLOWED_PROTOCOLS",
    "ALLOWED_RUN_KINDS",
    "ALLOWED_SCOPES",
    "ALLOWED_SPLITS",
    "ALLOWED_STAGES",
    "ALLOWED_STRATEGIES",
    "ExperimentTask",
    "ExperimentTaskError",
    "RuntimeInputs",
]
