"""MetaFi SSL 配置加载：合并 YAML、应用覆盖并生成稳定指纹。"""
from __future__ import annotations
from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
from pathlib import Path
from typing import Any
import yaml

RESULT_ROOT = "result_metafi_ssl"

class ExperimentConfigError(ValueError):
    """配置文件无法用于当前实验。"""

def _read_yaml(path: Path) -> Mapping[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ExperimentConfigError(f"无法读取配置文件：{path}") from exc
    if not isinstance(data, Mapping):
        raise ExperimentConfigError(f"配置根节点必须是字典：{path}")
    return data

def _merge(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    """递归合并字典；列表和标量由后面的配置覆盖。"""
    for key, value in source.items():
        if isinstance(target.get(key), Mapping) and isinstance(value, Mapping):
            child = copy.deepcopy(dict(target[key]))
            _merge(child, value)
            target[key] = child
        else:
            target[key] = copy.deepcopy(value)

def _validate(config: Mapping[str, Any]) -> None:
    """只检查会让实验跑错的关键字段。"""
    if config.get("result_root", RESULT_ROOT) != RESULT_ROOT:
        raise ExperimentConfigError(f"result_root 必须保持为 {RESULT_ROOT!r}")
    choices = {
        "method": {"sup", "simclr", "moco", "swav", "relpos", "mfm", "mae"},
        "encoder_arch": {"metafi_r18", "metafi_r34"},
        "data_scope": {"strict", "transductive"},
        "target_space": {"absolute", "root_relative"},
        "fine_tune_strategy": {"matched", "transfer", "sup_differential"},
        "loss_name": {"mse", "mse_bone"},
        "protocol": {"protocol1", "protocol2", "protocol3"},
        "split": {"random_split", "cross_subject_split", "cross_scene_split"},
    }
    for field, allowed in choices.items():
        value = config.get(field)
        if value is not None and value not in allowed:
            raise ExperimentConfigError(f"{field} 的值不支持：{value!r}")
    for field in ("protocols", "splits", "metric_names", "label_budgets"):
        value = config.get(field)
        if value is not None and not isinstance(value, list):
            raise ExperimentConfigError(f"{field} 必须是列表")

def load_experiment_config(paths: Sequence[Path], overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """按顺序加载配置层，最后应用命令行覆盖。"""
    if isinstance(paths, (str, bytes)) or not isinstance(paths, Sequence):
        raise TypeError("paths 必须是 pathlib.Path 序列")
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, Mapping):
        raise TypeError("overrides 必须是字典")
    merged: dict[str, Any] = {}
    for path in paths:
        if not isinstance(path, Path):
            raise TypeError("每个配置路径都必须是 pathlib.Path")
        _merge(merged, _read_yaml(path))
    _merge(merged, overrides)
    _validate(merged)
    return merged

def config_fingerprint(config: Mapping[str, Any]) -> str:
    """为配置生成稳定 SHA-256 指纹。"""
    if not isinstance(config, Mapping):
        raise TypeError("config 必须是字典")
    try:
        payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("config 必须能够序列化为有限 JSON") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

__all__ = ["ExperimentConfigError", "config_fingerprint", "load_experiment_config"]
