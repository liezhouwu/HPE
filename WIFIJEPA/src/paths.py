from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
_ARTIFACTS = {
    "manifest": "data_manifest.json",
    "leakage_audit": "leakage_audit.json",
    "label_manifest": "fewshot_manifest.json",
    "axis_stats": "label_stats.json",
}


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def _identity(config: dict) -> tuple[str, str, str, int, str]:
    exp = config["experiment"]
    return (
        exp["protocol"],
        str(exp.get("scope", "strict")),
        exp["split"],
        int(exp["seed"]),
        str(exp["label_budget"]),
    )


def _audit_dir(config: dict) -> Path:
    protocol, scope, split, seed, budget = _identity(config)
    root = repo_path(config["experiment"].get("audit_root", "WIFIJEPA/audit"))
    return root / protocol / scope / split / f"seed{seed}" / budget


def _check_audit(directory: Path, config: dict) -> None:
    from mmfi_wifi.data_manifest import DataManifest, audit_manifest

    protocol, scope, split, seed, budget = _identity(config)
    manifest = DataManifest.read_json(directory / _ARTIFACTS["manifest"])
    if (manifest.protocol, manifest.scope, manifest.split, manifest.seed) != (
        protocol, scope, split, seed
    ):
        raise ValueError(f"WIFIJEPA audit 与当前配置不一致: {directory}")
    labels = json.loads((directory / _ARTIFACTS["label_manifest"]).read_text(encoding="utf-8"))
    if labels.get("mode") != budget:
        raise ValueError(f"WIFIJEPA label manifest 不是 {budget}: {directory}")
    audit = json.loads((directory / _ARTIFACTS["leakage_audit"]).read_text(encoding="utf-8"))
    if not audit.get("passed") or audit.get("data_manifest_fingerprint") != manifest.fingerprint():
        raise ValueError(f"WIFIJEPA leakage audit 无效: {directory}")
    label_audit = audit_manifest(manifest)
    if not label_audit.passed:
        raise ValueError("data manifest 边界审计失败: " + "; ".join(label_audit.violations))
    stats = json.loads((directory / _ARTIFACTS["axis_stats"]).read_text(encoding="utf-8"))
    if stats.get("label_manifest_fingerprint") != labels.get("fingerprint"):
        raise ValueError(f"label stats 与 label manifest 不匹配: {directory}")


def audit_paths(config: dict) -> dict[str, Path]:
    """按当前协议、划分、seed 和标签量准备 WIFIJEPA 本地审计文件。"""
    target = _audit_dir(config)
    target_files = {key: target / name for key, name in _ARTIFACTS.items()}
    if all(path.is_file() for path in target_files.values()):
        _check_audit(target, config)
        return target_files
    if target.exists():
        raise FileExistsError(f"WIFIJEPA audit 目录不完整，请检查: {target}")

    protocol, scope, split, seed, budget = _identity(config)
    kshot = budget.removesuffix("shot")
    legacy = (
        ROOT / "result_metafi_ssl" / "runs" / protocol / scope / split
        / f"seed{seed}" / "audit" / f"b{kshot}s"
    )
    legacy_files = {key: legacy / name for key, name in _ARTIFACTS.items()}
    target.parent.mkdir(parents=True, exist_ok=True)
    if all(path.is_file() for path in legacy_files.values()):
        shutil.copytree(legacy, target)
    else:
        from scripts.metafi_ssl.audit_data import run_audit

        baseline = repo_path(config.get("baseline_config", "configs/baseline_config.yaml"))
        run_audit(SimpleNamespace(
            dataset_root=str(repo_path(config["experiment"]["dataset_root"])),
            config_file=str(baseline),
            protocol=protocol,
            split=split,
            scope=scope,
            seed=seed,
            label_budget=budget,
            output_dir=str(target),
        ))
    _check_audit(target, config)
    return {key: target / name for key, name in _ARTIFACTS.items()}


def _fingerprint(value: object) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:10]


def pretrain_output(config: dict) -> Path:
    exp = config["experiment"]
    protocol, _, split, seed, _ = _identity(config)
    audit = audit_paths(config)["manifest"]
    from mmfi_wifi.data_manifest import DataManifest

    data_fp = DataManifest.read_json(audit).fingerprint()
    settings = {
        "input": config["input"],
        "model": config["model"],
        "mask": config["mask"],
        "pretrain": config["pretrain"],
        "manifest": data_fp,
    }
    fraction = int(float(config["pretrain"]["sequence_fraction"]) * 100)
    epochs = int(config["pretrain"]["epochs"])
    masked = int(config["mask"]["masked_links"])
    tag = f"u{fraction:02d}-e{epochs}-m{masked}-{_fingerprint(settings)}"
    root = repo_path(exp.get("pretrain_root", "WIFIJEPA/results/pretrain"))
    return root / protocol / split / f"seed{seed}" / tag


def pretrain_checkpoint(config: dict) -> Path:
    configured = config["experiment"].get("pretrain_checkpoint")
    return repo_path(configured) if configured else pretrain_output(config) / "encoder.pth"


def pretrain_manifest(config: dict) -> Path:
    configured = config["experiment"].get("pretrain_manifest")
    return repo_path(configured) if configured else pretrain_output(config) / "pretrain_manifest.json"


def finetune_output(config: dict) -> Path:
    exp = config["experiment"]
    protocol, _, split, seed, budget = _identity(config)
    mode = exp.get("mode", "supervised")
    method = "wifi_jepa_amp" if mode == "jepa" else "sup_structured"
    from mmfi_wifi.data_manifest import DataManifest
    manifest_path = audit_paths(config)["manifest"]
    settings = {
        "mode": mode,
        "finetune": config.get("finetune", {}),
        "manifest": DataManifest.read_json(manifest_path).fingerprint(),
        "pretrain_checkpoint": str(pretrain_checkpoint(config)) if mode == "jepa" else None,
    }
    root = repo_path(exp.get("finetune_root", "WIFIJEPA/results/finetune"))
    return (
        root / method / protocol / split / f"seed{seed}" / budget
        / _fingerprint(settings)
    )
