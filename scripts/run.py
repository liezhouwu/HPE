"""try3 鐨勭粺涓€瀹為獙鍏ュ彛銆?

鍩虹嚎锛歱ython scripts/run.py baseline single|all
SSL锛?python scripts/run.py ssl supervised|simclr|moco|swav|relpos|mfm|mae single|all
鍙敤 --split 瑕嗙洊鍗曞垝鍒嗭紝--protocol 瑕嗙洊榛樿鍗忚銆?
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from argparse import Namespace
from collections.abc import Mapping
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from scripts.metafi_ssl.audit_data import run_audit
from scripts.metafi_ssl.pretrain import (
    run_pretraining, _extract_pretrain_config, _legacy_pretrain_config,
    _canonical_fingerprint, make_pretrain_identity,
)
from pose_ssl.metafi.pretrain_checkpoint import load_checkpoint, STATUS_COMPLETE
from mmfi_wifi.run_identity import RunIdentity
from scripts.metafi_ssl.train_supervised import run_matched
from mmfi_wifi.data_manifest import DataManifest, audit_manifest, build_data_manifest
from pose_ssl.metafi.pretrain_subset import make_small_pretrain_manifest, selection_summary
from scripts.report_results import generate_report


SPLITS = ("random_split", "cross_subject_split", "cross_scene_split")
SSL_METHODS = ("simclr", "moco", "swav", "relpos", "mfm", "mae")
ARTIFACTS = ("data_manifest.json", "leakage_audit.json", "fewshot_manifest.json", "label_stats.json")


def _load_config(name: str) -> dict[str, Any]:
    path = ROOT / "configs" / f"{name}_config.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"配置必须是 YAML 字典：{path}")
    return data


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _splits(config: Mapping[str, Any], mode: str, override: str | None) -> tuple[str, ...]:
    if mode == "single":
        split = override or config.get("single_split")
        if split not in SPLITS:
            raise ValueError("single_split 蹇呴』鏄?random_split銆乧ross_subject_split 鎴?cross_scene_split")
        return (split,)
    if mode == "all":
        values = tuple(config.get("all_splits", SPLITS))
        if not values or any(value not in SPLITS for value in values):
            raise ValueError("all_splits must contain valid split names")
        return values
    raise ValueError("妯″紡鍙兘鏄?single 鎴?all")


def _runtime(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _run(command: list[str]) -> None:
    print("[杩愯] " + " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def _baseline(config: Mapping[str, Any], mode: str, protocol: str, split_override: str | None) -> None:
    runtime = _runtime(config, "runtime")
    dataset_root = str(_path(str(config["dataset_root"])))
    config_file = str(ROOT / "configs" / "baseline_config.yaml")
    splits = _splits(config, mode, split_override)
    script = ROOT / "scripts" / "reproduction" / ("train.py" if mode == "single" else "run_all.py")
    command = [sys.executable, str(script), dataset_root, config_file, "--protocol", protocol, "--device", str(config.get("device", "cuda")), "--num_workers", str(int(runtime.get("num_workers", 0))), "--val_every", str(int(runtime.get("val_every", 5)))]
    if mode == "single":
        command.extend(("--split", splits[0]))
    else:
        command.extend(("--splits", *splits))
    command.append("--amp" if runtime.get("amp", False) else "--no_amp")
    _run(command)


def _baseline_snapshot(config):
    return yaml.safe_load(_path(str(config["baseline_config"])).read_text(encoding="utf-8"))


def _experiment_root(config: Mapping[str, Any], split: str) -> Path:
    """Return the canonical root for one protocol/scope/split/seed boundary."""

    return ROOT / "result_metafi_ssl" / "runs" / str(config["protocol"]) / str(config["scope"]) / split / f"seed{config['seed']}"


def _budget_tag(label_budget: object) -> str:
    if not isinstance(label_budget, str) or not label_budget:
        raise ValueError("label_budget must be a non-empty string")
    return "b" + label_budget.replace("shot", "s").replace("%", "pct")


def _small_finetune_tag(
    label_budget: str, config: Mapping[str, Any], stage: str, options: Mapping[str, Any]
) -> str:
    """Generate readable tag for small-sample finetune/supervised runs.

    Format: b4s-strategy-optimizer-scheduler-ft25-cXXXXXXXXXXX
    Example: b4s-matched-adamw-cosine-ft25-cc3c4a8fc71
    """
    budget_prefix = _budget_tag(label_budget)
    section = _small_section(config, stage, options)

    strategy = section.get("strategy", "matched")
    optimizer = section.get("optimizer", "adamw")
    scheduler = section.get("scheduler", "cosine")
    epochs = int(section.get("epochs", 25))

    # Build identity for digest
    identity = {
        "label_budget": label_budget,
        "stage": stage,
        "section": section,
        "baseline": _baseline_snapshot(config),
    }
    digest = _config_digest(identity)

    return f"{budget_prefix}-{strategy}-{optimizer}-{scheduler}-ft{epochs}-c{digest}"


def _full_pretrain_tag(config: Mapping[str, Any], method: str) -> str:
    pretrain = dict(_runtime(config, "pretrain"))
    epochs = int(pretrain.get("epochs", 100))
    micro_batch = int(pretrain.get("micro_batch", 1))

    # 鎻愬彇鍙鐨勫叧閿厤缃?
    optimizer = pretrain.get("optimizer", "adamw")
    scheduler = pretrain.get("scheduler", "cosine")

    # 鏋勫缓鍙鏍囩: u100-optimizer-scheduler-epochs-mb-digest
    # 渚嬪: u100-adamw-cosine-e100-mb12-c1a2b3c4d5
    identity = _extract_pretrain_config(config, method)
    readable_parts = [
        "u100",
        optimizer,
        scheduler,
        f"e{epochs}",
        f"mb{micro_batch}",
        f"c{_config_digest(identity)}"
    ]
    return "-".join(readable_parts)


def _full_finetune_tag(config: Mapping[str, Any], kind: str) -> str:
    section = dict(_runtime(config, "supervised" if kind == "supervised" else "finetune"))
    epochs = int(section.get("epochs", 100))

    # 鎻愬彇鍙鐨勫叧閿厤缃?
    strategy = section.get("strategy", "matched")
    optimizer = section.get("optimizer", "adamw")
    scheduler = section.get("scheduler", "cosine")

    # 鏋勫缓鍙鏍囩: budget-strategy-optimizer-scheduler-epochs-digest
    # 渚嬪: b4s-transfer-adamw-cosine-ft100-cd350c07ba7
    readable_parts = [
        _budget_tag(config['label_budget']),
        strategy,
        optimizer,
        scheduler,
        f"ft{epochs}",
        f"c{_config_digest({'section': section, 'baseline': _baseline_snapshot(config)})}"
    ]
    return "-".join(readable_parts)


def _run_dir(config: Mapping[str, Any], split: str, kind: str, method: str | None = None) -> Path:
    base = _experiment_root(config, split)
    if kind == "audit":
        return base / "audit" / _budget_tag(config["label_budget"])
    if kind == "pretrain":
        if method is None:
            raise ValueError("pretrain output requires a method")
        return base / "pretrain" / method / _full_pretrain_tag(config, method)
    if kind == "supervised":
        return base / "finetune" / "sup" / _full_finetune_tag(config, kind)
    if kind == "finetune":
        if method is None:
            raise ValueError("finetune output requires a method")
        return base / "finetune" / method / (_full_finetune_tag(config, kind) + "-p" + _config_digest(_extract_pretrain_config(config, method)))
    raise ValueError(f"unsupported result kind: {kind}")


def _artifacts_ready(directory: Path) -> bool:
    return all((directory / filename).is_file() for filename in ARTIFACTS)


def _completed_finetune(directory: Path) -> bool:
    """Return whether an identity-bound fine-tune result was fully published."""

    return all((directory / filename).is_file() for filename in ("done.txt", "final_report.json"))


def _resumable_finetune(directory: Path) -> bool:
    """Return whether an interrupted fine-tune has a committed recovery state."""

    return (directory / "last_state.pth").is_file()


def _resume_pretraining(directory: Path) -> bool:
    """Return whether a pretraining run has a checkpoint that can be resumed.

    A prior failure can leave only ``run_identity.json`` before the first
    checkpoint is published. That file cannot be resumed and prevents a fresh
    run in the same identity-bound directory, so remove precisely that empty
    setup state before starting over.
    """

    checkpoint = directory / "latest.pth"
    if checkpoint.is_file():
        return True
    if not directory.is_dir():
        return False
    children = tuple(directory.iterdir())
    if children and all(child.is_file() and child.name in {
        "run_identity.json", "config.yaml", "experiment_config.json", "data_manifest.json", "leakage_audit.json"
    } for child in children):
        for child in children:
            child.unlink()
        directory.rmdir()
        print(f"[restart] discarded checkpointless pretraining setup: {directory}", flush=True)
    return False


def _audit(config: Mapping[str, Any], split: str, audit_dir: Path) -> None:
    if _artifacts_ready(audit_dir):
        saved = DataManifest.read_json(audit_dir / "data_manifest.json")
        expected = build_data_manifest(str(_path(str(config["dataset_root"]))),
                                       _baseline_snapshot(config), config["protocol"], split,
                                       int(config["seed"]), config["scope"])
        if saved.fingerprint() != expected.fingerprint():
            raise ValueError("cached audit differs from current data split; use a new seed/boundary directory")
        print(f"[reuse] audit: {audit_dir}", flush=True)
        return

    args = Namespace(
        dataset_root=str(_path(str(config["dataset_root"]))),
        config_file=str(_path(str(config["baseline_config"]))),
        protocol=config["protocol"], split=split, scope=config["scope"],
        seed=int(config["seed"]), label_budget=config["label_budget"], output_dir=str(audit_dir),
    )
    if not audit_dir.exists():
        run_audit(args)
        return

    # Repair a partial audit atomically through a sibling staging directory;
    # the audit writer deliberately refuses to overwrite an existing directory.
    repair_dir = audit_dir.with_name(f".{audit_dir.name}.repair")
    if repair_dir.exists():
        raise ValueError(f"stale audit repair directory exists: {repair_dir}")
    args.output_dir = str(repair_dir)
    try:
        run_audit(args)
        for filename in ARTIFACTS:
            source = repair_dir / filename
            target = audit_dir / filename
            if target.exists():
                if target.read_bytes() != source.read_bytes():
                    raise ValueError(f"partial audit artifact conflicts with regenerated artifact: {target}")
            else:
                source.replace(target)
    finally:
        if repair_dir.exists():
            shutil.rmtree(repair_dir)
    if not _artifacts_ready(audit_dir):
        raise ValueError(f"audit repair did not publish all artifacts: {audit_dir}")
    print(f"[repair] audit repaired: {audit_dir}", flush=True)

def _matched_args(config: Mapping[str, Any], split: str, audit_dir: Path, output_dir: Path, section: Mapping[str, Any]) -> Namespace:
    return Namespace(
        experiment_source=dict(config),
        dataset_root=str(_path(str(config["dataset_root"]))),
        config_file=str(_path(str(config["baseline_config"]))),
        manifest=str(audit_dir / "data_manifest.json"),
        label_manifest=str(audit_dir / "fewshot_manifest.json"),
        leakage_audit=str(audit_dir / "leakage_audit.json"),
        axis_stats=str(audit_dir / "label_stats.json"),
        label_budget=config["label_budget"], protocol=config["protocol"], split=split,
        seed=int(config["seed"]), output_dir=str(output_dir), loss_name=None, bone_set=None,
        strategy=section.get("strategy", "matched"),
        strategy_params=section.get("strategy_params", {}),
        optimizer=section.get("optimizer", "adamw"),
        learning_rate=float(section.get("learning_rate", 3e-4)),
        weight_decay=float(section.get("weight_decay", 0.0)),
        sgd_momentum=float(section.get("sgd_momentum", 0.9)),
        scheduler=section.get("scheduler", "cosine"),
        lr_warmup_epochs=int(section.get("lr_warmup_epochs", 0)),
        lr_min=float(section.get("lr_min", 1e-6)),
        lr_milestones=list(section.get("lr_milestones", [])),
        lr_gamma=float(section.get("lr_gamma", 0.5)),
        device=config.get("device", "cuda"),
        num_workers=int(section.get("num_workers", 0)), epochs=section.get("epochs"),
        max_train_batches=None, max_val_batches=None, val_every=int(section.get("val_every", 5)),
        no_amp=not bool(section.get("amp", True)),
    )


def _pretrain_runtime_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten the pretraining section for the pretraining runner."""

    runtime = dict(config)
    runtime.update(_runtime(config, "pretrain"))
    return runtime


def _reusable_pretraining(directory, config, method, manifest_path, *, legacy_config=None):
    """Find an exact completed cache, including verifiable pre-snapshot runs."""
    spec = _extract_pretrain_config(config, method)
    pretrain = spec["pretrain"]
    manifest = DataManifest.read_json(manifest_path)
    candidates = [directory]
    if directory.parent.is_dir():
        candidates += sorted(path for path in directory.parent.iterdir() if path != directory and path.is_dir())
    for candidate in candidates:
        if not all((candidate / name).is_file() for name in ("latest.pth", "done.txt", "run_identity.json")):
            continue
        saved = RunIdentity.from_dict(json.loads((candidate / "run_identity.json").read_text(encoding="utf-8")))
        boundary = make_pretrain_identity(manifest, method, int(config["seed"]), saved.config_fingerprint)
        if saved != boundary:
            continue
        checkpoint = load_checkpoint(candidate / "latest.pth", saved, expected_status=STATUS_COMPLETE)
        execution = checkpoint.get("execution_parameters", {})
        expected = {
            "requested_micro_batch": pretrain["micro_batch"],
            "requested_gradient_accumulation": pretrain["gradient_accumulation"],
            "max_batches": pretrain["max_batches"], "num_workers": pretrain["num_workers"],
            "use_amp": pretrain["amp"], "profile_cuda": pretrain["profile_cuda"],
            "device_type": str(config.get("device", "cuda")).split(":")[0],
        }
        if any(execution.get(key) != value for key, value in expected.items()):
            continue
        if checkpoint["next_epoch"] != int(pretrain["epochs"]):
            continue
        fingerprint = _canonical_fingerprint({"method": method, "config": spec,
                                              "execution_parameters": execution})
        fingerprints = {fingerprint}
        legacy_source = legacy_config or config
        if legacy_source.get("pretrain"):
            fingerprints.add(_canonical_fingerprint({
                "method": method, "config": _legacy_pretrain_config(legacy_source),
                "execution_parameters": execution,
            }))
        if saved.config_fingerprint in fingerprints:
            print(f"[reuse] verified pretraining: {candidate}", flush=True)
            return candidate
    return None


def _ssl_one(config: Mapping[str, Any], method: str, split: str) -> None:
    audit_dir = _run_dir(config, split, "audit")
    _audit(config, split, audit_dir)
    if method == "supervised":
        supervised_dir = _run_dir(config, split, "supervised")
        if _completed_finetune(supervised_dir):
            print(f"[reuse] completed full supervised result: {supervised_dir}", flush=True)
            return
        run_matched(
            _matched_args(config, split, audit_dir, supervised_dir, _runtime(config, "supervised")),
            encoder_checkpoint=None,
            method="sup",
            resume=_resumable_finetune(supervised_dir),
        )
        return

    pretrain = _runtime(config, "pretrain")
    pretrain_dir = _run_dir(config, split, "pretrain", method)
    ssl_config = _pretrain_runtime_config(config)
    completed_pretrain = _reusable_pretraining(
        pretrain_dir, ssl_config, method, audit_dir / "data_manifest.json")
    if completed_pretrain:
        pretrain_dir = completed_pretrain
        print(f"[reuse] completed full pretraining: {pretrain_dir}", flush=True)
    else:
        run_pretraining(
            dataset_root=str(_path(str(config["dataset_root"]))), config=ssl_config, method_name=method,
            manifest_path=audit_dir / "data_manifest.json", audit_path=audit_dir / "leakage_audit.json",
            output_dir=pretrain_dir, device=config.get("device", "cuda"), seed=int(config["seed"]),
            epochs=pretrain.get("epochs"), max_batches=pretrain.get("max_batches"), micro_batch=pretrain.get("micro_batch"),
            gradient_accumulation=int(pretrain.get("gradient_accumulation", 1)), num_workers=int(pretrain.get("num_workers", 0)),
            use_amp=bool(pretrain.get("amp", True)), profile_cuda=bool(pretrain.get("profile_cuda", False)),
            resume=_resume_pretraining(pretrain_dir),
        )
    fine_tune = _runtime(config, "finetune")
    finetune_dir = _run_dir(config, split, "finetune", method)
    if _completed_finetune(finetune_dir):
        print(f"[reuse] completed full fine-tune result: {finetune_dir}", flush=True)
        return
    run_matched(
        _matched_args(config, split, audit_dir, finetune_dir, fine_tune),
        encoder_checkpoint=pretrain_dir / "latest.pth",
        method="ssl",
        resume=_resumable_finetune(finetune_dir),
    )



def _small_options(config: Mapping[str, Any]) -> Mapping[str, Any]:
    options = _runtime(config, "small_sample")
    fraction = options.get("pretrain_sequence_fraction")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0.0 < float(fraction) <= 1.0:
        raise ValueError("small_sample.pretrain_sequence_fraction must be in (0, 1]")
    return options


_PRETRAIN_IMPLICIT_DEFAULTS = {
    "optimizer": "adamw",
    "sgd_momentum": 0.9,
    "scheduler": "cosine",
    "lr_milestones": [60, 80],
    "lr_gamma": 0.1,
    "lr_min": 0.0,
}
_DOWNSTREAM_IMPLICIT_DEFAULTS = {
    "optimizer": "adamw",
    "learning_rate": 3e-4,
    "weight_decay": 0.0,
    "sgd_momentum": 0.9,
    "scheduler": "cosine",
    "lr_warmup_epochs": 0,
    "lr_min": 1e-6,
    "lr_milestones": [20, 40, 60, 80],
    "lr_gamma": 0.5,
}
_SUPERVISED_STRATEGY_DEFAULTS = {
    "sup_decoder_lr": 1e-4,
    "sup_encoder_layer12_lr": 5e-6,
    "sup_encoder_layer3_lr": 1e-5,
    "sup_encoder_high_lr": 3e-5,
}
_FINETUNE_STRATEGY_DEFAULTS = {
    "transfer_decoder_warmup_lr": 3e-4,
    "transfer_decoder_lr": 1e-4,
    "transfer_encoder_layer12_lr": 5e-6,
    "transfer_encoder_layer3_lr": 1e-5,
    "transfer_encoder_high_lr": 3e-5,
}


def _strip_implicit_defaults(section: Mapping[str, Any], defaults: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(section)
    for key, default in defaults.items():
        if result.get(key) == default:
            result.pop(key)
    return result


def _small_binding_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the config that defines a shared small-sample experiment."""

    bound = dict(config)
    bound.pop("label_budget", None)
    bound["pretrain"] = _strip_implicit_defaults(_runtime(config, "pretrain"), _PRETRAIN_IMPLICIT_DEFAULTS)
    supervised = _strip_implicit_defaults(_runtime(config, "supervised"), _DOWNSTREAM_IMPLICIT_DEFAULTS)
    supervised["strategy_params"] = _strip_implicit_defaults(
        supervised.get("strategy_params", {}), _SUPERVISED_STRATEGY_DEFAULTS
    )
    if not supervised["strategy_params"]:
        supervised.pop("strategy_params")
    finetune = _strip_implicit_defaults(_runtime(config, "finetune"), _DOWNSTREAM_IMPLICIT_DEFAULTS)
    finetune["strategy_params"] = _strip_implicit_defaults(
        finetune.get("strategy_params", {}), _FINETUNE_STRATEGY_DEFAULTS
    )
    if not finetune["strategy_params"]:
        finetune.pop("strategy_params")
    bound["supervised"] = supervised
    bound["finetune"] = finetune
    bound["small_sample"] = dict(_runtime(config, "small_sample"))
    return bound


def _config_digest(value: object) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:10]


def _small_tag(options: Mapping[str, Any], config: Mapping[str, Any] | None = None) -> str:
    fraction = round(float(options["pretrain_sequence_fraction"]) * 100)
    epochs = int(options.get("pretrain_epochs", 10))
    base = f"small-u{fraction}-e{epochs}"
    if config is None:
        return base
    return f"{base}-c{_config_digest(_small_binding_config(config))}"


def _small_root(config: Mapping[str, Any], split: str, options: Mapping[str, Any]) -> Path:
    return _experiment_root(config, split) / "small" / _small_tag(options, config)


def _small_pretrain_tag(
    config: Mapping[str, Any], options: Mapping[str, Any], method: str
) -> str:
    fraction = round(float(options["pretrain_sequence_fraction"]) * 100)
    epochs = int(options.get("pretrain_epochs", 10))
    pretrain = _small_section(config, "pretrain", options)
    identity = {"fraction": float(options["pretrain_sequence_fraction"]),
                "spec": _extract_pretrain_config({**config, "pretrain": pretrain}, method)}

    # Extract readable components
    optimizer = pretrain.get("optimizer", "adamw")
    scheduler = pretrain.get("scheduler", "cosine")

    return f"u{fraction}-{optimizer}-{scheduler}-e{epochs}-c{_config_digest(identity)}"


def _small_pretrain_root(
    config: Mapping[str, Any], split: str, options: Mapping[str, Any], method: str
) -> Path:
    return _experiment_root(config, split) / "pretrain" / method / _small_pretrain_tag(config, options, method)


def _small_pretrain_manifest_root(
    config: Mapping[str, Any], split: str, options: Mapping[str, Any], method: str
) -> Path:
    return _experiment_root(config, split) / "pretrain_manifests" / method / _small_pretrain_tag(config, options, method)


def _bind_small_root(config: Mapping[str, Any], root: Path) -> None:
    """Bind an automatically named result root to its config digest."""

    metadata_path = root / "small_run_config.json"
    fingerprint = _config_digest(_small_binding_config(config))
    payload = {"schema_version": 1, "config_fingerprint": fingerprint, "config": _small_binding_config(config)}
    if metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(existing, Mapping) or existing.get("config_fingerprint") != fingerprint:
            raise ValueError("automatic small result name is already bound to different parameters")
        return
    root.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _small_pretrain_manifest(
    config: Mapping[str, Any],
    split: str,
    audit_dir: Path,
    options: Mapping[str, Any],
    method: str,
) -> tuple[Path, Path]:
    checkpoint_root = _small_pretrain_root(config, split, options, method)
    manifest_root = _small_pretrain_manifest_root(config, split, options, method)
    manifest_path = manifest_root / "pretrain_manifest.json"
    audit_path = manifest_root / "pretrain_audit.json"
    if manifest_path.is_file() and audit_path.is_file():
        return manifest_path, audit_path

    # A failed run from the former layout may have placed only these metadata
    # files beside the checkpoint output. Move them out before the pretrainer
    # checks that its output directory is empty.
    for filename in ("pretrain_manifest.json", "pretrain_audit.json"):
        stale = checkpoint_root / filename
        target = manifest_root / filename
        if stale.is_file():
            if stale.is_file():
                if target.exists() and target.read_text(encoding="utf-8") != stale.read_text(encoding="utf-8"):
                    raise ValueError(f"conflicting pretrain metadata: {target}")
                if target.exists():
                    stale.unlink()
                else:
                    stale.replace(target)
    if manifest_path.is_file() and audit_path.is_file():
        return manifest_path, audit_path

    full_manifest = DataManifest.read_json(audit_dir / "data_manifest.json")
    subset = make_small_pretrain_manifest(
        full_manifest,
        fraction=float(options["pretrain_sequence_fraction"]),
        seed=int(config["seed"]),
        )
    audit = audit_manifest(subset)
    if not audit.passed:
        raise ValueError("灏忔牱鏈璁粌鏁版嵁瀹¤澶辫触: " + "; ".join(audit.violations))
    manifest_root.mkdir(parents=True, exist_ok=True)
    subset.write_json(manifest_path)
    payload = {
        "passed": True,
        "data_manifest_fingerprint": subset.fingerprint(),
        "counts": audit.counts,
        "selection": {"fraction": float(options["pretrain_sequence_fraction"]), **selection_summary(subset)},
    }
    audit_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = selection_summary(subset)
    print("[小样本] {} 条 sequence | {} | {}".format(summary["pretrain_sequences"], _small_tag(options), split), flush=True)
    return manifest_path, audit_path


def _small_section(config: Mapping[str, Any], stage: str, options: Mapping[str, Any]) -> dict[str, Any]:
    section = dict(_runtime(config, stage))
    section.update(_runtime(options, stage))
    epoch_key = f"{stage}_epochs"
    if epoch_key in options:
        section["epochs"] = options[epoch_key]
    return section


def _small_one(config: Mapping[str, Any], method: str, split: str) -> None:
    options = _small_options(config)
    audit_dir = _run_dir(config, split, "audit")
    _audit(config, split, audit_dir)
    root = _small_root(config, split, options)
    _bind_small_root(config, root)
    if method == "supervised":
        supervised_dir = root / "finetune" / "sup" / _small_finetune_tag(config["label_budget"], config, "supervised", options)
        if _completed_finetune(supervised_dir):
            print(f"[reuse] completed small-sample supervised result: {supervised_dir}", flush=True)
        else:
            run_matched(
                _matched_args(config, split, audit_dir, supervised_dir, _small_section(config, "supervised", options)),
                encoder_checkpoint=None,
                method="sup",
                resume=_resumable_finetune(supervised_dir),
            )
        return

    manifest_path, subset_audit_path = _small_pretrain_manifest(config, split, audit_dir, options, method)
    pretrain = _small_section(config, "pretrain", options)
    ssl_config = _pretrain_runtime_config({**config, "pretrain": pretrain})
    pretrain_dir = _small_pretrain_root(config, split, options, method)
    completed_pretrain = _reusable_pretraining(
        pretrain_dir, ssl_config, method, manifest_path, legacy_config=config)
    if completed_pretrain:
        pretrain_dir = completed_pretrain
        pretrain_checkpoint = pretrain_dir / "latest.pth"
        print(f"[reuse] completed small-sample pretraining: {pretrain_dir}", flush=True)
    else:
        resume_pretraining = _resume_pretraining(pretrain_dir)
        run_pretraining(
            dataset_root=str(_path(str(config["dataset_root"]))), config=ssl_config, method_name=method,
            manifest_path=manifest_path, audit_path=subset_audit_path, output_dir=pretrain_dir,
            device=config.get("device", "cuda"), seed=int(config["seed"]),
            epochs=pretrain.get("epochs"), max_batches=pretrain.get("max_batches"), micro_batch=pretrain.get("micro_batch"),
            gradient_accumulation=int(pretrain.get("gradient_accumulation", 1)), num_workers=int(pretrain.get("num_workers", 0)),
            use_amp=bool(pretrain.get("amp", True)), profile_cuda=bool(pretrain.get("profile_cuda", False)),
            resume=resume_pretraining,
        )
        pretrain_checkpoint = pretrain_dir / "latest.pth"
    finetune_dir = root / "finetune" / method / _small_finetune_tag(config["label_budget"], config, "finetune", options)
    if _completed_finetune(finetune_dir):
        print(f"[reuse] completed small-sample fine-tune result: {finetune_dir}", flush=True)
        return
    run_matched(
        _matched_args(config, split, audit_dir, finetune_dir, _small_section(config, "finetune", options)),
        encoder_checkpoint=pretrain_checkpoint,
        method="ssl",
        pretrain_manifest=manifest_path,
        resume=_resumable_finetune(finetune_dir),
    )


def _small(config: Mapping[str, Any], method: str, mode: str, split_override: str | None) -> None:
    # baseline 鏄皬鏍锋湰闅忔満鍒濆鍖栫洃鐫ｅ鐓х殑鏄撹鍒悕銆?
    if method == "baseline":
        method = "supervised"
    if method not in ("supervised", *SSL_METHODS):
        raise ValueError("small method must be baseline, supervised, simclr, moco, swav, relpos, mfm, or mae")
    if method != "supervised" and method not in config.get("methods", {}):
        raise ValueError(f"small_ssl_config.yaml 缂哄皯 {method} 鏂规硶閰嶇疆")
    for split in _splits(config, mode, split_override):
        _small_one(config, method, split)


def _ssl(config: Mapping[str, Any], method: str, mode: str, split_override: str | None) -> None:
    if method not in ("supervised", *SSL_METHODS):
        raise ValueError("SSL method must be supervised, simclr, moco, swav, relpos, mfm, or mae")
    if method != "supervised" and method not in config.get("methods", {}):
        raise ValueError(f"ssl_config.yaml 缂哄皯 {method} 鏂规硶閰嶇疆")
    for split in _splits(config, mode, split_override):
        _ssl_one(config, method, split)


def main() -> int:
    parser = argparse.ArgumentParser(description="缁熶竴杩愯瀹樻柟鍩虹嚎鍜?SSL 瀹為獙")
    parser.add_argument("workflow", choices=("baseline", "ssl", "small"))
    parser.add_argument("target", help="baseline 浣跨敤 single/all锛泂sl 浣跨敤 supervised 鎴?SSL 鏂规硶")
    parser.add_argument("mode", nargs="?", help="ssl 鏃舵寚瀹?single 鎴?all")
    parser.add_argument("--protocol", choices=("protocol1", "protocol2", "protocol3"))
    parser.add_argument("--split", choices=SPLITS)
    parser.add_argument("--label-budget", help="override downstream label budget, e.g. 2shot or 4subjects")
    args = parser.parse_args()

    if args.workflow == "baseline":
        if args.mode is not None or args.target not in {"single", "all"}:
            parser.error("鐢ㄦ硶锛歱ython scripts/run.py baseline single|all")
        config = _load_config("baseline")
        _baseline(config, args.target, args.protocol or str(config["protocol"]), args.split)
    else:
        if args.mode not in {"single", "all"}:
            parser.error("鐢ㄦ硶锛歱ython scripts/run.py ssl|small baseline|supervised|simclr|moco|swav|relpos|mfm|mae single|all")
        config = _load_config("small_ssl" if args.workflow == "small" else "ssl")
        if args.protocol:
            config["protocol"] = args.protocol
        if args.label_budget:
            config["label_budget"] = args.label_budget
        if args.workflow == "small":
            _small(config, args.target, args.mode, args.split)
        else:
            _ssl(config, args.target, args.mode, args.split)
    generate_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
