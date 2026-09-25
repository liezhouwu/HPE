"""One-command orchestration for the ViT-csi-small MAE line.

    python scripts/vit_ssl/run.py <dataset_root> configs/vit_ssl_small_config.yaml [single|all]

``single`` (default) runs one split (``--split`` or the config's ``single_split``);
``all`` runs every split listed in the config's ``all_splits`` in sequence, each with
its own audit manifest, pretraining run and fine-tune arms.

``--protocol`` / ``--split`` / ``--label-budget`` default to the config's
``protocol`` / ``single_split`` / ``label_budget``; ``--sequence-fraction`` defaults to
``small_sample.pretrain_sequence_fraction`` (1.0 = full data).

Steps per split (each resumable by re-running the same command):

1. reuse an existing audited manifest/audit pair from the MetaFi audit tree;
2. ViT-MAE pretraining (``scripts/vit_ssl/pretrain.py``);
3. Sup-ViT fine-tune control (no pretraining checkpoint);
4. MAE-ViT fine-tune from step 2's ``latest.pth``.

Everything lands under ``result_metafi_ssl/vit/``; the MetaFi method directories
and the small-sample workflow are never touched.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import yaml

_BASE = Path(__file__).resolve().parents[2]
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))

from mmfi_wifi.data_manifest import DataManifest
from scripts.vit_ssl.finetune import _engine_config, _section, run_finetune
from scripts.vit_ssl.pretrain import METHOD_NAME, _resolve_sequence_fraction, run_pretraining


VIT_ROOT = Path("result_metafi_ssl") / "vit"


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ViT-MAE pipeline end to end")
    parser.add_argument("dataset_root")
    parser.add_argument("config_file")
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("single", "all"),
        default="single",
        help="single=只跑一个划分（默认，取 --split 或配置 single_split）；all=依次跑配置 all_splits 的全部划分",
    )
    parser.add_argument("--protocol", default=None, help="缺省取配置的 protocol")
    parser.add_argument("--split", default=None, help="缺省取配置的 single_split")
    parser.add_argument("--label-budget", default=None, help="缺省取配置的 label_budget")
    parser.add_argument("--audit-dir", default=None, help="默认在 result_metafi_ssl 审计树中查找")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sequence-fraction",
        type=float,
        default=None,
        help="预训练只用该比例的无标注 sequence（<1 进入小样本布局）；缺省读 small_sample 配置",
    )
    parser.add_argument("--epochs", type=int, default=None, help="覆盖预训练 epoch 数")
    parser.add_argument("--finetune-epochs", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None, help="预训练冒烟用")
    parser.add_argument("--max-train-batches", type=int, default=None, help="微调冒烟用")
    parser.add_argument("--skip-sup", action="store_true")
    parser.add_argument("--skip-mae", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args(argv)


def _load_config(path: str | Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"unable to read config file: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("config root must be a mapping")
    return payload


def _budget_tag(label_budget: str) -> str:
    """Mirror the MetaFi audit-tree budget tag (``1shot`` -> ``b1s``)."""

    return "b" + label_budget.replace("shot", "s").replace("%", "pct")


def _find_audit_dir(args: argparse.Namespace) -> Path:
    """Locate the audited manifest/audit pair produced by the MetaFi audit step."""

    if args.audit_dir:
        candidate = Path(args.audit_dir)
        if not (candidate / "data_manifest.json").is_file():
            raise FileNotFoundError(f"audit directory lacks data_manifest.json: {candidate}")
        return candidate
    tag = _budget_tag(args.label_budget)
    pattern = (
        f"result_metafi_ssl/runs/{args.protocol}/*/{args.split}/seed*/audit/{tag}/data_manifest.json"
    )
    matches = sorted(_BASE.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"no audited manifest found for {args.protocol}/{args.split}/{args.label_budget}; "
            "run the MetaFi audit step first (python scripts/run.py ssl supervised single) "
            "or pass --audit-dir"
        )
    if len(matches) > 1:
        raise ValueError(
            "multiple audited manifests match; pass --audit-dir explicitly: "
            + ", ".join(str(path.parent) for path in matches)
        )
    return matches[0].parent


def _short(value: object) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]


def _pretrain_tag(
    epochs: int,
    micro_batch: int,
    accumulation: int,
    *,
    fraction: float,
    seed: int,
    model_config: Mapping[str, Any] | None = None,
) -> str:
    """Readable pretraining tag; ``fraction == 1.0`` reproduces the full-data tag.

    The hash covers the model geometry (``vit_mae``), so changing the patch grid or the
    decoder yields a different directory instead of colliding with an existing run whose
    identity no longer matches.
    """

    payload: dict[str, object] = {
        "epochs": epochs,
        "micro_batch": micro_batch,
        "accumulation": accumulation,
        "vit_mae": dict(model_config or {}),
    }
    if fraction == 1.0:
        return f"{METHOD_NAME}-e{epochs}-mb{micro_batch}x{accumulation}-{_short(payload)}"
    payload.update({"fraction": float(fraction), "seed": seed})
    return f"u{round(fraction * 100)}-{METHOD_NAME}-e{epochs}-{_short(payload)}"


def _layout(
    protocol: str, scope: str, split: str, seed: int, fraction: float, tag: str
) -> dict[str, Path]:
    """Return the pretrain / manifest / finetune root directories of this variant.

    ``fraction == 1.0`` keeps the full-data layout (``vit/pretrain`` + ``vit/runs``);
    small-sample runs live below ``vit/small/<protocol>/<scope>/<split>/seed<seed>/<tag>``.
    Manifests are always a sibling of the run directory: the runner requires an empty
    output directory.
    """

    if fraction == 1.0:
        tree = _BASE / VIT_ROOT
        pretrain_dir = tree / "pretrain" / protocol / scope / split / f"seed{seed}" / tag
        return {
            "pretrain_dir": pretrain_dir,
            "manifest_dir": pretrain_dir.parent / "pretrain_manifests" / tag,
            "finetune_root": tree / "runs" / protocol / scope / split,
        }
    variant_root = _BASE / VIT_ROOT / "small" / protocol / scope / split / f"seed{seed}" / tag
    return {
        "pretrain_dir": variant_root / "pretrain",
        "manifest_dir": variant_root / "pretrain_manifests",
        "finetune_root": variant_root / "finetune",
    }


def _resolve_stage_options(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    """Fill protocol / split / label budget from the config when the CLI omits them."""

    resolved = {
        "protocol": args.protocol if args.protocol else config.get("protocol"),
        "split": args.split if args.split else config.get("single_split"),
        "label_budget": args.label_budget if args.label_budget else config.get("label_budget"),
    }
    for name, value in resolved.items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be provided on the command line or in the config")
        setattr(args, name, value)


def _splits(config: Mapping[str, Any], mode: str, override: str | None) -> tuple[str, ...]:
    """Return the splits to run: one for ``single``, the configured list for ``all``."""

    if mode == "single":
        split = override or config.get("single_split")
        if not isinstance(split, str) or not split:
            raise ValueError("single 模式需要 --split 或配置里的 single_split")
        return (split,)
    if override:
        raise ValueError("all 模式不接受 --split；请改用 single 模式")
    configured = config.get("all_splits")
    if not isinstance(configured, (list, tuple)) or not configured:
        raise ValueError("all 模式需要配置里非空的 all_splits 列表")
    if any(not isinstance(split, str) or not split for split in configured):
        raise ValueError("all_splits 必须是非空字符串列表")
    return tuple(configured)


def _finetune_state(directory: Path) -> str:
    """Classify one fine-tune arm directory: complete / resumable / fresh / incomplete.

    ``prepare_run_directory`` refuses to write into a non-empty directory, so a
    re-run of the same command must recognise finished arms instead of colliding
    with them.
    """

    directory = Path(directory)
    if not directory.is_dir():
        return "fresh"
    if (directory / "done.txt").is_file() and (directory / "summary_absolute.json").is_file():
        return "complete"
    if (directory / "last_state.pth").is_file():
        return "resumable"
    if not any(directory.iterdir()):
        return "fresh"
    return "incomplete"


_FINETUNE_COMPARED_KEYS = (
    "protocol",
    "split_to_use",
    "init_rand_seed",
    "num_epochs",
    "optimizer",
    "learning_rate",
    "weight_decay",
    "sgd_momentum",
    "scheduler",
    "lr_warmup_epochs",
    "lr_min",
    "lr_milestones",
    "lr_gamma",
    "dropout_p",
    "selection_metric",
    "early_stopping_patience",
    "use_epoch_patience",
    "target_space",
    "fine_tune_strategy",
    "modality",
    "data_unit",
    "amp_dtype",
)


def _assert_finetune_config_matches(directory: Path, engine_config: Mapping[str, Any]) -> None:
    """Fail closed when an existing arm was produced by a different fine-tune config."""

    stored_path = Path(directory) / "config.yaml"
    if not stored_path.is_file():
        raise ValueError(f"已有的微调目录缺少 config.yaml，无法确认配置一致: {directory}")
    stored = yaml.safe_load(stored_path.read_text(encoding="utf-8"))
    if not isinstance(stored, Mapping):
        raise ValueError(f"{stored_path} 不是配置字典")
    drifted = [key for key in _FINETUNE_COMPARED_KEYS if stored.get(key) != engine_config.get(key)]
    if drifted:
        raise ValueError(
            "该目录里已有配置不同的微调结果（差异字段: " + ", ".join(drifted) + "）；"
            f"请改回原配置或换新目录后再跑: {directory}"
        )


def _run_one_split(args: argparse.Namespace, config: Mapping[str, Any], split: str) -> None:
    """Run audit reuse -> pretraining -> both fine-tune arms for one split."""

    args.split = split
    pretrain_config = config.get("pretrain", {})
    if not isinstance(pretrain_config, Mapping):
        raise ValueError("pretrain config must be a mapping")
    scope = str(config.get("scope", "strict"))
    audit_dir = _find_audit_dir(args)
    print(f"[审计] {split}: 复用 {audit_dir}", flush=True)

    epochs = int(args.epochs if args.epochs is not None else pretrain_config.get("epochs", 350))
    micro_batch = pretrain_config.get("micro_batch", 32)
    accumulation = pretrain_config.get("gradient_accumulation", 8)
    fraction = _resolve_sequence_fraction(config, args.sequence_fraction)
    model_config = config.get("vit_mae", {})
    if not isinstance(model_config, Mapping):
        raise ValueError("vit_mae config must be a mapping")
    pretrain_tag = _pretrain_tag(
        epochs, micro_batch, accumulation, fraction=fraction, seed=args.seed, model_config=model_config
    )
    layout = _layout(args.protocol, scope, args.split, args.seed, fraction, pretrain_tag)
    pretrain_dir = layout["pretrain_dir"]
    try:
        checkpoint = run_pretraining(
            dataset_root=args.dataset_root,
            config=config,
            manifest_path=audit_dir / "data_manifest.json",
            audit_path=audit_dir / "leakage_audit.json",
            output_dir=pretrain_dir,
            device=args.device,
            seed=args.seed,
            epochs=args.epochs,
            max_batches=args.max_batches,
            use_amp=not args.no_amp,
            resume=args.resume or (pretrain_dir / "latest.pth").is_file(),
            sequence_fraction=fraction,
            manifest_dir=layout["manifest_dir"],
        )
    except ValueError as error:
        if "resume RunIdentity mismatch" in str(error):
            raise ValueError(
                f"{pretrain_dir} 里已有一份预训练，但它的配置指纹与本轮不一致，无法直接复用/续训。"
                "常见原因：改了预训练真正用到的参数（vit_mae 的 patch/decoder、epochs、"
                "micro_batch/accumulation、fraction、学习率或调度器）。"
                "下游参数（label_budget、finetune、all_splits 等）不影响预训练，可以安全复用。"
                "请改回原配置，或确认后删除该目录再跑。"
            ) from error
        raise
    print(f"[预训练] {checkpoint}", flush=True)
    if fraction < 1.0:
        subset = DataManifest.read_json(layout["manifest_dir"] / "pretrain_manifest.json")
        bound = DataManifest.read_json(pretrain_dir / "data_manifest.json")
        if bound.fingerprint() != subset.fingerprint():
            raise ValueError(
                "pretraining run did not record the small-sample subset manifest: "
                f"{pretrain_dir / 'data_manifest.json'}"
            )

    shared = {
        "dataset_root": args.dataset_root,
        "config_file": args.config_file,
        "manifest": str(audit_dir / "data_manifest.json"),
        "leakage_audit": str(audit_dir / "leakage_audit.json"),
        "label_manifest": str(audit_dir / "fewshot_manifest.json"),
        "axis_stats": str(audit_dir / "label_stats.json"),
        "audit_dir": None,
        "protocol": args.protocol,
        "split": args.split,
        "label_budget": args.label_budget,
        "device": args.device,
        "seed": args.seed,
        "epochs": args.finetune_epochs,
        "max_train_batches": args.max_train_batches,
        "pretrain_manifest": str(pretrain_dir / "data_manifest.json"),
        "resume": args.resume,
        "no_amp": args.no_amp,
    }
    finetune_epochs = args.finetune_epochs if args.finetune_epochs is not None else config.get("finetune", {}).get("epochs", 25)
    for method, pretrain_checkpoint in (("sup_vit", None), (METHOD_NAME, checkpoint / "latest.pth")):
        if (method == "sup_vit" and args.skip_sup) or (method != "sup_vit" and args.skip_mae):
            continue
        finetune_tag = (
            f"b{args.label_budget}-ft{finetune_epochs}-"
            f"{_short({'method': method, 'budget': args.label_budget, 'pretrain': pretrain_tag, 'seed': args.seed})}"
        )
        output_dir = layout["finetune_root"] / method / finetune_tag
        state = _finetune_state(output_dir)
        if state in {"complete", "resumable"}:
            _assert_finetune_config_matches(
                output_dir,
                _engine_config(
                    config,
                    protocol=args.protocol,
                    split=split,
                    seed=args.seed,
                    epochs=finetune_epochs,
                    finetune=_section(config, "finetune"),
                ),
            )
        if state == "complete":
            print(f"[复用] {split} | {method} | 已完成，跳过 | {output_dir}", flush=True)
            continue
        if state == "incomplete":
            raise FileExistsError(
                f"{split} | {method} 的目录既没有完成产物也没有 last_state.pth，无法安全继续: {output_dir}；"
                "确认后手动删除该目录再重跑，或加上 --resume 从已有状态续训"
            )
        namespace = argparse.Namespace(
            **{
                **shared,
                "output_dir": str(output_dir),
                "pretrain_checkpoint": None if pretrain_checkpoint is None else str(pretrain_checkpoint),
                "num_workers": None,
                "val_every": None,
                "max_val_batches": None,
                "resume": args.resume or state == "resumable",
            }
        )
        if state == "resumable":
            print(f"[续训] {split} | {method} | 从 last_state.pth 继续 | {output_dir}", flush=True)
        summary = run_finetune(namespace)
        print(
            f"[微调] {split} | {method} | test MPJPE {summary['test_mpjpe_mm']:.1f}mm "
            f"PA-MPJPE {summary['test_pampjpe_mm']:.1f}mm | {output_dir}",
            flush=True,
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = _load_config(args.config_file)
    requested_split = args.split
    _resolve_stage_options(args, config)
    splits = _splits(config, args.mode, requested_split)
    print(f"[ViT-MAE] mode={args.mode} | splits={list(splits)}", flush=True)
    for index, split in enumerate(splits, start=1):
        print(f"[ViT-MAE] === 划分 {index}/{len(splits)}: {split} ===", flush=True)
        _run_one_split(args, config, split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
