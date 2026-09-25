"""
try3 — 批量实验 CLI
=====================
对指定 protocol 运行官方划分 (S1/S2/S3)，并在同一次 run 内汇总结果。

用法:
    python scripts/reproduction/run_all.py ../my_dataset ./config.yaml --protocol protocol3
    python scripts/reproduction/run_all.py ../my_dataset ./config.yaml --protocol protocol3 --splits random_split
    python scripts/reproduction/run_all.py ../my_dataset ./config.yaml --protocol protocol3 \
        --run_dir ./result/runs/protocol3/20260727_120000 --splits cross_subject_split
    python scripts/reproduction/run_all.py ../my_dataset ./config.yaml --protocol protocol3 \
        --run_dir ./result/runs/protocol3/20260727_120000 --resume --splits random_split

默认结果目录:
    result/runs/<protocol>/<timestamp>/
    ├── run_manifest.json
    ├── random_split/
    ├── cross_subject_split/
    ├── cross_scene_split/
    ├── _comparison_curves.png
    └── _all_results.json
"""

import os
import sys
import json
import csv
import argparse
import time
from datetime import datetime

# UTF-8 patch (仅主进程)
import builtins as _bi
_orig_open = _bi.open
def _utf8(file, mode='r', buffering=-1, encoding=None,
          errors=None, newline=None, closefd=True, opener=None):
    if encoding is None and 'b' not in mode:
        encoding = 'utf-8'
    return _orig_open(file, mode, buffering, encoding, errors, newline, closefd, opener)
_bi.open = _utf8

import yaml
import torch

_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _BASE)

from mmfi_wifi.engine import config_fingerprint, setup_cuda, train_one_experiment

ALL_SPLITS = ["random_split", "cross_subject_split", "cross_scene_split"]
MANIFEST_NAME = "run_manifest.json"


def _write_json(path, value):
    """Write JSON atomically so interrupted orchestration does not truncate metadata."""
    temp_path = f"{path}.tmp"
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, default=str)
    os.replace(temp_path, path)


def _default_run_root(protocol):
    """Return a timestamped path that does not currently exist."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    candidate = os.path.join(_BASE, "result", "runs", protocol, timestamp)
    suffix = 1
    while os.path.exists(candidate):
        candidate = os.path.join(
            _BASE, "result", "runs", protocol, f"{timestamp}_{suffix}")
        suffix += 1
    return candidate


def _manifest_config(base_config, protocol, epochs):
    """Build the split-independent effective config represented by a run root."""
    config = dict(base_config)
    config['protocol'] = protocol
    config.pop('split_to_use', None)
    if epochs is not None:
        config['num_epochs'] = epochs
    return config


def prepare_manifest(run_root, protocol, fingerprint, execution_fingerprint,
                     execution_spec, require_existing=False):
    """Create a run manifest or validate the manifest already in the run root."""
    manifest_path = os.path.join(run_root, MANIFEST_NAME)

    if require_existing and not os.path.isdir(run_root):
        raise ValueError(f"恢复目录不存在: {run_root}")

    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"无法读取 run manifest: {manifest_path}: {exc}") from exc

        if manifest.get('protocol') != protocol:
            raise ValueError(
                f"run protocol 不匹配: manifest={manifest.get('protocol')!r}, "
                f"requested={protocol!r}")
        if manifest.get('config_fingerprint') != fingerprint:
            raise ValueError(
                "run config fingerprint 不匹配；请使用创建该 run 的配置，"
                "或不指定 --run_dir 创建新 run")
        if manifest.get('execution_fingerprint') != execution_fingerprint:
            raise ValueError(
                "run 执行参数不匹配（数据根、AMP、验证频率或 batch 限制不同）")
        return manifest

    if require_existing:
        raise ValueError(f"恢复目录缺少 {MANIFEST_NAME}: {run_root}")

    if os.path.isdir(run_root) and os.listdir(run_root):
        raise ValueError(
            f"已有非空目录缺少 {MANIFEST_NAME}，拒绝作为 run root: {run_root}")

    os.makedirs(run_root, exist_ok=True)
    manifest = {
        'schema_version': 1,
        'protocol': protocol,
        'config_fingerprint': fingerprint,
        'execution_fingerprint': execution_fingerprint,
        'execution': execution_spec,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'completed_splits': [],
    }
    _write_json(manifest_path, manifest)
    return manifest


def scan_completed_results(run_root, protocol, config_fingerprints_expected=None,
                           execution_fingerprint_expected=None):
    """Load completed official splits from this run root only."""
    all_results = {}
    for split_name in ALL_SPLITS:
        split_dir = os.path.join(run_root, split_name)
        summary_path = os.path.join(split_dir, 'summary.json')
        if not os.path.isfile(summary_path):
            continue
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[WARN] 跳过无效 summary: {summary_path}: {exc}", flush=True)
            continue

        if summary.get('protocol') != protocol or summary.get('split') != split_name:
            raise ValueError(
                f"summary 与 run 不匹配: {summary_path} "
                f"({summary.get('protocol')!r}/{summary.get('split')!r})")
        expected_config = ((config_fingerprints_expected or {}).get(split_name))
        if (expected_config is not None and
                summary.get('config_fingerprint') != expected_config):
            raise ValueError(f"summary 配置指纹与 run 不匹配: {summary_path}")
        execution_path = os.path.join(split_dir, 'execution.json')
        if execution_fingerprint_expected is not None:
            if not os.path.isfile(execution_path):
                raise ValueError(f"已完成 split 缺少 execution.json: {split_dir}")
            with open(execution_path, 'r', encoding='utf-8') as f:
                split_execution = json.load(f)
            if config_fingerprint(split_execution) != execution_fingerprint_expected:
                raise ValueError(f"split 执行参数与 run 不匹配: {split_dir}")

        # Never follow a result_dir embedded in a copied or moved summary. This
        # keeps aggregate reads and comparisons inside the selected run root.
        summary['result_dir'] = split_dir
        all_results[split_name] = summary
    return all_results


def plot_comparison(all_results, result_root, protocol):
    """已完成划分的 MPJPE/PA-MPJPE 对比曲线。"""
    if not all_results:
        return
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Cross-Split Comparison — {protocol}', fontsize=14, fontweight='bold')
    colors = {'random_split': 'blue', 'cross_subject_split': 'green', 'cross_scene_split': 'red'}

    for metric_idx, (metric_key, title) in enumerate([('mpjpe_mm', 'MPJPE'),
                                                      ('pampjpe_mm', 'PA-MPJPE')]):
        ax = axes[metric_idx]
        for split_name, result in all_results.items():
            csv_path = os.path.join(result['result_dir'], 'metrics.csv')
            if not os.path.exists(csv_path):
                continue
            epochs, vals = [], []
            with open(csv_path, 'r', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    v = row.get(metric_key, 'nan')
                    if v != 'nan':
                        epochs.append(int(row['epoch']))
                        vals.append(float(v))
            if epochs:
                ax.plot(epochs, vals, color=colors.get(split_name, 'gray'),
                        linewidth=2, label=f"{split_name} (best {min(vals):.0f}mm)")
        ax.set_xlabel('Epoch')
        ax.set_ylabel(f'{title} (mm)')
        if ax.lines:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title(title)

    plt.tight_layout()
    out_path = os.path.join(result_root, "_comparison_curves.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] 对比曲线: {out_path}", flush=True)


def rebuild_aggregates(run_root, protocol, manifest):
    """Rebuild run-wide outputs from all completed split summaries on disk."""
    all_results = scan_completed_results(
        run_root, protocol,
        config_fingerprints_expected=manifest.get('split_config_fingerprints', {}),
        execution_fingerprint_expected=manifest.get('execution_fingerprint'))
    summary_path = os.path.join(run_root, "_all_results.json")
    _write_json(summary_path, all_results)
    plot_comparison(all_results, run_root, protocol)

    manifest['completed_splits'] = [
        split_name for split_name in ALL_SPLITS if split_name in all_results
    ]
    manifest['updated_at'] = datetime.now().isoformat(timespec='seconds')
    _write_json(os.path.join(run_root, MANIFEST_NAME), manifest)
    return all_results, summary_path


def print_results(all_results, protocol, total_time, summary_path):
    print(f"\n{'=' * 78}", flush=True)
    print(f"本次调用完成 ({total_time / 60:.0f} min) — {protocol}", flush=True)
    print(f"run 内已完成划分: {len(all_results)}/{len(ALL_SPLITS)}", flush=True)
    print(f"{'=' * 78}", flush=True)
    print(f"  {'Split':22s} {'Best MPJPE':>11s} {'Best PA':>9s} "
          f"{'Test MPJPE':>11s} {'Test PA':>9s}", flush=True)
    print(f"  {'-' * 66}", flush=True)
    for split_name in ALL_SPLITS:
        if split_name not in all_results:
            continue
        result = all_results[split_name]
        print(f"  {split_name:22s} {result['best_mpjpe_mm']:9.1f}mm "
              f"{result['best_pampjpe_mm']:7.1f}mm "
              f"{result['test_mpjpe_mm']:9.1f}mm "
              f"{result['test_pampjpe_mm']:7.1f}mm", flush=True)
    print(f"{'=' * 78}", flush=True)
    print(f"汇总报告: {summary_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="try3 — 在隔离的 run root 中运行并汇总多个划分")
    parser.add_argument("dataset_root", type=str)
    parser.add_argument("config_file", type=str)
    parser.add_argument("--protocol", type=str, default="protocol3",
                        choices=["protocol1", "protocol2", "protocol3"])
    parser.add_argument("--splits", type=str, nargs="+", default=ALL_SPLITS,
                        choices=ALL_SPLITS, help="只运行指定划分")
    parser.add_argument(
        "--run_dir", type=str, default=None, metavar="RUN_DIR",
        help="使用已有 run root 追加划分；省略时创建唯一时间戳目录")
    parser.add_argument(
        "--resume", action="store_true",
        help="恢复 --run_dir 中选定的划分（必须同时指定 --run_dir）")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="不给时用 config.num_workers (默认 8); 显式传入可覆盖")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--val_every", type=int, default=None,
                        help="不给时用 config.val_every (默认 2); 显式传入可覆盖")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    args = parser.parse_args()

    if args.resume and not args.run_dir:
        parser.error("--resume 必须与 --run_dir 一起使用")

    with open(args.config_file, 'r', encoding='utf-8') as f:
        base_config = yaml.load(f, Loader=yaml.FullLoader)

    # num_workers / val_every: CLI 显式传入优先, 否则回落 config, 再否则旧默认。
    num_workers = args.num_workers if args.num_workers is not None else base_config.get('num_workers', 8)
    val_every = args.val_every if args.val_every is not None else base_config.get('val_every', 2)

    protocol = args.protocol
    manifest_config = _manifest_config(base_config, protocol, args.epochs)
    fingerprint = config_fingerprint(manifest_config)
    run_root = (os.path.abspath(os.path.expanduser(args.run_dir))
                if args.run_dir else _default_run_root(protocol))
    effective_amp = bool(args.amp and torch.cuda.is_available() and args.device.startswith('cuda'))
    execution_spec = {
        'dataset_root': os.path.abspath(args.dataset_root),
        'num_workers': num_workers,
        'use_amp': effective_amp,
        'val_every': val_every,
        'max_train_batches': args.max_train_batches,
        'max_val_batches': args.max_val_batches,
    }
    execution_fingerprint = config_fingerprint(execution_spec)

    try:
        manifest = prepare_manifest(
            run_root, protocol, fingerprint, execution_fingerprint, execution_spec,
            require_existing=args.resume)
    except ValueError as exc:
        parser.error(str(exc))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    setup_cuda(device)

    print("=" * 60, flush=True)
    print(f"try3 批量实验 — {protocol}", flush=True)
    print(f"划分: {args.splits}", flush=True)
    print(f"模式: {'恢复' if args.resume else '新训练'}", flush=True)
    print(f"run 根目录: {run_root}", flush=True)
    print(f"val_every={val_every} num_workers={num_workers}", flush=True)
    print("=" * 60, flush=True)

    t_total = time.time()
    all_results = {}
    summary_path = os.path.join(run_root, "_all_results.json")
    try:
        for split_name in args.splits:
            print(f"\n{'#' * 60}", flush=True)
            print(f"# {'恢复' if args.resume else '开始'}: {protocol} / {split_name}", flush=True)
            print(f"{'#' * 60}", flush=True)

            config = dict(base_config)
            config['protocol'] = protocol
            config['split_to_use'] = split_name
            if args.epochs is not None:
                config['num_epochs'] = args.epochs

            split_fingerprint = config_fingerprint(config)
            known_fingerprints = manifest.setdefault('split_config_fingerprints', {})
            known = known_fingerprints.get(split_name)
            if known is not None and known != split_fingerprint:
                raise ValueError(f"{split_name} 配置与 run manifest 不一致")
            known_fingerprints[split_name] = split_fingerprint
            _write_json(os.path.join(run_root, MANIFEST_NAME), manifest)

            result_dir = os.path.join(run_root, split_name)
            train_one_experiment(
                args.dataset_root, config, result_dir, device,
                num_workers=num_workers, use_amp=args.amp,
                val_every=val_every,
                max_train_batches=args.max_train_batches,
                max_val_batches=args.max_val_batches,
                log_prefix="  ", resume=args.resume)
    finally:
        # Scan disk instead of aggregating only this invocation. Thus adding one
        # split cannot erase completed siblings, even if a later split fails.
        all_results, summary_path = rebuild_aggregates(
            run_root, protocol, manifest)

    print_results(all_results, protocol, time.time() - t_total, summary_path)


if __name__ == '__main__':
    main()
