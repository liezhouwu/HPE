"""
try3 — 单次训练 CLI
=====================
在指定 protocol/split 下完成一次完整训练 + held-out 评估。
核心逻辑在 mmfi_wifi.engine, 本文件仅负责参数解析。

用法:
    python scripts/reproduction/train.py ../my_dataset ./config.yaml
    python scripts/reproduction/train.py ../my_dataset ./config.yaml --output_dir ./result/my_run
    python scripts/reproduction/train.py ../my_dataset ./config.yaml --resume ./result/my_run
    python scripts/reproduction/train.py ../my_dataset ./config.yaml --split cross_subject_split --protocol protocol3
    python scripts/reproduction/train.py ../my_dataset ./config.yaml --epochs 2 --max_train_batches 50   # 冒烟
"""

import os
import sys
import argparse
from datetime import datetime

# UTF-8 patch (仅主进程, 解决 Windows GBK 控制台/文件编码问题)
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

from mmfi_wifi.engine import setup_cuda, train_one_experiment


def main():
    parser = argparse.ArgumentParser(description="try3 — 单次训练")
    parser.add_argument("dataset_root", type=str)
    parser.add_argument("config_file", type=str)
    parser.add_argument("--split", type=str, default=None,
                        choices=["random_split", "cross_subject_split",
                                 "cross_scene_split", "manual_split"],
                        help="覆盖 config 中的 split_to_use")
    parser.add_argument("--protocol", type=str, default=None,
                        choices=["protocol1", "protocol2", "protocol3"],
                        help="覆盖 config 中的 protocol")
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--output_dir", type=str, default=None, metavar="RUN_DIR",
        help="新实验的结果目录；目录非空时由训练引擎拒绝覆盖")
    output_group.add_argument(
        "--resume", type=str, default=None, metavar="RUN_DIR",
        help="从指定结果目录恢复实验")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="不给时用 config.num_workers (默认 8); 显式传入可覆盖")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--val_every", type=int, default=None,
                        help="不给时用 config.val_every (默认 2); 显式传入可覆盖")
    parser.add_argument("--epochs", type=int, default=None,
                        help="覆盖 num_epochs (冒烟测试用)")
    parser.add_argument("--max_train_batches", type=int, default=None,
                        help="每 epoch 最多训练 N 个 batch (冒烟测试用)")
    parser.add_argument("--max_val_batches", type=int, default=None,
                        help="每次验证最多评估 N 个 batch (冒烟测试用)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}", flush=True)
    setup_cuda(device)

    with open(args.config_file, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.split:
        config['split_to_use'] = args.split
    if args.protocol:
        config['protocol'] = args.protocol
    if args.epochs is not None:
        config['num_epochs'] = args.epochs

    # num_workers / val_every: CLI 显式传入优先, 否则回落 config, 再否则旧默认。
    num_workers = args.num_workers if args.num_workers is not None else config.get('num_workers', 8)
    val_every = args.val_every if args.val_every is not None else config.get('val_every', 2)

    is_resume = args.resume is not None
    if is_resume:
        run_dir = os.path.abspath(os.path.expanduser(args.resume))
    elif args.output_dir:
        run_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = os.path.join(
            _BASE, "result", "runs", config['protocol'],
            config['split_to_use'], timestamp)

    summary = train_one_experiment(
        args.dataset_root, config, run_dir, device,
        num_workers=num_workers, use_amp=args.amp,
        val_every=val_every,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        resume=is_resume)

    print(f"\n{'=' * 60}", flush=True)
    print(f"实验完成: {config['protocol']} / {config['split_to_use']}", flush=True)
    print(f"  Best  (epoch {summary['best_epoch']}): MPJPE {summary['best_mpjpe_mm']:.1f}mm "
          f"PA-MPJPE {summary['best_pampjpe_mm']:.1f}mm", flush=True)
    print(f"  Test:  MPJPE {summary['test_mpjpe_mm']:.1f}mm "
          f"PA-MPJPE {summary['test_pampjpe_mm']:.1f}mm", flush=True)
    print(f"  结果目录: {run_dir}", flush=True)
    print(f"{'=' * 60}", flush=True)


if __name__ == '__main__':
    main()
