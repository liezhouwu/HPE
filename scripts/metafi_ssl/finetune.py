"""Fine-tune an audited SSL MetaFi encoder.

The epoch-aware strategy object is selected by the shared training engine;
this CLI forwards the configured strategy and optimizer settings to that engine.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from collections.abc import Sequence
import sys

_BASE = Path(__file__).resolve().parents[2]
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))

from pose_ssl.metafi.fine_tune_strategy import MATCHED_STRATEGY, TRANSFER_STRATEGY
from scripts.metafi_ssl.train_supervised import run_matched


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune an audited SSL MetaFi encoder")
    parser.add_argument("dataset_root")
    parser.add_argument("config_file")
    parser.add_argument("--encoder-checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--label-manifest", required=True)
    parser.add_argument("--leakage-audit", required=True)
    parser.add_argument("--axis-stats", required=True)
    parser.add_argument("--label-budget", required=True)
    parser.add_argument("--protocol", required=True, choices=("protocol1", "protocol2", "protocol3"))
    parser.add_argument("--split", required=True, choices=("random_split", "cross_subject_split", "cross_scene_split"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--strategy", required=True, choices=(MATCHED_STRATEGY, TRANSFER_STRATEGY))
    parser.add_argument("--optimizer", choices=("adamw", "sgd"), default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--sgd-momentum", type=float, default=None)
    parser.add_argument("--scheduler", choices=("cosine", "multistep", "constant"), default=None)
    parser.add_argument("--lr-warmup-epochs", type=int, default=None)
    parser.add_argument("--lr-min", type=float, default=None)
    parser.add_argument("--lr-milestones", type=int, nargs="*", default=None)
    parser.add_argument("--lr-gamma", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run_matched(args, encoder_checkpoint=args.encoder_checkpoint, method="ssl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
