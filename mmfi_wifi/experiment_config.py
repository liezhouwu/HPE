"""Immutable, self-contained configuration records for every training entry point."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

import torch


def file_sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_experiment_config(directory, *, stage, config, execution, context=None, artifacts=None):
    directory = Path(directory)
    target = directory / "experiment_config.json"
    # Resume retains the original invocation; engine/runner validate run identity.
    if target.is_file():
        return
    copied = {}
    for name, source in (artifacts or {}).items():
        source = Path(source)
        content = source.read_bytes()
        destination = directory / name
        if destination.exists() and destination.read_bytes() != content:
            raise ValueError(f"conflicting experiment artifact: {destination}")
        if not destination.exists():
            destination.write_bytes(content)
        copied[name] = {"source": str(source.resolve()), "sha256": hashlib.sha256(content).hexdigest()}
    versions = {}
    for package in ("torch", "torchvision", "numpy", "PyYAML"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    payload = {
        "schema_version": 1, "stage": stage,
        "config": config, "execution": execution, "context": context or {},
        "artifacts": copied,
        "environment": {"python": sys.version, "executable": sys.executable,
                        "platform": platform.platform(), "packages": versions,
                        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                        "cudnn_benchmark": torch.backends.cudnn.benchmark,
                        "cudnn_deterministic": torch.backends.cudnn.deterministic},
        "invocation": {"argv": sys.argv, "cwd": str(Path.cwd())},
    }
    temporary = target.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(target)
