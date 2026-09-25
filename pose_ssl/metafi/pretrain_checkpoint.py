"""Versioned, identity-bound checkpoints for MetaFi SSL pretraining.

The checkpoint intentionally contains both the complete SSL method state and an
encoder-only export. The former is used for exact epoch-boundary resume; the
latter is consumed by the established fine-tuning admission contract.  Runner
execution parameters are stored alongside the state and are also included in
the run identity fingerprint.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn

from mmfi_wifi.run_identity import RunIdentity


CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_KIND = "metafi_ssl_encoder"
STATUS_IN_PROGRESS = "in_progress"
STATUS_COMPLETE = "complete"
_VALID_STATUSES = frozenset({STATUS_IN_PROGRESS, STATUS_COMPLETE})
_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "checkpoint_kind",
        "status",
        "encoder_arch",
        "encoder_state_dict",
        "method_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "python_rng_state",
        "numpy_rng_state",
        "torch_rng_state",
        "cuda_rng_states",
        "train_generator_state",
        "identity",
        "identity_json",
        "next_epoch",
    }
)


def _clone_state(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: _clone_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state(item) for item in value)
    return copy.deepcopy(value)


def _numpy_rng_to_payload(state: tuple[Any, ...]) -> dict[str, Any]:
    if len(state) != 5:
        raise ValueError("NumPy RNG state must have five fields")
    bit_generator, keys, position, has_gauss, cached_gaussian = state
    keys_array = np.asarray(keys)
    if keys_array.ndim != 1 or not np.issubdtype(keys_array.dtype, np.integer):
        raise ValueError("NumPy RNG state keys must be a one-dimensional integer array")
    return {
        "bit_generator": str(bit_generator),
        "keys": [int(value) for value in keys_array.tolist()],
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _numpy_rng_from_payload(payload: object) -> tuple[Any, ...]:
    if not isinstance(payload, Mapping):
        raise ValueError("numpy_rng_state must be a mapping")
    required = {"bit_generator", "keys", "position", "has_gauss", "cached_gaussian"}
    if set(payload) != required:
        raise ValueError("numpy_rng_state fields are invalid")
    bit_generator = payload["bit_generator"]
    keys = payload["keys"]
    if not isinstance(bit_generator, str) or not isinstance(keys, list):
        raise ValueError("numpy_rng_state has invalid fields")
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in keys):
        raise ValueError("numpy_rng_state keys must contain integers")
    position = payload["position"]
    has_gauss = payload["has_gauss"]
    cached = payload["cached_gaussian"]
    if (
        not isinstance(position, int)
        or isinstance(position, bool)
        or not isinstance(has_gauss, int)
        or isinstance(has_gauss, bool)
        or not isinstance(cached, (int, float))
        or not np.isfinite(float(cached))
    ):
        raise ValueError("numpy_rng_state has invalid scalar fields")
    return (
        bit_generator,
        np.asarray(keys, dtype=np.uint32),
        position,
        has_gauss,
        float(cached),
    )


def capture_rng_state(train_generator: torch.Generator) -> dict[str, Any]:
    """Capture portable Python/NumPy/Torch/DataLoader RNG state."""

    if not isinstance(train_generator, torch.Generator):
        raise TypeError("train_generator must be a torch.Generator")
    cuda_states = [state.detach().cpu().clone() for state in torch.cuda.get_rng_state_all()]
    return {
        "python_rng_state": _clone_state(random.getstate()),
        "numpy_rng_state": _numpy_rng_to_payload(np.random.get_state()),
        "torch_rng_state": torch.get_rng_state().detach().cpu().clone(),
        "cuda_rng_states": cuda_states,
        "train_generator_state": train_generator.get_state().detach().cpu().clone(),
    }


def restore_rng_state(payload: Mapping[str, Any], train_generator: torch.Generator) -> None:
    """Restore exactly the state emitted by :func:`capture_rng_state`."""

    if not isinstance(train_generator, torch.Generator):
        raise TypeError("train_generator must be a torch.Generator")
    required = {
        "python_rng_state",
        "numpy_rng_state",
        "torch_rng_state",
        "cuda_rng_states",
        "train_generator_state",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"checkpoint RNG state missing fields: {sorted(missing)}")
    try:
        raw_python = payload["python_rng_state"]
        if not isinstance(raw_python, (tuple, list)):
            raise ValueError("python_rng_state must be a tuple or list")
        random.setstate(tuple(raw_python))
        np.random.set_state(_numpy_rng_from_payload(payload["numpy_rng_state"]))
        torch_state = payload["torch_rng_state"]
        train_state = payload["train_generator_state"]
        if not isinstance(torch_state, Tensor) or torch_state.dtype != torch.uint8:
            raise ValueError("torch_rng_state must be a uint8 tensor")
        if not isinstance(train_state, Tensor) or train_state.dtype != torch.uint8:
            raise ValueError("train_generator_state must be a uint8 tensor")
        torch.set_rng_state(torch_state.detach().cpu())
        cuda_states = payload["cuda_rng_states"]
        if not isinstance(cuda_states, list) or not all(isinstance(state, Tensor) for state in cuda_states):
            raise ValueError("cuda_rng_states must be a list of tensors")
        if torch.cuda.is_available() and cuda_states:
            torch.cuda.set_rng_state_all([state.detach().cpu() for state in cuda_states])
        train_generator.set_state(train_state.detach().cpu())
    except (TypeError, ValueError, RuntimeError, OverflowError) as error:
        raise ValueError("invalid checkpoint RNG state") from error


def _require_state_mapping(name: str, value: object, *, nonempty: bool = True) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    if nonempty and not value:
        raise ValueError(f"{name} must be non-empty")
    return value


def validate_checkpoint(
    checkpoint: Mapping[str, Any],
    expected_identity: RunIdentity,
    *,
    expected_status: Literal["in_progress", "complete"] | None = None,
) -> None:
    """Fail-closed validation of a schema-v1 pretraining checkpoint."""

    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping")
    if not isinstance(expected_identity, RunIdentity):
        raise TypeError("expected_identity must be a RunIdentity")
    missing = sorted(_REQUIRED_KEYS.difference(checkpoint))
    if missing:
        raise ValueError(f"checkpoint missing required fields: {missing}")
    if checkpoint["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint schema_version mismatch")
    if checkpoint["checkpoint_kind"] != CHECKPOINT_KIND:
        raise ValueError("checkpoint checkpoint_kind mismatch")
    status = checkpoint["status"]
    if status not in _VALID_STATUSES:
        raise ValueError("checkpoint status must be in_progress or complete")
    if expected_status is not None and status != expected_status:
        raise ValueError(
            f"checkpoint status mismatch: expected {expected_status!r}, got {status!r}"
        )
    if checkpoint["encoder_arch"] != expected_identity.encoder_arch:
        raise ValueError("checkpoint encoder_arch mismatch")
    next_epoch = checkpoint["next_epoch"]
    if not isinstance(next_epoch, int) or isinstance(next_epoch, bool) or next_epoch < 0:
        raise ValueError("checkpoint next_epoch must be a non-negative integer")

    identity = RunIdentity.from_dict(checkpoint["identity"])
    if identity != expected_identity:
        raise ValueError("checkpoint RunIdentity mismatch")
    if checkpoint["identity_json"] != expected_identity.canonical_json():
        raise ValueError("checkpoint identity_json is not canonical or does not match")
    if checkpoint["encoder_arch"] != identity.encoder_arch:
        raise ValueError("checkpoint encoder_arch disagrees with identity")

    _require_state_mapping("encoder_state_dict", checkpoint["encoder_state_dict"])
    _require_state_mapping("method_state_dict", checkpoint["method_state_dict"])
    _require_state_mapping("optimizer_state_dict", checkpoint["optimizer_state_dict"], nonempty=False)
    _require_state_mapping("scheduler_state_dict", checkpoint["scheduler_state_dict"], nonempty=False)
    _require_state_mapping("scaler_state_dict", checkpoint["scaler_state_dict"], nonempty=False)
    if not isinstance(checkpoint["cuda_rng_states"], list):
        raise ValueError("cuda_rng_states must be a list")
    _numpy_rng_from_payload(checkpoint["numpy_rng_state"])
    if not isinstance(checkpoint["python_rng_state"], (tuple, list)):
        raise ValueError("python_rng_state must be a tuple or list")
    for field in ("torch_rng_state", "train_generator_state"):
        value = checkpoint[field]
        if not isinstance(value, Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
            raise ValueError(f"{field} must be a rank-1 uint8 tensor")
    for state in checkpoint["cuda_rng_states"]:
        if not isinstance(state, Tensor) or state.dtype != torch.uint8 or state.ndim != 1:
            raise ValueError("cuda_rng_states must contain rank-1 uint8 tensors")


def build_checkpoint_payload(
    *,
    identity: RunIdentity,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    next_epoch: int,
    status: Literal["in_progress", "complete"],
    train_generator: torch.Generator | None = None,
    execution_parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a self-contained schema-v1 checkpoint payload."""

    if not isinstance(identity, RunIdentity):
        raise TypeError("identity must be a RunIdentity")
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    if status not in _VALID_STATUSES:
        raise ValueError("status must be in_progress or complete")
    if not isinstance(next_epoch, int) or isinstance(next_epoch, bool) or next_epoch < 0:
        raise ValueError("next_epoch must be a non-negative integer")
    if train_generator is None:
        train_generator = torch.Generator(device="cpu").manual_seed(0)

    exporter = getattr(model, "export_encoder_state_dict", None)
    if not callable(exporter):
        raise TypeError("model must expose export_encoder_state_dict()")
    encoder_state = exporter()
    if not isinstance(encoder_state, Mapping) or not encoder_state:
        raise ValueError("model export_encoder_state_dict() must return a non-empty mapping")
    rng = capture_rng_state(train_generator)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_kind": CHECKPOINT_KIND,
        "status": status,
        "encoder_arch": identity.encoder_arch,
        "encoder_state_dict": _clone_state(encoder_state),
        "method_state_dict": _clone_state(model.state_dict()),
        "optimizer_state_dict": _clone_state(optimizer.state_dict()),
        "scheduler_state_dict": _clone_state(scheduler.state_dict()),
        "scaler_state_dict": _clone_state(scaler.state_dict()),
        **rng,
        "identity": identity.to_dict(),
        "identity_json": identity.canonical_json(),
        "next_epoch": next_epoch,
    }
    if execution_parameters is not None:
        if not isinstance(execution_parameters, Mapping) or not execution_parameters:
            raise ValueError("execution_parameters must be a non-empty mapping")
        payload["execution_parameters"] = _clone_state(execution_parameters)
    validate_checkpoint(payload, identity, expected_status=status)
    return payload


def save_checkpoint_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomically write a validated-in-memory checkpoint payload."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(
    path: str | Path,
    expected_identity: RunIdentity,
    *,
    expected_status: Literal["in_progress", "complete"] | None = None,
) -> dict[str, Any]:
    """Load and validate a schema-v1 checkpoint using the safe loader."""

    source = Path(path)
    try:
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"unable to load pretraining checkpoint: {source}") from error
    if not isinstance(checkpoint, Mapping):
        raise ValueError("pretraining checkpoint must be a mapping")
    validate_checkpoint(checkpoint, expected_identity, expected_status=expected_status)
    return dict(checkpoint)


def checkpoint_fingerprint(checkpoint: Mapping[str, Any]) -> str:
    """Return a stable fingerprint of identity metadata, not tensor bytes."""

    identity_json = checkpoint.get("identity_json")
    if not isinstance(identity_json, str):
        raise ValueError("checkpoint identity_json is required")
    return hashlib.sha256(identity_json.encode("utf-8")).hexdigest()
