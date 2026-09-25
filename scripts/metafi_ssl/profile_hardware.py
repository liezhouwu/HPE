"""Pre-run CUDA capacity profiling for the MetaFi SSL runners.

This module owns only the bounded pre-run probe.  The pretraining runner
invokes it before a formal epoch starts, and the selected values are recorded
with the hardware/configuration identity so a later run can reject stale
profiles instead of silently changing execution identity.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


RESULT_ROOT_NAME = "result_metafi_ssl"
HARDWARE_PROFILE_DIR = "hardware_profiles"
PROFILE_SCHEMA_VERSION = 1
PROFILE_FIELDS = frozenset({
    "schema_version",
    "device_name",
    "total_memory_bytes",
    "method",
    "view_policy",
    "micro_batch",
    "gradient_accumulation",
    "peak_memory_bytes",
    "encoder_arch",
    "input_shape",
    "amp_dtype",
    "config_fingerprint",
})
DEFAULT_PROBE_STEPS = 3
_MIN_HEADROOM_NUMERATOR = 9
_MIN_HEADROOM_DENOMINATOR = 10
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY_PLACEHOLDERS = frozenset({"disabled", "n/a", "na", "none", "null", "placeholder", "unknown"})


def _require_identity_string(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise HardwareProfileError(f"{field_name} must be a non-blank string")
    if value.strip().casefold() in _IDENTITY_PLACEHOLDERS:
        raise HardwareProfileError(f"{field_name} must identify a configured value")


class HardwareProfileError(ValueError):
    """Raised when profiling or profile persistence cannot proceed safely."""


class HardwareProfileIdentityError(HardwareProfileError):
    """Raised when a saved profile does not match the requested run identity."""


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    """A reusable, identity-bound result of a pre-run memory probe."""

    device_name: str
    total_memory_bytes: int
    method: str
    view_policy: str
    micro_batch: int
    gradient_accumulation: int
    peak_memory_bytes: int
    encoder_arch: str
    input_shape: tuple[int, ...]
    amp_dtype: str
    config_fingerprint: str

    def __post_init__(self) -> None:
        for field_name in ("device_name", "method", "view_policy", "encoder_arch", "amp_dtype"):
            _require_identity_string(field_name, getattr(self, field_name))
        for field_name in (
            "total_memory_bytes",
            "micro_batch",
            "gradient_accumulation",
            "peak_memory_bytes",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise HardwareProfileError(f"{field_name} must be a non-negative integer")
        if self.micro_batch < 1 or self.gradient_accumulation < 1:
            raise HardwareProfileError("micro_batch and gradient_accumulation must be positive")
        if not isinstance(self.input_shape, tuple) or not self.input_shape or any(
            type(value) is not int or value < 1 for value in self.input_shape
        ):
            raise HardwareProfileError("input_shape must be a non-empty tuple of positive integers")
        if not _SHA256_RE.fullmatch(self.config_fingerprint):
            raise HardwareProfileError("config_fingerprint must be a lowercase SHA-256")

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON payload for this profile."""

        return {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "device_name": self.device_name,
            "total_memory_bytes": self.total_memory_bytes,
            "method": self.method,
            "view_policy": self.view_policy,
            "micro_batch": self.micro_batch,
            "gradient_accumulation": self.gradient_accumulation,
            "peak_memory_bytes": self.peak_memory_bytes,
            "encoder_arch": self.encoder_arch,
            "input_shape": list(self.input_shape),
            "amp_dtype": self.amp_dtype,
            "config_fingerprint": self.config_fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HardwareProfile":
        if not isinstance(payload, Mapping):
            raise HardwareProfileError("hardware profile must be a JSON object")
        if set(payload) != PROFILE_FIELDS:
            missing = sorted(PROFILE_FIELDS.difference(payload), key=repr)
            extra = sorted(set(payload).difference(PROFILE_FIELDS), key=repr)
            raise HardwareProfileError(
                f"hardware profile schema keys mismatch; missing={missing}, extra={extra}"
            )
        if type(payload["schema_version"]) is not int or payload["schema_version"] != PROFILE_SCHEMA_VERSION:
            raise HardwareProfileError("unsupported hardware profile schema_version")
        try:
            input_shape = payload["input_shape"]
            if not isinstance(input_shape, list):
                raise TypeError("input_shape must be a JSON list")
            return cls(
                device_name=payload["device_name"],
                total_memory_bytes=payload["total_memory_bytes"],
                method=payload["method"],
                view_policy=payload["view_policy"],
                micro_batch=payload["micro_batch"],
                gradient_accumulation=payload["gradient_accumulation"],
                peak_memory_bytes=payload["peak_memory_bytes"],
                encoder_arch=payload["encoder_arch"],
                input_shape=tuple(input_shape),
                amp_dtype=payload["amp_dtype"],
                config_fingerprint=payload["config_fingerprint"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise HardwareProfileError("invalid hardware profile fields") from error


@dataclass(frozen=True, slots=True)
class _ProbeResult:
    success: bool
    peak_memory_bytes: int | None


def _normalise_probe_result(result: object) -> _ProbeResult:
    if isinstance(result, Mapping):
        success = result.get("success", True)
        if type(success) is not bool:
            raise HardwareProfileError("probe success must be bool")
        peak = result.get("peak_memory_bytes")
    elif isinstance(result, bool):
        success = result
        peak = None
    elif result is None:
        success = True
        peak = None
    else:
        success = getattr(result, "success", True)
        peak = getattr(result, "peak_memory_bytes", None)
        if type(success) is not bool:
            raise HardwareProfileError("probe success must be bool")

    if peak is not None and (type(peak) is not int or peak < 0):
        raise HardwareProfileError("probe peak_memory_bytes must be a non-negative integer")
    return _ProbeResult(success=success, peak_memory_bytes=peak)


def _default_clear_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError as error:
        raise HardwareProfileError("CUDA cache control is unavailable") from error
    except RuntimeError as error:
        raise HardwareProfileError("CUDA cache could not be cleared") from error


def _default_reset_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError as error:
        raise HardwareProfileError("CUDA peak statistics are unavailable") from error
    except RuntimeError as error:
        raise HardwareProfileError("CUDA peak statistics could not be reset") from error


def _default_read_peak() -> int:
    try:
        import torch

        if not torch.cuda.is_available():
            raise HardwareProfileError("CUDA memory statistics are unavailable")
        allocated = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        if type(allocated) is not int or type(reserved) is not int:
            raise HardwareProfileError("CUDA memory statistics are invalid")
        return max(allocated, reserved)
    except ImportError as error:
        raise HardwareProfileError("CUDA memory statistics are unavailable") from error
    except RuntimeError as error:
        raise HardwareProfileError("CUDA memory statistics could not be read") from error


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"


def _safe_profile_filename(filename: str) -> str:
    if not isinstance(filename, str) or not filename:
        raise HardwareProfileError("profile filename must be non-empty")
    path = Path(filename)
    if path.name != filename or path.suffix.casefold() != ".json" or ".." in path.parts:
        raise HardwareProfileError("profile filename must be one safe .json path segment")
    return filename


def _profile_output_dir(repository_root: Path) -> Path:
    if not isinstance(repository_root, Path):
        raise TypeError("repository_root must be a pathlib.Path")
    if repository_root.name.casefold() in {"result", RESULT_ROOT_NAME.casefold()}:
        raise HardwareProfileError("repository_root must not be a result directory")
    output_dir = repository_root / RESULT_ROOT_NAME / HARDWARE_PROFILE_DIR
    try:
        from mmfi_wifi.run_identity import assert_new_result_root

        assert_new_result_root(output_dir)
    except (ImportError, OSError, TypeError, ValueError) as error:
        raise HardwareProfileError(str(error)) from error
    return output_dir


def _atomic_write_new(path: Path, text: str) -> None:
    """Publish one complete file without ever replacing an existing target."""

    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # A hard-link create is an exclusive, same-directory publication
            # primitive on Windows and POSIX: an existing target cannot win a
            # race by being silently replaced.
            os.link(temporary, path)
        except FileExistsError as error:
            raise HardwareProfileError(
                f"refusing to overwrite existing hardware profile: {path}"
            ) from error
    finally:
        if temporary.exists():
            temporary.unlink()


def write_hardware_profile(
    profile: HardwareProfile,
    repository_root: Path,
    *,
    filename: str | None = None,
) -> Path:
    """Write a new profile only below ``result_metafi_ssl/hardware_profiles``."""

    if not isinstance(profile, HardwareProfile):
        raise TypeError("profile must be a HardwareProfile")
    output_dir = _profile_output_dir(repository_root)
    if profile.total_memory_bytes < 1:
        raise HardwareProfileError("persisted hardware profiles require positive total memory")
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_filename = f"{profile.method}.json" if filename is None else filename
    target = output_dir / _safe_profile_filename(filename or f"{profile.method}.json")
    _atomic_write_new(target, _canonical_json(profile.to_dict()))
    return target


def load_hardware_profile(path: Path) -> HardwareProfile:
    """Load and validate one canonical hardware profile JSON file."""

    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HardwareProfileError(f"unable to read hardware profile: {path}") from error
    profile = HardwareProfile.from_dict(payload)
    if raw != _canonical_json(profile.to_dict()):
        raise HardwareProfileError("hardware profile JSON is not canonical")
    return profile


def _identity_fields(
    *,
    device_name: str,
    total_memory_bytes: int,
    encoder_arch: str,
    input_shape: Sequence[int],
    amp_dtype: str,
    method: str,
    view_policy: str,
    config_fingerprint: str,
) -> dict[str, object]:
    return {
        "device_name": device_name,
        "total_memory_bytes": total_memory_bytes,
        "encoder_arch": encoder_arch,
        "input_shape": tuple(input_shape),
        "amp_dtype": amp_dtype,
        "method": method,
        "view_policy": view_policy,
        "config_fingerprint": config_fingerprint,
    }


def validate_profile_identity(
    profile: HardwareProfile,
    *,
    device_name: str,
    total_memory_bytes: int,
    encoder_arch: str,
    input_shape: Sequence[int],
    amp_dtype: str,
    method: str,
    view_policy: str,
    config_fingerprint: str,
) -> bool:
    """Require all hardware and method/config identity fields to match."""

    if not isinstance(profile, HardwareProfile):
        raise TypeError("profile must be a HardwareProfile")
    expected = _identity_fields(
        device_name=device_name,
        total_memory_bytes=total_memory_bytes,
        encoder_arch=encoder_arch,
        input_shape=input_shape,
        amp_dtype=amp_dtype,
        method=method,
        view_policy=view_policy,
        config_fingerprint=config_fingerprint,
    )
    actual = _identity_fields(
        device_name=profile.device_name,
        total_memory_bytes=profile.total_memory_bytes,
        encoder_arch=profile.encoder_arch,
        input_shape=profile.input_shape,
        amp_dtype=profile.amp_dtype,
        method=profile.method,
        view_policy=profile.view_policy,
        config_fingerprint=profile.config_fingerprint,
    )
    for field_name, expected_value in expected.items():
        if actual[field_name] != expected_value:
            raise HardwareProfileIdentityError(
                f"hardware profile identity mismatch for {field_name}: "
                f"saved={actual[field_name]!r}, requested={expected_value!r}"
            )
    return True


def find_safe_micro_batch(
    candidates: Sequence[int],
    target_effective_batch: int,
    probe: Callable[..., object],
    *,
    device_name: str = "unknown",
    total_memory_bytes: int = 0,
    method: str = "unknown",
    view_policy: str = "unknown",
    encoder_arch: str = "unknown",
    input_shape: Sequence[int] = (),
    amp_dtype: str = "disabled",
    config_fingerprint: str = "",
    probe_steps: int = DEFAULT_PROBE_STEPS,
    clear_cache: Callable[[], None] | None = None,
    reset_peak: Callable[[], None] | None = None,
    read_peak: Callable[[], int] | None = None,
) -> HardwareProfile:
    """Select the largest safe candidate during the pre-run phase only.

    Candidates are tried in descending order.  A probe exception, explicit
    probe failure, missing peak under a memory-headroom check, or a peak over
    90% of total memory causes a downgrade.  If no candidate passes, this
    function raises instead of returning a speculative execution setting.
    """

    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise TypeError("candidates must be a sequence of positive integers")
    if type(target_effective_batch) is not int or target_effective_batch < 1:
        raise HardwareProfileError("target_effective_batch must be a positive integer")
    if not callable(probe):
        raise TypeError("probe must be callable")
    if type(probe_steps) is not int or not 1 <= probe_steps <= DEFAULT_PROBE_STEPS:
        raise HardwareProfileError("probe_steps must be between 1 and 3")
    if type(total_memory_bytes) is not int or total_memory_bytes < 0:
        raise HardwareProfileError("total_memory_bytes must be a non-negative integer")

    valid_candidates: set[int] = set()
    for candidate in candidates:
        if type(candidate) is int and candidate > 0 and target_effective_batch % candidate == 0:
            valid_candidates.add(candidate)
    ordered = sorted(valid_candidates, reverse=True)
    if not ordered:
        raise HardwareProfileError("no safe micro-batch: no candidate divides target effective batch")

    clear = clear_cache or _default_clear_cache
    reset = reset_peak or _default_reset_peak
    read = read_peak or _default_read_peak
    for candidate in ordered:
        clear()
        reset()
        try:
            result = _normalise_probe_result(probe(candidate, steps=probe_steps))
            if not result.success:
                continue
            if read_peak is not None:
                peak = read()
            elif result.peak_memory_bytes is not None:
                # An explicit probe peak is a trusted custom reader result and
                # remains usable in synthetic environments without CUDA hooks.
                peak = result.peak_memory_bytes
            else:
                peak = read()
            if peak is None:
                if total_memory_bytes:
                    continue
                peak = 0
            if type(peak) is not int or peak < 0:
                continue
            if total_memory_bytes and peak * _MIN_HEADROOM_DENOMINATOR > total_memory_bytes * _MIN_HEADROOM_NUMERATOR:
                continue
            return HardwareProfile(
                device_name=device_name,
                total_memory_bytes=total_memory_bytes,
                method=method,
                view_policy=view_policy,
                micro_batch=candidate,
                gradient_accumulation=target_effective_batch // candidate,
                peak_memory_bytes=peak,
                encoder_arch=encoder_arch,
                input_shape=tuple(input_shape),
                amp_dtype=amp_dtype,
                config_fingerprint=config_fingerprint,
            )
        except Exception:
            # OOM and all other probe failures are conservative downgrade signals.
            continue
        finally:
            clear()

    raise HardwareProfileError("no safe micro-batch: all candidates failed the pre-run probe")


__all__ = [
    "HardwareProfile",
    "HardwareProfileError",
    "HardwareProfileIdentityError",
    "find_safe_micro_batch",
    "load_hardware_profile",
    "validate_profile_identity",
    "write_hardware_profile",
]
