"""Immutable sequence-level manifests and fail-closed leakage audits.

A manifest deliberately stores logical MM-Fi sequence identities rather than
machine-specific filesystem paths.  This makes its SHA-256 fingerprint stable
across Windows and remote Linux training environments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Collection, Literal, Mapping

from .data import decode_config, scene_for_subject
from .sequence_keys import SequenceKey, partition_train_select

DataScope = Literal["strict", "transductive"]

_SCHEMA_VERSION = 1
_VALID_SCOPES = frozenset({"strict", "transductive"})

# MM-Fi's official S2 cross-subject protocol.  This audit contract must remain
# independent of a possibly malformed or incomplete manifest.test_keys set.
OFFICIAL_S2_HELD_OUT_SUBJECTS = frozenset({
    "S05", "S10", "S15", "S20", "S25", "S30", "S35", "S40",
})


def _normalise_keys(
    keys: Collection[SequenceKey],
    *,
    field_name: str,
) -> frozenset[SequenceKey]:
    """Return immutable keys while rejecting non-sequence identities early."""

    result = frozenset(keys)
    invalid = [key for key in result if not isinstance(key, SequenceKey)]
    if invalid:
        raise TypeError(f"{field_name} 只能包含 SequenceKey")
    return result


def _serialise_keys(keys: Collection[SequenceKey]) -> list[list[str]]:
    """Produce order-independent canonical JSON values for sequence keys."""

    return [[key.scene, key.subject, key.action] for key in sorted(keys)]


def _deserialise_keys(value: Any, *, field_name: str) -> frozenset[SequenceKey]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} 必须是序列三元组列表")

    keys: list[SequenceKey] = []
    for triplet in value:
        if not (
            isinstance(triplet, list)
            and len(triplet) == 3
            and all(isinstance(part, str) for part in triplet)
        ):
            raise ValueError(f"{field_name} 包含非法序列三元组: {triplet!r}")
        keys.append(SequenceKey(*triplet))
    return frozenset(keys)


@dataclass(frozen=True)
class DataManifest:
    """Immutable sequence-level data boundary for one experiment identity."""

    protocol: str
    split: str
    seed: int
    scope: DataScope
    official_train_keys: Collection[SequenceKey]
    internal_train_keys: Collection[SequenceKey]
    pretrain_keys: Collection[SequenceKey]
    select_keys: Collection[SequenceKey]
    test_keys: Collection[SequenceKey]

    def __post_init__(self) -> None:
        if not self.protocol:
            raise ValueError("protocol 不能为空")
        if not self.split:
            raise ValueError("split 不能为空")
        if self.scope not in _VALID_SCOPES:
            raise ValueError(f"scope 必须是 strict 或 transductive，实际 {self.scope!r}")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed 必须是 int")

        for field_name in (
            "official_train_keys",
            "internal_train_keys",
            "pretrain_keys",
            "select_keys",
            "test_keys",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalise_keys(getattr(self, field_name), field_name=field_name),
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical, path-independent JSON payload."""

        return {
            "schema_version": _SCHEMA_VERSION,
            "protocol": self.protocol,
            "split": self.split,
            "seed": self.seed,
            "scope": self.scope,
            "official_train_keys": _serialise_keys(self.official_train_keys),
            "internal_train_keys": _serialise_keys(self.internal_train_keys),
            "pretrain_keys": _serialise_keys(self.pretrain_keys),
            "select_keys": _serialise_keys(self.select_keys),
            "test_keys": _serialise_keys(self.test_keys),
        }

    def fingerprint(self) -> str:
        """Return SHA-256 of canonical JSON, never depending on set iteration."""

        canonical = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def write_json(self, path: str | Path) -> None:
        """Write an inspectable manifest JSON document with its fingerprint."""

        destination = Path(path)
        payload = self.to_dict()
        payload["fingerprint"] = self.fingerprint()
        destination.write_text(
            json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def read_json(cls, path: str | Path) -> "DataManifest":
        """Load a manifest and reject stale or manually altered content."""

        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"manifest JSON 无法解析: {source}") from error
        if not isinstance(payload, Mapping):
            raise ValueError("manifest 根节点必须是对象")
        if payload.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError(f"不支持的 manifest schema_version: {payload.get('schema_version')!r}")

        required = (
            "protocol",
            "split",
            "seed",
            "scope",
            "official_train_keys",
            "internal_train_keys",
            "pretrain_keys",
            "select_keys",
            "test_keys",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f"manifest 缺少字段: {', '.join(missing)}")

        manifest = cls(
            protocol=payload["protocol"],
            split=payload["split"],
            seed=payload["seed"],
            scope=payload["scope"],
            official_train_keys=_deserialise_keys(
                payload["official_train_keys"], field_name="official_train_keys"
            ),
            internal_train_keys=_deserialise_keys(
                payload["internal_train_keys"], field_name="internal_train_keys"
            ),
            pretrain_keys=_deserialise_keys(payload["pretrain_keys"], field_name="pretrain_keys"),
            select_keys=_deserialise_keys(payload["select_keys"], field_name="select_keys"),
            test_keys=_deserialise_keys(payload["test_keys"], field_name="test_keys"),
        )
        declared = payload.get("fingerprint")
        if declared is not None and declared != manifest.fingerprint():
            raise ValueError("manifest fingerprint 不匹配")
        return manifest


@dataclass(frozen=True)
class LeakageAudit:
    """Complete audit result; callers must fail closed when ``passed`` is false."""

    passed: bool
    counts: dict[str, int] = field(default_factory=dict)
    violations: tuple[str, ...] = ()


def _overlap_count(left: Collection[SequenceKey], right: Collection[SequenceKey]) -> int:
    return len(frozenset(left) & frozenset(right))


def audit_manifest(
    manifest: DataManifest,
    labeled_keys: Collection[SequenceKey] = (),
) -> LeakageAudit:
    """Audit all sequence-level boundaries and return every detected violation.

    This function only reports failures.  Training entry points are expected to
    raise on ``not audit.passed`` before allocating a model or writing a
    checkpoint.
    """

    labeled = _normalise_keys(labeled_keys, field_name="labeled_keys")
    official = manifest.official_train_keys
    internal = manifest.internal_train_keys
    pretrain = manifest.pretrain_keys
    select = manifest.select_keys
    test = manifest.test_keys
    violations: list[str] = []

    def violation(condition: bool, text: str) -> None:
        if condition:
            violations.append(text)

    counts = {
        "official_train_sequences": len(official),
        "internal_train_sequences": len(internal),
        "pretrain_sequences": len(pretrain),
        "select_sequences": len(select),
        "test_sequences": len(test),
        "labeled_sequences": len(labeled),
        "overlap_official_train_test": _overlap_count(official, test),
        "overlap_internal_train_select": _overlap_count(internal, select),
        "overlap_internal_train_test": _overlap_count(internal, test),
        "overlap_select_test": _overlap_count(select, test),
        "overlap_pretrain_select": _overlap_count(pretrain, select),
        "overlap_pretrain_test": _overlap_count(pretrain, test),
        "overlap_labeled_select": _overlap_count(labeled, select),
        "overlap_labeled_test": _overlap_count(labeled, test),
        "labeled_outside_internal_train": len(labeled - internal),
        "pretrain_outside_internal_train": len(pretrain - internal),
        "internal_outside_official_train": len(internal - official),
        "select_outside_official_train": len(select - official),
        "s2_test_subject_contamination": 0,
        "s3_e04_contamination": 0,
    }

    violation(not official, "official_train_keys 不能为空")
    violation(not internal, "internal_train_keys 不能为空")
    violation(not select, "select_keys 不能为空")
    violation(not test, "test_keys 不能为空")
    violation(counts["overlap_official_train_test"] > 0, "official_train_keys 与 test_keys 重叠")
    violation(counts["overlap_internal_train_select"] > 0, "internal_train_keys 与 select_keys 重叠")
    violation(counts["overlap_internal_train_test"] > 0, "internal_train_keys 与 test_keys 重叠")
    violation(counts["overlap_select_test"] > 0, "select_keys 与 test_keys 重叠")
    violation(counts["internal_outside_official_train"] > 0, "internal_train_keys 不属于 official_train_keys")
    violation(counts["select_outside_official_train"] > 0, "select_keys 不属于 official_train_keys")
    violation((internal | select) != official, "official_train_keys 必须等于 internal_train_keys 与 select_keys 的并集")
    violation(counts["overlap_pretrain_select"] > 0, "pretrain_keys 与 select_keys 重叠")
    violation(counts["overlap_labeled_select"] > 0, "labeled_keys 与 select_keys 重叠")
    violation(counts["overlap_labeled_test"] > 0, "labeled_keys 与 test_keys 重叠")
    violation(counts["labeled_outside_internal_train"] > 0, "labeled_keys 不属于 internal_train_keys")

    if manifest.scope == "strict":
        # strict 允许使用 internal_train 的固定子集，便于可复现的小样本预训练；
        # 关键边界仍是不读取 select 或 test。
        violation(not pretrain, "strict 模式的 pretrain_keys 不能为空")
        violation(counts["pretrain_outside_internal_train"] > 0, "strict 模式的 pretrain_keys 必须属于 internal_train_keys")
        violation(counts["overlap_pretrain_test"] > 0, "strict 模式的 pretrain_keys 与 test_keys 重叠")
    else:
        expected_pretrain = internal | test
        violation(
            pretrain != expected_pretrain,
            "transductive 模式的 pretrain_keys 必须等于 internal_train_keys 与 test_keys 的并集",
        )

    if manifest.split == "cross_subject_split":
        held_out_subjects = OFFICIAL_S2_HELD_OUT_SUBJECTS
        training_side = official | internal | select
        if manifest.scope == "strict":
            training_side |= pretrain
        counts["s2_test_subject_contamination"] = len(
            {key for key in training_side if key.subject in held_out_subjects}
        )
        violation(
            counts["s2_test_subject_contamination"] > 0,
            "cross_subject_split 的训练侧包含测试被试",
        )

    if manifest.split == "cross_scene_split":
        training_side = official | internal | select
        if manifest.scope == "strict":
            training_side |= pretrain
        counts["s3_e04_contamination"] = len(
            {key for key in training_side if key.scene == "E04"}
        )
        violation(
            counts["s3_e04_contamination"] > 0,
            "cross_scene_split 的训练侧包含 E04",
        )

    return LeakageAudit(
        passed=not violations,
        counts=counts,
        violations=tuple(violations),
    )


def _keys_from_decoded_data_form(data_form: Mapping[str, Collection[str]]) -> frozenset[SequenceKey]:
    """Build logical sequence identities from ``decode_config`` output.

    ``decode_config`` is the single authority for the official protocol and
    split semantics, including S1's action-specific subject permutations.
    """

    return frozenset(
        SequenceKey(scene_for_subject(subject), subject, action)
        for subject, actions in data_form.items()
        for action in actions
    )


def build_data_manifest(
    dataset_root: str,
    config: Mapping[str, Any],
    protocol: str,
    split: str,
    seed: int,
    scope: DataScope,
) -> DataManifest:
    """Build one immutable data boundary from the official split config.

    The manifest stores logical sequence keys rather than filesystem paths, so
    the same boundary works on local Windows and remote Linux dataset roots.
    ``dataset_root`` is intentionally accepted for the public experiment API;
    construction uses the official config and does not scan or serialize paths.
    """

    if not isinstance(dataset_root, str):
        raise TypeError("dataset_root 必须是 str")
    if not isinstance(protocol, str) or not protocol:
        raise ValueError("protocol 必须是非空字符串")
    if not isinstance(split, str) or not split:
        raise ValueError("split 必须是非空字符串")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed 必须是 int")

    resolved_config = dict(config)
    resolved_config["protocol"] = protocol
    resolved_config["split_to_use"] = split
    decoded = decode_config(resolved_config)
    official_train = _keys_from_decoded_data_form(
        decoded["train_dataset"]["data_form"]
    )
    test = _keys_from_decoded_data_form(decoded["val_dataset"]["data_form"])
    internal_train, select = partition_train_select(
        official_train,
        float(resolved_config.get("val_fraction", 0.1)),
        seed,
    )
    pretrain = internal_train if scope == "strict" else internal_train | test

    return DataManifest(
        protocol=protocol,
        split=split,
        seed=seed,
        scope=scope,
        official_train_keys=official_train,
        internal_train_keys=internal_train,
        pretrain_keys=pretrain,
        select_keys=select,
        test_keys=test,
    )
