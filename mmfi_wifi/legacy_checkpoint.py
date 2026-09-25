"""Explicit compatibility loader for legacy flat MetaFi checkpoints."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn


LEGACY_PREFIX_MAP: dict[str, str] = {
    "encoder_conv1_p1.": "encoder.encoder_conv1_p1.",
    "encoder_bn1_p1.": "encoder.encoder_bn1_p1.",
    "encoder_layer1_p1.": "encoder.encoder_layer1_p1.",
    "encoder_layer2_p1.": "encoder.encoder_layer2_p1.",
    "encoder_layer3_p1.": "encoder.encoder_layer3_p1.",
    "encoder_layer4_p1.": "encoder.encoder_layer4_p1.",
    "tf.": "encoder.tf.",
    "bn2.": "encoder.bn2.",
    "decode.": "decoder.decode.",
    "bn1.": "decoder.bn1.",
}

LEGACY_EXACT_MAP: dict[str, str] = {
    "out_scale": "decoder.out_scale",
    "out_shift": "decoder.out_shift",
}


def strip_orig_mod_prefix(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Return a copy with optional torch.compile ``_orig_mod.`` prefixes removed."""
    return {
        key[len("_orig_mod.") :] if key.startswith("_orig_mod.") else key: value
        for key, value in state.items()
    }


def is_legacy_metafi_state_dict(state: Mapping[str, Tensor]) -> bool:
    """Whether normalized keys use the pre-split flat MetaFi layout."""
    keys = strip_orig_mod_prefix(state)
    return any(
        key in LEGACY_EXACT_MAP
        or any(key.startswith(prefix) for prefix in LEGACY_PREFIX_MAP)
        for key in keys
    )


def map_legacy_to_composed(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Map every legacy MetaFi key to the nested encoder/decoder layout.

    Unknown keys fail closed rather than being silently dropped.
    """
    mapped: dict[str, Tensor] = {}
    for key, value in strip_orig_mod_prefix(state).items():
        if key in LEGACY_EXACT_MAP:
            target_key = LEGACY_EXACT_MAP[key]
        else:
            target_key = ""
            for legacy_prefix, composed_prefix in LEGACY_PREFIX_MAP.items():
                if key.startswith(legacy_prefix):
                    target_key = composed_prefix + key[len(legacy_prefix) :]
                    break
            if not target_key:
                raise KeyError(f"未知 legacy MetaFi checkpoint 键: {key}")
        if target_key in mapped:
            raise KeyError(f"legacy MetaFi checkpoint 映射重复目标键: {target_key}")
        mapped[target_key] = value
    return mapped


def load_metafi_state_dict(
    model: nn.Module,
    state: Mapping[str, Tensor],
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    """Load either a legacy flat or current nested MetaFi state dictionary.

    The two optional output-affine parameters may be absent in checkpoints
    created before they were added. All other differences are rejected when
    ``strict`` is true.
    """
    normalized = strip_orig_mod_prefix(state)
    prepared = map_legacy_to_composed(normalized) if is_legacy_metafi_state_dict(normalized) else normalized
    missing, unexpected = model.load_state_dict(prepared, strict=False)
    allowed_missing = {"decoder.out_scale", "decoder.out_shift"}
    disallowed_missing = set(missing) - allowed_missing
    if strict and (unexpected or disallowed_missing):
        raise RuntimeError(
            "MetaFi checkpoint 与模型结构不匹配: "
            f"missing={sorted(disallowed_missing)} unexpected={sorted(unexpected)}"
        )
    return list(missing), list(unexpected)
