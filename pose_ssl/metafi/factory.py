"""Explicit construction registry for MetaFi SSL methods.

The registry is fixed and local: a method must be registered explicitly rather
than discovered dynamically, so a typo cannot silently select another method.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from mmfi_wifi.metafi_encoder import MetaFiEncoder

from .base import MetaFiSSLMethod
from .mae import MetaFiMAE
from .mfm import MetaFiMFM
from .moco import MoCoMetaFi
from .relpos import RelPosMetaFi
from .simclr import SimCLRMetaFi
from .swav import SwAVMetaFi, SwAVViewPolicy


MetaFiSSLConstructor = Callable[[MetaFiEncoder, Mapping[str, Any]], MetaFiSSLMethod]


def _reject_unknown(config: Mapping[str, Any], allowed: set[str], method: str) -> None:
    unknown = set(config).difference(allowed)
    if unknown:
        raise ValueError(f"unknown {method} MetaFi config fields: {sorted(unknown)!r}")


def _build_simclr(encoder: MetaFiEncoder, config: Mapping[str, Any]) -> MetaFiSSLMethod:
    _reject_unknown(
        config,
        {
            "temperature", "temporal_weights", "augmentation", "augmentation_seed",
            "projector_hidden_dim", "projector_out_dim",
        },
        "SimCLR",
    )
    return SimCLRMetaFi(
        encoder,
        temperature=config.get("temperature", 0.1),
        temporal_weights=config.get("temporal_weights", {}),
        augmentation_config=config.get("augmentation", {}),
        augmentation_seed=config.get("augmentation_seed", 0),
        projector_hidden_dim=config.get("projector_hidden_dim", 512),
        projector_out_dim=config.get("projector_out_dim", 128),
    )


def _build_mae(encoder: MetaFiEncoder, config: Mapping[str, Any]) -> MetaFiSSLMethod:
    _reject_unknown(
        config,
        {
            "mask_ratio", "patch_height", "patch_width", "decoder_hidden_dim",
            "patch_norm", "use_mask_token", "augmentation_seed",
        },
        "MAE",
    )
    return MetaFiMAE(
        encoder,
        mask_ratio=config.get("mask_ratio", 0.75),
        patch_height=config.get("patch_height", 6),
        patch_width=config.get("patch_width", 5),
        decoder_hidden_dim=config.get("decoder_hidden_dim", 256),
        patch_norm=config.get("patch_norm", True),
        use_mask_token=config.get("use_mask_token", True),
        augmentation_seed=config.get("augmentation_seed", 0),
    )


def _build_mfm(encoder: MetaFiEncoder, config: Mapping[str, Any]) -> MetaFiSSLMethod:
    _reject_unknown(
        config,
        {
            "objective", "mask_ratio", "mask_modes", "augmentation_seed", "ema_momentum",
            "ema_schedule", "predictor_hidden_dim", "projector_hidden_dim",
            "masked_region_weight", "collapse_std_threshold", "collapse_window",
            "raw_decoder_hidden_dim",
        },
        "MFM",
    )
    objective = config.get("objective", "feature")
    mask_modes = config.get("mask_modes", ("joint",))
    if isinstance(mask_modes, str):
        mask_modes = (mask_modes,)
    if not isinstance(mask_modes, (tuple, list)):
        raise TypeError("MFM mask_modes must be a sequence of strings")
    return MetaFiMFM(
        encoder,
        objective=objective,
        mask_ratio=config.get("mask_ratio", 0.40),
        mask_modes=tuple(mask_modes),
        augmentation_seed=config.get("augmentation_seed", 0),
        ema_momentum=config.get("ema_momentum", 0.996),
        ema_schedule=config.get("ema_schedule", "constant"),
        predictor_hidden_dim=config.get("predictor_hidden_dim", 256),
        projector_hidden_dim=config.get("projector_hidden_dim"),
        masked_region_weight=config.get("masked_region_weight", 2.0),
        collapse_std_threshold=config.get("collapse_std_threshold", 1e-6),
        collapse_window=config.get("collapse_window", 3),
        raw_decoder_hidden_dim=config.get("raw_decoder_hidden_dim", 64),
    )

def _build_moco(encoder: MetaFiEncoder, config: Mapping[str, Any]) -> MetaFiSSLMethod:
    _reject_unknown(
        config,
        {
            "projector_dim", "projector_hidden_dim", "queue_size", "momentum", "temperature",
            "temporal_lambda", "temporal_weights", "augmentation", "augmentation_seed",
        },
        "MoCo",
    )
    return MoCoMetaFi(
        encoder,
        projector_dim=config.get("projector_dim", 128),
        projector_hidden_dim=config.get("projector_hidden_dim", 512),
        queue_size=config.get("queue_size", 65_536),
        momentum=config.get("momentum", 0.999),
        temperature=config.get("temperature", 0.2),
        temporal_lambda=config.get("temporal_lambda", 0.2),
        temporal_weights=config.get("temporal_weights", {}),
        augmentation_config=config.get("augmentation", {}),
        augmentation_seed=config.get("augmentation_seed", 0),
    )



def _build_relpos(encoder: MetaFiEncoder, config: Mapping[str, Any]) -> MetaFiSSLMethod:
    _reject_unknown(
        config,
        {"hidden_dim", "temporal_consistency", "temporal_coefficient", "augmentation_seed"},
        "RelPos",
    )
    return RelPosMetaFi(
        encoder,
        hidden_dim=config.get("hidden_dim", 256),
        temporal_consistency=config.get("temporal_consistency", False),
        temporal_coefficient=config.get("temporal_coefficient", 0.2),
        augmentation_seed=config.get("augmentation_seed", 0),
    )



def _build_swav(encoder: MetaFiEncoder, config: Mapping[str, Any]) -> MetaFiSSLMethod:
    _reject_unknown(
        config,
        {
            "view_policy", "local_mask_config", "augmentation", "augmentation_seed",
            "projector_hidden_dim", "projector_out_dim", "n_prototypes", "temperature",
            "sinkhorn_epsilon", "sinkhorn_iterations", "temporal_lambda", "temporal_weights",
        },
        "SwAV",
    )
    if "view_policy" not in config:
        raise ValueError("SwAV MetaFi config must explicitly provide view_policy; valid methods require an explicit resource policy")
    profile = config["view_policy"]
    local_mask_config = config.get("local_mask_config")
    if not isinstance(profile, str):
        raise TypeError("SwAV view_policy must be a string")
    if local_mask_config is not None and not isinstance(local_mask_config, Mapping):
        raise TypeError("SwAV local_mask_config must be a mapping")
    return SwAVMetaFi(
        encoder,
        view_policy=SwAVViewPolicy.from_resource_profile(
            profile, local_mask_config=local_mask_config
        ),
        augmentation_config=config.get("augmentation", {}),
        augmentation_seed=config.get("augmentation_seed", 0),
        projector_hidden_dim=config.get("projector_hidden_dim", 512),
        projector_out_dim=config.get("projector_out_dim", 128),
        n_prototypes=config.get("n_prototypes", 256),
        temperature=config.get("temperature", 0.1),
        sinkhorn_epsilon=config.get("sinkhorn_epsilon", 0.05),
        sinkhorn_iterations=config.get("sinkhorn_iterations", 3),
        temporal_lambda=config.get("temporal_lambda", 0.2),
        temporal_weights=config.get("temporal_weights", {}),
    )


_METHOD_CONSTRUCTORS: dict[str, MetaFiSSLConstructor] = {
    "mae": _build_mae,
    "mfm": _build_mfm,
    "moco": _build_moco,
    "relpos": _build_relpos,
    "simclr": _build_simclr,
    "swav": _build_swav,
}


def build_metafi_ssl_method(
    name: str,
    encoder: MetaFiEncoder,
    config: Mapping[str, Any],
) -> MetaFiSSLMethod:
    """Build one explicitly registered MetaFi SSL method."""

    if not isinstance(name, str):
        raise ValueError("SSL method name must be a string")
    if not isinstance(encoder, MetaFiEncoder):
        raise TypeError("encoder must be a MetaFiEncoder")
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    constructor = _METHOD_CONSTRUCTORS.get(name)
    if constructor is None:
        valid_methods = ", ".join(sorted(_METHOD_CONSTRUCTORS)) or "(none registered)"
        raise ValueError(f"unknown MetaFi SSL method {name!r}; valid methods: {valid_methods}")
    return constructor(encoder, config)
