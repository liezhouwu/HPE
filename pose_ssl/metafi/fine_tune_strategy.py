"""Deterministic MetaFi fine-tuning schedules.

These strategies intentionally own only parameter freezing, BatchNorm mode, and
optimizer-group construction.  An epoch-aware trainer applies the selected
strategy before each epoch; keeping that transition logic here makes the
matched causal control and the transfer ablation independently testable.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch.nn as nn

from mmfi_wifi.metafi_pose_model import MetaFiPoseModel


MATCHED_STRATEGY = "matched"
TRANSFER_STRATEGY = "transfer"
SUP_DIFFERENTIAL_LR_STRATEGY = "sup-differential-lr"
STRATEGY_NAMES = (
    MATCHED_STRATEGY,
    TRANSFER_STRATEGY,
    SUP_DIFFERENTIAL_LR_STRATEGY,
)


ParameterGroup = dict[str, object]


def set_batchnorm_mode(module: nn.Module, frozen: bool) -> None:
    """Freeze or restore BatchNorm running-stat updates under ``module``.

    ``requires_grad=False`` does not prevent BatchNorm buffers from changing in
    train mode.  A frozen encoder must therefore keep all its BatchNorm modules
    in evaluation mode.  When a partition becomes trainable again, its
    BatchNorm modules are returned to train mode; callers use these schedules
    only during fine-tuning, where the containing model is in train mode.
    """

    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            child.train(not frozen)


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = enabled


def _parameters(*modules: nn.Module) -> list[nn.Parameter]:
    """Return a deterministic, duplicate-free trainable parameter list."""

    result: list[nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            identifier = id(parameter)
            if identifier not in seen:
                seen.add(identifier)
                result.append(parameter)
    return result


def _group(name: str, learning_rate: float, modules: Iterable[nn.Module]) -> ParameterGroup:
    params = _parameters(*tuple(modules))
    if not params:
        raise ValueError(f"parameter group {name!r} is empty")
    return {"name": name, "params": params, "lr": learning_rate}


def _encoder_low_modules(model: MetaFiPoseModel) -> tuple[nn.Module, ...]:
    encoder = model.encoder
    return (
        encoder.encoder_conv1,
        encoder.encoder_bn1,
        encoder.encoder_layer1,
        encoder.encoder_layer2,
    )


def _encoder_layer3_modules(model: MetaFiPoseModel) -> tuple[nn.Module, ...]:
    return (model.encoder.encoder_layer3,)


def _encoder_high_modules(model: MetaFiPoseModel) -> tuple[nn.Module, ...]:
    encoder = model.encoder
    return (encoder.encoder_layer4,)


def _assert_complete_partition(model: MetaFiPoseModel, groups: list[ParameterGroup]) -> None:
    """Reject strategy bugs that omit or duplicate a trainable parameter."""

    grouped = [parameter for group in groups for parameter in group["params"]]  # type: ignore[index]
    grouped_ids = [id(parameter) for parameter in grouped]
    expected_ids = [id(parameter) for parameter in model.parameters() if parameter.requires_grad]
    if len(grouped_ids) != len(set(grouped_ids)) or set(grouped_ids) != set(expected_ids):
        raise RuntimeError("fine-tune strategy parameter groups are not a complete partition")


@dataclass(frozen=True)
class MatchedFineTuneStrategy:
    """The causal control: all parameters train at the original baseline LR."""

    baseline_lr: float = 3e-4
    name: str = MATCHED_STRATEGY

    def configure(self, model: MetaFiPoseModel, epoch: int) -> list[ParameterGroup]:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        _set_requires_grad(model, True)
        set_batchnorm_mode(model, frozen=False)
        groups = [_group("all", self.baseline_lr, (model,))]
        _assert_complete_partition(model, groups)
        return groups

    def apply(self, model: MetaFiPoseModel, epoch: int) -> list[ParameterGroup]:
        return self.configure(model, epoch)


@dataclass(frozen=True)
class TransferFineTuneStrategy:
    """Three-phase SSL transfer schedule with explicit encoder partitions."""

    decoder_warmup_lr: float = 3e-4
    decoder_lr: float = 1e-4
    encoder_layer12_lr: float = 5e-6
    encoder_layer3_lr: float = 1e-5
    encoder_high_lr: float = 3e-5
    name: str = TRANSFER_STRATEGY

    def configure(self, model: MetaFiPoseModel, epoch: int) -> list[ParameterGroup]:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")

        low = _encoder_low_modules(model)
        layer3 = _encoder_layer3_modules(model)
        high = _encoder_high_modules(model)

        _set_requires_grad(model.encoder, False)
        _set_requires_grad(model.decoder, True)
        set_batchnorm_mode(model.encoder, frozen=True)
        set_batchnorm_mode(model.decoder, frozen=False)

        if epoch <= 2:
            groups = [_group("decoder", self.decoder_warmup_lr, (model.decoder,))]
        elif epoch <= 7:
            _set_requires_grad(model.encoder.encoder_layer4, True)
            set_batchnorm_mode(model.encoder.encoder_layer4, frozen=False)
            groups = [
                _group("encoder_high", self.encoder_high_lr, high),
                _group("decoder", self.decoder_lr, (model.decoder,)),
            ]
        else:
            _set_requires_grad(model.encoder, True)
            set_batchnorm_mode(model.encoder, frozen=False)
            groups = [
                _group("encoder_layer12", self.encoder_layer12_lr, low),
                _group("encoder_layer3", self.encoder_layer3_lr, layer3),
                _group("encoder_high", self.encoder_high_lr, high),
                _group("decoder", self.decoder_lr, (model.decoder,)),
            ]

        _assert_complete_partition(model, groups)
        return groups

    def apply(self, model: MetaFiPoseModel, epoch: int) -> list[ParameterGroup]:
        return self.configure(model, epoch)


@dataclass(frozen=True)
class SupDifferentialLRStrategy:
    """Random-init control using transfer's final differential LR groups."""

    decoder_lr: float = 1e-4
    encoder_layer12_lr: float = 5e-6
    encoder_layer3_lr: float = 1e-5
    encoder_high_lr: float = 3e-5
    name: str = SUP_DIFFERENTIAL_LR_STRATEGY

    def configure(self, model: MetaFiPoseModel, epoch: int) -> list[ParameterGroup]:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        _set_requires_grad(model, True)
        set_batchnorm_mode(model, frozen=False)
        groups = [
            _group("encoder_layer12", self.encoder_layer12_lr, _encoder_low_modules(model)),
            _group("encoder_layer3", self.encoder_layer3_lr, _encoder_layer3_modules(model)),
            _group("encoder_high", self.encoder_high_lr, _encoder_high_modules(model)),
            _group("decoder", self.decoder_lr, (model.decoder,)),
        ]
        _assert_complete_partition(model, groups)
        return groups

    def apply(self, model: MetaFiPoseModel, epoch: int) -> list[ParameterGroup]:
        return self.configure(model, epoch)


def select_fine_tune_strategy(name: str, config: Mapping[str, object] | None = None):
    """Build a named strategy from optional YAML-provided parameters."""

    config = {} if config is None else config
    raw_params = config.get("strategy_params", {})
    if not isinstance(raw_params, Mapping):
        raise TypeError("strategy_params must be a mapping")
    params = raw_params

    def value(key: str, default: float) -> float:
        raw = params.get(key, config.get(key, default))
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TypeError(f"fine-tune strategy parameter {key!r} must be numeric")
        return float(raw)

    if name == MATCHED_STRATEGY:
        return MatchedFineTuneStrategy(
            baseline_lr=value("baseline_lr", float(config.get("learning_rate", 3e-4)))
        )
    if name == TRANSFER_STRATEGY:
        return TransferFineTuneStrategy(
            decoder_warmup_lr=value("transfer_decoder_warmup_lr", 3e-4),
            decoder_lr=value("transfer_decoder_lr", 1e-4),
            encoder_layer12_lr=value("transfer_encoder_layer12_lr", 5e-6),
            encoder_layer3_lr=value("transfer_encoder_layer3_lr", 1e-5),
            encoder_high_lr=value("transfer_encoder_high_lr", 3e-5),
        )
    if name == SUP_DIFFERENTIAL_LR_STRATEGY:
        return SupDifferentialLRStrategy(
            decoder_lr=value("sup_decoder_lr", 1e-4),
            encoder_layer12_lr=value("sup_encoder_layer12_lr", 5e-6),
            encoder_layer3_lr=value("sup_encoder_layer3_lr", 1e-5),
            encoder_high_lr=value("sup_encoder_high_lr", 3e-5),
        )
    raise ValueError(f"unsupported fine-tune strategy: {name!r}")
