"""Shared contracts for MetaFi-R34 self-supervised learning methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from .pretrain_data import PretrainBatch


@dataclass(frozen=True)
class SSLStepOutput:
    """One self-supervised optimization step and scalar logging metrics."""

    loss: Tensor
    metrics: dict[str, float]


class MetaFiSSLMethod(nn.Module, ABC):
    """Base contract shared by all MetaFi self-supervised methods.

    Concrete methods own a :class:`mmfi_wifi.metafi_encoder.MetaFiEncoder` and
    any pretraining-only modules.  They export only the encoder state because
    projectors, predictors, queues, and prototypes are not used downstream by
    pose fine-tuning.
    """

    @abstractmethod
    def forward(self, batch: PretrainBatch) -> SSLStepOutput:
        """Compute one SSL loss from a CSI-only pretraining batch."""

    @abstractmethod
    def export_encoder_state_dict(self) -> dict[str, Tensor]:
        """Return the downstream-exportable MetaFiEncoder state dictionary."""
