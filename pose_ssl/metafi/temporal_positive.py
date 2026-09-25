"""Temporal-positive weighting for MetaFi SSL objectives."""

from __future__ import annotations

import math
from numbers import Real
from typing import Mapping


def temporal_weight(delta_t: int, weights: Mapping[int, float]) -> float:
    """Return the declared weak-positive weight for a signed frame offset.

    The anchor itself has weight ``1.0``.  Neighbor weights are indexed by the
    absolute frame distance; missing distances intentionally contribute zero.
    """

    if isinstance(delta_t, bool) or not isinstance(delta_t, int):
        raise TypeError("delta_t must be an integer frame offset")
    if not isinstance(weights, Mapping):
        raise TypeError("weights must map positive frame distances to weights")

    normalized: dict[int, float] = {}
    for distance, value in weights.items():
        if isinstance(distance, bool) or not isinstance(distance, int) or distance <= 0:
            raise ValueError("temporal positive distances must be positive integers")
        if not isinstance(value, Real) or not math.isfinite(value) or value < 0:
            raise ValueError("temporal positive weights must be finite and non-negative")
        normalized[distance] = float(value)

    distance = abs(delta_t)
    return 1.0 if distance == 0 else normalized.get(distance, 0.0)
