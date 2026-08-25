"""Weighted mean of the observations, renormalised."""

from __future__ import annotations

import numpy as np

from shipvision.errors import ConfigurationError
from shipvision.registry import PYTHON
from shipvision.reid.aggregation.base import AGGREGATORS, FeatureAggregator
from shipvision.reid.distance import normalize

__all__ = ["MeanAggregator"]


@AGGREGATORS.register("mean", backend=PYTHON, aliases=("average",))
class MeanAggregator(FeatureAggregator):
    """The baseline: average the vectors, then renormalise.

    Renormalising is not cosmetic. The mean of unit vectors has length equal to how much
    they agree — three near-identical views average to length ~1, three scattered ones to
    ~0.4 — so without it the "distance" to a gallery entry would encode how consistent that
    entry's crops were rather than how much the query resembles it. Two identities would be
    ranked by their own internal agreement.

    Online :meth:`update` is a *running* mean, kept exactly by carrying the accumulated
    weight rather than by re-averaging normalised partial results. Averaging the renormalised
    running vector with each new observation is the tempting one-liner and it is wrong: it
    weights the most recent observation as heavily as the entire history, which makes this
    an EMA with an accidental coefficient. Use :class:`EmaAggregator` if that is what you
    want, deliberately.
    """

    def __init__(self) -> None:
        self._weight = 0.0
        self._sum: np.ndarray | None = None

    def aggregate(
        self, vectors: np.ndarray, *, weights: np.ndarray | None = None
    ) -> np.ndarray:
        rows = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
        if rows.shape[0] == 0:
            raise ConfigurationError("cannot aggregate zero vectors")
        if weights is None:
            summed = rows.sum(axis=0)
        else:
            w = np.asarray(weights, dtype=np.float32).reshape(-1)
            if w.shape[0] != rows.shape[0]:
                raise ConfigurationError(f"{w.shape[0]} weights for {rows.shape[0]} vectors")
            if np.any(w < 0):
                raise ConfigurationError("weights must be non-negative")
            if not np.any(w > 0):
                raise ConfigurationError("at least one weight must be positive")
            summed = w @ rows
        return normalize(summed, copy=False)

    def update(
        self, current: np.ndarray | None, observation: np.ndarray, *, weight: float = 1.0
    ) -> np.ndarray:
        if weight < 0:
            raise ConfigurationError(f"weight must be non-negative, got {weight}")
        row = np.asarray(observation, dtype=np.float32).reshape(-1)
        if current is None or self._sum is None or self._sum.shape != row.shape:
            self._sum = row * weight
            self._weight = weight
        else:
            self._sum = self._sum + row * weight
            self._weight += weight
        if self._weight <= 0.0:
            # Every observation so far had zero weight; there is nothing to represent yet,
            # so hand back the observation rather than a zero vector that matches nothing.
            return normalize(row)
        return normalize(self._sum / self._weight)
