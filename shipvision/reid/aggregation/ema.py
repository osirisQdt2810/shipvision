"""Exponential moving average — the online default."""

from __future__ import annotations

import numpy as np

from shipvision.errors import ConfigurationError
from shipvision.registry import PYTHON
from shipvision.reid.aggregation.base import AGGREGATORS, FeatureAggregator
from shipvision.reid.distance import normalize

__all__ = ["EmaAggregator"]


@AGGREGATORS.register("ema", backend=PYTHON, aliases=("exponential",))
class EmaAggregator(FeatureAggregator):
    """``f <- alpha * f + (1 - alpha) * observation``, renormalised.

    What a live track wants, and what a running mean is not. A ship tracked for twenty
    minutes accumulates thousands of crops; under a mean, the appearance from when it
    entered the frame outvotes everything since, so the gallery entry describes a vessel
    that is no longer what the camera sees. An EMA forgets at a fixed rate, so the entry
    tracks the current appearance while still being averaged enough to survive one bad crop.

    `alpha` is the memory. 0.9 is the usual starting point (roughly a ten-frame horizon);
    higher is steadier and slower to adapt, lower is the opposite. `alpha = 1` never
    updates, which is a configuration error rather than a valid choice — it silently turns
    the whole feature bank into "whatever the first crop looked like".

    O(1) in time and memory per update, which is what makes it affordable at 50 cameras
    times 20 fps times 15 objects: no history is kept, so nothing grows.
    """

    def __init__(self, *, alpha: float = 0.9) -> None:
        if not 0.0 <= alpha < 1.0:
            raise ConfigurationError(
                f"alpha must be in [0, 1) — 1.0 would never incorporate an observation; "
                f"got {alpha}"
            )
        self.alpha = float(alpha)

    def aggregate(
        self, vectors: np.ndarray, *, weights: np.ndarray | None = None
    ) -> np.ndarray:
        """Fold the rows in order, oldest first, so the last row weighs the most.

        Order matters here and does not for a mean. Rows are taken as chronological
        because that is the only order an EMA has an opinion about.
        """
        rows = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
        if rows.shape[0] == 0:
            raise ConfigurationError("cannot aggregate zero vectors")
        w = (
            np.ones(rows.shape[0], dtype=np.float32)
            if weights is None
            else np.asarray(weights, dtype=np.float32).reshape(-1)
        )
        if w.shape[0] != rows.shape[0]:
            raise ConfigurationError(f"{w.shape[0]} weights for {rows.shape[0]} vectors")

        current: np.ndarray | None = None
        for row, weight in zip(rows, w, strict=True):
            current = self.update(current, row, weight=float(weight))
        assert current is not None
        return current

    def update(
        self, current: np.ndarray | None, observation: np.ndarray, *, weight: float = 1.0
    ) -> np.ndarray:
        if weight < 0:
            raise ConfigurationError(f"weight must be non-negative, got {weight}")
        row = np.asarray(observation, dtype=np.float32).reshape(-1)
        if current is None:
            return normalize(row)

        previous = np.asarray(current, dtype=np.float32).reshape(-1)
        if previous.shape != row.shape:
            raise ConfigurationError(
                f"cannot update a {previous.shape[0]}-d vector with a {row.shape[0]}-d one"
            )
        # Quality scales how far the update moves, not the mixing coefficient itself: a
        # weight of 0 must leave the vector exactly where it was, and 1 must give the
        # configured alpha. Folding weight into alpha instead would let a low-quality crop
        # *increase* the step, which is precisely backwards.
        effective = 1.0 - (1.0 - self.alpha) * weight
        return normalize(effective * previous + (1.0 - effective) * row, copy=False)
