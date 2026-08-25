"""Cosine appearance similarity, hard-thresholded, with same-camera pairs excluded.

The baseline matcher, and the only one that works on an uncalibrated site. It answers "do
these two crops look like the same object" and nothing else, which makes it the right thing
to compose a geometric gate on top of rather than a competitor to it.

**The hard threshold is not a tuning nicety.** Everything below ``appearance_threshold`` is
set to exactly zero, which the base class then turns into "never merge". Without it,
average-linkage clustering is free to chain: A resembles B a little, B resembles C a little,
and a threshold on the *average* distance groups all three even though A and C are strangers.
Zeroing weak evidence means a chain has to be built out of links that each stand on their own.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from shipvision.errors import ConfigurationError
from shipvision.mtmc.base import BaseMatcher
from shipvision.mtmc.core.appearance.utils import stack_embeddings
from shipvision.mtmc.frames import TrackObservation
from shipvision.mtmc.registry import MTMC_MATCHERS
from shipvision.registry import PYTHON
from shipvision.reid.distance import cosine_similarity

__all__ = ["AppearanceMatcher"]


@MTMC_MATCHERS.register("appearance", backend=PYTHON, aliases=("aic",))
class AppearanceMatcher(BaseMatcher):
    """Cosine similarity between track embeddings, thresholded, camera-masked.

    Cosine only, with no metric switch. On L2-normalised vectors euclidean distance is a
    monotone function of cosine distance and therefore ranks identically — see
    :func:`shipvision.reid.distance.euclidean_distance` — so a metric option would change
    nothing except the scale the threshold is expressed in. The reference implementation had
    that switch and shipped two configurations whose thresholds (0.55 and 0.86) were not
    comparable, which is a way to misconfigure a system rather than a way to improve one.
    """

    def __init__(self, *, appearance_threshold: float = 0.86) -> None:
        """
        Args:
            appearance_threshold: minimum cosine similarity for a pair to be considered at
                all. The reference's production value is 0.86; its appearance-only variant
                used a looser bar because it had no geometry to fall back on.
        """
        if not -1.0 <= appearance_threshold <= 1.0:
            raise ConfigurationError(
                f"appearance_threshold is a cosine similarity and must be in [-1, 1], got "
                f"{appearance_threshold}"
            )
        self.appearance_threshold = float(appearance_threshold)

    def similarities(self, observations: Sequence[TrackObservation]) -> np.ndarray:
        """``(n, n)`` thresholded cosine similarity. Zero means "no appearance evidence".

        Deliberately *not* camera-masked: the mask belongs to exactly one place
        (:meth:`~shipvision.mtmc.base.BaseMatcher.to_distance`), and having it applied twice
        is how it ends up applied zero times after a refactor. Composed by
        :class:`~shipvision.mtmc.core.gated.matcher.GatedMatcher`, which needs the raw
        appearance evidence before deciding whether geometry vetoes it.
        """
        features = stack_embeddings(observations)
        if features.size == 0:
            return np.zeros((len(observations), len(observations)), dtype=np.float32)
        similarity = cosine_similarity(features, features)
        return np.where(similarity > self.appearance_threshold, similarity, 0.0).astype(
            np.float32
        )

    def build(self, observations: Sequence[TrackObservation]) -> np.ndarray:
        return self.to_distance(
            self.similarities(observations), self.mergeable_mask(observations)
        )

    def __repr__(self) -> str:
        return (
            f"<AppearanceMatcher appearance_threshold={self.appearance_threshold} "
            f"backend={self.backend}>"
        )
