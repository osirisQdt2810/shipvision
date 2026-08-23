"""Cosine appearance similarity, hard-thresholded, with same-camera pairs excluded.

The baseline builder, and the only one that works on an uncalibrated site. It answers "do
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

from shipvision.errors import ConfigurationError, DimensionMismatchError, TrackingError
from shipvision.mtmc.frames import TrackObservation
from shipvision.mtmc.matrix.base import MATRIX_BUILDERS, BaseMatrixBuilder
from shipvision.registry import PYTHON
from shipvision.reid.distance import cosine_similarity, normalize

__all__ = ["AppearanceMatrixBuilder", "stack_embeddings"]


def stack_embeddings(observations: Sequence[TrackObservation]) -> np.ndarray:
    """``(n, d)`` float32 L2-normalised embeddings, or a typed failure.

    All-or-nothing, following :attr:`shipvision.types.Detections.embeddings`: one track
    without an embedding would make one row of the similarity matrix meaningless while the
    rest looked fine, and a matrix nobody can reason about is worse than a refusal.

    Normalised here even though :mod:`shipvision.types` says embeddings are stored
    normalised. This is once per synchronised group over at most a few hundred rows — not
    once per query against a gallery, which is the case that convention exists to protect —
    and an un-normalised vector turns cosine similarity into a dot product of arbitrary
    scale, which does not fail, it just makes every threshold in this package mean something
    different.
    """
    if not observations:
        return np.zeros((0, 0), dtype=np.float32)
    missing = [str(o.key) for o in observations if o.embedding is None]
    if missing:
        raise TrackingError(
            f"appearance matching needs an embedding on every track; {len(missing)} have "
            f"none (first: {missing[0]}). Either run the re-ID stage before MTMC or choose a "
            f"builder that does not use appearance"
        )
    widths = {int(o.embedding.shape[-1]) for o in observations}  # type: ignore[union-attr]
    if len(widths) > 1:
        raise DimensionMismatchError(
            f"embeddings of {sorted(widths)} dimensions arrived in one synchronised group; "
            f"these cameras are not running the same re-ID model"
        )
    stacked = np.stack([np.asarray(o.embedding, dtype=np.float32) for o in observations])
    return normalize(stacked)


@MATRIX_BUILDERS.register("appearance", backend=PYTHON, aliases=("aic",))
class AppearanceMatrixBuilder(BaseMatrixBuilder):
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
        (:meth:`BaseMatrixBuilder.to_distance`), and having it applied twice is how it ends
        up applied zero times after a refactor. Composed by
        :class:`~shipvision.mtmc.matrix.gated.GatedMatrixBuilder`, which needs the raw
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
            f"<AppearanceMatrixBuilder appearance_threshold={self.appearance_threshold} "
            f"backend={self.backend}>"
        )
