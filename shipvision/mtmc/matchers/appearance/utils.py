"""Getting this instant's embeddings into one array, or refusing to.

Separate from the matcher because it is the half with an opinion about *input*, and the
opinion is a refusal: a group where one track has no embedding, or where two cameras are
running different re-ID models, cannot produce a similarity matrix anybody can reason about.
Keeping it here means the check has a test that does not need a matcher, and means a future
matcher that also consumes embeddings inherits the same refusal rather than writing a second,
subtly different one.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from shipvision.errors import DimensionMismatchError, TrackingError
from shipvision.mtmc.frames import TrackObservation
from shipvision.reid.distance import normalize

__all__ = ["stack_embeddings"]


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
            f"matcher that does not use appearance"
        )
    widths = {int(o.embedding.shape[-1]) for o in observations}  # type: ignore[union-attr]
    if len(widths) > 1:
        raise DimensionMismatchError(
            f"embeddings of {sorted(widths)} dimensions arrived in one synchronised group; "
            f"these cameras are not running the same re-ID model"
        )
    stacked = np.stack([np.asarray(o.embedding, dtype=np.float32) for o in observations])
    return normalize(stacked)
