"""How fast a track's appearance vector should follow the crop it just saw.

A fixed EMA rate is the usual choice and it is wrong in both directions at once. Too fast and
one badly-cropped frame — a person half behind a pillar, a ship clipped by the frame edge —
becomes the reference every future match is scored against, and the identity walks away from
itself. Too slow and a track that genuinely changes appearance, which everything does as it
turns or moves through light, never catches up.

The rate should depend on how much the crop is worth. This is the *dynamic appearance* rule
from the internal C++ DeepSORTv2, ported here because it is the genuinely reusable idea in
that codebase: a detection is trustworthy when it is **confident** and when it is
**isolated**, and a crop of a crowded box contains as much of the neighbour as of the subject.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from shipvision.errors import ConfigurationError
from shipvision.mot.association.costs import appearance_cost
from shipvision.types import iou_matrix

__all__ = ["dynamic_appearance_momentum", "isolation", "pairwise_appearance"]


def isolation(boxes: np.ndarray) -> np.ndarray:
    """``(n,)`` in ``[0, 1]``: ``1 - `` the largest IoU with any *other* box.

    One means the detection has the frame to itself, zero means something else occupies the
    same pixels. A single detection scores one, which is correct and is the reason the
    diagonal has to be excluded rather than merely ignored — ``max`` over a row that includes
    the self-IoU of 1.0 is always 1.0, and every detection would look maximally crowded.
    """
    boxes = np.atleast_2d(np.asarray(boxes, dtype=np.float32))
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if boxes.shape[0] == 1:
        return np.ones((1,), dtype=np.float32)
    overlaps = iou_matrix(boxes, boxes)
    np.fill_diagonal(overlaps, -1.0)
    return np.clip(1.0 - overlaps.max(axis=1), 0.0, 1.0).astype(np.float32)


def dynamic_appearance_momentum(
    boxes: np.ndarray,
    scores: np.ndarray,
    *,
    min_momentum: float = 0.9,
    max_momentum: float = 0.95,
    conf_range: tuple[float, float] = (0.5, 0.8),
    isolation_range: tuple[float, float] = (0.5, 0.8),
) -> np.ndarray:
    """``(n,)`` per-detection EMA **retention**, high meaning "barely update".

    Both signals are mapped to a retention independently and the **larger** wins, then the
    result is capped. Taking the maximum rather than a blend is what makes the rule
    conservative in the right way: a detection needs to be confident *and* isolated to move a
    track's appearance, and failing either one is enough to hold it back. A weighted blend
    would let a very confident detection in the middle of a crowd overwrite a track's
    appearance with a crop that is half somebody else.

    Args:
        boxes: ``(n, 4)`` xyxy for this frame's detections. Crowding is measured among them.
        scores: ``(n,)`` detector confidence.
        min_momentum: retention for a detection that is both confident and isolated. Still
            high — 0.9 means a track's appearance is a ten-frame average even at its most
            responsive, because a single crop is never worth more than that.
        max_momentum: the cap. Below 1.0 on purpose: a track whose appearance can never
            update at all is a track that will eventually fail to match itself.
        conf_range: ``(lower, upper)`` confidence band the mapping is linear across.
        isolation_range: ``(lower, upper)`` isolation band, in the units of :func:`isolation`.

    Returns:
        ``(n,)`` float32 in ``[min_momentum, max_momentum]``.
    """
    if not 0.0 <= min_momentum <= max_momentum < 1.0:
        raise ConfigurationError(
            f"need 0 <= min_momentum ({min_momentum}) <= max_momentum ({max_momentum}) < 1"
        )
    for label, (lower, upper) in (
        ("conf_range", conf_range),
        ("isolation_range", isolation_range),
    ):
        if not lower < upper:
            raise ConfigurationError(f"{label} must be increasing, got ({lower}, {upper})")

    boxes = np.atleast_2d(np.asarray(boxes, dtype=np.float32))
    confidences = np.asarray(scores, dtype=np.float32).reshape(-1)
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if confidences.shape[0] != boxes.shape[0]:
        raise ConfigurationError(f"{boxes.shape[0]} boxes but {confidences.shape[0]} scores")

    span = 1.0 - min_momentum
    from_confidence = min_momentum + span * (1.0 - _ramp(confidences, conf_range))
    from_isolation = min_momentum + span * (1.0 - _ramp(isolation(boxes), isolation_range))
    return np.minimum(np.maximum(from_confidence, from_isolation), max_momentum).astype(
        np.float32
    )


def _ramp(values: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    """Clamp into ``bounds`` and rescale to ``[0, 1]``.

    Clamping before rescaling rather than after: a confidence of 0.95 and one of 0.99 are not
    meaningfully different qualities of crop, and letting the band extend to the extremes
    would spend most of its range distinguishing them.
    """
    lower, upper = bounds
    return (np.clip(values, lower, upper) - lower) / (upper - lower)


def pairwise_appearance(
    track_embeddings: np.ndarray | None,
    rows: Sequence[int],
    detection_embeddings: Sequence[np.ndarray | None],
) -> np.ndarray | None:
    """``(len(rows), len(detection_embeddings))`` cosine distance, or ``None`` if either side
    lacks an embedding.

    ``None`` means "there is no appearance evidence on this frame", and it is a distinct
    answer from any cost matrix. The tempting alternative — a zero — asserts that every pair
    looks identical, which is the strongest claim available and made on no evidence at all;
    the tracker that receives ``None`` falls back to geometry instead.

    All-or-nothing on purpose. A partially-populated matrix would have to invent a value for
    the missing pairs, and any value invented there is a claim about identity.

    Shared by BoT-SORT and DeepSORTv2, which is why it lives here rather than in either
    algorithm's ``utils.py``: they had the same six lines each, and the day one of them fixed
    a bug in its copy the two trackers would have started disagreeing about what "no
    appearance" means.

    Args:
        track_embeddings: ``(n, d)`` L2-normalised track vectors, or ``None`` when the pool
            cannot supply one for every track.
        rows: which track indices to score, as indices into ``track_embeddings``.
        detection_embeddings: one vector per detection column, any of which may be ``None``.
    """
    if track_embeddings is None:
        return None
    if any(embedding is None for embedding in detection_embeddings):
        return None
    return appearance_cost(track_embeddings[list(rows)], np.stack(list(detection_embeddings)))
