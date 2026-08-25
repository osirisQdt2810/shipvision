"""Neighbourhood suppression: greedy, plus a demand for corroboration.

Its own module rather than a branch inside :mod:`shipvision.imgproc.nms.greedy` because it
answers a different question. Greedy NMS asks "which of these overlapping boxes is best";
this asks "is this cluster of boxes evidence at all". A cluster of overlapping proposals is;
a single confident box with nothing agreeing with it often is not, and on open water — where a
detector's false positives are isolated rather than clustered — that distinction is worth
having behind a parameter.
"""

from __future__ import annotations

import numpy as np

from shipvision.types import iou_matrix

__all__ = ["neighborhood"]


def neighborhood(
    boxes: np.ndarray,
    scores: np.ndarray,
    order: np.ndarray,
    *,
    iou_threshold: float,
    min_neighbors: int,
    min_score_sum: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy NMS that publishes a survivor only if enough boxes agreed with it.

    ``min_score_sum`` is measured over the anchor *plus* its suppressed neighbours, so it is a
    statement about the cluster rather than about the best box in it. The neighbours are
    consumed either way: an anchor that fails the test does not hand them a second chance,
    because they are the same object.

    With ``min_neighbors=0`` and ``min_score_sum=0.0`` — the values the C++ reference
    hard-coded at its only call site — every anchor passes and this is exactly classic NMS.
    That is worth knowing rather than hiding: the method as shipped in the reference could
    never behave differently from the default, which is why the parameters are exposed here.

    Returns:
        ``(indices, scores)``. Scores are the originals; this method removes boxes and does
        not decay anything.
    """
    live = np.ones(order.size, dtype=bool)
    working = scores[order]
    kept: list[int] = []
    kept_scores: list[float] = []

    while True:
        alive = np.flatnonzero(live)
        if alive.size == 0:
            break
        best = int(alive[np.argmax(working[alive])])
        live[best] = False
        score_sum = float(working[best])
        neighbours = 0

        rest = np.flatnonzero(live)
        if rest.size:
            overlaps = iou_matrix(boxes[order[best]][None, :], boxes[order[rest]])[0]
            hit = rest[overlaps > np.float32(iou_threshold)]
            neighbours = int(hit.size)
            score_sum += float(working[hit].sum())
            live[hit] = False

        if neighbours >= min_neighbors and score_sum >= min_score_sum:
            kept.append(int(order[best]))
            kept_scores.append(float(working[best]))

    return np.asarray(kept, dtype=np.int64), np.asarray(kept_scores, dtype=np.float32)
