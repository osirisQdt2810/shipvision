"""Greedy suppression: classic NMS and the two soft variants.

One loop for all three, because they differ only in the weight an overlapping box's score is
multiplied by. Splitting them into three functions would triple the number of places the
departure rule and the tie-break live, and those — not the weight — are where the subtle
disagreements come from.

The loop is over *survivors*, not over candidates, and each iteration's overlap test is one
vectorised call. So the Python cost is proportional to the handful of objects in a frame
rather than to the thousands of raw proposals, which is what makes a numpy implementation
usable for more than testing.
"""

from __future__ import annotations

import numpy as np

from shipvision.imgproc.nms.candidates import CLASSIC, GAUSS, LINEAR
from shipvision.types import iou_matrix

__all__ = ["decay_weights", "greedy"]


def greedy(
    boxes: np.ndarray,
    scores: np.ndarray,
    order: np.ndarray,
    *,
    iou_threshold: float,
    method: str,
    sigma: float,
    score_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick the best box, punish what it overlaps, repeat.

    A candidate leaves the pool when its decayed score drops below ``score_threshold`` **or
    reaches zero**. The "or zero" half is what makes ``classic`` — whose weight is 0 — plain
    greedy NMS at every threshold including 0.0, with no special case anywhere; a box worth
    nothing is not a detection. Without it, ``classic`` at ``score_threshold=0.0`` would
    return every suppressed box with a score of 0.0 attached.

    Args:
        boxes: ``(n, 4)`` xyxy float32, indexed by ``order``.
        scores: ``(n,)`` float32, indexed by ``order``.
        order: admitted candidates, descending score. From
            :func:`~shipvision.imgproc.nms.candidates.prepare`.
        iou_threshold: punish when ``iou > iou_threshold``, strictly.
        method: ``"classic"``, ``"linear"`` or ``"gauss"``.
        sigma: the gaussian's width. Read by ``"gauss"`` only.
        score_threshold: the floor a decayed score must stay at or above.

    Returns:
        ``(indices, scores)``, aligned, in descending order of the score each survivor held
        when it was picked.
    """
    live = np.ones(order.size, dtype=bool)
    working = scores[order].astype(np.float32, copy=True)
    kept: list[int] = []
    kept_scores: list[float] = []

    while True:
        alive = np.flatnonzero(live)
        if alive.size == 0:
            break
        # argmax takes the first maximum, and `order` is already descending, so a tie is
        # resolved towards the lower input index.
        best = int(alive[np.argmax(working[alive])])
        live[best] = False
        kept.append(int(order[best]))
        kept_scores.append(float(working[best]))

        rest = np.flatnonzero(live)
        if rest.size == 0:
            break
        overlaps = iou_matrix(boxes[order[best]][None, :], boxes[order[rest]])[0]
        # WHICH BOXES GET DECAYED depends on the method, and this is the one place the three
        # genuinely differ in *shape* rather than in the weight they apply.
        #
        # `classic` and `linear` act only above the threshold: classic removes there and
        # nowhere else, and linear's `1 - iou` is the paper's Eq. (3), which is explicitly
        # piecewise — identity below N_t, decay above. Both need the gate.
        #
        # `gauss` does not. Eq. (4) is `s_i <- s_i * exp(-iou(M, b_i)^2 / sigma)` for every
        # b_i not yet kept, with no N_t anywhere; removing the discontinuity at N_t is the
        # paper's *stated reason* for preferring it over the linear rule. This code used to
        # gate all three, which did two things to gauss at once: it removed every decay in the
        # 0-to-N_t band — where soft-NMS does most of its work on a crowded quayside — and it
        # put the discontinuity back at N_t, so the method was linear-with-a-different-curve
        # rather than the paper's. Two moored vessels at IoU 0.40 under sigma=0.5 came back at
        # 0.85 where the paper says 0.617, and gauss and linear returned identical results.
        # No test could see it because every gauss case used an IoU above the gate.
        if method == GAUSS:
            punished = np.ones_like(overlaps, dtype=bool)
        else:
            punished = overlaps > np.float32(iou_threshold)
        if not punished.any():
            continue

        hit = rest[punished]
        working[hit] = working[hit] * decay_weights(
            overlaps[punished], method=method, sigma=sigma
        )
        live[hit] = (working[hit] >= np.float32(score_threshold)) & (working[hit] > 0.0)

    return np.asarray(kept, dtype=np.int64), np.asarray(kept_scores, dtype=np.float32)


def decay_weights(overlaps: np.ndarray, *, method: str, sigma: float) -> np.ndarray:
    """The factor an overlapping box's score is multiplied by.

    Zero for ``classic`` — removal expressed as a weight, so there is one loop instead of
    two — ``1 - iou`` for ``linear`` and ``exp(-iou^2 / sigma)`` for ``gauss``. The last two
    are the soft-NMS paper's Eq. (3) and Eq. (4), and the C++ reference computed the same
    expressions. Which boxes each is *applied to* is decided by the caller: linear is
    piecewise and only ever sees overlaps above the threshold; gauss sees every live box.
    """
    if method == CLASSIC:
        return np.zeros_like(overlaps)
    if method == LINEAR:
        return (np.float32(1.0) - overlaps).astype(np.float32)
    return np.exp(-(overlaps * overlaps) / np.float32(sigma)).astype(np.float32)
