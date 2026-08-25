"""DeepSORTv2's four stage costs, its dynamic EMA rate, and its border rule.

Four costs rather than one because the cascade's *ordering* is the design, and an ordering is
only readable if each stage is a named thing that says what evidence it is allowed to see. The
primitives they compose stay shared: GIoU and IoU in
:mod:`shipvision.tracking.association.costs`, "is there any appearance evidence" in
:func:`~shipvision.tracking.association.appearance.pairwise_appearance` (BoT-SORT needs it
too), and the dynamic-rate rule itself in
:func:`~shipvision.tracking.association.appearance.dynamic_appearance_momentum`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from shipvision.tracking.association import (
    INFEASIBLE,
    dynamic_appearance_momentum,
    gate_cost,
    giou_cost,
    iou_cost,
)
from shipvision.tracking.motion.kalman import CHI2_INV_95_4DOF
from shipvision.types import Detection

__all__ = [
    "dynamic_momentum",
    "off_border",
    "stage_a_cost",
    "stage_b_cost",
    "stage_c_cost",
    "stage_d_cost",
]


def stage_a_cost(
    track_boxes: np.ndarray,
    detection_boxes: np.ndarray,
    *,
    appearance: np.ndarray | None,
    appearance_weight: float,
    appearance_gate: float,
    giou_gate: float,
    gating_distances: np.ndarray,
) -> np.ndarray:
    """GIoU blended with appearance, then gated on **both** independently.

    The conjunction is deliberate and is where the C++ reference contradicts itself: its loop
    path requires both gates and its vectorised path requires either. A cost matrix whose
    gates can each be satisfied by ignoring the other is not gated.

    GIoU rather than IoU because IoU is flat at zero for every non-overlapping pair, so it
    cannot rank two equally-unmatched candidates — and ranking them is exactly what a cascade
    band is for.

    Args:
        track_boxes: ``(n, 4)`` xyxy predictions for this band's tracks.
        detection_boxes: ``(m, 4)`` xyxy detections still unmatched.
        appearance: ``(n, m)`` cosine distances, or ``None`` when this frame has no appearance
            evidence. ``None`` falls back to gated geometry alone; the alternative — treating
            a missing appearance distance as zero — asserts that every pair looks identical,
            which is the strongest claim available and made on no evidence at all.
        appearance_weight: how much of the fused cost is appearance rather than GIoU. High
            (0.9) reads as extreme until you remember the gates: GIoU has already vetoed
            anything geometrically impossible, so what is left for the cost to decide *is* an
            appearance question.
        appearance_gate: cosine distance above which a pair is forbidden.
        giou_gate: GIoU cost above which a pair is forbidden. The range is ``[0, 2]``, so 1.2
            admits pairs that do not overlap at all but are close.
        gating_distances: ``(n, m)`` squared Mahalanobis distances. Required, not optional:
            stage A is where a confirmed track gets first refusal on the best evidence in the
            frame, and an ungated first refusal is how one crowded frame swaps two identities.
    """
    geometry = giou_cost(track_boxes, detection_boxes)
    if appearance is None:
        return gate_cost(geometry, gating_distances, CHI2_INV_95_4DOF)

    fused = appearance_weight * appearance + (1.0 - appearance_weight) * geometry
    forbidden = (geometry > giou_gate) | (appearance > appearance_gate)
    fused = np.where(forbidden, INFEASIBLE, fused).astype(np.float32)
    return gate_cost(fused, gating_distances, CHI2_INV_95_4DOF)


def stage_b_cost(
    track_boxes: np.ndarray,
    detection_boxes: np.ndarray,
    *,
    appearance: np.ndarray | None,
    appearance_gate: float,
    gating_distances: np.ndarray,
) -> np.ndarray:
    """IoU against the prediction, with appearance demoted to a veto.

    The demotion is the difference from stage A and the reason both exist: these tracks lost
    stage A, so their appearance has gone stale and blending it into the cost would rank them
    by how out-of-date their gallery vector is. It is still worth a *veto* — a stale vector
    that is wildly wrong is still evidence of the wrong object.

    Args:
        track_boxes: ``(n, 4)`` xyxy predictions for the stage-A leftovers seen recently.
        detection_boxes: ``(m, 4)`` xyxy detections still unmatched.
        appearance: ``(n, m)`` cosine distances, or ``None`` for no appearance evidence.
        appearance_gate: cosine distance above which a pair is forbidden.
        gating_distances: ``(n, m)`` squared Mahalanobis distances.
    """
    cost = iou_cost(track_boxes, detection_boxes)
    if appearance is not None:
        cost = np.where(appearance > appearance_gate, INFEASIBLE, cost).astype(np.float32)
    return gate_cost(cost, gating_distances, CHI2_INV_95_4DOF)


def stage_c_cost(observed_boxes: np.ndarray, detection_boxes: np.ndarray) -> np.ndarray:
    """IoU against the last observation. No motion gate, by design (OC-SORT's OCR).

    The prediction is what already failed in stages A and B, and the filter's covariance after
    a gap is wide enough that its gate admits almost anything. Both are reasons to leave the
    filter out of this stage entirely rather than to consult it more carefully.

    Args:
        observed_boxes: ``(n, 4)`` xyxy last observations. Not predictions.
        detection_boxes: ``(m, 4)`` xyxy detections still unmatched.
    """
    return iou_cost(observed_boxes, detection_boxes)


def stage_d_cost(track_boxes: np.ndarray, detection_boxes: np.ndarray) -> np.ndarray:
    """IoU, and nothing else. A tentative track has no history worth gating on.

    Args:
        track_boxes: ``(n, 4)`` xyxy predictions for the tentative tracks.
        detection_boxes: ``(m, 4)`` xyxy detections the first three stages did not take.
    """
    return iou_cost(track_boxes, detection_boxes)


def dynamic_momentum(
    detections: Sequence[Detection], *, bounds: tuple[float, float]
) -> np.ndarray | None:
    """One EMA retention per detection, high where the detection is hard to trust.

    ``None`` when there are no detections — a legitimate absence, and distinct from an array
    of rates. :meth:`~shipvision.tracking.pool.TrackPool.apply_matches` reads it as "use the
    pool's configured rate", which is the honest degradation; inventing a rate for a frame
    with nothing in it would mean an empty frame silently changed how every gallery vector
    ages.

    Args:
        detections: this frame's kept detections, in the order the columns refer to. The order
            matters: the returned array is indexed by column, so a reordering here would apply
            one detection's confidence to another's track.
        bounds: ``(min, max)`` retention. ``min`` is used for a detection the frame gives no
            reason to distrust; ``max`` for one that is unconfident or crowded.
    """
    if not detections:
        return None
    boxes = np.stack([d.box for d in detections])
    scores = np.array([d.score for d in detections], dtype=np.float32)
    low, high = bounds
    return dynamic_appearance_momentum(boxes, scores, min_momentum=low, max_momentum=high)


def off_border(
    observed_boxes: np.ndarray,
    rows: Sequence[int],
    *,
    height: int,
    width: int,
    border_fraction: float,
) -> list[int]:
    """The subset of ``rows`` whose last observation is not against the frame edge.

    An object leaving the frame is half out of it, so its last observation is a truncated box
    that overlaps whatever else is at that edge. Recovering on that evidence swaps identities
    between everything entering and everything leaving — the one failure that makes the
    recovery stage worse than not having it.

    A frame size of zero means the caller did not supply one, and ``rows`` comes back
    unchanged. Guessing the frame size from the boxes would make the rule depend on where the
    objects happen to be, which is a rule that works until the first frame where everyone
    stands on one side.

    Args:
        observed_boxes: ``(n, 4)`` xyxy last observations for the **whole pool**, indexed by
            the values in ``rows`` rather than pre-sliced.
        rows: candidate track indices.
        height: frame height in pixels, or ``0`` when unknown.
        width: frame width in pixels, or ``0`` when unknown.
        border_fraction: how close to the edge counts as "near the border", as a fraction of
            the smaller frame dimension.
    """
    if not rows or height <= 0 or width <= 0:
        return list(rows)
    margin = border_fraction * min(height, width)
    return [
        row
        for row in rows
        if not (
            observed_boxes[row][0] < margin
            or observed_boxes[row][1] < margin
            or width - observed_boxes[row][2] < margin
            or height - observed_boxes[row][3] < margin
        )
    ]
