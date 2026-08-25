"""ByteTrack's two-tier detection split and the cost matrix for each tier.

The split is the algorithm — everything else in the paper follows from having two tiers — so
it lives here as a named function rather than as two comprehensions inside ``update``.

The primitives the costs compose (:func:`~shipvision.tracking.association.costs.iou_cost`,
:func:`~shipvision.tracking.association.costs.fuse_score`,
:func:`~shipvision.tracking.association.costs.gate_cost`) are shared and stay in
``association/``.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from shipvision.tracking.association import fuse_score, gate_cost, gated_iou_cost, iou_cost
from shipvision.tracking.motion.kalman import CHI2_INV_95_4DOF
from shipvision.types import Detection

__all__ = ["high_score_cost", "low_score_cost", "split_by_score"]


def split_by_score(
    detections: Iterable[Detection], *, low_threshold: float, track_threshold: float
) -> tuple[list[Detection], list[Detection]]:
    """Partition into ``(high, low)``; anything under ``low_threshold`` is dropped entirely.

    Both lists preserve input order, and the caller relies on that: the column indices the
    solver returns are positions in these lists, so a reordering here would associate the
    wrong detections while every test still passed.

    Args:
        detections: this frame's detections, already tagged.
        low_threshold: below this a detection is noise and is discarded.
        track_threshold: at or above this a detection is "high score" and may start a track.
    """
    high: list[Detection] = []
    low: list[Detection] = []
    for detection in detections:
        if detection.score >= track_threshold:
            high.append(detection)
        elif detection.score >= low_threshold:
            low.append(detection)
    return high, low


def high_score_cost(
    track_boxes: np.ndarray,
    detection_boxes: np.ndarray,
    detection_scores: np.ndarray,
    *,
    gating_distances: np.ndarray | None = None,
) -> np.ndarray:
    """Stage one: ``1 - IoU`` scaled by detector confidence, then gated.

    Args:
        track_boxes: ``(n, 4)`` xyxy predictions for the tracks being matched.
        detection_boxes: ``(m, 4)`` xyxy high-score detections.
        detection_scores: ``(m,)`` detector confidence, folded into the similarity so that a
            0.95 box outbids a 0.55 box at equal overlap.
        gating_distances: ``(n, m)`` squared Mahalanobis distances, or ``None`` to skip.
    """
    cost = fuse_score(iou_cost(track_boxes, detection_boxes), detection_scores)
    if gating_distances is not None:
        cost = gate_cost(cost, gating_distances, CHI2_INV_95_4DOF)
    return cost


def low_score_cost(
    track_boxes: np.ndarray,
    detection_boxes: np.ndarray,
    *,
    gating_distances: np.ndarray | None = None,
) -> np.ndarray:
    """Stage two: overlap alone, because nothing else is trustworthy at this score.

    Confidence is not fused here and appearance is not consulted. Both are unreliable on a
    0.3 detection, and folding an unreliable signal into the cost is how the second stage
    starts inventing matches instead of rescuing them.

    The expression is identical to
    :func:`shipvision.tracking.core.sort.utils.association_cost`, so both delegate to the one
    shared implementation instead of keeping a copy each. Two copies is how the second pass
    and the SORT baseline it is measured against drift apart while both still pass their
    tests; the thresholds that make the two stages *different* are the caller's, and stay in
    the tracker.

    Args:
        track_boxes: ``(n, 4)`` xyxy predictions for the tracks still unmatched.
        detection_boxes: ``(m, 4)`` xyxy low-score detections.
        gating_distances: ``(n, m)`` squared Mahalanobis distances, or ``None`` to skip.
    """
    return gated_iou_cost(
        track_boxes,
        detection_boxes,
        gating_distances=gating_distances,
        threshold=CHI2_INV_95_4DOF,
    )
