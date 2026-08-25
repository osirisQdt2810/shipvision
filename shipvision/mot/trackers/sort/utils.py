"""SORT's cost matrix. One stage, so one function.

The primitives it composes are shared and stay in ``association/`` and ``motion/``. What is
SORT's own is *which* of them it uses and in what order, because that ordering is the
algorithm — so the composition gets a name at SORT's own address while the arithmetic stays
in one place for every tracker that needs it.
"""

from __future__ import annotations

import numpy as np

from shipvision.mot.association import gated_iou_cost
from shipvision.mot.motion.kalman import CHI2_INV_95_4DOF

__all__ = ["association_cost"]


def association_cost(
    track_boxes: np.ndarray,
    detection_boxes: np.ndarray,
    *,
    gating_distances: np.ndarray | None = None,
) -> np.ndarray:
    """``1 - IoU``, with the motion model given a veto.

    Delegates rather than reimplements: ByteTrack's second stage is the same expression, and
    two copies of it is how the baseline and the tracker measured against it drift apart
    without either one's tests noticing. See
    :func:`~shipvision.mot.association.costs.gated_iou_cost`.

    Args:
        track_boxes: ``(n, 4)`` xyxy predictions, one per track being matched.
        detection_boxes: ``(m, 4)`` xyxy detections that passed ``det_threshold``.
        gating_distances: ``(n, m)`` squared Mahalanobis distances, or ``None`` to skip the
            gate. ``None`` rather than a sentinel distance because the caller that turns
            gating off should not have to pay for computing the distances first — and
            because "the gate is off" and "the gate passed everything" are different states.
    """
    return gated_iou_cost(
        track_boxes,
        detection_boxes,
        gating_distances=gating_distances,
        threshold=CHI2_INV_95_4DOF,
    )
