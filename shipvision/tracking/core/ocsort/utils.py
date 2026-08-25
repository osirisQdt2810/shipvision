"""OC-SORT's two cost matrices: the primary association and the recovery pass.

The primitives are shared (:mod:`shipvision.tracking.association`,
:mod:`shipvision.tracking.motion`). What is OC-SORT's own — and what these two functions say
out loud — is *which* signals each stage is allowed to see. The recovery stage's omissions are
the algorithm, not an oversight, which is why it is a named function here rather than a bare
``iou_cost`` call inside ``update``.
"""

from __future__ import annotations

import numpy as np

from shipvision.errors import ConfigurationError
from shipvision.tracking.association import direction_cost, gate_cost, iou_cost
from shipvision.tracking.motion.kalman import CHI2_INV_95_4DOF

__all__ = ["primary_cost", "recovery_cost"]


def primary_cost(
    track_boxes: np.ndarray,
    detection_boxes: np.ndarray,
    *,
    headings: np.ndarray | None = None,
    origins: np.ndarray | None = None,
    momentum_weight: float = 0.0,
    gating_distances: np.ndarray | None = None,
) -> np.ndarray:
    """IoU against the prediction, nudged by whether the candidate is *ahead* (OCM).

    The momentum term is added, not fused: it is a tie-breaker between geometrically plausible
    candidates rather than a cost in its own right, which is why its weight is small by
    default. Two objects passing each other are geometrically interchangeable at the moment
    they overlap and are *not* interchangeable in heading.

    Args:
        track_boxes: ``(n, 4)`` xyxy predictions for the tracks being matched.
        detection_boxes: ``(m, 4)`` xyxy detections.
        headings: ``(n, 2)`` unit vectors from
            :meth:`~shipvision.tracking.pool.TrackPool.directions`, or ``None`` when the
            momentum term is off.
        origins: ``(n, 4)`` xyxy last observations the headings were measured *to*, or
            ``None`` when the momentum term is off.
        momentum_weight: how much the direction term counts against IoU. Zero disables it.
        gating_distances: ``(n, m)`` squared Mahalanobis distances, or ``None`` to skip the
            gate.

    Raises:
        ConfigurationError: ``momentum_weight`` is positive but the observations it is
            measured from were not supplied. Refusing beats silently dropping the term: a
            tracker that quietly stops being observation-centric still tracks, just worse, and
            nothing in its output says which of the three fixes went missing.
    """
    cost = iou_cost(track_boxes, detection_boxes)
    if momentum_weight > 0.0:
        if headings is None or origins is None:
            raise ConfigurationError(
                "momentum_weight > 0 needs headings and origins; the momentum term is "
                "measured between two real observations and cannot be read off the filter"
            )
        cost = cost + momentum_weight * direction_cost(headings, origins, detection_boxes)
    if gating_distances is not None:
        cost = gate_cost(cost, gating_distances, CHI2_INV_95_4DOF)
    return cost


def recovery_cost(observed_boxes: np.ndarray, detection_boxes: np.ndarray) -> np.ndarray:
    """IoU against the **last observation**, with no motion model and no gate (OCR).

    Both omissions are the point, and naming them here is why this exists as a function rather
    than as a bare ``iou_cost`` call in ``update``. The prediction is what already failed in
    the primary stage, so reusing it would just fail again; and gating on a filter whose
    covariance grew through the gap either admits everything or vetoes the one honest
    candidate. This is the stage that catches an object which stopped moving while it was
    hidden — the filter carried the old velocity and its prediction has walked off, while the
    object is still standing where it was last seen.

    Args:
        observed_boxes: ``(n, 4)`` xyxy last observations, from
            :meth:`~shipvision.tracking.pool.TrackPool.observed_boxes`. Not predictions.
        detection_boxes: ``(m, 4)`` xyxy detections the primary stage did not take.
    """
    return iou_cost(observed_boxes, detection_boxes)
