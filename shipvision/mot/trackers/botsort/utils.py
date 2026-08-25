"""BoT-SORT's first-stage cost: minimum fusion of geometry and appearance.

The primitives are shared — :func:`~shipvision.mot.association.costs.min_fuse` is where
the fusion rule lives and :func:`~shipvision.mot.association.appearance.pairwise_appearance`
is where "there is no appearance evidence" is decided, both because DeepSORTv2 needs them too.
What is BoT-SORT's own is the *shape* of the stage: fuse by minimum, fall back to geometry
when there is nothing to fuse, and gate afterwards.
"""

from __future__ import annotations

import numpy as np

from shipvision.mot.association import gate_cost, iou_cost, min_fuse
from shipvision.mot.motion.kalman import CHI2_INV_95_4DOF

__all__ = ["first_cost"]


def first_cost(
    track_boxes: np.ndarray,
    detection_boxes: np.ndarray,
    *,
    appearance: np.ndarray | None,
    motion_gate: float,
    appearance_gate: float,
    appearance_weight: float,
    gating_distances: np.ndarray | None = None,
) -> np.ndarray:
    """Minimum of the gated IoU cost and the gated, halved appearance cost.

    Note what is *not* here: :func:`~shipvision.mot.association.costs.fuse_score`, which
    ByteTrack uses to scale similarity by detector confidence. Folding confidence into a cost
    that is already a minimum of two gated terms double-counts it — the high-score stage has
    by definition already filtered on confidence — and it pushes the fused cost above the
    appearance gate for exactly the medium-confidence detections appearance is supposed to
    rescue. That omission is why this is BoT-SORT's own function rather than a call to
    ByteTrack's.

    Args:
        track_boxes: ``(n, 4)`` xyxy predictions for the tracks being matched.
        detection_boxes: ``(m, 4)`` xyxy high-score detections.
        appearance: ``(n, m)`` cosine distances, or ``None`` when either side of this frame
            lacks an embedding. ``None`` falls back to geometry alone rather than treating a
            missing appearance distance as zero — a zero would mean "identical appearance",
            which is the strongest possible claim, made on no evidence.
        motion_gate: IoU cost above which a pair contributes no motion term.
        appearance_gate: cosine distance above which a pair contributes no appearance term.
        appearance_weight: the paper halves the cosine distance before the minimum, because
            ``1 - IoU`` and a cosine distance are not on the same scale.
        gating_distances: ``(n, m)`` squared Mahalanobis distances, or ``None`` to skip.
    """
    motion = iou_cost(track_boxes, detection_boxes)
    if appearance is None:
        cost = motion
    else:
        cost = min_fuse(
            motion,
            appearance,
            motion_gate=motion_gate,
            appearance_gate=appearance_gate,
            appearance_weight=appearance_weight,
        )
    if gating_distances is not None:
        cost = gate_cost(cost, gating_distances, CHI2_INV_95_4DOF)
    return cost
