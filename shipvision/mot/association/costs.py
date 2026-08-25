"""The cost matrices: what makes two boxes "the same object".

Building a cost is domain knowledge. Solving the assignment is a solved problem, and it lives
next door in :mod:`~shipvision.mot.association.solver` — the split is deliberate, because
every tracker here differs in *which of these it combines* and none of them differ in how the
Hungarian algorithm works.

Everything is vectorised and returns ``(n_tracks, n_detections)`` float32. A Python loop over
tracks and detections is the classic way a tracker becomes the bottleneck at fifty cameras,
and it buys nothing: the pairs are independent, which is what numpy is for.
"""

from __future__ import annotations

import numpy as np

from shipvision.types import iou_matrix

__all__ = [
    "INFEASIBLE",
    "appearance_cost",
    "direction_cost",
    "fuse_score",
    "gate_cost",
    "gated_iou_cost",
    "giou_cost",
    "giou_matrix",
    "iou_cost",
    "min_fuse",
]

#: The cost given to a pair a gate has forbidden. Large enough that the solver will never
#: choose it over any real alternative, finite so the solver still terminates on a matrix that
#: is entirely gated — ``np.inf`` makes ``linear_sum_assignment`` raise instead.
#:
#: Defined here rather than in the solver because it is a *cost*, and the functions that
#: produce it are these.
INFEASIBLE = 1e5


def iou_cost(track_boxes: np.ndarray, detection_boxes: np.ndarray) -> np.ndarray:
    """``1 - IoU``. The workhorse: cheap, and right whenever motion is small between frames."""
    return (1.0 - iou_matrix(track_boxes, detection_boxes)).astype(np.float32)


def giou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Generalised IoU, ``(n, m)`` in ``[-1, 1]``.

    IoU minus the fraction of the smallest enclosing box that neither input covers. The
    reason to reach for it over plain IoU is that IoU is *flat at zero* for every
    non-overlapping pair, so an assignment on IoU alone cannot tell "just missed" from "the
    other side of the frame". GIoU keeps decreasing as the boxes separate, which is what
    lets a cascade rank two equally-unmatched candidates.
    """
    boxes_a = np.atleast_2d(np.asarray(a, dtype=np.float32))
    boxes_b = np.atleast_2d(np.asarray(b, dtype=np.float32))
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float32)

    iou = iou_matrix(boxes_a, boxes_b)
    area_a = np.prod(np.clip(boxes_a[:, 2:] - boxes_a[:, :2], 0.0, None), axis=1)
    area_b = np.prod(np.clip(boxes_b[:, 2:] - boxes_b[:, :2], 0.0, None), axis=1)

    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    overlap = np.prod(np.clip(bottom_right - top_left, 0.0, None), axis=2)
    union = area_a[:, None] + area_b[None, :] - overlap

    hull_top_left = np.minimum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    hull_bottom_right = np.maximum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    hull = np.prod(np.clip(hull_bottom_right - hull_top_left, 0.0, None), axis=2)

    return (iou - (hull - union) / np.maximum(hull, 1e-9)).astype(np.float32)


def giou_cost(track_boxes: np.ndarray, detection_boxes: np.ndarray) -> np.ndarray:
    """``1 - GIoU``, so the range is ``[0, 2]`` rather than ``[0, 1]``.

    Worth remembering when picking a threshold: a GIoU cost of 1.0 is "no overlap and
    touching", not "no overlap at all".
    """
    return (1.0 - giou_matrix(track_boxes, detection_boxes)).astype(np.float32)


def appearance_cost(
    track_embeddings: np.ndarray, detection_embeddings: np.ndarray
) -> np.ndarray:
    """Cosine distance between L2-normalised embeddings.

    Assumes normalised inputs and does not renormalise. That is a contract, not laziness:
    silently normalising here would hide a caller feeding raw logits, and the cost would
    look plausible while meaning nothing. This library normalises once, on the way in.
    """
    if len(track_embeddings) == 0 or len(detection_embeddings) == 0:
        return np.zeros((len(track_embeddings), len(detection_embeddings)), dtype=np.float32)
    similarity = (
        np.asarray(track_embeddings, dtype=np.float32)
        @ np.asarray(detection_embeddings, dtype=np.float32).T
    )
    return np.clip(1.0 - similarity, 0.0, 2.0).astype(np.float32)


def fuse_score(cost: np.ndarray, detection_scores: np.ndarray) -> np.ndarray:
    """Fold detector confidence into an IoU cost.

    A high-IoU match with a 0.3-confidence detection is weaker evidence than the same IoU
    with a 0.9 one, and an assignment that ignores that will happily attach an identity to a
    barely-there box. ByteTrack's formulation: similarity is scaled by the score.
    """
    if cost.size == 0:
        return cost
    similarity = (1.0 - cost) * np.asarray(detection_scores, dtype=np.float32)[None, :]
    return (1.0 - similarity).astype(np.float32)


def gate_cost(
    cost: np.ndarray, gating_distances: np.ndarray, threshold: float, *, weight: float = 0.0
) -> np.ndarray:
    """Forbid pairs the motion model says are impossible.

    Gating before assignment rather than weighting inside it. A detection the filter says
    cannot belong to this track should not be selectable at *any* price — leaving it
    selectable means one crowded frame can hand an identity to the wrong object, and an ID
    switch is not recoverable the way a missed frame is.

    ``weight`` optionally blends a little of the motion distance into the cost, which breaks
    ties between two appearance-equal candidates in favour of the one the filter expected.
    """
    gated = cost.copy()
    gated[gating_distances > threshold] = INFEASIBLE
    if weight > 0.0:
        feasible = gating_distances <= threshold
        gated[feasible] = (1 - weight) * gated[feasible] + weight * gating_distances[feasible]
    return gated


def gated_iou_cost(
    track_boxes: np.ndarray,
    detection_boxes: np.ndarray,
    *,
    gating_distances: np.ndarray | None = None,
    threshold: float,
) -> np.ndarray:
    """``1 - IoU``, with the motion model given a veto. Two algorithms share this exactly.

    SORT's only association stage and ByteTrack's low-score second stage are the same three
    lines. They lived as two copies until the algorithms became packages, and the copies are
    what this function exists to prevent: the two are a plausible place to "fix" a gate, and
    fixing one of them silently makes ByteTrack's second pass and the SORT baseline it is
    measured against stop being comparable — with both trackers still passing every test.

    ``gating_distances=None`` means *the gate is off*, which is a different state from a gate
    that admitted everything, and is why the parameter is not a sentinel distance: a caller
    that has switched gating off should not first have to pay for the distances.

    ``threshold`` is required rather than defaulted to
    :data:`~shipvision.mot.motion.kalman.CHI2_INV_95_4DOF` so that ``association`` keeps
    knowing nothing about ``motion``. The chi-square value belongs to the filter that produced
    the distances, and a default here would quietly outlive a change to the state dimension.

    Args:
        track_boxes: ``(n, 4)`` xyxy predictions, one per track being matched.
        detection_boxes: ``(m, 4)`` xyxy detections.
        gating_distances: ``(n, m)`` squared Mahalanobis distances, or ``None`` to skip.
        threshold: gating distance above which a pair is forbidden.
    """
    cost = iou_cost(track_boxes, detection_boxes)
    if gating_distances is None:
        return cost
    return gate_cost(cost, gating_distances, threshold)


def min_fuse(
    motion: np.ndarray,
    appearance: np.ndarray,
    *,
    motion_gate: float,
    appearance_gate: float,
    appearance_weight: float = 0.5,
) -> np.ndarray:
    """BoT-SORT's fusion: element-wise **minimum** of two independently gated costs.

    The alternative — a weighted sum — is what DeepSORT does, and it has a specific failure:
    a pair that is unambiguous on one signal is dragged over the threshold by the other. Two
    people in identical uniforms have a near-zero appearance distance, so a sum lets
    appearance veto a geometrically obvious match; a person turning a corner has poor
    overlap, so a sum lets geometry veto an appearance-obvious match. Taking the minimum
    means *either* signal on its own is enough, and the two gates are what stop that from
    becoming "anything matches anything".

    Args:
        motion: ``(n, m)`` geometric cost, normally ``1 - IoU``.
        appearance: ``(n, m)`` cosine distance, or an all-zero array when no embeddings
            exist. Pass ``motion_gate``-passing zeros only if you mean "appearance agrees".
        motion_gate: pairs whose motion cost exceeds this contribute nothing.
        appearance_gate: pairs whose appearance cost exceeds this contribute nothing.
        appearance_weight: the paper halves the appearance distance before the minimum,
            because a cosine distance and ``1 - IoU`` are not on the same scale and the raw
            cosine term would almost never win.

    Returns:
        ``(n, m)`` fused cost. A pair failing both gates gets 1.0, which every caller's
        threshold rejects.
    """
    if motion.size == 0:
        return motion.astype(np.float32)
    admitted_motion = np.where(motion <= motion_gate, motion, 1.0)
    admitted_appearance = np.where(
        (appearance <= appearance_gate) & (motion <= motion_gate),
        appearance_weight * appearance,
        1.0,
    )
    return np.minimum(admitted_motion, admitted_appearance).astype(np.float32)


def direction_cost(
    directions: np.ndarray, origins: np.ndarray, detection_boxes: np.ndarray
) -> np.ndarray:
    """OC-SORT's observation-centric momentum: does this candidate lie the way we were going?

    ``(n, m)`` in ``[0, 1]``, the angle between the direction a track has been travelling and
    the direction from where it was last seen to the candidate detection, normalised by
    ``pi``. Zero means "straight ahead", one means "directly backwards", and a track with no
    direction yet scores zero for everything so the term is neutral rather than obstructive.

    This is the piece that resolves the crossing case without appearance. Two objects passing
    each other are geometrically interchangeable at the moment they overlap; they are *not*
    interchangeable in heading, and heading is measured between two real observations rather
    than read off a filter that has been extrapolating. That distinction is the whole of
    "observation-centric": a filter's velocity after a gap is a guess conditioned on its own
    earlier guesses, while the displacement between two detections is a measurement.

    Args:
        directions: ``(n, 2)`` unit vectors. ``(0, 0)`` for a track with no history.
        origins: ``(n, 4)`` xyxy of the observation the direction was measured *to*.
        detection_boxes: ``(m, 4)`` xyxy candidates.
    """
    dirs = np.atleast_2d(np.asarray(directions, dtype=np.float32))
    starts = np.atleast_2d(np.asarray(origins, dtype=np.float32))
    dets = np.atleast_2d(np.asarray(detection_boxes, dtype=np.float32))
    if dirs.size == 0 or dets.size == 0:
        return np.zeros((dirs.shape[0], dets.shape[0]), dtype=np.float32)

    start_centres = (starts[:, :2] + starts[:, 2:]) * 0.5
    det_centres = (dets[:, :2] + dets[:, 2:]) * 0.5
    offsets = det_centres[None, :, :] - start_centres[:, None, :]
    norms = np.linalg.norm(offsets, axis=2, keepdims=True)
    candidate = offsets / np.maximum(norms, 1e-6)

    cosine = np.sum(candidate * dirs[:, None, :], axis=2)
    cost = np.arccos(np.clip(cosine, -1.0, 1.0)) / np.pi

    # A track without a measured direction, or a detection sitting exactly on the origin,
    # carries no information. Scoring it zero keeps the term additive-neutral; scoring it
    # 0.5 (the mean) would quietly penalise every newly-born track.
    unknown = np.linalg.norm(dirs, axis=1) < 1e-6
    cost[unknown, :] = 0.0
    cost[norms[..., 0] < 1e-6] = 0.0
    return cost.astype(np.float32)
