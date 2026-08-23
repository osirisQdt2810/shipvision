"""SORT: Kalman prediction, IoU cost, one Hungarian assignment per frame."""

from __future__ import annotations

import numpy as np

from shipvision.errors import ConfigurationError
from shipvision.registry import PYTHON
from shipvision.tracking.association import associate, gate_cost, iou_cost
from shipvision.tracking.base import TRACKERS, BaseTracker
from shipvision.tracking.motion.kalman import CHI2_INV_95_4DOF
from shipvision.tracking.pool import TrackPool
from shipvision.types import Detections, Track

__all__ = ["SortTracker"]


@TRACKERS.register("sort", backend=PYTHON)
class SortTracker(BaseTracker):
    """The baseline every other tracker here is measured against.

    Predict where each track went, score every ``(track, detection)`` pair by IoU, solve the
    assignment once, and age out whatever did not match. It is a hundred lines and it is
    remarkably hard to beat when the detector is good and the frame rate is high.

    Where it fails is instructive, because it is the same place every simple tracker fails: a
    detection that drops below the confidence threshold for a few frames — a person walking
    behind a pillar — takes its track with it. That is precisely what
    :class:`~shipvision.tracking.trackers.bytetrack.ByteTrackTracker` addresses, and this class exists
    partly so that claim can be tested rather than asserted.

    Args:
        det_threshold: detections below this confidence are discarded outright. Not optional
            in practice: without it the tracker associates and then *publishes* every noise
            box the detector emits, and a tracker that invents identities is worse than no
            tracker. It is also the precise limitation ByteTrack removes.
        iou_threshold: an association needs at least this much overlap. Higher is stricter:
            fewer ID switches, more fragmented tracks.
        max_age: frames a confirmed track survives without a match before it is dropped.
        min_hits: matches before a track is published. Withholding a track for a frame or two
            is what keeps a one-frame false positive from becoming an identity downstream.
        gate: reject associations the motion model calls impossible, before the assignment
            sees them. Off means one crowded frame can hand an identity to the wrong object.
    """

    def __init__(
        self,
        *,
        det_threshold: float = 0.5,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
        gate: bool = True,
    ) -> None:
        if not 0.0 <= det_threshold <= 1.0:
            raise ConfigurationError(f"det_threshold must be in [0, 1], got {det_threshold}")
        if not 0.0 < iou_threshold <= 1.0:
            raise ConfigurationError(f"iou_threshold must be in (0, 1], got {iou_threshold}")
        super().__init__(TrackPool(max_age=max_age, min_hits=min_hits))
        self._det_threshold = det_threshold
        self._max_cost = 1.0 - iou_threshold
        self._gate = gate

    def update(self, detections: Detections, *, image: np.ndarray | None = None) -> list[Track]:
        self.begin(detections)
        kept = detections.filter(min_score=self._det_threshold)
        boxes = kept.boxes

        rows = list(range(self.pool_size))
        if rows and len(kept):
            cost = iou_cost(self._pool.boxes(), boxes)
            if self._gate:
                cost = gate_cost(
                    cost, self._pool.gating_distance(boxes, rows), CHI2_INV_95_4DOF
                )
            matches, unmatched_rows, unmatched_cols = associate(cost, self._max_cost)
        else:
            matches, unmatched_rows, unmatched_cols = [], rows, list(range(len(kept)))

        self._pool.apply_matches(matches, kept.items)
        self._pool.mark_missed(unmatched_rows)
        self._pool.spawn(kept.items, unmatched_cols)
        self._pool.sweep()
        return self._pool.output()

    def describe(self) -> str:
        return "SORT: Kalman + IoU + one Hungarian assignment per frame"
