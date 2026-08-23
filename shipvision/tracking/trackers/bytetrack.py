"""ByteTrack: associate the confident detections, then give the rest a second chance.

Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", ECCV
2022. Written from the paper.
"""

from __future__ import annotations

import numpy as np

from shipvision.errors import ConfigurationError
from shipvision.registry import PYTHON
from shipvision.tracking.association import (
    associate_subset,
    fuse_score,
    gate_cost,
    iou_cost,
)
from shipvision.tracking.base import TRACKERS, BaseTracker
from shipvision.tracking.motion.kalman import CHI2_INV_95_4DOF
from shipvision.tracking.pool import TrackPool
from shipvision.types import Detection, Detections, Track, TrackState

__all__ = ["ByteTrackTracker"]


@TRACKERS.register("bytetrack", backend=PYTHON, aliases=("byte",))
class ByteTrackTracker(BaseTracker):
    """Two-stage association, and the second stage is the whole idea.

    Every tracker throws away low-confidence detections, because most of them are noise.
    ByteTrack's observation is that *some* of them are not: when a tracked person walks behind
    a pillar, the detector does not stop seeing them — it sees them at 0.3 instead of 0.9.
    Discarding that box loses the track; matching it against an *existing* track keeps the
    identity through the occlusion.

    The asymmetry is what makes it safe. High-score detections may start new tracks; low-score
    ones may only continue existing ones. So a low-confidence false positive can never create
    an identity, and the cost of being wrong is one frame of a slightly misplaced box rather
    than a spurious object.

    Stage one associates confirmed and tentative tracks with the high-score detections, with
    the detector's confidence folded into the cost. Stage two takes the tracks that found
    nothing and offers them the low-score leftovers on IoU alone — appearance and confidence
    are both unreliable at that score, and overlap is the only signal left worth trusting.

    Args:
        track_threshold: at or above this a detection is "high score" and may start a track.
        low_threshold: below this a detection is discarded entirely.
        match_threshold: minimum IoU for stage one.
        second_match_threshold: minimum IoU for stage two. Deliberately stricter — the
            evidence is weaker, so the geometry has to be better.
        max_age: frames a confirmed track survives without a match before it is dropped.
        min_hits: matches before a track is published.
        gate: apply the Mahalanobis gate before each assignment.
        embedding_momentum: EMA retention for a track's appearance vector. ByteTrack's own
            association is purely geometric, but a track still *carries* an appearance vector
            when the detections have one, because the cross-camera tier downstream needs it —
            and averaging it here, once, is cheaper than re-deriving it there.
    """

    def __init__(
        self,
        *,
        track_threshold: float = 0.5,
        low_threshold: float = 0.1,
        match_threshold: float = 0.2,
        second_match_threshold: float = 0.5,
        max_age: int = 30,
        min_hits: int = 3,
        gate: bool = True,
        embedding_momentum: float = 0.9,
    ) -> None:
        if not low_threshold < track_threshold <= 1.0:
            raise ConfigurationError(
                f"need 0 <= low_threshold ({low_threshold}) < track_threshold "
                f"({track_threshold}) <= 1"
            )
        super().__init__(
            TrackPool(max_age=max_age, min_hits=min_hits, embedding_momentum=embedding_momentum)
        )
        self._track_threshold = track_threshold
        self._low_threshold = low_threshold
        self._max_cost = 1.0 - match_threshold
        self._second_max_cost = 1.0 - second_match_threshold
        self._gate = gate

    def update(self, detections: Detections, *, image: np.ndarray | None = None) -> list[Track]:
        self.begin(detections)
        self._compensate(image)
        high = [d for d in detections if d.score >= self._track_threshold]
        low = [d for d in detections if self._low_threshold <= d.score < self._track_threshold]

        rows = list(range(self.pool_size))
        matches, unmatched_rows, unmatched_high = associate_subset(
            lambda r, c: self._first_cost(r, c, high),
            self._max_cost,
            rows,
            list(range(len(high))),
        )
        self._pool.apply_matches(matches, high)

        # Stage two, over CONFIRMED and LOST tracks only. A tentative track rescued by a
        # 0.3-confidence box is two weak pieces of evidence agreeing with each other, which
        # is not evidence — and it is how a noise track becomes a published identity.
        eligible = [
            row
            for row in unmatched_rows
            if self._pool.tracks[row].state in (TrackState.CONFIRMED, TrackState.LOST)
        ]
        second_matches, still_unmatched, _ = associate_subset(
            lambda r, c: self._second_cost(r, c, low),
            self._second_max_cost,
            eligible,
            list(range(len(low))),
        )
        self._pool.apply_matches(second_matches, low)

        missed = still_unmatched + [row for row in unmatched_rows if row not in eligible]
        self._pool.mark_missed(missed)

        # Only high-score detections may start a track. This is the asymmetry that keeps a
        # low-confidence false positive from ever becoming an identity.
        self._pool.spawn(high, unmatched_high)
        self._pool.sweep()
        return self._pool.output()

    def _compensate(self, image: np.ndarray | None) -> None:
        """Warp the predictions into this frame's coordinates. ByteTrack does nothing here.

        The extension point exists because BoT-SORT's first contribution is *exactly* this
        step and nothing else, so a subclass that fills it in is a faithful reading of the
        paper rather than a fork of the association logic. ByteTrack assumes a bolted-down
        camera, which is the right assumption for most of a fifty-camera installation and the
        wrong one for a PTZ head.
        """

    def _first_cost(
        self, rows: list[int], columns: list[int], high: list[Detection]
    ) -> np.ndarray:
        boxes = np.stack([high[c].box for c in columns])
        scores = np.array([high[c].score for c in columns], dtype=np.float32)
        cost = fuse_score(iou_cost(self._pool.boxes()[rows], boxes), scores)
        if self._gate:
            cost = gate_cost(cost, self._pool.gating_distance(boxes, rows), CHI2_INV_95_4DOF)
        return cost

    def _second_cost(
        self, rows: list[int], columns: list[int], low: list[Detection]
    ) -> np.ndarray:
        boxes = np.stack([low[c].box for c in columns])
        cost = iou_cost(self._pool.boxes()[rows], boxes)
        if self._gate:
            cost = gate_cost(cost, self._pool.gating_distance(boxes, rows), CHI2_INV_95_4DOF)
        return cost

    def describe(self) -> str:
        return (
            "ByteTrack: high-score association, then a second pass over the low-score leftovers"
        )
