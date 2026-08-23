"""OC-SORT: stop trusting the filter's extrapolation, trust the last thing you saw.

Cao et al., "Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking",
CVPR 2023. Written from the paper.

SORT's Kalman filter is *estimation-centric*: while a track is unobserved it keeps producing
state from its own previous state, and every frame of that compounds. Three consequences, and
OC-SORT is three named fixes for them:

**ORU** — observation-centric re-update. When a gapped track is re-found, the single distant
measurement corrects the *position* but also drives an enormous *velocity* correction,
because the covariance has been inflating for the whole gap. The next prediction overshoots,
misses, and the track is lost again — permanently, this time, and a new identity is born.
Rewind to the last real observation, interpolate the measurements the detector would have
produced, and run the filter through them instead.

**OCR** — observation-centric recovery. A second association that ignores the prediction
entirely and matches unmatched tracks against their *last observation*. This is what catches
the object that stopped moving while it was hidden: the filter carried the old velocity and
its prediction has walked off, while the object is still standing where it was last seen.

**OCM** — observation-centric momentum. A direction-consistency term in the cost, measured
between two real observations rather than read off the filter. Displacement between two
detections is a measurement; a filter's velocity after a gap is a guess conditioned on its own
earlier guesses.

Deliberately not implemented here: the paper's optional BYTE-style low-score second
association (that is what :class:`~shipvision.tracking.trackers.bytetrack.ByteTrackTracker` is for, and
combining the two is a fourth tracker, not a flag), and the "OC-SORT + appearance" variants
from later papers.
"""

from __future__ import annotations

import numpy as np

from shipvision.errors import ConfigurationError
from shipvision.registry import PYTHON
from shipvision.tracking.association import (
    associate_subset,
    direction_cost,
    gate_cost,
    iou_cost,
)
from shipvision.tracking.base import TRACKERS, BaseTracker
from shipvision.tracking.motion.kalman import CHI2_INV_95_4DOF
from shipvision.tracking.pool import TrackPool
from shipvision.types import Detection, Detections, Track, TrackState

__all__ = ["OcSortTracker"]


@TRACKERS.register("ocsort", backend=PYTHON, aliases=("oc", "oc_sort"))
class OcSortTracker(BaseTracker):
    """SORT's association, plus the three observation-centric corrections.

    Each of the three can be switched off independently. That is not configurability for its
    own sake: it is the only way to say which of them is earning its keep on a given camera,
    and the tests use it to prove that the ORU scenario is won by ORU rather than by one of
    the others quietly rescuing it.

    Args:
        det_threshold: detections below this are discarded. OC-SORT is a single-threshold
            tracker; the low-score second pass is ByteTrack's idea and lives there.
        iou_threshold: minimum overlap for the primary association.
        recovery_iou_threshold: minimum overlap for OCR. Stricter than the primary threshold
            by default, because OCR is deliberately matching against a *stale* box and the
            geometry has to be convincing to make up for it.
        delta_t: how many frames back the momentum term measures heading over. One frame of
            displacement at 20 fps is mostly detector jitter, so the paper measures over a
            span; three is its default.
        momentum_weight: how much the direction term counts against IoU. Small on purpose —
            it is a tie-breaker between geometrically plausible candidates, not a cost in its
            own right. Setting it high makes the tracker refuse to follow anything that
            changes direction.
        max_age: frames a confirmed track survives without a match. OC-SORT's whole point is
            that this can be generous, so the default is longer than SORT's.
        min_hits: matches before a track is published.
        gate: apply the Mahalanobis gate to the primary association. Never to OCR — the
            filter's opinion is exactly what OCR is overruling.
        re_update: enable ORU.
        recover: enable OCR.
    """

    def __init__(
        self,
        *,
        det_threshold: float = 0.5,
        iou_threshold: float = 0.3,
        recovery_iou_threshold: float = 0.5,
        delta_t: int = 3,
        momentum_weight: float = 0.2,
        max_age: int = 30,
        min_hits: int = 3,
        gate: bool = True,
        re_update: bool = True,
        recover: bool = True,
    ) -> None:
        if not 0.0 <= det_threshold <= 1.0:
            raise ConfigurationError(f"det_threshold must be in [0, 1], got {det_threshold}")
        if not 0.0 < iou_threshold <= 1.0:
            raise ConfigurationError(f"iou_threshold must be in (0, 1], got {iou_threshold}")
        if not 0.0 < recovery_iou_threshold <= 1.0:
            raise ConfigurationError(
                f"recovery_iou_threshold must be in (0, 1], got {recovery_iou_threshold}"
            )
        if delta_t < 1:
            raise ConfigurationError(f"delta_t must be >= 1, got {delta_t}")
        if not 0.0 <= momentum_weight <= 1.0:
            raise ConfigurationError(
                f"momentum_weight must be in [0, 1], got {momentum_weight}"
            )
        super().__init__(
            TrackPool(
                max_age=max_age,
                min_hits=min_hits,
                # delta_t + 1 observations is the smallest ring that can measure a heading
                # over delta_t frames. Bounded, because this process runs for weeks.
                observation_history=delta_t + 1,
                re_update=re_update,
            )
        )
        self._det_threshold = det_threshold
        self._max_cost = 1.0 - iou_threshold
        self._recovery_max_cost = 1.0 - recovery_iou_threshold
        self._delta_t = delta_t
        self._momentum_weight = momentum_weight
        self._gate = gate
        self._recover = recover

    def update(self, detections: Detections, *, image: np.ndarray | None = None) -> list[Track]:
        self.begin(detections)
        kept = detections.filter(min_score=self._det_threshold).items

        rows = list(range(self.pool_size))
        matches, unmatched_rows, unmatched_cols = associate_subset(
            lambda r, c: self._primary_cost(r, c, kept),
            self._max_cost,
            rows,
            list(range(len(kept))),
        )
        self._pool.apply_matches(matches, kept)

        if self._recover:
            # OCR sees only tracks that have earned trust. Offering a tentative track a
            # stale-box match is the same "two weak signals agreeing" mistake ByteTrack's
            # second stage avoids, and here it would additionally resurrect noise.
            eligible = [
                row
                for row in unmatched_rows
                if self._pool.tracks[row].state in (TrackState.CONFIRMED, TrackState.LOST)
            ]
            recovered, still_unmatched, unmatched_cols = associate_subset(
                lambda r, c: self._recovery_cost(r, c, kept),
                self._recovery_max_cost,
                eligible,
                unmatched_cols,
            )
            self._pool.apply_matches(recovered, kept)
            unmatched_rows = still_unmatched + [r for r in unmatched_rows if r not in eligible]

        self._pool.mark_missed(unmatched_rows)
        self._pool.spawn(kept, unmatched_cols)
        self._pool.sweep()
        return self._pool.output()

    def _primary_cost(
        self, rows: list[int], columns: list[int], kept: list[Detection]
    ) -> np.ndarray:
        """IoU against the prediction, nudged by whether the candidate is *ahead*."""
        boxes = np.stack([kept[c].box for c in columns])
        cost = iou_cost(self._pool.boxes()[rows], boxes)
        if self._momentum_weight > 0.0:
            headings = self._pool.directions(self._delta_t)[rows]
            origins = self._pool.observed_boxes()[rows]
            cost = cost + self._momentum_weight * direction_cost(headings, origins, boxes)
        if self._gate:
            cost = gate_cost(cost, self._pool.gating_distance(boxes, rows), CHI2_INV_95_4DOF)
        return cost

    def _recovery_cost(
        self, rows: list[int], columns: list[int], kept: list[Detection]
    ) -> np.ndarray:
        """IoU against the **last observation**, with no motion model and no gate.

        Both omissions are the point. The prediction is what failed in the primary stage, so
        reusing it would just fail again; and gating on a filter whose covariance grew through
        the gap either admits everything or vetoes the one honest candidate.
        """
        boxes = np.stack([kept[c].box for c in columns])
        return iou_cost(self._pool.observed_boxes()[rows], boxes)

    def describe(self) -> str:
        return "OC-SORT: observation-centric momentum, recovery and re-update over SORT"
