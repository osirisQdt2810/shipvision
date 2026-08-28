"""ByteTrack: associate the confident detections, then give the rest a second chance.

Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", ECCV
2022. Written from the paper.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from shipvision.errors import ConfigurationError
from shipvision.mot.association import associate_subset
from shipvision.mot.backends.native import NativeTracker, require_extension, validate_lifecycle
from shipvision.mot.base import BaseTracker
from shipvision.mot.registry import TRACKERS
from shipvision.mot.trackers.bytetrack.tracklet import new_pool
from shipvision.mot.trackers.bytetrack.utils import (
    high_score_cost,
    low_score_cost,
    split_by_score,
)
from shipvision.registry import NATIVE, PYTHON
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
            new_pool(max_age=max_age, min_hits=min_hits, embedding_momentum=embedding_momentum)
        )
        self._track_threshold = track_threshold
        self._low_threshold = low_threshold
        self._max_cost = 1.0 - match_threshold
        self._second_max_cost = 1.0 - second_match_threshold
        self._gate = gate

    def update(self, detections: Detections, *, image: np.ndarray | None = None) -> list[Track]:
        self.begin(detections)
        self._compensate(image)
        high, low = split_by_score(
            detections,
            low_threshold=self._low_threshold,
            track_threshold=self._track_threshold,
        )

        rows = list(range(self.pool_size))
        matches, unmatched_rows, unmatched_high = self._associate(
            lambda r, c: self._first_cost(r, c, high),
            self._max_cost,
            rows,
            list(range(len(high))),
            high,
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
        second_matches, still_unmatched, _ = self._associate(
            lambda r, c: self._second_cost(r, c, low),
            self._second_max_cost,
            eligible,
            list(range(len(low))),
            low,
        )
        self._pool.apply_matches(second_matches, low)

        missed = still_unmatched + [row for row in unmatched_rows if row not in eligible]
        self._pool.mark_missed(missed)

        # Only high-score detections may start a track. This is the asymmetry that keeps a
        # low-confidence false positive from ever becoming an identity.
        self._pool.spawn(high, unmatched_high)
        self._pool.sweep()
        return self._pool.output()

    def _associate(
        self,
        build_cost: Callable[[Sequence[int], Sequence[int]], np.ndarray],
        max_cost: float,
        rows: Sequence[int],
        columns: Sequence[int],
        detections: Sequence[Detection],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """How one association stage is solved, so a subclass can change it once for both.

        ByteTrack and BoT-SORT hand the whole sub-problem to
        :func:`~shipvision.mot.association.solver.associate_subset`. McByte decides part of it
        first; a hook rather than two overridden stages because the rule applies identically
        to both, and restating the loop twice to say so would make it a fork.

        ``detections`` are this stage's own, in the caller's column order, for a subclass that
        needs the boxes rather than only the cost they produced.
        """
        return associate_subset(build_cost, max_cost, rows, columns)

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
        return high_score_cost(
            self._pool.boxes()[rows],
            boxes,
            scores,
            gating_distances=(self._pool.gating_distance(boxes, rows) if self._gate else None),
        )

    def _second_cost(
        self, rows: list[int], columns: list[int], low: list[Detection]
    ) -> np.ndarray:
        boxes = np.stack([low[c].box for c in columns])
        return low_score_cost(
            self._pool.boxes()[rows],
            boxes,
            gating_distances=(self._pool.gating_distance(boxes, rows) if self._gate else None),
        )

    def describe(self) -> str:
        return (
            "ByteTrack: high-score association, then a second pass over the low-score leftovers"
        )


# -- the compiled implementation ------------------------------------------------------------
#
# Same algorithm, same registry name, `native` backend, and in this file rather than beside
# it: bytetrack is one tracker with two implementations, and splitting them by language splits
# them by the least interesting thing about them. The readable class above is the
# specification; the parity tests assert the two agree.
#
# Only `deepsortv2` is what `motservice` actually runs — its README says "currently supports
# only deepsort". This one is kept because it is written and tested (V50), not because
# anything downstream selects it. New compiled work goes to what the services use.
#
# The extension probe and the marshalling are in `mot/backends/base.py`: not per algorithm,
# and five copies would be five places to disagree about what an empty detection set is.


@TRACKERS.register("bytetrack", backend=NATIVE)
class NativeByteTrackTracker(NativeTracker):
    """ByteTrack with its two association stages in C++. See
    :class:`~shipvision.mot.trackers.bytetrack.tracker.ByteTrackTracker` for the algorithm.

    ``embedding_momentum`` is handled on this side rather than passed to the C++ session, for
    the reason in the module docstring — but it is still a constructor keyword with the same
    name and default, because the registry may hand this class to anything that was written
    against the numpy one.
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
        """
        Raises:
            BackendUnavailableError: the extension is not built, or is older than the trackers.
            ConfigurationError: the two score thresholds are the wrong way round, which would
                leave the high tier empty and mean no track is ever born.
        """
        native = require_extension("ByteTrackTracker")
        validate_lifecycle(max_age, min_hits)
        if not low_threshold < track_threshold <= 1.0:
            raise ConfigurationError(
                f"need 0 <= low_threshold ({low_threshold}) < track_threshold "
                f"({track_threshold}) <= 1"
            )
        super().__init__(
            native.ByteTrackTracker(
                track_threshold=float(track_threshold),
                low_threshold=float(low_threshold),
                match_threshold=float(match_threshold),
                second_match_threshold=float(second_match_threshold),
                max_age=int(max_age),
                min_hits=int(min_hits),
                gate=bool(gate),
            ),
            embedding_momentum=embedding_momentum,
        )

    def describe(self) -> str:
        return (
            "ByteTrack: high-score association, then a second pass over the low-score "
            "leftovers, in C++"
        )
