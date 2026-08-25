"""SORT: Kalman prediction, IoU cost, one Hungarian assignment per frame."""

from __future__ import annotations

import numpy as np

from shipvision.errors import ConfigurationError
from shipvision.mot.association import associate
from shipvision.mot.backends.native import NativeTracker, require_extension, validate_lifecycle
from shipvision.mot.base import BaseTracker
from shipvision.mot.registry import TRACKERS
from shipvision.mot.trackers.sort.tracklet import new_pool
from shipvision.mot.trackers.sort.utils import association_cost
from shipvision.registry import NATIVE, PYTHON
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
    :class:`~shipvision.mot.trackers.bytetrack.tracker.ByteTrackTracker` addresses, and this
    class exists partly so that claim can be tested rather than asserted.

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
        super().__init__(new_pool(max_age=max_age, min_hits=min_hits))
        self._det_threshold = det_threshold
        self._max_cost = 1.0 - iou_threshold
        self._gate = gate

    def update(self, detections: Detections, *, image: np.ndarray | None = None) -> list[Track]:
        self.begin(detections)
        kept = detections.filter(min_score=self._det_threshold)
        boxes = kept.boxes

        rows = list(range(self.pool_size))
        if rows and len(kept):
            gating = self._pool.gating_distance(boxes, rows) if self._gate else None
            cost = association_cost(self._pool.boxes(), boxes, gating_distances=gating)
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


# -- the compiled implementation ------------------------------------------------------------
#
# Same algorithm, same registry name, `native` backend, and in this file rather than beside
# it: sort is one tracker with two implementations, and splitting them by language splits
# them by the least interesting thing about them. The readable class above is the
# specification; the parity tests assert the two agree.
#
# Only `deepsortv2` is what `motservice` actually runs — its README says "currently supports
# only deepsort". This one is kept because it is written and tested (V50), not because
# anything downstream selects it. New compiled work goes to what the services use.
#
# The extension probe and the marshalling are in `mot/backends/base.py`: not per algorithm,
# and five copies would be five places to disagree about what an empty detection set is.


@TRACKERS.register("sort", backend=NATIVE)
class NativeSortTracker(NativeTracker):
    """SORT with its per-frame work in C++. See
    :class:`~shipvision.mot.trackers.sort.tracker.SortTracker` for the algorithm.

    The keyword arguments are the numpy tracker's, exactly — same names, same defaults, same
    validation. That is not politeness: :mod:`shipvision.tune` validates a search space against
    whichever class the registry resolves, so a native tracker that quietly accepted a
    different set would let a study tune parameters the tracker it actually ran did not have.
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
        """
        Raises:
            BackendUnavailableError: the extension is not built, or is older than the trackers.
            ConfigurationError: an argument cannot work. Checked here as well as in the C++,
                because a caller who wrote ``iou_threshold=1.5`` deserves the same typed
                failure whichever backend the registry resolved.
        """
        native = require_extension("SortTracker")
        validate_lifecycle(max_age, min_hits)
        if not 0.0 <= det_threshold <= 1.0:
            raise ConfigurationError(f"det_threshold must be in [0, 1], got {det_threshold}")
        if not 0.0 < iou_threshold <= 1.0:
            raise ConfigurationError(f"iou_threshold must be in (0, 1], got {iou_threshold}")
        super().__init__(
            native.SortTracker(
                det_threshold=float(det_threshold),
                iou_threshold=float(iou_threshold),
                max_age=int(max_age),
                min_hits=int(min_hits),
                gate=bool(gate),
            )
        )

    def describe(self) -> str:
        return "SORT: Kalman + IoU + one assignment per frame, in C++"
