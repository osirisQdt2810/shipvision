"""SORT: the baseline, and the properties that make it a *useful* baseline.

Its behaviour is only interesting where it differs from the four trackers built on it, and
those differences are asserted in the other files. What is here is the two things SORT owns —
the hard confidence threshold and the single-assignment structure — plus the failure the
others exist to fix, stated once so nobody has to rediscover it.
"""

from __future__ import annotations

import pytest

from shipvision.errors import ConfigurationError
from shipvision.mot import TRACKERS
from tests.mot.conftest import all_ids, det, drive, frame


class TestTheConfidenceThreshold:
    """SORT's one filter, and the reason it is not optional."""

    def test_a_detection_below_the_threshold_is_invisible(self) -> None:
        """Not "matched more weakly" — discarded before the cost matrix exists. That is the
        precise limitation ByteTrack removes, so it is worth asserting rather than assuming.
        """
        tracker = TRACKERS.build("sort", min_hits=1, det_threshold=0.5)
        published = drive(tracker, [[det(400, 600, score=0.49)] for _ in range(6)])
        assert all(step == [] for step in published)
        assert tracker.pool_size == 0

    def test_dropping_below_the_threshold_mid_track_is_the_same_as_disappearing(self) -> None:
        """The failure ByteTrack was written for, recorded here as SORT's actual behaviour.

        The detector never stopped seeing the object — it saw it at 0.3. SORT cannot tell that
        apart from an empty frame, so the track ages out and the object returns as somebody
        new.
        """
        tracker = TRACKERS.build("sort", min_hits=2, max_age=3, det_threshold=0.5)
        before = all_ids(drive(tracker, [[det(100 + 5 * f, 200, score=0.9)] for f in range(6)]))
        drive(tracker, [[det(130 + 5 * f, 200, score=0.3)] for f in range(6)], start=6)
        after = all_ids(
            drive(tracker, [[det(160 + 5 * f, 200, score=0.9)] for f in range(6)], start=12)
        )
        assert before and after
        assert before.isdisjoint(after)

    def test_a_threshold_outside_zero_one_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="det_threshold"):
            TRACKERS.build("sort", det_threshold=1.5)


class TestTheSingleAssignment:
    """One solve per frame, which means one detection per track and one track per detection."""

    def test_two_detections_on_one_object_produce_two_tracks_not_one(self) -> None:
        """A duplicate detection the NMS missed. The assignment is one-to-one, so the second
        box cannot also feed the existing track — it starts its own, which is the honest
        outcome and is visible downstream rather than silent.
        """
        tracker = TRACKERS.build("sort", min_hits=1, iou_threshold=0.3)
        drive(tracker, [[det(400, 600)]])
        tracker.update(frame([det(400, 600), det(404, 600)], 1))
        assert tracker.pool_size == 2

    def test_the_globally_cheapest_pairing_wins_not_the_first_plausible_one(self) -> None:
        """Two objects, and each one's *own* detection is not its nearest. Greedy matching
        takes the near pair first and forces the far one; the solver takes the total.
        """
        tracker = TRACKERS.build("sort", min_hits=1, iou_threshold=0.1)
        drive(tracker, [[det(300, 500, w=100), det(420, 500, w=100)]])
        first = sorted(tracker.tracks, key=lambda t: float(t.box[0]))
        left_id, right_id = first[0].track_id, first[1].track_id

        # Both shift right by 40: the left object now overlaps where the right one was.
        tracks = tracker.update(frame([det(340, 500, w=100), det(460, 500, w=100)], 1))
        by_x = sorted(tracks, key=lambda t: float(t.box[0]))
        assert [t.track_id for t in by_x] == [left_id, right_id]


class TestTheMotionGate:
    """Off is a real option, so what it buys has to be measurable."""

    def test_gating_refuses_a_jump_the_filter_calls_impossible(self) -> None:
        """One crowded frame can otherwise hand an identity to the wrong object, and an ID
        switch is not recoverable the way a missed frame is.
        """
        established = 500.0

        def run(*, gate: bool) -> bool:
            tracker = TRACKERS.build(
                "sort", min_hits=1, max_age=10, iou_threshold=0.1, gate=gate
            )
            drive(tracker, [[det(established, 500, w=200, h=140)] for _ in range(8)])
            original = tracker.tracks[0].track_id
            tracks = tracker.update(frame([det(established + 110.0, 500, w=200, h=140)], 8))
            return bool(tracks) and tracks[0].track_id == original

        assert not run(gate=True), "the gate let through a 110 px jump on a settled track"
        assert run(gate=False), (
            "without the gate the jump should be accepted on IoU alone; if it is not, this "
            "test is measuring the IoU threshold rather than the gate"
        )
