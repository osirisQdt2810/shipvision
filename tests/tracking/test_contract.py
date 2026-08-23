"""What every tracker must do, whatever algorithm it is. Parametrised over the registry.

These are the invariants a new tracker inherits by being registered. That is deliberate: a
tracker added tomorrow gets held to them without anybody remembering to write them again, and
the failure mode this protects against is real — a new tracker that publishes on frame one, or
that never frees a dead track, looks fine on its own scenario test and breaks a fifty-camera
process a week later.

Every test here calls ``update(detections)`` with the tagged container and nothing else, so it
also asserts that the container alone is a sufficient input for all five.

The parametrisation is applied to the whole class rather than to each method, which is the
same thing to pytest and keeps the intent — *this claim holds for every tracker* — in one
place.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipvision.errors import ConfigurationError, TrackingError
from shipvision.tracking import TRACKERS
from shipvision.types import Detections, FrameTag, TrackState
from tests.tracking.conftest import all_ids, det, drive, frame, ids

NAMES = TRACKERS.names()


@pytest.mark.parametrize("name", NAMES)
class TestBasicTracking:
    """The three things a tracker is for: one identity per object, no identity before it is
    earned, and no swap when two objects meet.
    """

    def test_a_stationary_object_keeps_one_identity(self, name: str) -> None:
        tracker = TRACKERS.build(name, min_hits=2)
        seen = all_ids(drive(tracker, [[det(100, 200)] for _ in range(12)]))
        assert (
            len(seen) == 1
        ), f"{name} produced {len(seen)} identities for one stationary object"

    def test_a_track_is_withheld_until_it_has_earned_confirmation(self, name: str) -> None:
        """A single detection is not evidence of an object. Publishing on frame one means
        emitting an identity for every false positive, and downstream cannot tell."""
        tracker = TRACKERS.build(name, min_hits=3)
        published = drive(tracker, [[det(100, 200)], [det(101, 200)], [det(102, 200)]])
        assert published[0] == []
        assert published[1] == []
        assert len(published[2]) == 1

    def test_two_objects_crossing_do_not_swap_identities(self, name: str) -> None:
        """The canonical tracking failure.

        Two objects approach, pass within a few pixels of each other, and separate. Association
        on position alone is ambiguous at the crossing point; the motion model is what resolves
        it, because each object continues in the direction it was already going.
        """
        tracker = TRACKERS.build(name, min_hits=2)
        left_id: int | None = None
        right_id: int | None = None

        for frame_id in range(24):
            left_x = 60 + frame_id * 8  # moving right
            right_x = 244 - frame_id * 8  # moving left
            tracks = tracker.update(frame([det(left_x, 200), det(right_x, 200)], frame_id))

            if frame_id < 6 or not tracks:
                continue
            by_x = sorted(tracks, key=lambda t: float(t.box[0]))
            if frame_id < 10:  # before they cross, learn which id is which
                left_id, right_id = by_x[0].track_id, by_x[-1].track_id
            elif frame_id > 16 and len(tracks) == 2:  # after they separate
                # They have swapped sides, so the LEFTMOST track is now the one that started
                # on the right. Identities must travel with the objects, not with the side.
                assert (
                    by_x[0].track_id == right_id
                ), f"{name} swapped identities at the crossing"
                assert by_x[-1].track_id == left_id


@pytest.mark.parametrize("name", NAMES)
class TestLifecycle:
    """Ageing, dying, and releasing memory. A tracker that merely stops *publishing* a dead
    track leaks one row of a dense Kalman array per object that ever appeared, and a process
    here runs for weeks.
    """

    def test_a_track_survives_a_gap_and_then_dies(self, name: str) -> None:
        tracker = TRACKERS.build(name, min_hits=2, max_age=5)
        drive(tracker, [[det(100 + f * 4, 200)] for f in range(6)])
        established = all_ids(drive(tracker, [[det(124, 200)]], start=6))
        assert len(established) == 1

        drive(tracker, [[] for _ in range(7, 20)], start=7)  # far longer than max_age

        # min_hits withholds the new track for a frame or two; drive it until it appears.
        revived = drive(tracker, [[det(140 + f * 4, 200)] for f in range(5)], start=20)
        assert revived[-1], "a detection after the gap should eventually produce a track"
        assert all_ids(revived).isdisjoint(established), (
            f"{name} revived a dead identity; a re-appearing object must get a new id, and "
            f"re-identifying it across the gap is the multi-camera tier's job, not this one's"
        )

    def test_an_empty_frame_is_information_not_a_no_op(self, name: str) -> None:
        """Nothing detected still ages every track. A tracker that skips the update keeps dead
        objects alive forever."""
        tracker = TRACKERS.build(name, min_hits=1, max_age=2)
        drive(tracker, [[det(100, 200)], [det(100, 200)]])
        published = drive(tracker, [[] for _ in range(7)], start=2)
        assert published[-1] == []

    def test_the_pool_is_empty_after_max_age_frames_of_nothing(self, name: str) -> None:
        """Publication stopping is not the same as memory being released.

        The leak is invisible in the output, which is exactly what makes asserting it here
        worth a separate property rather than trusting the previous test to cover it.
        """
        tracker = TRACKERS.build(name, min_hits=1, max_age=3)
        drive(tracker, [[det(100, 200)], [det(102, 200)], [det(104, 200)]])
        assert tracker.pool_size == 1

        drive(tracker, [[] for _ in range(6)], start=3)
        assert (
            tracker.pool_size == 0
        ), f"{name} still holds {tracker.pool_size} track(s) six frames after max_age=3 expired"

    def test_states_move_in_one_direction(self, name: str) -> None:
        """A confirmed track that stops being seen becomes LOST, then REMOVED, and is never
        confirmed again without a detection.

        The original form of this test asserted over ``update()``'s return value, which is
        empty once the track is lost — so it passed vacuously. Reading the pool instead is
        what makes it say anything.
        """
        tracker = TRACKERS.build(name, min_hits=2, max_age=2)
        drive(tracker, [[det(100, 200)], [det(100, 200)]])
        assert [t.state for t in tracker.tracks] == [TrackState.CONFIRMED]

        drive(tracker, [[]], start=2)
        assert [t.state for t in tracker.tracks] == [TrackState.LOST]

        drive(tracker, [[], []], start=3)
        assert tracker.tracks == []

    def test_a_lost_track_is_never_published(self, name: str) -> None:
        """A LOST track's box is a prediction no detector saw. Emitting it as an observation is
        how a phantom object drifts across a scene."""
        tracker = TRACKERS.build(name, min_hits=1, max_age=10)
        drive(tracker, [[det(100 + 5 * f, 200)] for f in range(4)])
        published = drive(tracker, [[] for _ in range(5)], start=4)
        assert ids(published) == [set()] * 5
        assert tracker.pool_size == 1, "the track should still be alive, just unpublished"

    def test_reset_forgets_everything(self, name: str) -> None:
        tracker = TRACKERS.build(name, min_hits=1)
        first = all_ids(drive(tracker, [[det(100, 200)]]))
        tracker.reset()
        second = all_ids(drive(tracker, [[det(100, 200)]]))
        assert first and second and first.isdisjoint(second)
        assert tracker.pool_size == 1


@pytest.mark.parametrize("name", NAMES)
class TestIdentity:
    """Track ids are process-wide and single-use. Two cameras' output meets downstream, and
    making that meeting possible is the entire point of the cross-camera tier.
    """

    def test_ids_are_never_reused_within_one_tracker(self, name: str) -> None:
        """Ten objects appearing and dying one after another must yield ten identities.

        Recycling a dead id is the worst kind of correct-looking bug: downstream stitches two
        unrelated trajectories into one, and the cross-camera tier then propagates that
        identity everywhere.
        """
        tracker = TRACKERS.build(name, min_hits=1, max_age=1)
        seen: set[int] = set()
        frame_id = 0
        for cycle in range(10):
            for _ in range(2):
                seen |= all_ids(drive(tracker, [[det(100 + 300 * cycle, 200)]], start=frame_id))
                frame_id += 1
            for _ in range(4):  # let it die before the next object appears
                drive(tracker, [[]], start=frame_id)
                frame_id += 1
        assert len(seen) == 10, f"{name} produced {len(seen)} ids for 10 successive objects"

    def test_ids_are_unique_across_tracker_instances(self, name: str) -> None:
        """Two cameras' output meets downstream. If both call their first object "1", the
        multi-camera tier cannot tell them apart — and that is the whole tier."""
        a = TRACKERS.build(name, min_hits=1)
        b = TRACKERS.build(name, min_hits=1)
        ids_a = all_ids(drive(a, [[det(100, 200)]], camera="cam-a"))
        ids_b = all_ids(drive(b, [[det(100, 200)]], camera="cam-b"))
        assert ids_a and ids_b and ids_a.isdisjoint(ids_b)


@pytest.mark.parametrize("name", NAMES)
class TestTheTag:
    """``(camera_id, frame_id)`` travels inside the input so the output cannot disagree with
    it, and the two ways a caller can break that are refused loudly rather than absorbed.
    """

    def test_every_published_track_carries_the_input_tag(self, name: str) -> None:
        """A mis-tagged result is worse than a dropped one.

        Dropped is counted; mis-tagged is a real-looking detection on a camera where nothing
        happened. This asserts nobody rebuilt the tag from a counter on the way out — the
        frame ids here are deliberately non-consecutive, because a dropped frame is normal.
        """
        tracker = TRACKERS.build(name, min_hits=1)
        for frame_id in (0, 1, 5, 9):
            tag = FrameTag(camera_id="berth-7", frame_id=frame_id, timestamp=1.5 * frame_id)
            published = tracker.update(
                Detections(tag=tag, items=[det(100 + frame_id, 200)], height=720, width=1280)
            )
            assert published, f"{name} published nothing on frame {frame_id}"
            assert all(t.tag == tag for t in published)
            assert all(t.camera_id == "berth-7" for t in published)

    def test_a_second_camera_on_one_instance_is_refused(self, name: str) -> None:
        """The failure this replaces was silent, which is why it raises rather than degrades.

        One tracker shared between two cameras does not track worse — it associates camera A's
        objects with camera B's and hands the result an identity, which downstream reads as a
        real detection somewhere nothing happened.
        """
        tracker = TRACKERS.build(name, min_hits=1)
        drive(tracker, [[det(100, 200)]], camera="cam-a")
        with pytest.raises(TrackingError, match="one camera"):
            drive(tracker, [[det(100, 200)]], camera="cam-b", start=1)

    def test_a_replayed_frame_is_refused(self, name: str) -> None:
        """A frame_id that does not advance double-ages every track and double-counts the hit
        that promotes one, so accepting it quietly changes which identities exist."""
        tracker = TRACKERS.build(name, min_hits=1)
        drive(tracker, [[det(100, 200)]], start=7)
        with pytest.raises(TrackingError, match="frame_id must advance"):
            drive(tracker, [[det(100, 200)]], start=7)


@pytest.mark.parametrize("name", NAMES)
class TestTheCallSignature:
    """One call form for all five, so a deployment can swap them from config."""

    def test_an_image_is_optional_for_every_tracker(self, name: str) -> None:
        """Only BoT-SORT can use pixels, but no tracker may *require* them.

        An evaluation over a directory of MOT ground-truth files has boxes and no frames, and
        that evaluation is the only way to know whether a change to the association helped.
        """
        tracker = TRACKERS.build(name, min_hits=1)
        assert tracker.update(frame([det(100, 200)], 0))
        assert tracker.update(frame([det(101, 200)], 1), image=np.zeros((64, 64, 3), np.uint8))


class TestTheRegistry:
    """Selection by name, and a bad configuration that stops the process at start-up rather
    than on frame 40 000.
    """

    def test_it_lists_and_builds_everything(self) -> None:
        assert {"sort", "bytetrack", "ocsort", "botsort", "deepsortv2"} == set(NAMES)
        for name in NAMES:
            tracker = TRACKERS.build(name)
            assert tracker.describe()
            assert tracker.name == name
            assert repr(tracker)

    def test_aliases_resolve_to_the_same_class(self) -> None:
        for alias, name in (("byte", "bytetrack"), ("oc", "ocsort"), ("dsv2", "deepsortv2")):
            assert TRACKERS.get(alias) is TRACKERS.get(name)

    def test_an_unknown_tracker_names_the_alternatives(self) -> None:
        with pytest.raises(ConfigurationError, match="available:"):
            TRACKERS.build("nonexistent")

    def test_contradictory_configuration_is_refused_at_construction(self) -> None:
        with pytest.raises(ConfigurationError):
            TRACKERS.build("sort", iou_threshold=1.5)
        with pytest.raises(ConfigurationError):
            TRACKERS.build("bytetrack", track_threshold=0.1, low_threshold=0.5)
        with pytest.raises(ConfigurationError):
            TRACKERS.build("ocsort", delta_t=0)
        with pytest.raises(ConfigurationError):
            TRACKERS.build("botsort", appearance_gate=0.0)
        with pytest.raises(ConfigurationError):
            TRACKERS.build("deepsortv2", cascade_stride=0)
        with pytest.raises(ConfigurationError):
            TRACKERS.build("sort", max_age=0)

    def test_a_typo_in_a_config_key_is_not_silently_dropped(self) -> None:
        """A dropped key means the algorithm runs with a default nobody chose, and the run
        looks successful."""
        with pytest.raises(TypeError):
            TRACKERS.build("bytetrack", track_thresold=0.5)
