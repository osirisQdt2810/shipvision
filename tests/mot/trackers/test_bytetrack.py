"""The two properties that actually distinguish ByteTrack from SORT.

Everything else the two share, so a test that both pass proves nothing about either. These
two are the whole of the paper's contribution, and each of them also asserts that **SORT
fails** — a comparison against a baseline that was not given its own paper's configuration
measures the handicap, not the algorithm.
"""

from __future__ import annotations

from shipvision.mot import TRACKERS
from tests.mot.conftest import det, drive


def _occlusion_run(
    name: str, *, occluded_frames: int, max_age: int
) -> tuple[set[int], int, set[int]]:
    """Walk an object across the frame, dropping its confidence for a stretch in the middle.

    Returns the identities before the occlusion, how many frames the tracker published
    *during* it, and the identities after.
    """
    tracker = TRACKERS.build(name, min_hits=2, max_age=max_age)
    before: set[int] = set()
    during = 0
    after: set[int] = set()
    total = 12 + occluded_frames + 12

    for frame_id in range(total):
        occluded = 12 <= frame_id < 12 + occluded_frames
        tracks = drive(
            tracker,
            [[det(60 + frame_id * 5, 200, score=0.30 if occluded else 0.92)]],
            start=frame_id,
        )[0]
        if frame_id < 12:
            before |= {t.track_id for t in tracks}
        elif occluded:
            during += bool(tracks)
        elif frame_id >= 12 + occluded_frames + 6:
            after |= {t.track_id for t in tracks}
    return before, during, after


class TestTheSecondAssociation:
    """The two properties that actually distinguish ByteTrack from SORT, each with SORT
    asserted to fail.
    """

    def test_bytetrack_keeps_publishing_through_a_low_confidence_stretch(self) -> None:
        """The direct, observable difference.

        An object walks behind a pillar: the detector still sees it, at 0.30 instead of 0.92.
        SORT discards those boxes, so for six frames it reports nothing — downstream that is
        indistinguishable from the object having left. ByteTrack matches them against the
        existing track in its second pass and keeps reporting a position throughout.
        """
        _sort_before, sort_during, _sort_after = _occlusion_run(
            "sort", occluded_frames=6, max_age=30
        )
        _byte_before, byte_during, _byte_after = _occlusion_run(
            "bytetrack", occluded_frames=6, max_age=30
        )

        assert (
            sort_during == 0
        ), "SORT is expected to go silent; if it does not, this test proves nothing"
        assert byte_during == 6, (
            f"ByteTrack published on {byte_during} of 6 occluded frames; continuing to report a "
            f"position through an occlusion is the one thing it is for"
        )

    def test_bytetrack_survives_an_occlusion_longer_than_max_age(self) -> None:
        """And the identity difference, which only appears once the gap outlasts the age-out.

        With a generous `max_age` both trackers keep the identity — SORT's track goes LOST and
        is re-matched when the object reappears. Shorten `max_age` below the occlusion and
        SORT's track is deleted, so the object comes back as somebody new; ByteTrack never lost
        it, because it never stopped matching.
        """
        sort_before, _, sort_after = _occlusion_run("sort", occluded_frames=10, max_age=4)
        byte_before, _, byte_after = _occlusion_run("bytetrack", occluded_frames=10, max_age=4)

        assert sort_before and sort_after and byte_before and byte_after
        assert sort_before.isdisjoint(
            sort_after
        ), "SORT is expected to lose the identity across a gap longer than max_age"
        assert byte_before == byte_after, "ByteTrack lost an identity it never stopped seeing"


class TestTheAsymmetry:
    """Low-score detections may continue a track and may never start one. That is what makes
    the second pass safe rather than a licence to publish noise.
    """

    def test_a_low_score_detection_may_continue_a_track_but_never_start_one(self) -> None:
        """The asymmetry that makes the second pass safe.

        Without it, ByteTrack would publish an identity for every 0.3-confidence noise box the
        detector emits, and a tracker that invents identities is worse than no tracker. With it,
        the worst a bad low-score box can do is misplace an existing track by one frame.
        """
        tracker = TRACKERS.build(
            "bytetrack", min_hits=1, track_threshold=0.5, low_threshold=0.1
        )
        published = drive(tracker, [[det(400, 600, score=0.3)] for _ in range(8)])
        assert all(step == [] for step in published)
        assert tracker.pool_size == 0, "a low-score-only stream must not create a track at all"

    def test_the_second_pass_is_offered_only_to_tracks_that_have_earned_it(self) -> None:
        """A tentative track rescued by a 0.3-confidence box is two weak pieces of evidence
        agreeing with each other, which is not evidence — and it is how a noise track becomes a
        published identity. So a track must be CONFIRMED before the low-score pass will feed it.
        """
        tracker = TRACKERS.build("bytetrack", min_hits=3, max_age=10)
        # One confident frame: the track exists but is tentative.
        drive(tracker, [[det(300, 400, score=0.9)]])
        assert tracker.pool_size == 1

        # Now only low-score boxes in the same place. A tentative track must not be kept alive
        # by them, so it ages out on the very next frame.
        drive(tracker, [[det(300, 400, score=0.3)]], start=1)
        assert tracker.pool_size == 0
