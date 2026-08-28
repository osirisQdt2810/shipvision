"""McByte, against the BoT-SORT it is a diff of.

Two claims, and the second makes the first worth having. **It gains something**: where the
Hungarian total would trade an unambiguous pair away for two it then throws out, McByte keeps
it — on either association stage, and across a sequence that ends as one identity rather than
three. Every comparison asserts the baseline fails, so it measures an algorithm and not a
handicap. **It costs nothing**: locking off is byte-for-byte BoT-SORT, and locking on loses
none of BoT-SORT's associations over a busy forty-frame sequence.

Those gain scenarios are constructed, and say so: locking fires 125 times over that busy
sequence and changes nothing, so the frames it does change are built rather than found.

End-to-end numeric parity with the upstream port is not attempted: the filter state and the
lifecycles differ. The reference's *decisions* are pinned in ``test_mcbyte_association.py``
instead, on the pure functions where they are decidable.
"""

from __future__ import annotations

import numpy as np
import pytest

import shipvision.mot.trackers.mcbyte.tracker as mcbyte_module
from shipvision.errors import ConfigurationError
from shipvision.mot import TRACKERS
from shipvision.mot.motion import IDENTITY_AFFINE
from shipvision.types import Detection
from tests.mot.backends.conftest import assert_same_tracking
from tests.mot.conftest import all_ids, det, drive, frame

#: A stolen-pair frame, built backwards from the cost matrix that produces the failure. Two
#: objects settle at ``LEFT`` and ``RIGHT``; one detection lands close to the left one (IoU
#: 0.63) and one wide of both (0.35, 0.34). Only the close pair is affordable at a threshold
#: of 0.5 — but the other two are cheaper *together*, so the solver takes them and loses both.
WIDTH, HEIGHT, ROW = 100.0, 200.0, 400.0
LEFT, RIGHT = 500.0, 572.0
CLOSE, WIDE = 523.0, 452.0

#: A third object, far enough away to share no cost with the other two, and a score in
#: ByteTrack's low tier. Together they put the stolen pair in the *second* association: the
#: detector is still visibly working, so stage one runs, and it runs on the far box alone.
FAR, LOW_SCORE = 1400.0, 0.3


def settled_scene(*extra: Detection) -> list[Detection]:
    """The two objects the stolen pair is about, plus whatever else the scenario needs."""
    return [det(LEFT, ROW, w=WIDTH, h=HEIGHT), det(RIGHT, ROW, w=WIDTH, h=HEIGHT), *extra]


def settle(tracker: object, *extra: Detection) -> list[int]:
    """Six frames of stationary objects; returns the published ids, left to right."""
    published = drive(tracker, [settled_scene(*extra)] * 6)
    by_x = sorted(published[-1], key=lambda track: float(track.box[0]))
    assert len(by_x) == 2 + len(extra), "every object must be tracked before the stolen frame"
    return [track.track_id for track in by_x]


def stolen_frame(tracker: object) -> list:
    return tracker.update(
        frame([det(CLOSE, ROW, w=WIDTH, h=HEIGHT), det(WIDE, ROW, w=WIDTH, h=HEIGHT)], 6)
    )


#: How long the detector is confused, and how long a track remembers. The stretch outlasts
#: the memory, which is what turns a starved track into an identity switch — six frames is
#: 0.3 s at the twenty-per-second this library is sized for.
SETTLED, SPLIT, RECOVERED, SHORT_MEMORY = 6, 6, 6, 5


def split_detection_sequence() -> list[list[Detection]]:
    """Eighteen frames: two people settle, the detector splits over one of them, it recovers.

    The middle stretch is a split detection, which is an ordinary detector failure and not a
    matrix built backwards: one box lands nearly on the left-hand person and one wide of them,
    and the right-hand person is missed entirely. The near box is the only affordable
    candidate the left-hand track has, and it is exactly the one the solver trades away.
    """
    settled = settled_scene()
    split = [det(CLOSE, ROW, w=WIDTH, h=HEIGHT), det(WIDE, ROW, w=WIDTH, h=HEIGHT)]
    return [settled] * SETTLED + [split] * SPLIT + [settled] * RECOVERED


def busy_sequence() -> list[list]:
    """Forty frames with two objects crossing, one still, one low-score, one passing through."""
    frames = []
    for index in range(40):
        items = [
            det(200 + index * 12, 300, w=60, h=140),
            det(680 - index * 12, 300, w=60, h=140),
            det(400, 700, w=90, h=180),
            det(150 + index * 4, 900, 0.35, w=70, h=150),
        ]
        if 10 <= index < 25:
            items.append(det(1000 - index * 6, 520, w=80, h=160))
        frames.append(items)
    return frames


class TestClearMatchLocking:
    """The frame the paper is about, and the two arms that are asserted to lose it."""

    def test_the_only_affordable_pair_survives_a_solver_that_would_trade_it_away(
        self,
    ) -> None:
        tracker = TRACKERS.build("mcbyte", min_hits=2, match_threshold=0.5)
        left_id, _ = settle(tracker)

        published = stolen_frame(tracker)

        assert [track.track_id for track in published] == [left_id]
        centre = float(published[0].box[0] + published[0].box[2]) / 2.0
        assert centre == pytest.approx(CLOSE, abs=10.0)

    @pytest.mark.parametrize(
        ("name", "options"),
        [("mcbyte", {"lock_clear_matches": False}), ("botsort", {})],
    )
    def test_the_baseline_loses_it(self, name: str, options: dict) -> None:
        """Asserted, not assumed. A comparison whose baseline is never shown to fail is a
        comparison that would still read as a success if the feature did nothing."""
        tracker = TRACKERS.build(name, min_hits=2, match_threshold=0.5, **options)
        settle(tracker)

        assert stolen_frame(tracker) == []


class TestTheGainSurvivesAWholeSequence:
    """The same failure over eighteen frames, where the cost of it is an identity switch.

    One frame of a starved track is recoverable; a stretch of them longer than ``max_age``
    is not, and the person walks out of the tracker under a different number. ``max_age``
    decides only how long the loss takes to surface — the trade is what caused it, which is
    why turning locking off reproduces the baseline exactly.
    """

    def build(self, name: str, **options: object) -> object:
        return TRACKERS.build(
            name,
            backend="python",
            min_hits=2,
            max_age=SHORT_MEMORY,
            match_threshold=0.5,
            **options,
        )

    def left_hand_id(self, published: list) -> int:
        return min(published[SETTLED - 1], key=lambda track: float(track.box[0])).track_id

    def test_the_left_hand_person_keeps_one_identity_from_end_to_end(self) -> None:
        published = drive(self.build("mcbyte"), split_detection_sequence())
        left_id = self.left_hand_id(published)

        holding = [step for step in published[SETTLED - 1 :] if left_id in all_ids([step])]

        assert len(holding) == len(published) - SETTLED + 1, "the identity was dropped"
        assert len(all_ids(published)) == 4, "two people, one spurious box, one re-birth"

    @pytest.mark.parametrize(
        ("name", "options"),
        [("mcbyte", {"lock_clear_matches": False}), ("botsort", {})],
    )
    def test_the_baseline_publishes_that_person_under_three_identities(
        self, name: str, options: dict
    ) -> None:
        """And publishes nothing at all on the frame the trade happens, with two people
        standing in front of the camera — the starved pair is both of them at once."""
        published = drive(self.build(name, **options), split_detection_sequence())
        left_id = self.left_hand_id(published)

        assert published[SETTLED] == []
        assert left_id not in all_ids(published[SETTLED:])
        assert len(all_ids(published)) == 6


class TestLockingOnTheSecondAssociationStage:
    """The low-score pass, where losing a pair is losing an identity through an occlusion.

    Stage two exists because a tracked object behind a pillar is still detected, at 0.3
    instead of 0.9. That box is the *only* evidence the track has, and the solver will still
    spend it on a cheaper pair it then throws away — so the trade the paper is about is more
    expensive here than in stage one, not less.

    The two match thresholds are deliberately unequal, so locking applied to stage one only
    is a different answer here rather than the same answer twice.
    """

    def build(self, name: str, **options: object) -> object:
        return TRACKERS.build(
            name,
            backend="python",
            min_hits=2,
            max_age=10,
            match_threshold=0.5,
            second_match_threshold=0.45,
            **options,
        )

    def occluded_frame(self, tracker: object, *, missing: int) -> list[int]:
        """Drop the pair for ``missing`` frames, then give it back at a low score.

        ``missing=0`` leaves both tracks CONFIRMED and ``missing=2`` leaves them LOST; both
        are eligible for stage two, and a rule that only held for one of them would be a rule
        that expires exactly when the occlusion has lasted long enough to matter.
        """
        far = det(FAR, ROW, w=WIDTH, h=HEIGHT)
        for offset in range(missing):
            tracker.update(frame([far], 6 + offset))
        published = tracker.update(
            frame(
                [
                    det(CLOSE, ROW, LOW_SCORE, w=WIDTH, h=HEIGHT),
                    det(WIDE, ROW, LOW_SCORE, w=WIDTH, h=HEIGHT),
                    far,
                ],
                6 + missing,
            )
        )
        return sorted(track.track_id for track in published)

    @pytest.mark.parametrize(("missing", "state"), [(0, "confirmed"), (2, "lost")])
    def test_the_only_affordable_low_score_box_is_not_traded_away(
        self, missing: int, state: str
    ) -> None:
        tracker = self.build("mcbyte")
        left_id, _, far_id = settle(tracker, det(FAR, ROW, w=WIDTH, h=HEIGHT))

        published = self.occluded_frame(tracker, missing=missing)

        assert published == sorted([left_id, far_id]), f"the {state} track lost its only box"

    @pytest.mark.parametrize(
        ("name", "options"),
        [("mcbyte", {"lock_clear_matches": False}), ("botsort", {})],
    )
    def test_the_baseline_loses_it(self, name: str, options: dict) -> None:
        """Only the far object survives the frame: the solver spent both low-score boxes on
        the pair that is cheaper together and above the threshold apart."""
        tracker = self.build(name, **options)
        _, _, far_id = settle(tracker, det(FAR, ROW, w=WIDTH, h=HEIGHT))

        assert self.occluded_frame(tracker, missing=0) == [far_id]


class TestItDegradesToItsBaselineWithoutMasks:
    """Switched off it *is* BoT-SORT; switched on it takes nothing away."""

    def test_with_locking_off_it_is_bot_sort_frame_for_frame(self) -> None:
        frames = busy_sequence()

        reference = drive(TRACKERS.build("botsort", min_hits=2, max_age=10), frames)
        candidate = drive(
            TRACKERS.build("mcbyte", min_hits=2, max_age=10, lock_clear_matches=False),
            frames,
        )

        assert assert_same_tracking(reference, candidate) == 0.0

    def test_with_locking_on_it_loses_none_of_the_baselines_associations(
        self, monkeypatch
    ) -> None:
        """Every box BoT-SORT published is still published. The spy is there so the claim is
        not accidentally vacuous: a sequence in which nothing ever locks would pass this while
        proving that locking is unreachable."""
        frames = busy_sequence()
        locked: list[list[tuple[int, int]]] = []
        real = mcbyte_module.clear_matches

        def spy(cost, max_cost):
            locked.append(real(cost, max_cost))
            return locked[-1]

        monkeypatch.setattr(mcbyte_module, "clear_matches", spy)
        reference = drive(TRACKERS.build("botsort", min_hits=2, max_age=10), frames)
        candidate = drive(TRACKERS.build("mcbyte", min_hits=2, max_age=10), frames)

        assert sum(len(pairs) for pairs in locked) > 0, "no pair ever locked in this sequence"
        for index, (want, got) in enumerate(zip(reference, candidate, strict=True)):
            for track in want:
                assert any(
                    np.abs(track.box - other.box).max() < 1e-3 for other in got
                ), f"frame {index}: locking dropped the association at {track.box.tolist()}"


class TestOperability:
    """What an operator reading a log line and a reconnect handler both depend on."""

    def test_describe_names_the_camera_motion_model_and_whether_it_locks(self) -> None:
        locking = TRACKERS.build("mcbyte", cmc="external").describe()
        plain = TRACKERS.build("mcbyte", lock_clear_matches=False).describe()

        assert "external" in locking and "locked" in locking
        assert "none" in plain and "not locked" in plain

    def test_a_switch_that_is_not_a_bool_is_refused_rather_than_read_as_true(self) -> None:
        """``bool("false")`` is ``True``. A config file that quotes the value would otherwise
        pin the one switch this tracker is measured by permanently on, and say nothing."""
        with pytest.raises(ConfigurationError, match="lock_clear_matches"):
            TRACKERS.build("mcbyte", lock_clear_matches="false")

    def test_reset_clears_the_pool_and_the_camera_motion_estimate(self) -> None:
        """A reconnect breaks continuity in both. An affine pushed before the drop, applied
        after it, moves every prediction by a motion that did not happen."""
        tracker = TRACKERS.build("mcbyte", min_hits=1, cmc="external")
        drive(tracker, [[det(100, 200)], [det(102, 200)]])
        tracker.camera_motion.push(np.array([[1, 0, 40], [0, 1, 0]], np.float32))

        tracker.reset()

        assert tracker.pool_size == 0
        assert np.array_equal(tracker.camera_motion.estimate(None), IDENTITY_AFFINE)
