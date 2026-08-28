"""McByte, against the BoT-SORT it is a diff of.

Two claims, and the second is what makes the first worth having. **It gains something**: on a
frame where the Hungarian total prefers to trade an unambiguous pair away for two it will then
throw out, McByte keeps the pair — and the baseline is asserted to lose it, so the comparison
is an algorithm rather than a handicap. **It costs nothing**: with the locking switched off it
is byte-for-byte BoT-SORT, and with it on it loses none of BoT-SORT's associations over a busy
forty-frame sequence.

End-to-end numeric parity with the upstream port is out of scope and is not attempted; the
filter state is ``(cx, cy, aspect, height)`` here against the reference's ``(xc, yc, w, h)``,
and the lifecycles differ. The reference's *decisions* are pinned in
``test_mcbyte_association.py`` instead, on the pure functions where they are decidable.
"""

from __future__ import annotations

import numpy as np
import pytest

import shipvision.mot.trackers.mcbyte.tracker as mcbyte_module
from shipvision.mot import TRACKERS
from shipvision.mot.motion import IDENTITY_AFFINE
from tests.mot.backends.conftest import assert_same_tracking
from tests.mot.conftest import det, drive, frame

#: A stolen-pair frame, built backwards from the cost matrix that produces the failure. Two
#: objects settle at ``LEFT`` and ``RIGHT``; one detection lands close to the left one (IoU
#: 0.63) and one wide of both (0.35, 0.34). Only the close pair is affordable at a threshold
#: of 0.5 — but the other two are cheaper *together*, so the solver takes them and loses both.
WIDTH, HEIGHT, ROW = 100.0, 200.0, 400.0
LEFT, RIGHT = 500.0, 572.0
CLOSE, WIDE = 523.0, 452.0


def settle(tracker: object) -> tuple[int, int]:
    """Six frames of two stationary objects; returns ``(left_id, right_id)``."""
    published = drive(
        tracker,
        [[det(LEFT, ROW, w=WIDTH, h=HEIGHT), det(RIGHT, ROW, w=WIDTH, h=HEIGHT)]] * 6,
    )
    by_x = sorted(published[-1], key=lambda track: float(track.box[0]))
    assert len(by_x) == 2, "the scenario needs both objects tracked before the stolen frame"
    return by_x[0].track_id, by_x[1].track_id


def stolen_frame(tracker: object) -> list:
    return tracker.update(
        frame([det(CLOSE, ROW, w=WIDTH, h=HEIGHT), det(WIDE, ROW, w=WIDTH, h=HEIGHT)], 6)
    )


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

    def test_reset_clears_the_pool_and_the_camera_motion_estimate(self) -> None:
        """A reconnect breaks continuity in both. An affine pushed before the drop, applied
        after it, moves every prediction by a motion that did not happen."""
        tracker = TRACKERS.build("mcbyte", min_hits=1, cmc="external")
        drive(tracker, [[det(100, 200)], [det(102, 200)]])
        tracker.camera_motion.push(np.array([[1, 0, 40], [0, 1, 0]], np.float32))

        tracker.reset()

        assert tracker.pool_size == 0
        assert np.array_equal(tracker.camera_motion.estimate(None), IDENTITY_AFFINE)
