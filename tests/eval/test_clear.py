"""CLEAR, checked against arithmetic written out longhand.

Every number asserted here is derivable from ``MOTA = 1 - (FN + FP + IDSW) / GT`` and the
fixture's box positions. Where a test claims a matcher is right, it also builds the wrong
matcher and asserts a *different* answer — a test that only checks the right one cannot tell
whether the property is being enforced or is merely true by accident.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipvision.eval.association import align, match_preferring, solve_maximum
from shipvision.eval.metrics import clear_counts
from shipvision.eval.sequence import TrackSequence

from .conftest import frame, sequence


def copies(source, n: int):
    """``source`` with every box repeated ``n`` times under ``n`` distinct ids.

    Each copy gets its own id block, so the result is a tracker that publishes ``n`` separate
    identities on top of every object — the "flooded the scene" failure, as distinct from the
    "found nothing" one.
    """
    return frame(
        source.frame_id,
        [(100 * k + 71 + i, float(b[0])) for k in range(n) for i, b in enumerate(source.boxes)],
    )


def naive_clear(ground_truth: TrackSequence, predictions: TrackSequence) -> tuple[int, int]:
    """``(true_positives, id_switches)`` from a matcher that re-solves every frame.

    This is the wrong implementation, written out on purpose. It is the one a reader would
    produce from the MOTA formula alone, and the point of having it here is that the
    difference between it and :func:`clear_counts` is exactly the property the CLEAR protocol
    adds — which cannot be demonstrated without it.
    """
    aligned = align(ground_truth, predictions)
    last: dict[int, int] = {}
    true_positives = switches = 0
    for gt_ids, pred_ids, similarity in zip(
        aligned.gt_ids, aligned.pred_ids, aligned.similarity, strict=True
    ):
        if gt_ids.size == 0 or pred_ids.size == 0:
            continue
        rows, cols = solve_maximum(np.where(similarity >= 0.5, similarity, 0.0))
        for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
            gt, pred = int(gt_ids[row]), int(pred_ids[col])
            if gt in last and last[gt] != pred:
                switches += 1
            last[gt] = pred
            true_positives += 1
    return true_positives, switches


class TestAPerfectTracker:
    """Identical boxes under different ids. Every count is at its extreme."""

    def test_it_scores_mota_of_exactly_one(self, two_objects, perfect) -> None:
        counts = clear_counts(align(two_objects, perfect))

        assert (counts.true_positives, counts.false_positives, counts.false_negatives) == (
            6,
            0,
            0,
        )
        assert counts.id_switches == 0
        assert counts.mota == 1.0

    def test_motp_is_one_because_the_boxes_coincide(self, two_objects, perfect) -> None:
        assert clear_counts(align(two_objects, perfect)).motp == pytest.approx(1.0)

    def test_both_trajectories_are_mostly_tracked(self, two_objects, perfect) -> None:
        counts = clear_counts(align(two_objects, perfect))

        assert (counts.mostly_tracked, counts.partly_tracked, counts.mostly_lost) == (2, 0, 0)
        assert counts.fragmentations == 0


class TestOneIdentitySwitch:
    """Four frames, two objects, GT = 8. One switch must cost exactly ``1 / GT``."""

    def test_mota_drops_by_exactly_one_over_gt(
        self, four_frames, four_frames_perfect, four_frames_one_switch
    ) -> None:
        """MOTA = 1 - (0 + 0 + 1)/8 = 7/8 = 0.875, against 1.0 for the same boxes with one id."""
        clean = clear_counts(align(four_frames, four_frames_perfect))
        switched = clear_counts(align(four_frames, four_frames_one_switch))

        assert clean.mota == 1.0
        assert switched.id_switches == 1
        assert switched.mota == pytest.approx(7 / 8)
        assert clean.mota - switched.mota == pytest.approx(1 / switched.num_gt_dets)

    def test_the_switch_costs_no_true_positive(
        self, four_frames, four_frames_one_switch
    ) -> None:
        """Both halves of the split still localise the object, so TP is untouched at 8."""
        counts = clear_counts(align(four_frames, four_frames_one_switch))

        assert (counts.true_positives, counts.false_positives, counts.false_negatives) == (
            8,
            0,
            0,
        )

    def test_it_is_counted_once_not_once_per_remaining_frame(
        self, four_frames, four_frames_one_switch
    ) -> None:
        """The second and later frames under the new id agree with each other, so the charge is
        one. A ``last_matched`` that compared against the *original* id forever would charge
        two here, which is the bug this asserts against."""
        assert clear_counts(align(four_frames, four_frames_one_switch)).id_switches == 1

    def test_a_reborn_track_after_a_gap_is_still_a_switch(self, four_frames) -> None:
        """Object 2 vanishes for a frame and comes back under a new id. That is a switch even
        though the previous *frame* had no assignment to break — which is why the counter
        remembers the last match ever rather than the last frame."""
        predictions = sequence(
            "reborn",
            [
                frame(1, [(71, 0.0), (82, 210.0)]),
                frame(2, [(71, 0.0)]),
                frame(3, [(71, 0.0), (83, 230.0)]),
                frame(4, [(71, 0.0), (83, 240.0)]),
            ],
        )

        counts = clear_counts(align(four_frames, predictions))

        assert counts.id_switches == 1
        assert counts.false_negatives == 1  # object 2 on frame 2
        assert counts.mota == pytest.approx(1 - 2 / 8)


class TestAnEmptyTracker:
    """A tracker that publishes nothing. A real case; it must score, not crash."""

    def test_mota_is_zero_and_every_ground_truth_box_is_a_miss(self, two_objects) -> None:
        counts = clear_counts(align(two_objects, TrackSequence.empty("nothing", length=3)))

        assert counts.false_negatives == two_objects.num_detections == 6
        assert (counts.true_positives, counts.false_positives, counts.id_switches) == (0, 0, 0)
        assert counts.mota == 0.0

    def test_the_score_is_a_float_and_not_a_nan(self, two_objects) -> None:
        """A NaN here would propagate through the aggregate and poison every other sequence."""
        counts = clear_counts(align(two_objects, TrackSequence.empty("nothing", length=3)))

        assert np.isfinite(counts.mota)
        assert np.isfinite(counts.motp)
        assert counts.mostly_lost == 2


class TestDuplicatePredictions:
    """MOTA is unbounded below, and clamping it would hide the difference between a tracker
    that found nothing and one that flooded the scene. Those need different fixes."""

    def test_two_boxes_per_object_lands_exactly_on_zero(self, two_objects) -> None:
        """FP = GT, so MOTA = 1 - 6/6 = 0. The same score as finding nothing, from the
        opposite failure — which is the reason MOTA is never quoted without FP and FN."""
        doubled = sequence("doubled", [copies(f, 2) for f in two_objects])

        counts = clear_counts(align(two_objects, doubled))

        assert (counts.true_positives, counts.false_positives) == (6, 6)
        assert counts.mota == 0.0

    def test_three_boxes_per_object_drives_it_below_zero(self, two_objects) -> None:
        """FP = 2 * GT, so MOTA = 1 - 12/6 = -1. Asserted as ``< 0`` *and* as exactly -1: the
        first is the property, the second proves the property is not an accident of rounding."""
        tripled = sequence("tripled", [copies(f, 3) for f in two_objects])

        counts = clear_counts(align(two_objects, tripled))

        assert (counts.true_positives, counts.false_positives) == (6, 12)
        assert counts.mota < 0.0
        assert counts.mota == pytest.approx(-1.0)


class TestThePreviousFrameIsPreferred:
    """The property that separates a CLEAR implementation from a plausible one.

    Two people cross. Both predictions are admissible for both ground-truth objects and the
    geometry favours the swap. A matcher that re-solves from scratch records two identity
    switches; the tracker did not switch anything.
    """

    def test_the_correct_matcher_reports_no_switch(self, crossing) -> None:
        ground_truth, predictions = crossing

        counts = clear_counts(align(ground_truth, predictions))

        assert counts.id_switches == 0
        assert counts.true_positives == 4
        assert counts.mota == 1.0

    def test_a_free_re_solve_invents_two(self, crossing) -> None:
        """The naive matcher, run on the same data, disagrees. Without this half the test above
        would pass on a scenario where no matcher could have gone wrong."""
        ground_truth, predictions = crossing

        true_positives, switches = naive_clear(ground_truth, predictions)

        assert true_positives == 4
        assert switches == 2

    def test_the_carried_pair_wins_even_though_the_swap_scores_higher(self) -> None:
        """Directly on the matcher, with the totals written out: 0.579 + 0.579 = 1.158 for the
        carried assignment against 0.875 + 0.875 = 1.750 for the swap. The carried one must
        still win, or 'prefer' means nothing."""
        similarity = np.array([[22 / 38, 28 / 32], [28 / 32, 22 / 38]])
        carried = np.array([0, 1], dtype=np.int64)

        rows, cols = match_preferring(similarity, carried, threshold=0.5)

        assert list(zip(rows.tolist(), cols.tolist(), strict=True)) == [(0, 0), (1, 1)]
        assert similarity[rows, cols].sum() < similarity[[0, 1], [1, 0]].sum()

    def test_with_no_history_the_same_matcher_takes_the_better_geometry(self) -> None:
        """``previous = -1`` must behave as a plain maximum-IoU solve, or the preference is not
        a preference but a bias."""
        similarity = np.array([[22 / 38, 28 / 32], [28 / 32, 22 / 38]])

        rows, cols = match_preferring(similarity, np.array([-1, -1]), threshold=0.5)

        assert list(zip(rows.tolist(), cols.tolist(), strict=True)) == [(0, 1), (1, 0)]

    def test_the_preference_expires_after_one_frame(self, four_frames) -> None:
        """A prediction whose box was good five frames ago says nothing about where the object
        is now. The preference resolves a *current* ambiguity; it does not reward history."""
        predictions = sequence(
            "gap",
            [
                frame(1, [(71, 0.0), (82, 210.0)]),
                frame(2, [(71, 0.0)]),
                frame(3, [(71, 0.0), (82, 230.0)]),
                frame(4, [(71, 0.0), (82, 240.0)]),
            ],
        )

        counts = clear_counts(align(four_frames, predictions))

        # Same id resumed, so no switch — but the missed frame is still a miss.
        assert (counts.id_switches, counts.false_negatives) == (0, 1)


class TestTheThresholdIsACliff:
    """0.5 is a cliff, not a slope: a box at 0.49 is a false positive *and* a miss."""

    def test_a_box_just_inside_counts_and_just_outside_does_not(self) -> None:
        """For 30-wide boxes an offset of 10 gives IoU 20/40 = 0.5 exactly, and 11 gives
        19/41 = 0.463. So the two cases differ by one pixel of translation and by two errors."""
        ground_truth = sequence("cliff", [frame(1, [(1, 0.0)])])

        inside = clear_counts(align(ground_truth, sequence("p", [frame(1, [(71, 10.0)])])))
        outside = clear_counts(align(ground_truth, sequence("p", [frame(1, [(71, 11.0)])])))

        assert (inside.true_positives, inside.false_positives, inside.false_negatives) == (
            1,
            0,
            0,
        )
        assert (outside.true_positives, outside.false_positives, outside.false_negatives) == (
            0,
            1,
            1,
        )
        assert outside.mota == pytest.approx(-1.0)

    def test_exactly_at_the_threshold_is_a_match(self) -> None:
        """A box read back from a text file and one computed in float32 can differ in the last
        bit, so an IoU of exactly 0.5 must count — or a tracker scores differently depending on
        whether its output went through a file."""
        similarity = np.array([[0.5]])

        rows, _ = match_preferring(similarity, np.array([-1]), threshold=0.5)

        assert rows.size == 1
