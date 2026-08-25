"""HOTA, its two halves, and the threshold sweep.

The tests that matter here are the ones about *structure*: that the sweep has nineteen
thresholds, that the average is of the geometric means rather than the other way round, and
that DetA and AssA move independently — because a change that trades one for the other is
invisible in HOTA alone and that is the report's main blind spot.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipvision.eval.association import align, solve_maximum
from shipvision.eval.metrics import ALPHAS, HotaCounts, hota_counts
from shipvision.eval.sequence import TrackSequence

from .conftest import frame, sequence


class TestTheThresholdSweep:
    def test_it_is_nineteen_thresholds_from_five_to_ninety_five_percent(self) -> None:
        """A twentieth threshold at 1.0 would be unreachable by any real box and would drag
        every score down by 1/20 of itself, which is larger than the differences a tuning study
        is asked to resolve."""
        assert len(ALPHAS) == 19
        assert ALPHAS[0] == pytest.approx(0.05)
        assert ALPHAS[-1] == pytest.approx(0.95)
        assert np.allclose(np.diff(ALPHAS), 0.05)

    def test_the_counts_arrays_are_one_per_threshold(self, two_objects, perfect) -> None:
        counts = hota_counts(align(two_objects, perfect))

        for array in (counts.true_positives, counts.false_negatives, counts.false_positives):
            assert array.shape == (19,)

    def test_a_wrong_length_array_is_refused_at_construction(self) -> None:
        with pytest.raises(Exception, match="thresholds"):
            HotaCounts(true_positives=np.zeros(18))


class TestAPerfectTracker:
    def test_every_score_is_exactly_one(self, two_objects, perfect) -> None:
        """Identical boxes give IoU 1.0, which clears even the 0.95 threshold, so TP = GT at
        every alpha: DetA = 6/6 = 1, AssA = 1, HOTA = sqrt(1) = 1, LocA = 1."""
        counts = hota_counts(align(two_objects, perfect))

        assert counts.hota == 1.0
        assert counts.det_a == 1.0
        assert counts.ass_a == 1.0
        assert counts.loc_a == pytest.approx(1.0)


class TestAnEmptyTracker:
    def test_hota_is_zero_and_loca_is_one_rather_than_a_nan(self, two_objects) -> None:
        """No true positives means no localisation error to report, so LocA is 1.0 by
        convention — TrackEval's convention, kept so the numbers are comparable. The important
        half is that nothing here is NaN: a NaN in one sequence poisons the aggregate for all
        of them."""
        counts = hota_counts(align(two_objects, TrackSequence.empty("nothing", length=3)))

        assert counts.hota == 0.0
        assert counts.det_a == 0.0
        assert counts.ass_a == 0.0
        assert counts.loc_a == pytest.approx(1.0)
        assert np.all(counts.false_negatives == 6)


class TestDetectionAndAssociationAreSeparable:
    """Two failures of the same size in different halves. HOTA barely distinguishes them; DetA
    and AssA do, which is why all three are reported."""

    def test_an_identity_churn_moves_assa_and_leaves_deta_alone(self) -> None:
        """Ten frames, one object, boxes exact either way. Perfect ids against a new id every
        frame: detection is identical, association collapses."""
        ground_truth = sequence("one", [frame(t, [(1, 0.0)]) for t in range(1, 11)])
        stable = hota_counts(
            align(ground_truth, sequence("s", [frame(t, [(71, 0.0)]) for t in range(1, 11)]))
        )
        churn = hota_counts(
            align(
                ground_truth, sequence("c", [frame(t, [(100 + t, 0.0)]) for t in range(1, 11)])
            )
        )

        assert churn.det_a == pytest.approx(stable.det_a)
        assert churn.ass_a < 0.15 < stable.ass_a
        assert churn.hota < stable.hota

    def test_a_missed_frame_moves_deta_and_barely_touches_assa(self) -> None:
        """Half the frames dropped, one identity throughout. DetA halves; AssA stays high
        because the frames that were found are all attributed to the same identity."""
        ground_truth = sequence("one", [frame(t, [(1, 0.0)]) for t in range(1, 11)])
        partial = sequence("p", [frame(t, [(71, 0.0)]) for t in range(1, 11, 2)])

        counts = hota_counts(align(ground_truth, partial))

        assert counts.det_a == pytest.approx(0.5)
        assert counts.ass_a == pytest.approx(0.5)  # 5 matches / (10 gt + 5 pred - 5)
        assert counts.loc_a == pytest.approx(1.0)


class TestTheAveragingOrder:
    """HOTA is the mean of ``sqrt(DetA_a * AssA_a)``, not ``sqrt(mean(DetA) * mean(AssA))``.
    By Cauchy-Schwarz the second is never smaller, so getting the order wrong never
    under-reports — which is exactly why nobody notices."""

    def test_the_two_orders_differ_and_the_reported_one_is_the_lower(self) -> None:
        """Built to make the curves cross: the boxes are offset so that detection degrades with
        alpha while association does not."""
        ground_truth = sequence(
            "drift", [frame(t, [(1, 0.0), (2, 300.0)]) for t in range(1, 8)]
        )
        predictions = sequence(
            "drift-pred",
            [frame(t, [(71, 0.0), (82, 300.0 + 12.0)]) for t in range(1, 8)],
        )

        counts = hota_counts(align(ground_truth, predictions))
        wrong_order = float(np.sqrt(counts.det_a_curve.mean() * counts.ass_a_curve.mean()))

        assert counts.hota == pytest.approx(float(counts.hota_curve.mean()))
        assert counts.hota < wrong_order
        assert wrong_order - counts.hota > 1e-3

    def test_hota_is_the_geometric_mean_at_every_threshold(self, two_objects) -> None:
        predictions = sequence(
            "offset", [frame(f.frame_id, [(71, 4.0), (82, 204.0)]) for f in two_objects]
        )

        counts = hota_counts(align(two_objects, predictions))

        assert np.allclose(counts.hota_curve, np.sqrt(counts.det_a_curve * counts.ass_a_curve))


class TestTheSequenceWideAlignment:
    """The property that makes HOTA higher-order: an ambiguous frame resolves the way the rest
    of the sequence says it should, not by a hair of geometry in that frame alone."""

    def test_a_single_ambiguous_frame_follows_the_majority(self) -> None:
        """Five frames. On four of them prediction 71 is plainly object 1 and 82 is object 2.
        On the fifth the two objects are close enough that the swapped pairing has the higher
        IoU. A per-frame matcher takes the swap and loses two matched pairs' worth of
        association; the sequence-wide alignment keeps the majority reading.

        The assertion is at alpha 0.5, where both readings of frame 5 are admissible (0.579 and
        0.875 IoU): AssA is 1.0 there only if all five frames were attributed to the same pair
        of trajectories. It is *not* 1.0 averaged over all nineteen thresholds, because above
        0.579 frame 5's carried pair stops being a true positive at all while its two
        detections stay in the trajectory lengths — that is detection loss showing up in the
        association denominator, not a matching mistake."""
        ground_truth = sequence(
            "majority",
            [frame(t, [(1, 0.0), (2, 200.0)]) for t in range(1, 5)]
            + [frame(5, [(1, 0.0), (2, 10.0)])],
        )
        predictions = sequence(
            "majority-pred",
            [frame(t, [(71, 0.0), (82, 200.0)]) for t in range(1, 5)]
            + [frame(5, [(71, 8.0), (82, 2.0)])],
        )

        counts = hota_counts(align(ground_truth, predictions))

        assert counts.at(0.5)["AssA"] == pytest.approx(1.0)
        assert counts.at(0.5)["TP"] == 10.0

    def test_the_frame_really_is_ambiguous(self) -> None:
        """Without this the test above could be passing on a frame where no matcher could have
        gone wrong. A plain maximum-IoU solve on frame 5 alone takes the swap."""
        similarity = np.array([[22 / 38, 28 / 32], [28 / 32, 22 / 38]])

        rows, cols = solve_maximum(np.where(similarity >= 0.5, similarity, 0.0))

        assert list(zip(rows.tolist(), cols.tolist(), strict=True)) == [(0, 1), (1, 0)]


class TestAggregationOfTheAssociationTerm:
    """AssA is an average over pairs of trajectories, so it cannot be a ratio of summed
    detection counts. It is stored pre-multiplied by the TP count it averages over, and the
    test is that adding two sequences reproduces the detection-weighted average."""

    def test_summing_reproduces_the_tp_weighted_average(self, two_objects, perfect) -> None:
        long_gt = sequence("long", [frame(t, [(1, 0.0)]) for t in range(1, 21)], length=20)
        long_pred = sequence(
            "long-p", [frame(t, [(100 + t, 0.0)]) for t in range(1, 21)], length=20
        )

        good = hota_counts(align(two_objects, perfect))
        bad = hota_counts(align(long_gt, long_pred))
        total = good + bad

        expected = (
            good.ass_a_curve * good.true_positives + bad.ass_a_curve * bad.true_positives
        ) / np.maximum(1.0, good.true_positives + bad.true_positives)

        assert np.allclose(total.ass_a_curve, expected)
        assert total.ass_a == pytest.approx(float(expected.mean()))
        assert not total.ass_a == pytest.approx((good.ass_a + bad.ass_a) / 2)


class TestTheHalfThresholdView:
    def test_at_reports_the_nearest_threshold_and_its_scores(
        self, two_objects, perfect
    ) -> None:
        """A report that wants the 0.5 column — the one comparable with CLEAR — should not have
        to know which index that is."""
        at_half = hota_counts(align(two_objects, perfect)).at(0.5)

        assert at_half["alpha"] == pytest.approx(0.5)
        assert at_half["TP"] == 6.0
        assert at_half["HOTA"] == pytest.approx(1.0)
