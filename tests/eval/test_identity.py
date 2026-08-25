"""IDF1, and the specific way re-implementations get it wrong.

The failure is always the same shape: a per-frame count of agreements instead of one global
matching between whole trajectories. It never crashes and it always inflates, so the only
test that catches it is one where the two answers differ and the global one is asserted.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipvision.eval.association import align
from shipvision.eval.metrics import identity_counts
from shipvision.eval.sequence import TrackSequence

from .conftest import frame, sequence


def per_frame_idf1(ground_truth: TrackSequence, predictions: TrackSequence) -> float:
    """IDF1 computed the wrong way: count the frames on which *any* prediction covered a
    ground-truth box. Written out so the disagreement below is demonstrable rather than
    asserted; this is what a reader produces from the formula without the paper."""
    aligned = align(ground_truth, predictions)
    hits = 0
    for _, _, similarity in zip(
        aligned.gt_ids, aligned.pred_ids, aligned.similarity, strict=True
    ):
        if similarity.size:
            hits += int(np.count_nonzero((similarity >= 0.5).any(axis=1)))
    false_negatives = aligned.num_gt_dets - hits
    false_positives = aligned.num_pred_dets - hits
    return 2 * hits / (2 * hits + false_positives + false_negatives)


class TestAPerfectTracker:
    def test_idf1_is_exactly_one(self, two_objects, perfect) -> None:
        counts = identity_counts(align(two_objects, perfect))

        assert counts.true_positives == 6
        assert (counts.false_positives, counts.false_negatives) == (0, 0)
        assert counts.idf1 == 1.0
        assert counts.idp == counts.idr == 1.0


class TestGlobalVersusPerFrameMatching:
    """One object tracked as two half-length identities. The two implementations must disagree,
    and the global one must be the lower — that is the whole content of the metric."""

    def test_the_global_matching_credits_only_one_half(self, split_track) -> None:
        """4 GT detections, 4 predicted, one trajectory each side. The matching pairs the GT
        trajectory with one of the two halves, so IDTP = 2, IDFN = 4 - 2 = 2, IDFP = 4 - 2 = 2,
        and IDF1 = 2*2 / (2*2 + 2 + 2) = 4/8 = 0.5."""
        ground_truth, predictions = split_track

        counts = identity_counts(align(ground_truth, predictions))

        assert counts.true_positives == 2
        assert (counts.false_positives, counts.false_negatives) == (2, 2)
        assert counts.idf1 == pytest.approx(0.5)

    def test_the_per_frame_count_would_have_said_one(self, split_track) -> None:
        """The wrong implementation reports a perfect score for a tracker that lost the identity
        halfway through. Twice the truth, and nothing about the output looks wrong."""
        ground_truth, predictions = split_track

        assert per_frame_idf1(ground_truth, predictions) == pytest.approx(1.0)
        assert identity_counts(align(ground_truth, predictions)).idf1 == pytest.approx(0.5)

    def test_a_tracker_that_switches_every_frame_scores_near_zero(self) -> None:
        """Ten frames, ten predicted identities, boxes exact. Every frame is localised perfectly
        and the identity is worthless: IDTP = 1, IDF1 = 2/(2 + 9 + 9) = 0.1."""
        ground_truth = sequence("one", [frame(t, [(1, 0.0)]) for t in range(1, 11)])
        predictions = sequence("churn", [frame(t, [(100 + t, 0.0)]) for t in range(1, 11)])

        counts = identity_counts(align(ground_truth, predictions))

        assert counts.true_positives == 1
        assert counts.idf1 == pytest.approx(0.1)
        assert per_frame_idf1(ground_truth, predictions) == pytest.approx(1.0)


class TestOneIdentitySwitch:
    def test_the_drop_is_the_length_of_the_shorter_half(
        self, four_frames, four_frames_perfect, four_frames_one_switch
    ) -> None:
        """8 GT detections over 4 frames. Object 1 is one clean trajectory (4 frames); object 2
        splits into 2 + 2, and the matching keeps one of them. So IDTP = 4 + 2 = 6, IDFN = 2,
        IDFP = 2, and IDF1 = 12/16 = 0.75 against 1.0 for the unsplit version.

        Note it is *not* the same drop MOTA takes for the same event (1/8): MOTA charges one
        error at the moment of the switch, IDF1 charges every frame the wrong half covered.
        That difference is the reason both are reported."""
        clean = identity_counts(align(four_frames, four_frames_perfect))
        switched = identity_counts(align(four_frames, four_frames_one_switch))

        assert clean.idf1 == 1.0
        assert switched.true_positives == 6
        assert switched.idf1 == pytest.approx(0.75)


class TestDegenerateSequences:
    def test_an_empty_tracker_scores_zero_with_every_box_missed(self, two_objects) -> None:
        counts = identity_counts(align(two_objects, TrackSequence.empty("nothing", length=3)))

        assert (counts.true_positives, counts.false_positives) == (0, 0)
        assert counts.false_negatives == 6
        assert counts.idf1 == 0.0

    def test_predictions_with_no_ground_truth_are_all_false_positives(self, perfect) -> None:
        counts = identity_counts(align(TrackSequence.empty("nothing", length=3), perfect))

        assert (counts.true_positives, counts.false_negatives) == (0, 0)
        assert counts.false_positives == 6
        assert counts.idf1 == 0.0

    def test_two_empty_sequences_give_zero_rather_than_a_division_by_zero(self) -> None:
        counts = identity_counts(
            align(TrackSequence.empty("a", length=5), TrackSequence.empty("b", length=5))
        )

        assert counts.idf1 == 0.0
        assert np.isfinite(counts.idp) and np.isfinite(counts.idr)


class TestOneToManyCoOccurrence:
    """In a crowd one ground-truth box can be covered by two predictions at once. Both pairings
    are candidates, and the global matching has to choose — it must not count both."""

    def test_a_duplicated_prediction_cannot_be_credited_twice(self) -> None:
        """Two frames, one object, two predictions on it every frame. IDTP = 2 (one pairing),
        IDFP = 4 - 2 = 2, IDFN = 0, so IDF1 = 4/(4 + 2) = 2/3."""
        ground_truth = sequence("one", [frame(t, [(1, 0.0)]) for t in (1, 2)])
        predictions = sequence("doubled", [frame(t, [(71, 0.0), (82, 2.0)]) for t in (1, 2)])

        counts = identity_counts(align(ground_truth, predictions))

        assert counts.true_positives == 2
        assert (counts.false_positives, counts.false_negatives) == (2, 0)
        assert counts.idf1 == pytest.approx(2 / 3)
