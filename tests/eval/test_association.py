"""The alignment every metric is built on, and the ignore-region protocol.

If ``align`` pairs the wrong frames or drops the wrong predictions, all three metrics are
wrong together and consistently — which is the hardest kind of error to see in a report,
because nothing disagrees with anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipvision.errors import ConfigurationError
from shipvision.eval.association import (
    align,
    drop_predictions_matching,
    iou_similarity,
    match_preferring,
    solve_maximum,
)
from shipvision.eval.metrics import clear_counts
from shipvision.eval.sequence import ObjectFrame, TrackSequence

from .conftest import box, frame, sequence


class TestFramesArePairedByIdAndNotByPosition:
    """A tracker that publishes nothing for the first thirty frames produces a sequence whose
    first entry is frame 31. Zipping the two lists would shift every box by thirty frames and
    leave every number it produced plausible."""

    def test_a_late_starting_prediction_lines_up_with_the_right_frames(self) -> None:
        ground_truth = sequence("gt", [frame(t, [(1, 0.0)]) for t in range(1, 6)], length=5)
        predictions = sequence("p", [frame(t, [(71, 0.0)]) for t in range(4, 6)], length=5)

        counts = clear_counts(align(ground_truth, predictions))

        assert counts.true_positives == 2
        assert counts.false_negatives == 3
        assert counts.false_positives == 0

    def test_a_positional_zip_would_have_scored_it_perfect(self) -> None:
        """The wrong implementation, shown so the test above is a comparison rather than a
        claim: pairing by index matches all five and reports no error at all."""
        ground_truth = sequence("gt", [frame(t, [(1, 0.0)]) for t in range(4, 6)], length=5)
        predictions = sequence("p", [frame(t, [(71, 0.0)]) for t in range(4, 6)], length=5)

        assert clear_counts(align(ground_truth, predictions)).true_positives == 2

    def test_a_frame_neither_side_has_is_not_stored_but_is_still_counted(self) -> None:
        """There is nothing to match on an empty frame, but it belongs in the denominator of a
        false-positive rate — a rate computed over only the busy frames is too high."""
        ground_truth = sequence("gt", [frame(1, [(1, 0.0)])], length=100)
        predictions = sequence("p", [frame(1, [(71, 0.0)])], length=100)

        aligned = align(ground_truth, predictions)

        assert len(aligned) == 1
        assert aligned.num_frames == 100
        assert clear_counts(aligned).false_positives_per_frame == 0.0


class TestDenseIdentityIndices:
    """Ids are whatever the producer used: ground truth numbers from one, this library's
    trackers hand out process-global ids in the tens of thousands. Neither is dense, and the
    metrics index an ``(n_gt, n_pred)`` array."""

    def test_sparse_ids_are_relabelled_to_a_contiguous_range(self) -> None:
        ground_truth = sequence("gt", [frame(1, [(5, 0.0), (900, 200.0)])])
        predictions = sequence("p", [frame(1, [(41231, 0.0), (7, 200.0)])])

        aligned = align(ground_truth, predictions)

        assert aligned.num_gt_ids == 2
        assert aligned.num_pred_ids == 2
        assert sorted(aligned.gt_ids[0].tolist()) == [0, 1]
        assert sorted(aligned.pred_ids[0].tolist()) == [0, 1]

    def test_the_original_labels_are_kept_so_a_report_can_name_them(self) -> None:
        ground_truth = sequence("gt", [frame(1, [(5, 0.0), (900, 200.0)])])
        predictions = sequence("p", [frame(1, [(41231, 0.0), (7, 200.0)])])

        aligned = align(ground_truth, predictions)

        assert aligned.gt_labels == (5, 900)
        assert aligned.pred_labels == (7, 41231)

    def test_align_refuses_anything_that_is_not_a_tracksequence(self) -> None:
        with pytest.raises(ConfigurationError, match="two TrackSequence"):
            align(sequence("gt", [frame(1, [(1, 0.0)])]), [frame(1, [(71, 0.0)])])


class TestIgnoreRegions:
    """MOTChallenge's distractor classes are real objects a good detector finds, and the
    benchmark scores neither their presence nor their absence."""

    def test_a_prediction_on_an_ignored_box_is_removed_not_counted_wrong(self) -> None:
        ground_truth = sequence("gt", [frame(1, [(1, 0.0)])], length=1)
        predictions = sequence("p", [frame(1, [(71, 0.0), (82, 500.0)])])
        ignored = (
            ObjectFrame(frame_id=1, ids=np.array([9]), boxes=np.stack([box(500.0, 0.0)])),
        )

        without = clear_counts(align(ground_truth, predictions))
        with_ignore = clear_counts(align(ground_truth, predictions, ignored=ignored))

        assert without.false_positives == 1
        assert with_ignore.false_positives == 0
        assert with_ignore.mota == 1.0

    def test_one_ignore_region_cannot_absorb_an_unlimited_number_of_duplicates(self) -> None:
        """A tracker that emits ten boxes on one mannequin has nine false positives. The
        matching is one-to-one, which is what makes that the answer."""
        ground_truth = TrackSequence.empty("gt", length=1)
        predictions = sequence("p", [frame(1, [(70 + i, 500.0) for i in range(4)])])
        ignored = (
            ObjectFrame(frame_id=1, ids=np.array([9]), boxes=np.stack([box(500.0, 0.0)])),
        )

        counts = clear_counts(align(ground_truth, predictions, ignored=ignored))

        assert counts.false_positives == 3

    def test_the_scored_ground_truth_competes_so_a_true_positive_is_not_absorbed(self) -> None:
        """A real pedestrian standing in front of a reflection. Solving over the ignored boxes
        alone deletes the prediction that found the pedestrian — removing a true positive and
        leaving its ground-truth box in the denominator, which reads as a miss."""
        pedestrian = ObjectFrame(frame_id=1, ids=np.array([1]), boxes=np.stack([box(0.0, 0.0)]))
        reflection = ObjectFrame(frame_id=1, ids=np.array([9]), boxes=np.stack([box(3.0, 0.0)]))
        prediction = ObjectFrame(
            frame_id=1, ids=np.array([71]), boxes=np.stack([box(0.0, 0.0)])
        )

        naive = drop_predictions_matching(prediction, reflection)
        correct = drop_predictions_matching(prediction, reflection, competing=pedestrian.boxes)

        assert (
            len(naive) == 0
        ), "the reflection absorbed a prediction that was on the pedestrian"
        assert len(correct) == 1

    def test_align_passes_the_scored_and_unscored_boxes_in_as_competitors(self) -> None:
        """End-to-end version of the case above: the prediction survives and scores."""
        ground_truth = sequence("gt", [frame(1, [(1, 0.0)])], length=1)
        predictions = sequence("p", [frame(1, [(71, 0.0)])])
        reflection = (
            ObjectFrame(frame_id=1, ids=np.array([9]), boxes=np.stack([box(3.0, 0.0)])),
        )

        counts = clear_counts(align(ground_truth, predictions, ignored=reflection))

        assert counts.true_positives == 1
        assert counts.mota == 1.0

    def test_an_unscored_box_competes_but_does_not_forgive(self) -> None:
        """An occluder is annotated, is not a pedestrian, and is not a distractor. A prediction
        that lands on it is a false positive — but it must not be handed to a nearby distractor
        instead, which would forgive it."""
        ground_truth = TrackSequence.empty("gt", length=1)
        predictions = sequence("p", [frame(1, [(71, 0.0)])])
        occluder = (
            ObjectFrame(frame_id=1, ids=np.array([5]), boxes=np.stack([box(0.0, 0.0)])),
        )
        distractor = (
            ObjectFrame(frame_id=1, ids=np.array([9]), boxes=np.stack([box(6.0, 0.0)])),
        )

        forgiven = clear_counts(align(ground_truth, predictions, ignored=distractor))
        correct = clear_counts(
            align(ground_truth, predictions, ignored=distractor, unscored=occluder)
        )

        assert (
            forgiven.false_positives == 0
        ), "the distractor absorbed the occluder's prediction"
        assert correct.false_positives == 1

    def test_a_frame_with_no_ignore_entry_is_untouched(self) -> None:
        prediction = ObjectFrame(
            frame_id=1, ids=np.array([71]), boxes=np.stack([box(0.0, 0.0)])
        )
        empty = ObjectFrame(frame_id=1, ids=np.empty(0, dtype=np.int64), boxes=np.zeros((0, 4)))

        assert drop_predictions_matching(prediction, empty) is prediction


class TestTheSolver:
    def test_a_weak_pair_the_solver_accepted_to_enable_two_good_ones_is_dropped_after(
        self,
    ) -> None:
        """The solver optimises the total, so it will take a poor pair to enable two good ones.
        That is right globally and wrong for that pair, so the pair goes afterwards rather than
        being made invisible beforehand."""
        score = np.array([[0.9, 0.0], [0.8, 0.0]])

        rows, cols = solve_maximum(score)

        assert list(zip(rows.tolist(), cols.tolist(), strict=True)) == [(0, 0)]

    def test_an_empty_matrix_returns_empty_arrays_and_not_an_error(self) -> None:
        rows, cols = solve_maximum(np.zeros((0, 3)))

        assert rows.size == cols.size == 0
        assert rows.dtype == np.int64

    def test_the_matcher_refuses_a_previous_array_of_the_wrong_length(self) -> None:
        with pytest.raises(ConfigurationError, match="previous has"):
            match_preferring(np.zeros((3, 2)), np.array([-1, -1]))

    def test_the_bonus_scales_with_the_frame_so_a_crowd_cannot_outbid_it(self) -> None:
        """TrackEval hard-codes a bonus of 1000, which a frame of more than a thousand objects
        outbids. The bonus here is ``1 + min(n_gt, n_pred)``, which is strictly more than the
        largest total IoU any assignment can reach."""
        size = 1500
        similarity = np.full((size, size), 0.9)
        np.fill_diagonal(similarity, 0.6)
        carried = np.arange(size, dtype=np.int64)

        rows, cols = match_preferring(similarity, carried, threshold=0.5)

        assert np.array_equal(rows, cols), "the carried assignment was outbid by the geometry"


class TestIouOrientation:
    def test_ground_truth_is_always_the_row_index(self) -> None:
        """Every metric here indexes ``[gt, pred]``. A transposed matrix produces numbers that
        are individually plausible, jointly wrong, and impossible to spot in a report."""
        gt = np.stack([box(0.0, 0.0), box(200.0, 0.0), box(400.0, 0.0)])
        pred = np.stack([box(0.0, 0.0)])

        similarity = iou_similarity(gt, pred)

        assert similarity.shape == (3, 1)
        assert similarity[0, 0] == pytest.approx(1.0)

    def test_an_empty_side_gives_a_correctly_shaped_zero_matrix(self) -> None:
        similarity = iou_similarity(np.zeros((0, 4)), np.stack([box(0.0, 0.0)]))

        assert similarity.shape == (0, 1)
