"""The survivor budget — ``max_output``, applied after suppression, never during it.

A cap is a *budget*, not a filter: it exists because a downstream tensor has a fixed number of
rows, not because the box it drops failed a test. So the two things worth pinning are that it
cannot change which boxes were suppressed, and that what it keeps is the top of the answer by
**final** score — which for a soft method is not the same list as the top by input score.

Three places implement it and they must agree: the shared numpy dispatcher truncates the pair
it is about to return, the torch backend slices ``torchvision.ops.nms``'s output, and the
native backend hands the number to the C++ sweep, which stops early rather than finishing and
throwing the tail away. All three are the same claim, so every test here runs on every backend
this machine can build.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipvision.errors import ConfigurationError
from shipvision.imgproc.nms import suppress
from tests.imgproc.nms.conftest import (
    ANCHOR,
    DEGENERATE,
    FAR_AWAY,
    HALF_SHIFTED,
    INSIDE_SMALL,
    run,
)

FOUR_SURVIVORS = [ANCHOR, INSIDE_SMALL, FAR_AWAY, DEGENERATE]
"""Four boxes whose pairwise IoUs are all at or below 0.04, so at ``iou_threshold=0.5``
classic NMS suppresses nothing and every survivor exists to be counted. Built from this
directory's shared geometry so the overlaps stay exact fractions."""

FOUR_SCORES = [0.9, 0.6, 0.8, 0.7]
"""Distinct, and deliberately not in input order: descending score is ``0, 2, 3, 1``, so a cap
that sliced the *input* rather than the answer would be caught here."""

DESCENDING = [0, 2, 3, 1]
"""The uncapped answer, spelled out — the tests below are about its prefixes."""


class TestTheCapCountsSurvivors:
    def test_a_cap_below_the_survivor_count_leaves_exactly_that_many(self, ops) -> None:
        kept = run(ops, FOUR_SURVIVORS, FOUR_SCORES, iou_threshold=0.5, max_output=2)

        assert len(kept) == 2

    def test_and_they_are_the_highest_scored_survivors(self, ops) -> None:
        """Two claims in one list: the right boxes, in the right order.

        Comparing against a set would let a backend that kept the two *lowest* pass whenever
        the caller only counted rows, which is what a fixed-size downstream tensor does.
        """
        kept = run(ops, FOUR_SURVIVORS, FOUR_SCORES, iou_threshold=0.5, max_output=2)

        assert kept == [0, 2]

    @pytest.mark.parametrize("cap", [0, 1, 2, 3, 4, 5, 100])
    def test_every_cap_is_a_prefix_of_the_uncapped_answer(self, ops, cap: int) -> None:
        """The strongest form of "it truncates rather than re-runs".

        A cap that re-entered the suppression loop with a budget could return a different
        *set* — dropping a box early changes what it would have suppressed — and the answer
        would still look sorted and plausible. Requiring a prefix of the uncapped list rules
        that out for every cap at once, including the two boundaries.
        """
        uncapped = run(ops, FOUR_SURVIVORS, FOUR_SCORES, iou_threshold=0.5)
        capped = run(ops, FOUR_SURVIVORS, FOUR_SCORES, iou_threshold=0.5, max_output=cap)

        assert uncapped == DESCENDING
        assert capped == uncapped[:cap]

    def test_a_whole_valued_float_cap_is_the_same_answer_as_its_int(self, ops) -> None:
        """#12 round 1: `validate_max_output(2.0)` accepts a whole-valued float, and the
        python/torch paths then sliced with the caller's original object — TypeError on the
        first frame — while the native backend converted and ran correctly for weeks. The
        normalised value must be the one that slices, on every backend."""
        as_float = run(ops, FOUR_SURVIVORS, FOUR_SCORES, iou_threshold=0.5, max_output=2.0)

        assert as_float == run(
            ops, FOUR_SURVIVORS, FOUR_SCORES, iou_threshold=0.5, max_output=2
        )

    def test_a_cap_larger_than_the_survivor_count_is_a_no_op(self, ops) -> None:
        capped = run(ops, FOUR_SURVIVORS, FOUR_SCORES, iou_threshold=0.5, max_output=99)

        assert capped == run(ops, FOUR_SURVIVORS, FOUR_SCORES, iou_threshold=0.5)

    def test_none_is_the_uncapped_answer(self, ops) -> None:
        """The default, asserted against an explicit ``None`` as well as against omitting it:
        every existing caller passes neither, and both spellings must mean no cap."""
        omitted = run(ops, FOUR_SURVIVORS, FOUR_SCORES, iou_threshold=0.5)
        explicit = run(ops, FOUR_SURVIVORS, FOUR_SCORES, iou_threshold=0.5, max_output=None)

        assert explicit == omitted == DESCENDING

    def test_a_cap_of_zero_keeps_nothing(self, ops) -> None:
        """Unusual, but unambiguous, and the three backends agree on it — the C++ sweep's
        ``keep.size() < 0`` is false on the first iteration and a numpy ``[:0]`` is empty."""
        assert run(ops, FOUR_SURVIVORS, FOUR_SCORES, iou_threshold=0.5, max_output=0) == []

    def test_the_cap_does_not_rescue_a_suppressed_box(self, ops) -> None:
        """Room in the budget is not a reason to keep something suppression removed.

        Two boxes overlapping at IoU 1/3 and a cap of ten: the answer is still one box. A cap
        implemented as "take the top k of the input" — which is a real way to write it, and
        faster — returns both.
        """
        kept = run(ops, [ANCHOR, HALF_SHIFTED], [0.9, 0.8], iou_threshold=0.3, max_output=10)

        assert kept == [0]


class TestTheCapReadsTheFinalScore:
    """For a soft method the ranking it truncates is the *decayed* one.

    ``gauss`` decays every live candidate by its overlap with the box just kept, so a box can
    finish below one it started above. A cap taken over the input scores and a cap taken over
    the final scores therefore keep different boxes, and only one of them is this method's
    contract.
    """

    SOFT_BOXES = [ANCHOR, HALF_SHIFTED, FAR_AWAY]
    SOFT_SCORES = [0.9, 0.85, 0.8]
    """By input score the ranking is ``0, 1, 2``. Under ``gauss`` box 1 overlaps the winner at
    IoU 1/3 and is decayed to ~0.68, while box 2 overlaps nothing and keeps 0.8 — so the final
    ranking is ``0, 2, 1`` and a cap of two must drop the box with the *second highest* input
    score."""

    def test_the_uncapped_ranking_really_does_reorder(self, ops) -> None:
        """The premise of the test below, asserted so it cannot rot. Without it, a cap that
        read the input scores would pass whenever the decay happened not to reorder."""
        kept = run(
            ops, self.SOFT_BOXES, self.SOFT_SCORES, iou_threshold=0.3, method="gauss", sigma=0.5
        )

        assert kept == [0, 2, 1]

    def test_the_cap_keeps_the_top_two_after_the_decay(self, ops) -> None:
        kept = run(
            ops,
            self.SOFT_BOXES,
            self.SOFT_SCORES,
            iou_threshold=0.3,
            method="gauss",
            sigma=0.5,
            max_output=2,
        )

        assert kept == [0, 2]

    def test_nms_with_scores_cuts_both_arrays_together(self, ops) -> None:
        """A capped index list beside an uncapped score list is a misalignment nothing
        downstream can detect: every row still parses, and every score belongs to some box."""
        indices, scores = ops.nms_with_scores(
            np.array(self.SOFT_BOXES, dtype=np.float32),
            np.array(self.SOFT_SCORES, dtype=np.float32),
            iou_threshold=0.3,
            method="gauss",
            sigma=0.5,
            max_output=2,
        )

        assert indices.tolist() == [0, 2]
        assert scores.tolist() == pytest.approx([0.9, 0.8])

    @pytest.mark.parametrize("method", ["classic", "linear", "gauss", "neighborhood", "none"])
    def test_every_method_takes_the_cap(self, ops, method: str) -> None:
        """All five, through one seam. The cap lives in the dispatcher precisely so that
        adding a sixth method cannot forget it."""
        kept = run(
            ops,
            FOUR_SURVIVORS,
            FOUR_SCORES,
            iou_threshold=0.5,
            method=method,
            max_output=2,
        )

        assert len(kept) == 2


class TestTheSharedDispatcherAgrees:
    """``suppress`` is what the detection heads call without an ``ImageOps`` in the picture,
    so the cap has to mean the same thing one layer down."""

    @pytest.mark.parametrize("method", ["classic", "linear", "gauss", "neighborhood", "none"])
    def test_suppress_matches_the_backend(self, ops, method: str) -> None:
        boxes = np.array(FOUR_SURVIVORS, dtype=np.float32)
        scores = np.array(FOUR_SCORES, dtype=np.float32)

        expected, expected_scores = suppress(
            boxes, scores, iou_threshold=0.5, method=method, max_output=3
        )
        indices, kept_scores = ops.nms_with_scores(
            boxes, scores, iou_threshold=0.5, method=method, max_output=3
        )

        assert indices.tolist() == expected.tolist()
        assert kept_scores.tolist() == pytest.approx(expected_scores.tolist())

    def test_suppress_returns_a_pair_of_equal_length(self) -> None:
        indices, kept_scores = suppress(
            np.array(FOUR_SURVIVORS, dtype=np.float32),
            np.array(FOUR_SCORES, dtype=np.float32),
            iou_threshold=0.5,
            max_output=1,
        )

        assert indices.shape == kept_scores.shape == (1,)


class TestANonsenseCapIsRefused:
    """Refused at :func:`~shipvision.imgproc.nms.candidates.prepare`, which is the one function
    every method and every backend passes through — including the device kernel.

    The negative case is the reason the check exists rather than being left to a slice.
    Python's ``[:-1]`` drops the *worst* survivor and returns an almost-complete answer; the
    C++ sweep's ``keep.size() < max_output`` is false from the first iteration and returns
    nothing. Neither raises, and a caller comparing two backends would see one frame's worth of
    detections vanish on the GPU path only.
    """

    @pytest.mark.parametrize("cap", [-1, -100])
    def test_a_negative_cap_raises(self, ops, cap: int) -> None:
        with pytest.raises(ConfigurationError, match="max_output"):
            run(ops, FOUR_SURVIVORS, FOUR_SCORES, iou_threshold=0.5, max_output=cap)

    def test_a_fractional_cap_raises(self, ops) -> None:
        """A slice truncates 2.5 to 2 and the binding's ``int`` conversion is a third rule
        again, so the number is refused rather than adjudicated."""
        with pytest.raises(ConfigurationError, match="max_output"):
            run(ops, FOUR_SURVIVORS, FOUR_SCORES, iou_threshold=0.5, max_output=2.5)

    def test_the_check_happens_before_any_work(self, ops) -> None:
        """On an empty frame too, so a quiet camera does not hide a misconfiguration until the
        first frame that has objects in it."""
        with pytest.raises(ConfigurationError, match="max_output"):
            ops.nms(
                np.zeros((0, 4), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                iou_threshold=0.5,
                max_output=-1,
            )
