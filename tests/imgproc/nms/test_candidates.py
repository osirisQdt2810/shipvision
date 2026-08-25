"""Who competes, and in what order — ``shipvision.imgproc.nms.candidates``.

Admission and ordering are decided before any suppression happens, and both are shared by
every method and by the backends' accelerated paths. A device kernel handed a different
candidate set than the reference used is not comparable to it, so these two rules are pinned
here rather than inside any one method's tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipvision.errors import ConfigurationError, DimensionMismatchError
from shipvision.imgproc import METHODS
from tests.imgproc.nms.conftest import ANCHOR, FAR_AWAY, HALF_SHIFTED, INSIDE_SMALL, run

# ------------------------------------------------------------------------- admission


class TestAdmission:
    def test_the_score_threshold_is_inclusive(self, ops) -> None:
        """``score >= score_threshold``, matching the CUDA kernel.

        The C++ reference used a strict ``>``; matching the kernel is what keeps the backends
        comparable, and it is what makes the default of 0.0 admit every valid score instead of
        quietly dropping the ones that are exactly zero.
        """
        assert run(ops, [ANCHOR], [0.5], iou_threshold=0.5, score_threshold=0.5) == [0]
        assert run(ops, [ANCHOR], [0.5], iou_threshold=0.5, score_threshold=0.50001) == []

    def test_no_boxes_gives_no_indices(self, ops) -> None:
        keep = ops.nms(
            np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32), iou_threshold=0.5
        )

        assert keep.shape == (0,)
        assert keep.dtype == np.int64

    def test_one_box_survives_itself(self, ops) -> None:
        assert run(ops, [ANCHOR], [0.5], iou_threshold=0.5) == [0]

    # -------------------------------------------------------------------------- ordering

    def test_survivors_come_back_in_descending_score_order(self, ops) -> None:
        """Order is part of the answer, not a side effect: downstream code that takes the first
        *k* survivors depends on it."""
        boxes = np.array([ANCHOR, FAR_AWAY, INSIDE_SMALL], dtype=np.float32)
        scores = np.array([0.4, 0.9, 0.6], dtype=np.float32)

        keep, kept_scores = ops.nms_with_scores(boxes, scores, iou_threshold=0.5)

        assert keep.tolist() == [1, 2, 0]
        assert kept_scores.tolist() == pytest.approx([0.9, 0.6, 0.4])

    def test_a_tie_breaks_towards_the_lower_input_index(self, ops) -> None:
        """A stable descending sort, so the same input always gives the same output.

        An unstable sort would make a duplicated proposal resolve differently between runs, and the
        only visible symptom is a tracker that occasionally swaps two ids — a regression nobody can
        reproduce. ``torchvision.ops.nms`` sorts stably too, so the backends agree here.
        """
        assert run(ops, [ANCHOR, ANCHOR], [0.7, 0.7], iou_threshold=0.5) == [0]

    # -------------------------------------------------------------------------- no method

    def test_none_applies_the_score_threshold_and_nothing_else(self, ops) -> None:
        """Which makes it exactly what this module does, with no suppression on top.

        It exists so a benchmark can measure what suppression is worth rather than assume it.
        """
        keep = run(
            ops,
            [ANCHOR, ANCHOR, ANCHOR],
            [0.9, 0.2, 0.8],
            iou_threshold=0.5,
            method="none",
            score_threshold=0.5,
        )

        assert keep == [0, 2]

    def test_every_documented_method_runs(self, ops) -> None:
        boxes = np.array([ANCHOR, HALF_SHIFTED, FAR_AWAY], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

        for method in METHODS:
            keep = ops.nms(boxes, scores, iou_threshold=0.3, method=method)
            assert keep.dtype == np.int64
            # The anchor outscores everything it overlaps, so it survives every method.
            assert 0 in keep.tolist()

    # -------------------------------------------------------------------------- refusals

    def test_a_score_and_box_count_mismatch_is_refused(self, ops) -> None:
        """A broadcast here would silently score the wrong box, which is worse than a crash."""
        with pytest.raises(DimensionMismatchError):
            ops.nms(
                np.array([ANCHOR, FAR_AWAY], dtype=np.float32),
                np.array([0.9], dtype=np.float32),
                iou_threshold=0.5,
            )

    def test_an_unknown_method_is_refused_by_name(self, ops) -> None:
        with pytest.raises(ConfigurationError, match="unknown nms method"):
            ops.nms(
                np.array([ANCHOR], dtype=np.float32),
                np.array([0.9], dtype=np.float32),
                iou_threshold=0.5,
                method="softest",
            )

    def test_an_out_of_range_iou_threshold_is_refused(self, ops) -> None:
        with pytest.raises(ConfigurationError):
            ops.nms(
                np.array([ANCHOR], dtype=np.float32),
                np.array([0.9], dtype=np.float32),
                iou_threshold=1.5,
            )

    def test_gauss_refuses_a_non_positive_sigma(self, ops) -> None:
        """It is a divisor. Validated at the call rather than producing an inf weight."""
        with pytest.raises(ConfigurationError):
            ops.nms(
                np.array([ANCHOR], dtype=np.float32),
                np.array([0.9], dtype=np.float32),
                iou_threshold=0.5,
                method="gauss",
                sigma=0.0,
            )
