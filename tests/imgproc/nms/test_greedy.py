"""Classic and soft suppression — ``shipvision.imgproc.nms.greedy``.

One loop serves all three methods, so the tests are arranged the same way: first the overlap
test they share, then the decay weight that distinguishes them, then the departure rule that
decides whether a punished box is gone or merely demoted.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.imgproc.nms.conftest import (
    ANCHOR,
    DEGENERATE,
    HALF_SHIFTED,
    INSIDE_LARGE,
    INSIDE_SMALL,
    ONE_THIRD,
    run,
)

# --------------------------------------------------------------------- the overlap test


def test_overlap_above_the_threshold_suppresses(ops) -> None:
    assert run(ops, [ANCHOR, HALF_SHIFTED], [0.9, 0.8], iou_threshold=0.3) == [0]


def test_overlap_below_the_threshold_does_not(ops) -> None:
    assert run(ops, [ANCHOR, HALF_SHIFTED], [0.9, 0.8], iou_threshold=0.4) == [0, 1]


def test_overlap_exactly_at_the_threshold_survives(ops) -> None:
    """The test is ``iou > threshold``, strictly.

    Not a detail: it is what makes ``iou_threshold=1.0`` a documented no-op instead of a
    duplicate remover, and it is what the CUDA kernel, ``torchvision.ops.nms`` and the C++
    reference all do. Flipping it to ``>=`` changes the output of every real frame.
    """
    assert run(ops, [ANCHOR, HALF_SHIFTED], [0.9, 0.8], iou_threshold=ONE_THIRD) == [0, 1]


def test_a_threshold_of_one_suppresses_nothing_at_all(ops) -> None:
    assert run(ops, [ANCHOR, ANCHOR], [0.9, 0.8], iou_threshold=1.0) == [0, 1]


# ----------------------------------------------------------------------- containment


def test_a_small_box_inside_a_large_one_is_kept(ops) -> None:
    """IoU 0.04. NMS suppresses overlap, not containment — a person standing in front of a ship
    is a real second detection, and a suppressor that used intersection-over-area would delete
    it."""
    assert run(ops, [ANCHOR, INSIDE_SMALL], [0.9, 0.8], iou_threshold=0.5) == [0, 1]


def test_a_large_box_inside_a_slightly_larger_one_is_suppressed(ops) -> None:
    """IoU 0.81: the same containment relation at a different ratio. Both answers come from the
    same rule, which is the point."""
    assert run(ops, [ANCHOR, INSIDE_LARGE], [0.9, 0.8], iou_threshold=0.5) == [0]


# ------------------------------------------------------------------------ duplicates


def test_identical_boxes_collapse_to_the_highest_score(ops) -> None:
    assert run(ops, [ANCHOR, ANCHOR, ANCHOR], [0.3, 0.9, 0.6], iou_threshold=0.5) == [1]


def test_a_zero_area_box_does_not_divide_by_zero(ops) -> None:
    """Two degenerate boxes at the same spot have IoU 0/0. The guarded denominator turns that
    into 0, so neither suppresses the other and no NaN reaches the caller."""
    assert run(ops, [DEGENERATE, DEGENERATE], [0.9, 0.8], iou_threshold=0.5) == [0, 1]


# ------------------------------------------------------------------- the decay weight


def test_classic_removes_the_overlapping_box(ops) -> None:
    assert run(
        ops, [ANCHOR, HALF_SHIFTED], [0.9, 0.8], iou_threshold=0.3, method="classic"
    ) == [0]


@pytest.mark.parametrize("method", ["linear", "gauss"])
def test_a_soft_method_keeps_the_box_and_lowers_its_score(ops, method: str) -> None:
    """Soft-NMS's output is a re-weighted score, not a shorter list.

    With the default ``score_threshold=0.0`` every index comes back — which is exactly why
    ``nms_with_scores`` exists. A caller who reads only the indices of a soft method has read
    half the answer and applied no suppression at all.
    """
    boxes = np.array([ANCHOR, HALF_SHIFTED], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)

    keep, decayed = ops.nms_with_scores(
        boxes, scores, iou_threshold=0.3, method=method, sigma=0.5
    )

    assert keep.tolist() == [0, 1]
    assert ops.nms(boxes, scores, iou_threshold=0.3, method=method).tolist() == keep.tolist()
    assert decayed[0] == pytest.approx(0.9)
    assert decayed[1] < 0.8


def test_linear_decay_is_one_minus_the_overlap(ops) -> None:
    """``0.8 * (1 - 1/3)``, worked out by hand. The formula is the soft-NMS paper's, and the
    C++ reference computed the same thing."""
    keep, decayed = ops.nms_with_scores(
        np.array([ANCHOR, HALF_SHIFTED], dtype=np.float32),
        np.array([0.9, 0.8], dtype=np.float32),
        iou_threshold=0.3,
        method="linear",
    )

    assert keep.tolist() == [0, 1]
    assert decayed[1] == pytest.approx(0.8 * (1.0 - ONE_THIRD), abs=1e-6)


def test_gauss_decay_is_the_exponential_of_the_squared_overlap(ops) -> None:
    """``0.8 * exp(-(1/3)^2 / 0.5)``. Sigma is the gaussian's width, and only gauss reads it."""
    _, decayed = ops.nms_with_scores(
        np.array([ANCHOR, HALF_SHIFTED], dtype=np.float32),
        np.array([0.9, 0.8], dtype=np.float32),
        iou_threshold=0.3,
        method="gauss",
        sigma=0.5,
    )

    assert decayed[1] == pytest.approx(0.8 * np.exp(-(ONE_THIRD**2) / 0.5), abs=1e-6)


# --------------------------------------------------------------------- the departure rule


def test_a_soft_method_drops_a_box_it_decays_under_the_threshold(ops) -> None:
    """The score threshold is what turns soft-NMS from a re-ranker into a suppressor.

    ``0.8 * (1 - 1/3)`` is about 0.533, so a floor of 0.6 removes the box while a floor of 0.5
    keeps it. Nothing else in the input changes.
    """
    boxes = np.array([ANCHOR, HALF_SHIFTED], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)

    kept = ops.nms(boxes, scores, iou_threshold=0.3, method="linear", score_threshold=0.5)
    dropped = ops.nms(boxes, scores, iou_threshold=0.3, method="linear", score_threshold=0.6)

    assert kept.tolist() == [0, 1]
    assert dropped.tolist() == [0]


def test_a_soft_method_removes_a_perfect_duplicate_outright(ops) -> None:
    """Linear's weight at IoU 1.0 is zero, and a box worth zero is not a detection.

    This is the "or reaches zero" half of the departure rule. Without it the duplicate would
    come back with a score of 0.0 attached, and a caller thresholding at 0.0 would publish it —
    which is also, exactly, why ``classic`` needs no special case.
    """
    assert run(ops, [ANCHOR, ANCHOR], [0.9, 0.8], iou_threshold=0.3, method="linear") == [0]
