"""Corroboration-gated suppression — ``shipvision.imgproc.nms.neighborhood``.

The method the C++ reference shipped but could never exercise: it called the neighbourhood
variant with ``neighbors=0`` and ``minScoreSum=0``, so the gate always opened and the method
was indistinguishable from classic NMS. Both facts are asserted here — that the defaults
reproduce the reference, and that the parameters do something once raised.
"""

from __future__ import annotations

import numpy as np

from shipvision.types import iou_matrix
from tests.imgproc.nms.conftest import ANCHOR, FAR_AWAY, HALF_SHIFTED, INSIDE_LARGE


def test_the_default_parameters_reproduce_classic(ops) -> None:
    boxes = np.array([ANCHOR, HALF_SHIFTED, INSIDE_LARGE, FAR_AWAY], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32)

    assert ops.nms(boxes, scores, iou_threshold=0.5, method="neighborhood").tolist() == (
        ops.nms(boxes, scores, iou_threshold=0.5, method="classic").tolist()
    )


def test_a_box_nothing_corroborates_is_dropped(ops) -> None:
    """Raised to ``min_neighbors=1``, a lone proposal is no longer evidence.

    ``ANCHOR`` has ``INSIDE_LARGE`` agreeing with it at IoU 0.81, so it survives. ``FAR_AWAY``
    is on its own, so it does not — even though it cleared the score threshold. That is the
    whole reason the method exists, and with the reference's parameters it can never be
    observed.
    """
    boxes = np.array([ANCHOR, INSIDE_LARGE, FAR_AWAY], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

    keep = ops.nms(boxes, scores, iou_threshold=0.5, method="neighborhood", min_neighbors=1)

    assert keep.tolist() == [0]


def test_a_cluster_whose_scores_do_not_add_up_is_dropped(ops) -> None:
    """``min_score_sum`` reads the anchor plus its suppressed neighbours: 0.4 + 0.3 = 0.7, so a
    demand of 0.6 passes and 0.8 does not. It is a statement about the cluster, not about the
    best box in it."""
    boxes = np.array([ANCHOR, INSIDE_LARGE], dtype=np.float32)
    scores = np.array([0.4, 0.3], dtype=np.float32)

    assert ops.nms(
        boxes, scores, iou_threshold=0.5, method="neighborhood", min_score_sum=0.6
    ).tolist() == [0]
    assert (
        ops.nms(
            boxes, scores, iou_threshold=0.5, method="neighborhood", min_score_sum=0.8
        ).tolist()
        == []
    )


class TestARejectedAnchorConsumesItsNeighbours:
    """They are the same object. Releasing them would turn one rejected detection into two
    accepted ones, which is the opposite of what the gate is for.

    The case has to be built so that a released neighbour would *pass* a gate its anchor
    failed, or the assertion holds either way and proves nothing. The previous version of this
    test used ``[ANCHOR, INSIDE_LARGE]`` with scores 0.4 and 0.3 against
    ``min_score_sum=0.8``: releasing ``INSIDE_LARGE`` gives it a cluster of 0.3, which fails
    the same gate, so the answer was ``[]`` whether the neighbours were released or not.

    The chain below fixes that. Four 10x10 boxes in a row, three pixels apart, so overlap is
    transitive but not universal — ``iou(A, B) = 0.538`` and ``iou(B, D) = 0.818`` while
    ``iou(A, C) = 0.25`` and ``iou(A, D) = 0.429``. A therefore has one neighbour and B, if it
    were ever anchored, would have two. A released B passes a gate A failed, on either
    parameter, and the correct answer is still ``[]``.
    """

    CHAIN = [
        [0.0, 0.0, 10.0, 10.0],  # A: neighbours = {B}
        [3.0, 0.0, 13.0, 10.0],  # B: neighbours = {A, C, D}
        [6.0, 0.0, 16.0, 10.0],  # C: neighbours = {B, D}
        [4.0, 0.0, 14.0, 10.0],  # D: neighbours = {B, C}
    ]
    SCORES = [0.9, 0.8, 0.7, 0.6]

    def keep(self, ops, **kwargs) -> list[int]:
        return ops.nms(
            np.array(self.CHAIN, dtype=np.float32),
            np.array(self.SCORES, dtype=np.float32),
            iou_threshold=0.5,
            method="neighborhood",
            **kwargs,
        ).tolist()

    def test_the_neighbour_count_gate_does_not_release_them(self, ops) -> None:
        """A has one neighbour and fails ``min_neighbors=2``; B has three and would pass it.

        An implementation that put B back in the pool returns ``[1]`` here. This is the
        assertion the old test could not make.
        """
        assert self.keep(ops, min_neighbors=2) == []

    def test_the_score_sum_gate_does_not_release_them_either(self, ops) -> None:
        """A's cluster is 0.9 + 0.8 = 1.7 and fails a demand of 2.0; B's would be
        0.8 + 0.7 + 0.6 = 2.1 and would pass it. Same shape of failure, other parameter."""
        assert self.keep(ops, min_score_sum=2.0) == []

    def test_the_chain_really_is_asymmetric(self, ops) -> None:
        """The premise, asserted rather than asserted-in-a-comment.

        If the overlaps ever stopped being transitive-but-not-universal the two tests above
        would quietly become vacuous again, which is exactly the failure mode being fixed.
        """
        overlaps = iou_matrix(
            np.array(self.CHAIN, dtype=np.float32), np.array(self.CHAIN, dtype=np.float32)
        )

        assert overlaps[0, 1] > 0.5, "A must suppress B"
        assert overlaps[0, 2] < 0.5 and overlaps[0, 3] < 0.5, "A must not reach C or D"
        assert overlaps[1, 2] > 0.5 and overlaps[1, 3] > 0.5, "B must reach both"

    def test_lowering_the_gate_admits_the_whole_cluster(self, ops) -> None:
        """The control: with the gate open, A survives and its neighbours are still consumed,
        so a released neighbour is the only thing that could add an index."""
        assert self.keep(ops, min_neighbors=1) == [0, 2]
