"""Corroboration-gated suppression — ``shipvision.imgproc.nms.neighborhood``.

The method the C++ reference shipped but could never exercise: it called the neighbourhood
variant with ``neighbors=0`` and ``minScoreSum=0``, so the gate always opened and the method
was indistinguishable from classic NMS. Both facts are asserted here — that the defaults
reproduce the reference, and that the parameters do something once raised.
"""

from __future__ import annotations

import numpy as np

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


def test_a_rejected_anchor_does_not_hand_its_neighbours_a_second_chance(ops) -> None:
    """They are the same object. Releasing them would turn one rejected detection into two
    accepted ones, which is the opposite of what the gate is for."""
    boxes = np.array([ANCHOR, INSIDE_LARGE], dtype=np.float32)
    scores = np.array([0.4, 0.3], dtype=np.float32)

    keep = ops.nms(boxes, scores, iou_threshold=0.5, method="neighborhood", min_score_sum=0.8)

    assert keep.tolist() == []
