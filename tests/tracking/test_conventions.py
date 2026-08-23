"""The box conventions tracking is built on, asserted from tracking's point of view.

:mod:`shipvision.types` has its own tests and they are more thorough than these. These exist
anyway, and deliberately duplicate a little, because they are the four facts that every file
in this package silently assumes. If one of them changed, the symptom would not be a failing
type test — it would be five trackers that still run, still publish, and quietly track square
objects perfectly while falling apart on a ship. A test in *this* directory is where someone
debugging that would look.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipvision.errors import ConfigurationError
from shipvision.types import Detection, cxcyah_to_xyxy, iou_matrix, xyxy_to_cxcyah


class TestBoxConventions:
    """The parameterisations, and the ones a wrong converter would silently break."""

    def test_the_box_parameterisations_round_trip(self) -> None:
        """Every association step converts between these. A round trip that loses a pixel
        accumulates into drift that looks like a tracking failure."""
        boxes = np.array(
            [[10, 20, 110, 220], [0, 0, 4, 8], [50.5, 60.25, 90.75, 160.5]], np.float32
        )
        np.testing.assert_allclose(cxcyah_to_xyxy(xyxy_to_cxcyah(boxes)), boxes, atol=1e-4)

    def test_the_filter_state_is_aspect_and_height(self) -> None:
        """Not width and height: height is far more stable under occlusion, so a filter on it
        rides out a partial hide instead of fighting it."""
        state = xyxy_to_cxcyah(np.array([[10, 20, 110, 220]], np.float32))[0]
        assert state.tolist() == pytest.approx([60.0, 120.0, 0.5, 200.0])

    def test_a_detection_rejects_a_malformed_box(self) -> None:
        """The typed refusal is the point: a five-value "box" is a converter bug upstream, and
        letting it through produces a cost matrix that is wrong rather than absent."""
        with pytest.raises(ConfigurationError, match="xyxy"):
            Detection(box=np.zeros(5, np.float32), score=0.9)


class TestOverlap:
    """IoU on inputs whose answer can be worked out on paper, including the degenerate ones
    that reach the cost matrix on the same code path as a real box.
    """

    def test_iou_on_hand_computed_boxes(self) -> None:
        a = np.array([[0, 0, 10, 10]], np.float32)
        b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], np.float32)
        #   identical -> 1; half overlap -> 50/150; disjoint -> 0
        np.testing.assert_allclose(iou_matrix(a, b)[0], [1.0, 1 / 3, 0.0], atol=1e-6)

    def test_iou_handles_empty_and_degenerate_input(self) -> None:
        """A camera can legitimately see nothing, and a detector can legitimately emit a
        zero-area box. Both reach the cost matrix on the same code path as a real box."""
        empty = np.zeros((0, 4), np.float32)
        assert iou_matrix(empty, np.array([[0, 0, 1, 1]], np.float32)).shape == (0, 1)
        # A degenerate box has no area; it must score zero rather than a negative overlap.
        flat = np.array([[10, 10, 10, 10]], np.float32)
        assert iou_matrix(flat, np.array([[0, 0, 20, 20]], np.float32))[0, 0] == 0.0
