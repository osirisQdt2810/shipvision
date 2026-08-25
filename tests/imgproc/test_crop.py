"""Cropping: what happens at and past the frame border, and to a box with no area.

The interesting cases are all degenerate ones, because a detector produces them constantly. A
box on the edge of the frame has coordinates a pixel or two outside it — that is where the
objects entering and leaving a scene are, so refusing those boxes throws away exactly the
detections a tracker most needs. And a zero-area box is a division by the crop's span waiting
to happen. Neither may raise, and neither may quietly wrap around to the other side of the
image, which is what an unclamped index does.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipvision.errors import DimensionMismatchError
from shipvision.imgproc import IMGPROC
from shipvision.registry import PYTHON

TARGET = (32, 16)
MEAN = (10.0, 20.0, 30.0)
STD = (2.0, 4.0, 8.0)
"""Deliberately not the defaults. With ``mean=0`` a zero-value crop normalises to zero and a
test cannot tell "went through normalisation" from "never written"."""


@pytest.fixture()
def ops():
    return IMGPROC.build("default", backend=PYTHON)


@pytest.fixture()
def frame() -> np.ndarray:
    rng = np.random.default_rng(4)
    return rng.integers(0, 256, size=(120, 200, 3), dtype=np.uint8)


def _crop(ops, frame: np.ndarray, boxes: list[list[float]], **kwargs) -> np.ndarray:
    return ops.crop_batch(frame, np.array(boxes, dtype=np.float32), TARGET, **kwargs)


# ----------------------------------------------------------------------- the border


class TestTheBorder:
    def test_a_box_partly_outside_the_frame_is_clamped_not_dropped(self, ops, frame) -> None:
        """And clamped to exactly the same pixels an explicitly clamped box would read.

        Asserting equality against the clamped box, rather than merely that the call returned,
        is what rules out the two ways this goes wrong quietly: a negative index that wraps to
        the far side of the image, and a silently skipped crop that leaves the row at zero.
        """
        frame_h, frame_w = frame.shape[:2]

        overflowing = _crop(ops, frame, [[-25.0, -40.0, 60.0, 70.0]])
        clamped = _crop(ops, frame, [[0.0, 0.0, 60.0, 70.0]])

        assert overflowing.shape == (1, 3, *TARGET)
        assert np.array_equal(overflowing, clamped)
        assert np.isfinite(overflowing).all()
        assert (frame_h, frame_w) == (120, 200)

    def test_a_box_past_the_far_edge_clamps_to_the_last_addressable_pixel(
        self, ops, frame
    ) -> None:
        """``[0, extent - 1]``, because these are sampling coordinates and not box edges."""
        beyond = _crop(ops, frame, [[150.0, 80.0, 5_000.0, 5_000.0]])
        clamped = _crop(ops, frame, [[150.0, 80.0, 199.0, 119.0]])

        assert np.array_equal(beyond, clamped)

    def test_a_box_entirely_outside_the_frame_yields_a_blank_crop(self, ops, frame) -> None:
        """It clamps to a zero-area box at the corner, which is the degenerate case below — not
        an exception, and not a read from unmapped memory."""
        blank = _crop(ops, frame, [[-500.0, -500.0, -400.0, -400.0]], mean=MEAN, std=STD)

        expected = -np.asarray(MEAN, dtype=np.float32) / np.asarray(STD, dtype=np.float32)
        assert blank[0] == pytest.approx(np.broadcast_to(expected[:, None, None], (3, *TARGET)))

    # -------------------------------------------------------------------- no area at all

    def test_a_zero_area_box_does_not_divide_by_zero(self, ops, frame) -> None:
        """The crop's span is the divisor in ``low + (i + 0.5) * span / target``.

        A zero span would be a division by zero on the *inverse* formulation; on this one it
        collapses every sample onto one coordinate, which is worse — a plausible-looking crop of
        a single repeated pixel. Both backends short-circuit to source value zero instead, and
        that value is then normalised like any other, so the tensor stays finite and mean-shifted
        rather than becoming a hole.
        """
        crops = _crop(ops, frame, [[50.0, 50.0, 50.0, 50.0]], mean=MEAN, std=STD)

        expected = -np.asarray(MEAN, dtype=np.float32) / np.asarray(STD, dtype=np.float32)
        assert np.isfinite(crops).all()
        assert crops[0] == pytest.approx(np.broadcast_to(expected[:, None, None], (3, *TARGET)))

    def test_an_inside_out_box_is_treated_as_having_no_area(self, ops, frame) -> None:
        """``x2 < x1`` is what an xywh-to-xyxy converter emits when it subtracts instead of
        adding. It must not sample a negative span."""
        crops = _crop(ops, frame, [[80.0, 90.0, 20.0, 30.0]], mean=MEAN, std=STD)

        expected = -np.asarray(MEAN, dtype=np.float32) / np.asarray(STD, dtype=np.float32)
        assert np.isfinite(crops).all()
        assert crops[0] == pytest.approx(np.broadcast_to(expected[:, None, None], (3, *TARGET)))

    def test_a_blank_crop_does_not_contaminate_its_neighbours(self, ops, frame) -> None:
        """A degenerate box in the middle of a real batch must cost exactly its own row.

        Killing the batch over one bad box is the wrong trade — the frame's other fourteen people
        are fine — and so is letting the bad row's shape shift everything after it.
        """
        boxes = [[10.0, 10.0, 60.0, 60.0], [5.0, 5.0, 5.0, 5.0], [90.0, 40.0, 150.0, 100.0]]

        crops = _crop(ops, frame, boxes)
        alone = _crop(ops, frame, [boxes[0], boxes[2]])

        assert crops.shape == (3, 3, *TARGET)
        assert np.array_equal(crops[0], alone[0])
        assert np.array_equal(crops[2], alone[1])
        assert crops[1].min() == crops[1].max() == 0.0

    # ------------------------------------------------------------------ ordinary crops

    def test_no_boxes_gives_an_empty_batch_not_an_error(self, ops, frame) -> None:
        """A frame with nothing on it is the common case on most cameras, all night."""
        crops = ops.crop_batch(frame, np.zeros((0, 4), dtype=np.float32), TARGET)

        assert crops.shape == (0, 3, *TARGET)
        assert crops.dtype == np.float32

    def test_a_crop_of_a_uniform_region_is_that_colour_in_rgb(self, ops) -> None:
        """Pins the channel swap and the normalisation without any interpolation in the way."""
        frame = np.zeros((60, 60, 3), dtype=np.uint8)
        frame[..., 0] = 40  # blue
        frame[..., 1] = 80  # green
        frame[..., 2] = 160  # red

        crops = _crop(ops, frame, [[5.0, 5.0, 40.0, 40.0]], mean=MEAN, std=STD)

        assert crops[0, 0] == pytest.approx((160 - 10) / 2)
        assert crops[0, 1] == pytest.approx((80 - 20) / 4)
        assert crops[0, 2] == pytest.approx((40 - 30) / 8)

    def test_a_crop_follows_the_box_along_each_axis_independently(self, ops) -> None:
        """A horizontal ramp read left and right, and a vertical one read top and bottom.

        The cheapest test there is for a transposed crop: swap the x and y coordinates anywhere in
        the sampling and one of these two comparisons inverts, while a random-image parity test
        against a backend that made the *same* swap would still pass.
        """
        horizontal = np.zeros((100, 200, 3), dtype=np.uint8)
        horizontal[:] = np.arange(200, dtype=np.uint8)[None, :, None]
        vertical = np.transpose(horizontal, (1, 0, 2)).copy()

        left = _crop(ops, horizontal, [[0.0, 0.0, 40.0, 90.0]])
        right = _crop(ops, horizontal, [[150.0, 0.0, 190.0, 90.0]])
        top = _crop(ops, vertical, [[0.0, 0.0, 90.0, 40.0]])
        bottom = _crop(ops, vertical, [[0.0, 150.0, 90.0, 190.0]])

        assert left.mean() < right.mean()
        assert top.mean() < bottom.mean()

    def test_the_output_extent_is_the_target_not_the_box(self, ops, frame) -> None:
        """Boxes of wildly different sizes all come back the same shape, which is what makes them
        one batch for the embedding model."""
        crops = _crop(ops, frame, [[0.0, 0.0, 4.0, 4.0], [0.0, 0.0, 190.0, 110.0]])

        assert crops.shape == (2, 3, *TARGET)

    def test_a_malformed_box_array_is_refused(self, ops, frame) -> None:
        with pytest.raises(DimensionMismatchError):
            ops.crop_batch(frame, np.zeros((4, 5), dtype=np.float32), TARGET)

    def test_a_malformed_frame_is_refused(self, ops) -> None:
        with pytest.raises(DimensionMismatchError):
            ops.crop_batch(
                np.zeros((10, 10), dtype=np.uint8), np.zeros((1, 4), dtype=np.float32), TARGET
            )
