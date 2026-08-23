"""The sampling conventions, pinned against numbers written by hand.

Every other convention test in this directory uses an *exact* scale — 2 into 4, 1x2 into 2x4,
a span-4 crop into 2 output pixels — and at an exact scale the two candidate formulas for the
sampling ratio coincide. ``source_extent / resized_extent`` (the achieved ratio, which
:mod:`shipvision.imgproc.geometry` states is the rule, and which the CUDA kernel implements as
``view.height / view.out_h``) is equal to ``1 / scale`` whenever ``source * scale`` is already
whole. They are only different where it is not, and the fleet's own resolution is such a case:
1077x1920 into 512x512 gives ``scale = 0.2666667`` and a resized height of 287, so the
achieved ratio is ``1077 / 287 = 3.7526`` while ``1 / scale`` is ``3.75``.

That gap is not subtle where it matters — on a 1077x1920 frame into 512x512 the two produce
per-pixel differences up to **0.677** on a [0, 1] tensor, 677 times the parity suite's 1e-3
tolerance. But it was invisible to the offline tier, because the only tests that could see it
compared one backend against another and all three backends call the same
:func:`~shipvision.imgproc.geometry.resize_centres`. Change that function and every backend
moves together; the parity suite stays green and every box in production shifts.

So the assertions here compare against arithmetic, not against another implementation. The
numbers are worked out in the docstrings, the distinguishing cases have non-integer scales, and
one of them writes out the *wrong* formula explicitly and asserts the output does not match it.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipvision.imgproc import IMGPROC
from shipvision.imgproc.geometry import LetterboxGeometry, crop_centres, resize_centres
from shipvision.registry import PYTHON

FLEET_SOURCE = (1077, 1920)
"""1080p with three rows missing — the shape that makes the vertical extent round rather than
divide, and the one the parity suite already uses for that reason."""

FLEET_TARGET = (512, 512)


@pytest.fixture()
def ops():
    """The oracle. Every backend inherits its geometry, so pinning it pins all three."""
    return IMGPROC.build("default", backend=PYTHON)


def _bilinear(source: np.ndarray, src_y: float, src_x: float, channel: int) -> float:
    """One bilinear tap, written out here rather than imported.

    Deliberately a second implementation: importing the library's gather would make this file
    agree with the library by construction, which is the hole it exists to fill. Border
    handling is the kernel's — clamp the tap *index*, keep the unclamped ``floor`` in the
    weight.
    """
    height, width = source.shape[:2]
    y0, x0 = int(np.floor(src_y)), int(np.floor(src_x))
    y1, x1 = min(y0 + 1, height - 1), min(x0 + 1, width - 1)
    yc, xc = max(y0, 0), max(x0, 0)
    wy, wx = src_y - y0, src_x - x0
    top = source[yc, xc, channel] * (1.0 - wx) + source[yc, x1, channel] * wx
    bottom = source[y1, xc, channel] * (1.0 - wx) + source[y1, x1, channel] * wx
    return float(top * (1.0 - wy) + bottom * wy)


class TestResizeCentresAtANonIntegerScale:
    """Convention 1, at a scale where the achieved ratio and ``1 / scale`` disagree.

    1077 source rows into 287 resized rows. The ratio is ``1077 / 287 = 3.7526132...`` and the
    centre of output row *i* is ``(i + 0.5) * 1077 / 287 - 0.5``, multiplying before dividing
    exactly as ``imgproc_image_ops.cu`` does.
    """

    def test_the_first_centre_is_the_half_pixel_offset(self) -> None:
        """``0.5 * 1077 / 287 - 0.5 = 538.5 / 287 - 0.5 = 1.8763066 - 0.5``.

        ``1 / scale`` would give ``0.5 * 3.75 - 0.5 = 1.375``, which is 0.0013 lower — small
        at the top of the image, and the same error is 0.75 pixels at the bottom.
        """
        assert resize_centres(1077, 287)[0] == pytest.approx(1.3763067, abs=1e-6)

    def test_the_middle_centre_lands_exactly_half_a_pixel_above_the_source_middle(self) -> None:
        """287 is odd, so output row 143 is the middle one and ``i + 0.5 = 143.5``.

        ``143.5 * 1077 = 154549.5``, and ``154549.5 / 287`` is exactly ``538.5`` — half of
        1077 — so the centre is ``538.0``. An exact number by construction, which is why this
        index is the one worth asserting: ``1 / scale`` gives ``143.5 * 3.75 - 0.5 =
        537.625``, and 0.375 of a pixel cannot hide in a rounding argument.
        """
        centres = resize_centres(1077, 287)

        assert centres[143] == pytest.approx(538.0, abs=1e-4)
        assert centres[143] != pytest.approx(537.625, abs=1e-2)

    def test_the_last_centre_is_three_quarters_of_a_pixel_from_the_wrong_answer(self) -> None:
        """``286.5 * 1077 / 287 - 0.5 = 308560.5 / 287 - 0.5 = 1075.1237 - 0.5``.

        Against ``286.5 * 3.75 - 0.5 = 1073.875``. The error grows linearly down the image,
        which is why a convention mistake shifts every box on every camera by an amount that
        depends on where in the frame the object was.
        """
        centres = resize_centres(1077, 287)

        assert centres[286] == pytest.approx(1074.6237, abs=1e-3)
        assert centres[286] - 1073.875 == pytest.approx(0.7488, abs=1e-3)

    def test_an_exact_scale_hides_the_difference_entirely(self) -> None:
        """Which is why this file exists.

        1920 into 512 is exactly 3.75, so here the achieved ratio *is* ``1 / scale`` and both
        formulas agree to the bit. Every convention test written before this one used a case
        of this kind.
        """
        centres = resize_centres(1920, 512)
        one_over_scale = (np.arange(512, dtype=np.float32) + np.float32(0.5)) * np.float32(
            3.75
        ) - np.float32(0.5)

        assert centres.tolist() == pytest.approx(one_over_scale.tolist())

    def test_the_achieved_ratio_is_not_one_over_scale_at_the_fleet_resolution(self) -> None:
        """The premise, asserted so it cannot rot: 1077x1920 into 512x512 is a distinguishing
        case, and the resized extent is 287 rather than 287.2 rows."""
        geometry = LetterboxGeometry.plan(FLEET_SOURCE, FLEET_TARGET)

        assert geometry.resized_height == 287
        assert 1077 / geometry.resized_height == pytest.approx(3.7526132, abs=1e-6)
        assert 1.0 / geometry.scale == pytest.approx(3.75, abs=1e-6)


class TestLetterboxPixelsAtANonIntegerScale:
    """The same convention, asserted on the tensor a caller actually receives.

    ``resize_centres`` could be right and a backend could still ignore it, so these go through
    :meth:`~shipvision.imgproc.base.ImageOps.letterbox` and check individual pixel values.
    Both cases have a **non-integer** scale on the sampled axis and an **odd** total pad, so
    they pin conventions 1, 2 and 3 at once. ``std=(1, 1, 1)`` keeps the numbers in the 0-255
    source scale, where they can be read.
    """

    ROW = [0.0, 100.0, 200.0, 40.0]
    """Four source values with no symmetry, so a mirrored or shifted sample cannot coincide."""

    EXPECTED = [16.666668, 150.0, 66.666679]
    """The three output values, by hand.

    A 5x4 source into a 4x4 canvas scales by ``min(4/5, 4/4) = 0.8``, so the resized extent is
    ``round(4 * 0.8) = 3`` and the horizontal ratio is the achieved ``4 / 3 = 1.333...``, not
    ``1 / 0.8 = 1.25``. The centres are ``(i + 0.5) * 4 / 3 - 0.5`` = 0.1666667, 1.5,
    2.8333333, so:

    * ``0.1666667``: between source 0 and 100 at weight 1/6 -> ``100/6 = 16.666667``
    * ``1.5``:       midway between 100 and 200 -> ``150``
    * ``2.8333333``: between 200 and 40 at weight 5/6 -> ``200/6 + 40*5/6 = 66.666667``

    Under ``1 / scale`` the centres would be 0.125, 1.375, 2.625 and the values 12.5, 137.5
    and 100 — nowhere near, at any tolerance.
    """

    def test_a_horizontal_resize_reads_the_hand_computed_taps(self, ops) -> None:
        """5 rows into 4 leaves the vertical axis exact; only the horizontal one rounds."""
        image = np.zeros((5, 4, 3), dtype=np.uint8)
        image[:, :, 2] = np.array(self.ROW, dtype=np.uint8)  # BGR channel 2 -> RGB channel 0

        batch, (geometry,) = ops.letterbox(image, (4, 4), pad_value=7, std=(1.0, 1.0, 1.0))

        assert (geometry.resized_height, geometry.resized_width) == (4, 3)
        assert (geometry.pad_top, geometry.pad_left) == (0, 0)
        assert (geometry.pad_bottom, geometry.pad_right) == (0, 1)
        for row in range(4):
            assert batch[0, 0, row, 0:3].tolist() == pytest.approx(self.EXPECTED, abs=1e-4)

    def test_the_odd_pad_pixel_is_on_the_right(self, ops) -> None:
        """Convention 3, and it is what makes the case above unambiguous: the resized band is
        three columns wide, so the fourth column is a bar and nothing else can be."""
        image = np.zeros((5, 4, 3), dtype=np.uint8)
        image[:, :, 2] = np.array(self.ROW, dtype=np.uint8)

        batch, _ = ops.letterbox(image, (4, 4), pad_value=7, std=(1.0, 1.0, 1.0))

        assert np.array_equal(batch[0, :, :, 3], np.full((3, 4), 7.0, dtype=np.float32))

    def test_a_vertical_resize_reads_the_same_taps_down_the_other_axis(self, ops) -> None:
        """The transpose of the case above: a 4x5 source into 4x4, so the *vertical* extent is
        ``round(4 * 0.8) = 3`` and the odd pad row is at the bottom.

        Worth its own case because the two axes are separate expressions in every backend —
        ``view.height / view.out_h`` and ``view.width / view.out_w`` in the kernel — and a
        convention fixed on one axis only would still shift every box vertically.
        """
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        image[:, :, 2] = np.array(self.ROW, dtype=np.uint8)[:, None]

        batch, (geometry,) = ops.letterbox(image, (4, 4), pad_value=7, std=(1.0, 1.0, 1.0))

        assert (geometry.resized_height, geometry.resized_width) == (3, 4)
        assert (geometry.pad_top, geometry.pad_bottom) == (0, 1)
        assert batch[0, 0, 0:3, 0].tolist() == pytest.approx(self.EXPECTED, abs=1e-4)
        assert batch[0, 0, 3, :].tolist() == pytest.approx([7.0, 7.0, 7.0, 7.0])


class TestTheWrongRatioIsMeasurablyWrong:
    """The fleet resolution, against both candidate formulas at once.

    The two tests above are small enough to reason about; this one is the real frame. It
    computes an output row twice — once with the achieved ratio and once with ``1 / scale`` —
    and asserts the backend matches the first and *differs from the second*. Asserting the
    difference matters as much as asserting the match: it is the evidence that this input can
    tell the two conventions apart, so a future reader cannot conclude the case is vacuous.
    """

    def test_the_bottom_of_the_resized_band_matches_the_achieved_ratio(self, ops) -> None:
        rng = np.random.default_rng(1077)
        image = rng.integers(0, 256, size=(*FLEET_SOURCE, 3), dtype=np.uint8)
        geometry = LetterboxGeometry.plan(FLEET_SOURCE, FLEET_TARGET)
        batch, _ = ops.letterbox(image, FLEET_TARGET, std=(1.0, 1.0, 1.0))

        # Output row 286 is the last row of the resized band, where the two formulas are
        # furthest apart: 1074.6237 source rows against 1073.8748, three quarters of a pixel.
        resized_row = 286
        achieved_y = (resized_row + 0.5) * 1077 / 287 - 0.5
        mutated_y = (resized_row + 0.5) / (512 / 1920) - 0.5
        xs = [(x + 0.5) * 1920 / 512 - 0.5 for x in range(0, 512, 37)]

        actual = batch[0, 0, geometry.pad_top + resized_row, ::37]
        achieved = [_bilinear(image, achieved_y, x, 2) for x in xs]
        mutated = [_bilinear(image, mutated_y, x, 2) for x in xs]

        assert actual.tolist() == pytest.approx(achieved, abs=0.02)
        assert np.abs(actual - np.array(mutated)).max() > 5.0, (
            "this frame cannot distinguish the two conventions, so the test above proves "
            "nothing — pick a noisier image or a row further down"
        )

    def test_the_gap_between_the_conventions_dwarfs_the_parity_tolerance(self, ops) -> None:
        """Quantified, in the units the parity suite works in.

        The parity tests admit 1e-3 on a [0, 1] tensor. A convention error produces two orders
        of magnitude more than that on a noise frame, which is the argument that no tolerance
        can hide one while admitting float32 noise — and the reason it is worth stating as an
        assertion rather than as prose.
        """
        rng = np.random.default_rng(4)
        image = rng.integers(0, 256, size=(*FLEET_SOURCE, 3), dtype=np.uint8)
        geometry = LetterboxGeometry.plan(FLEET_SOURCE, FLEET_TARGET)
        batch, _ = ops.letterbox(image, FLEET_TARGET)

        row = 286
        mutated_y = (row + 0.5) / (512 / 1920) - 0.5
        xs = [(x + 0.5) * 1920 / 512 - 0.5 for x in range(0, 512, 11)]
        mutated = np.array([_bilinear(image, mutated_y, x, 2) for x in xs]) / 255.0

        deviation = np.abs(batch[0, 0, geometry.pad_top + row, ::11] - mutated).max()

        # 0.49 on this frame, against a parity tolerance of 1e-3. A backend that fell to the
        # 1/scale formula would drive this to ~0, which is what makes it a live assertion
        # rather than a note.
        assert deviation > 0.1, (
            f"output row {row} is only {deviation} away from what the 1/scale formula "
            f"produces, so the letterbox is sampling on the wrong ratio — see convention 1"
        )


class TestCropCentresAtANonIntegerSpan:
    """The crop convention has the same hole, and the same fix.

    Every existing crop-centre test uses a span that divides its target exactly. A span of 4
    into 3 output pixels does not, and ``low + (i + 0.5) * span / target - 0.5`` is the only
    formula that gives these three numbers.
    """

    def test_a_span_of_four_into_three_pixels(self) -> None:
        """``10 + (i + 0.5) * 4 / 3 - 0.5`` = 10.1666667, 11.5, 12.8333333.

        The origin is added before the half-pixel shift, so a crop of the whole frame reduces
        to a resize of it — which is the property the two functions have to share. A span
        divided by a target it does not divide evenly is the case the existing crop tests miss:
        ``[10, 14]`` into 2 pixels gives 10.5 and 12.5 under either arithmetic.
        """
        assert crop_centres(10.0, 14.0, 3).tolist() == pytest.approx(
            [10.166667, 11.5, 12.833333], abs=1e-5
        )

    def test_it_agrees_with_a_resize_of_the_whole_frame(self, ops) -> None:
        """The same non-integer ratio through both entry points.

        A crop covering ``[0, 4]`` of a 4-wide source into 3 output pixels must read the same
        taps as a resize of that source to 3 — and both must be the achieved ratio 4/3.
        """
        crop = crop_centres(0.0, 4.0, 3)

        assert crop.tolist() == pytest.approx(resize_centres(4, 3).tolist())
