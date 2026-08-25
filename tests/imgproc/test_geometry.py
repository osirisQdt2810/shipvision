"""The letterbox geometry: rounding, the pad split, the half-pixel rule, and the inverse.

This file exists because none of these can fail visibly. A letterboxed image with the pad on
the wrong side, or sampled half a pixel off, looks exactly like a correct one — and every box
the detector produces is then shifted by the same amount, on every camera, for as long as
nobody checks. So each convention in :mod:`shipvision.imgproc.geometry` is asserted here
against a number worked out by hand, plus one assertion against what ``letterbox`` actually
wrote — because a test on the geometry object alone cannot catch a backend that ignored it.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipvision.errors import ConfigurationError, DimensionMismatchError
from shipvision.imgproc import IMGPROC, LetterboxGeometry
from shipvision.imgproc.geometry import crop_centres, resize_centres
from shipvision.registry import PYTHON

# 1080x1920 -> 640x640: scale is exactly 1/3 and the vertical pad, 280, is even.
# 1077x1920 -> 512x512: scale is 4/15, so 1/scale is 3.75 and NOT an integer, and the
#   vertical pad is 225 — odd, so which side gets the extra pixel changes the answer.
EVEN_PAD = ((1080, 1920), (640, 640))
ODD_PAD = ((1077, 1920), (512, 512))


# --------------------------------------------------------------------- the conventions


class TestTheConventions:
    def test_scale_preserves_aspect_by_taking_the_smaller_ratio(self) -> None:
        geometry = LetterboxGeometry.plan(*EVEN_PAD)

        assert geometry.scale == pytest.approx(640 / 1920)
        assert (geometry.resized_height, geometry.resized_width) == (360, 640)

    def test_the_resized_extent_rounds_half_up_not_to_even(self) -> None:
        """Convention 2. ``numpy.round`` would answer 2 here, and the CUDA kernel answers 3.

        A 5x4 source into a 100x2 canvas scales by exactly 0.5, so the resized height is exactly
        2.5 — the one input where round-half-up and round-half-to-even disagree. The kernel uses
        ``lroundf``, which rounds half away from zero, so this library must too.
        """
        geometry = LetterboxGeometry.plan((5, 4), (100, 2))

        assert geometry.scale == 0.5
        assert np.round(5 * 0.5) == 2, "the case is only interesting while numpy still says 2"
        assert geometry.resized_height == 3

    def test_an_odd_total_pad_puts_the_extra_pixel_at_the_bottom(self) -> None:
        """Convention 3. 1077 -> 287 inside 512 leaves 225 rows of bar: 112 above, 113 below."""
        geometry = LetterboxGeometry.plan(*ODD_PAD)

        assert geometry.resized_height == 287
        assert geometry.pad_top == 112
        assert geometry.pad_bottom == 113
        assert geometry.pad_top + geometry.resized_height + geometry.pad_bottom == 512

    def test_the_horizontal_pad_splits_the_same_way(self) -> None:
        geometry = LetterboxGeometry.plan((1920, 1077), (512, 512))

        assert geometry.resized_width == 287
        assert geometry.pad_left == 112
        assert geometry.pad_right == 113

    def test_the_pad_split_is_visible_in_the_tensor_itself(self) -> None:
        """The arithmetic above, checked against what letterbox actually wrote.

        The bars are ``(pad_value - mean) / std``, and the rows immediately inside them are not,
        so this pins the split to the exact row — which a test on the geometry object alone
        cannot do, because that object is also what the implementation used.
        """
        ops = IMGPROC.build("default", backend=PYTHON)
        image = np.full((1077, 1920, 3), 200, dtype=np.uint8)

        batch, (geometry,) = ops.letterbox(image, (512, 512), pad_value=114)
        bar = 114.0 / 255.0

        assert geometry.pad_top == 112
        assert np.allclose(batch[0, :, :112, :], bar)
        assert np.allclose(batch[0, :, 512 - 113 :, :], bar)
        # And the two rows that bound the image are image, not bar.
        assert not np.allclose(batch[0, :, 112, :], bar)
        assert not np.allclose(batch[0, :, 512 - 114, :], bar)

    def test_the_channel_order_of_the_output_is_rgb(self) -> None:
        """Convention 4. A BGR frame in, an RGB tensor out, with mean/std in RGB order."""
        ops = IMGPROC.build("default", backend=PYTHON)
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[..., 0] = 10  # blue
        image[..., 1] = 20  # green
        image[..., 2] = 30  # red

        batch, _ = ops.letterbox(image, (16, 16), mean=(1.0, 2.0, 3.0), std=(10.0, 20.0, 30.0))

        assert batch[0, 0].mean() == pytest.approx((30 - 1) / 10)
        assert batch[0, 1].mean() == pytest.approx((20 - 2) / 20)
        assert batch[0, 2].mean() == pytest.approx((10 - 3) / 30)

    # ------------------------------------------------------------------ the half-pixel rule

    def test_resize_centres_follow_the_half_pixel_convention(self) -> None:
        """Convention 1, worked out by hand for a 2-pixel row stretched to 4.

        ``(i + 0.5) * 2 / 4 - 0.5`` gives -0.25, 0.25, 0.75, 1.25. The first and last land
        outside the source and clamp onto the end pixels; the middle two are ordinary
        interpolations. ``align_corners=True`` would give 0, 1/3, 2/3, 1 instead — visibly
        different, and wrong for every model this library serves.
        """
        assert resize_centres(2, 4).tolist() == [-0.25, 0.25, 0.75, 1.25]

    def test_letterbox_upsampling_matches_the_hand_computed_taps(self) -> None:
        """The centres above, turned into pixel values by hand.

        A 1x2 row of 0 and 200 into a 2x4 canvas scales by exactly 2 on both axes, so there is no
        pad and the four output columns sample at -0.25, 0.25, 0.75 and 1.25. The outer two clamp
        onto the end pixels and read them exactly; the inner two are 3:1 and 1:3 mixes. Bilinear
        with ``align_corners=True`` would read 0, 66.7, 133.3, 200 instead.
        """
        ops = IMGPROC.build("default", backend=PYTHON)
        image = np.zeros((1, 2, 3), dtype=np.uint8)
        image[0, 1, :] = 200

        batch, (geometry,) = ops.letterbox(image, (2, 4), std=(1.0, 1.0, 1.0))

        assert (geometry.pad_top, geometry.pad_left) == (0, 0)
        assert batch[0, 0, 0].tolist() == pytest.approx([0.0, 50.0, 150.0, 200.0])

    def test_crop_centres_offset_the_subregion_before_the_half_pixel_shift(self) -> None:
        """A crop of ``[10, 14]`` into two output pixels samples at 10.5 and 12.5.

        ``low + (i + 0.5) * span / target - 0.5``: the origin is added first, so a crop covering
        the whole frame reduces to the same arithmetic as a resize of it.
        """
        assert crop_centres(10.0, 14.0, 2).tolist() == [10.5, 12.5]

    # ----------------------------------------------------------------------- the inversion

    @pytest.mark.parametrize(("source_hw", "target_hw"), [EVEN_PAD, ODD_PAD])
    def test_invert_boxes_is_the_exact_inverse_of_the_forward_map(
        self, source_hw: tuple[int, int], target_hw: tuple[int, int]
    ) -> None:
        """``src * scale + pad`` then ``(dst - pad) / scale`` must be a round trip.

        Including for the odd-pad case, where ``pad_top`` is 112 and ``pad_bottom`` is 113: an
        implementation that used ``(target - resized) / 2`` as a float, or that took the bottom
        pad by mistake, is off by exactly one pixel here and by nothing at all in the even case.
        """
        geometry = LetterboxGeometry.plan(source_hw, target_hw)
        source = np.array(
            [[0.0, 0.0, 100.0, 200.0], [640.0, 300.0, 1900.0, 1000.0]], dtype=np.float32
        )

        mapped = np.empty_like(source)
        mapped[:, 0::2] = source[:, 0::2] * geometry.scale + geometry.pad_left
        mapped[:, 1::2] = source[:, 1::2] * geometry.scale + geometry.pad_top

        assert geometry.invert_boxes(mapped) == pytest.approx(source, abs=1e-3)

    @pytest.mark.parametrize(("source_hw", "target_hw"), [EVEN_PAD, ODD_PAD])
    def test_invert_boxes_recovers_a_geometrically_mapped_box_within_a_pixel(
        self, source_hw: tuple[int, int], target_hw: tuple[int, int]
    ) -> None:
        """The harder round trip: forward through the *achieved* resize, back through ``scale``.

        A detector never sees the source grid, so what it measures is the resized image — whose
        extent was rounded to whole pixels, making the true ratio ``source / resized`` rather
        than ``1 / scale``. The inverse uses ``scale``, so it carries that rounding, and this
        test pins how much: bounded by ``0.5 / scale`` source pixels, which is 0.75 for the
        1077 -> 512 case and exactly zero for the 1080 -> 640 one.
        """
        geometry = LetterboxGeometry.plan(source_hw, target_hw)
        source = np.array([[12.0, 30.0, 900.0, 1000.0]], dtype=np.float32)

        mapped = np.empty_like(source)
        mapped[:, 0::2] = (
            source[:, 0::2] * geometry.resized_width / geometry.source_width + geometry.pad_left
        )
        mapped[:, 1::2] = (
            source[:, 1::2] * geometry.resized_height / geometry.source_height
            + geometry.pad_top
        )

        recovered = geometry.invert_boxes(mapped)
        assert np.abs(recovered - source).max() < 1.0

    def test_invert_boxes_clips_to_the_source_extent(self) -> None:
        """A box over the letterbox bar maps outside the image, and must come back on it.

        Clipping to the continuous extent, not to ``extent - 1``: a detection that genuinely
        reaches the right edge of a 1920-wide frame has ``x2 == 1920``.
        """
        geometry = LetterboxGeometry.plan(*ODD_PAD)

        recovered = geometry.invert_boxes(np.array([[-50.0, 0.0, 10_000.0, 10_000.0]]))

        assert recovered[0].tolist() == [0.0, 0.0, 1920.0, 1077.0]

    def test_invert_boxes_accepts_an_empty_frame(self) -> None:
        geometry = LetterboxGeometry.plan(*EVEN_PAD)

        recovered = geometry.invert_boxes(np.zeros((0, 4), dtype=np.float32))

        assert recovered.shape == (0, 4)

    def test_invert_boxes_refuses_a_bare_box(self) -> None:
        """``(4,)`` is not ``(1, 4)``. Guessing which one the caller meant is how a whole batch
        gets treated as one box."""
        geometry = LetterboxGeometry.plan(*EVEN_PAD)

        with pytest.raises(DimensionMismatchError):
            geometry.invert_boxes(np.array([1.0, 2.0, 3.0, 4.0]))

    @pytest.mark.parametrize("shape", [(5, 2), (5, 17, 2)])
    def test_invert_points_preserves_shape(self, shape: tuple[int, ...]) -> None:
        """Both trailing layouts the contract names, on the odd-pad geometry.

        The point sits at the centre of the resized image on both axes, which for a 1077x1920
        source is a *different* network coordinate per axis — the width fills the canvas, so
        ``pad_left`` is 0 while ``pad_top`` is 112. A transform that used one pad for both axes
        passes on a square source and fails here.
        """
        geometry = LetterboxGeometry.plan(*ODD_PAD)
        points = np.zeros(shape, dtype=np.float32)
        points[..., 0] = geometry.pad_left + geometry.resized_width / 2
        points[..., 1] = geometry.pad_top + geometry.resized_height / 2

        recovered = geometry.invert_points(points)

        assert recovered.shape == shape
        assert recovered[..., 0] == pytest.approx(1920.0 / 2, abs=1.0)
        assert recovered[..., 1] == pytest.approx(1077.0 / 2, abs=1.0)

    def test_invert_points_carries_a_keypoint_confidence_through_untouched(self) -> None:
        """``(n, k, 3)`` keypoints keep their third column, so a pose head needs no reshaping."""
        geometry = LetterboxGeometry.plan(*EVEN_PAD)
        points = np.array([[[100.0, 200.0, 0.9], [110.0, 210.0, 0.4]]], dtype=np.float32)

        recovered = geometry.invert_points(points)

        assert recovered[..., 2].ravel() == pytest.approx([0.9, 0.4])
        assert recovered[0, 0, 0] == pytest.approx((100.0 - 0) / geometry.scale)
        assert recovered[0, 0, 1] == pytest.approx((200.0 - 140) / geometry.scale)

    # ------------------------------------------------------------------------- refusals

    def test_a_zero_target_is_refused_at_planning_time(self) -> None:
        with pytest.raises(ConfigurationError):
            LetterboxGeometry.plan((100, 100), (0, 640))

    def test_an_empty_source_is_refused_at_planning_time(self) -> None:
        with pytest.raises(DimensionMismatchError):
            LetterboxGeometry.plan((0, 100), (640, 640))

    def test_a_geometry_cannot_be_mutated_after_the_fact(self) -> None:
        """Frozen because post-processing must not be able to 'fix' the numbers that were used."""
        geometry = LetterboxGeometry.plan(*EVEN_PAD)

        with pytest.raises(AttributeError):
            geometry.scale = 1.0  # type: ignore[misc]
