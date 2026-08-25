"""Mask reconstruction, and the one ordering mistake that is invisible.

The prototype gemm is arithmetic anyone can check. The two resizes are not: crop-then-resize
and resize-then-crop both produce a plausible mask of exactly the right shape, and only one of
them puts it in the right place.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipvision.detection.heads import HEADS
from shipvision.detection.heads.masks import (
    bilinear_resize,
    box_crop_bounds,
    fuse_mask_logits,
    unpad_mask,
)
from shipvision.errors import ConfigurationError, DimensionMismatchError
from shipvision.imgproc.geometry import LetterboxGeometry

from .conftest import LANDSCAPE, NETWORK, detection_output, geometry, to_network_space

#: A quarter of the 640x640 network input, which is where YOLO26-seg's prototypes live.
PROTO = 160

#: For a 1080x1920 source in a 640x640 input the content occupies canvas rows [140, 500) —
#: exactly proto rows [35, 125), because the proto grid is the canvas divided by four. Using an
#: exact multiple keeps the expected answers stateable rather than approximate.
CONTENT_ROWS = (35, 125)


def banded_proto(bands, *, high=10.0, low=-10.0) -> np.ndarray:
    """A ``(PROTO, PROTO)`` logit plane that is ``high`` inside each row band and ``low`` outside."""
    plane = np.full((PROTO, PROTO), low, dtype=np.float32)
    for start, stop in bands:
        plane[start:stop, :] = high
    return plane


def resize_then_crop(logits, geom) -> np.ndarray:
    """The mistake, written out: upsample the padded canvas straight to the source extent.

    This is what forgetting step 4 looks like, and it is the most common form of the bug — the
    mask is stretched by ``target / resized`` on the letterboxed axis, which for a 1080p frame
    in a square input is 1.78x vertically, on every mask, forever.
    """
    from shipvision.detection.heads.masks import _sigmoid

    canvas = bilinear_resize(_sigmoid(logits), geom.target_height, geom.target_width)
    return bilinear_resize(canvas, geom.source_height, geom.source_width)


class TestMaskGeometry:
    """The un-padded region maps onto the whole source image, and nothing else does."""

    def test_the_content_band_fills_the_whole_frame(self) -> None:
        """A mask that covers exactly the letterboxed content must cover the entire source
        image afterwards — including its first and last row, which is where the two orderings
        differ."""
        geom = geometry(LANDSCAPE)

        mask = unpad_mask(banded_proto([CONTENT_ROWS]), geom)

        assert mask.shape == LANDSCAPE
        assert mask[0].mean() > 0.5
        assert mask[-1].mean() > 0.5
        assert mask.mean() > 0.99

    def test_resize_then_crop_gets_a_different_and_wrong_answer(self) -> None:
        """The discriminating case. Under the wrong order the top row of the source image reads
        the letterbox bar, so it is empty — and the whole mask is shifted and stretched."""
        geom = geometry(LANDSCAPE)
        plane = banded_proto([CONTENT_ROWS])

        correct = unpad_mask(plane, geom)
        wrong = resize_then_crop(plane, geom)

        assert correct[0, 0] > 0.5 > wrong[0, 0]
        assert np.abs(correct - wrong).max() > 0.9

    def test_a_band_lands_where_the_geometry_says_it_should(self) -> None:
        """Half the content band is half the image, exactly. Proto rows [35, 80) are canvas rows
        [140, 320), which is the top half of the content and so source rows [0, 540)."""
        geom = geometry(LANDSCAPE)

        mask = unpad_mask(banded_proto([(35, 80)]), geom)

        assert mask[:530].mean() > 0.99
        assert mask[550:].mean() < 0.01
        assert int(np.argmax(mask[:, 0] < 0.5)) == 540

    def test_a_square_source_needs_no_unpadding_and_is_unchanged_by_it(self) -> None:
        """The degenerate case worth pinning: with no bars, crop-then-resize and
        resize-then-crop agree, so a test that only used a square source would pass either way.
        """
        geom = geometry((1080, 1080))
        plane = banded_proto([(0, 80)])

        assert np.allclose(unpad_mask(plane, geom), resize_then_crop(plane, geom))


class TestBilinearResize:
    """Half-pixel centres from :mod:`shipvision.imgproc.geometry`, and a window that is exact."""

    def test_a_matching_extent_returns_the_values_untouched(self) -> None:
        plane = np.random.default_rng(0).standard_normal((7, 5)).astype(np.float32)

        assert np.array_equal(bilinear_resize(plane, 7, 5), plane)

    def test_a_window_is_bit_identical_to_slicing_the_full_output(self) -> None:
        """The identity the whole optimisation rests on: it is what makes it safe to skip
        materialising a full 1080x1920 plane per detection."""
        plane = np.random.default_rng(1).standard_normal((11, 9)).astype(np.float32)

        full = bilinear_resize(plane, 43, 37)
        windowed = bilinear_resize(plane, 43, 37, window=(5, 20, 3, 30))

        assert np.array_equal(full[5:20, 3:30], windowed)

    def test_a_constant_plane_stays_constant_at_any_scale(self) -> None:
        plane = np.full((3, 4), 0.375, dtype=np.float32)

        assert np.allclose(bilinear_resize(plane, 17, 23), 0.375)

    def test_a_ramp_upsampled_2x_matches_the_hand_computed_half_pixel_answer(self) -> None:
        """The half-pixel rule and the border clamp, both pinned by numbers worked out by hand.

        Output pixel ``i`` reads the source at ``(i + 0.5) * 4 / 8 - 0.5 = i/2 - 0.25``, so the
        first and last outputs land outside the source and read the clamped edge tap while the
        interior interpolates. Note that the edges are therefore *not* an extrapolated ramp —
        that is what ``align_corners=False`` with border clamping means, and a backend that
        extrapolated instead would fail here.
        """
        plane = np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float32)

        out = bilinear_resize(plane, 1, 8)

        assert out[0].tolist() == pytest.approx([0.0, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.0])

    @pytest.mark.parametrize(
        ("shape", "target", "window"),
        [
            ((0, 4), (8, 8), None),
            ((4, 4), (0, 8), None),
            ((4, 4), (8, 8), (0, 0, 0, 8)),
            ((4, 4), (8, 8), (0, 9, 0, 8)),
        ],
    )
    def test_an_impossible_request_is_refused(self, shape, target, window) -> None:
        plane = np.zeros(shape, dtype=np.float32)

        with pytest.raises(DimensionMismatchError):
            bilinear_resize(plane, target[0], target[1], window=window)

    def test_a_non_planar_input_is_refused(self) -> None:
        with pytest.raises(DimensionMismatchError, match="one \\(h, w\\) plane"):
            bilinear_resize(np.zeros((2, 4, 4), dtype=np.float32), 8, 8)


class TestMaskFusion:
    """``coefficients @ prototypes``, and the mismatch the reference logs and works around."""

    def test_the_fused_plane_is_the_weighted_sum_of_the_prototypes(self) -> None:
        prototypes = np.stack(
            [np.full((4, 4), 1.0), np.full((4, 4), 10.0), np.full((4, 4), 100.0)]
        ).astype(np.float32)
        coefficients = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 1.0]], dtype=np.float32)

        fused = fuse_mask_logits(coefficients, prototypes)

        assert fused.shape == (2, 4, 4)
        assert np.allclose(fused[0], 321.0)
        assert np.allclose(fused[1], 100.0)

    def test_a_coefficient_count_that_disagrees_with_the_basis_is_refused(self) -> None:
        with pytest.raises(DimensionMismatchError, match="same export"):
            fuse_mask_logits(
                np.zeros((2, 16), dtype=np.float32), np.zeros((32, 4, 4), dtype=np.float32)
            )

    def test_a_rank_mistake_is_refused_rather_than_broadcast(self) -> None:
        with pytest.raises(DimensionMismatchError):
            fuse_mask_logits(
                np.zeros(32, dtype=np.float32), np.zeros((32, 4, 4), dtype=np.float32)
            )


class TestBoxCropBounds:
    """Outward rounding, clipped to the frame, never empty."""

    def test_a_fractional_box_is_covered_outward(self) -> None:
        assert box_crop_bounds(np.array([10.2, 20.7, 30.1, 40.9]), 1080, 1920) == (
            20,
            41,
            10,
            31,
        )

    def test_a_box_past_the_edge_is_clipped(self) -> None:
        assert box_crop_bounds(np.array([-5.0, -5.0, 5000.0, 5000.0]), 100, 200) == (
            0,
            100,
            0,
            200,
        )

    def test_a_degenerate_box_still_gives_one_pixel(self) -> None:
        """A ``(0, 0)`` mask is a shape every consumer has to special-case; a 1x1 is not."""
        assert box_crop_bounds(np.array([50.0, 50.0, 50.0, 50.0]), 100, 100) == (50, 51, 50, 51)

    def test_a_box_at_the_far_corner_still_gives_one_pixel(self) -> None:
        assert box_crop_bounds(np.array([200.0, 100.0, 200.0, 100.0]), 100, 200) == (
            99,
            100,
            199,
            200,
        )


class TestSegmentationMasks:
    """End to end through the head: the mask is in the box's frame of reference, in the right place."""

    BOX = np.array([[100.0, 200.0, 400.0, 700.0]], dtype=np.float32)

    def seg_outputs(self, plane, geom, *, coefficient=1.0):
        """A one-coefficient segmentation output, so the fused mask *is* ``plane``."""
        detections = detection_output(
            to_network_space(self.BOX, geom), [0.9], [0.0], extra=[[coefficient]]
        )
        return [detections, plane[None, None]]

    def test_the_mask_has_the_shape_of_the_box_and_is_boolean(self) -> None:
        geom = geometry(LANDSCAPE)
        outputs = self.seg_outputs(banded_proto([CONTENT_ROWS]), geom)

        result = HEADS.build("yolo26_seg").decode([*outputs], [geom], [_tag()])[0]

        mask = result[0].mask
        assert mask is not None
        # floor(200)..ceil(700) rows, floor(100)..ceil(400) columns.
        assert mask.shape == (500, 300)
        assert mask.dtype == np.bool_
        assert mask.all()

    def test_the_mask_lands_on_the_half_of_the_box_the_prototype_marked(self) -> None:
        """A positional claim, not a coverage one — that is what tells a correct mask from a
        stretched or shifted one of the same shape.

        The box spans canvas columns [33.3, 133.3]; the prototype is marked over proto columns
        [8, 21), which is canvas [32, 84). So the mask must be full up to source column
        ``84 / 0.3333 - 100 = 152`` and empty after it, and 152 is asserted directly.
        """
        geom = geometry(LANDSCAPE)
        plane = np.full((PROTO, PROTO), -10.0, dtype=np.float32)
        plane[51:94, 8:21] = 10.0

        result = HEADS.build("yolo26_seg").decode(
            [*self.seg_outputs(plane, geom)], [geom], [_tag()]
        )[0]

        mask = result[0].mask
        assert mask.shape == (500, 300)
        assert mask[:, :150].all()
        assert not mask[:, 154:].any()
        assert int(np.argmax(~mask[250])) == 152

    def test_binarise_false_returns_the_probabilities(self) -> None:
        geom = geometry(LANDSCAPE)
        outputs = self.seg_outputs(banded_proto([CONTENT_ROWS]), geom)

        result = HEADS.build("yolo26_seg", binarise=False).decode([*outputs], [geom], [_tag()])[
            0
        ]

        mask = result[0].mask
        assert mask.dtype == np.float32
        assert 0.5 < float(mask.min()) <= float(mask.max()) <= 1.0

    def test_the_mask_threshold_is_inclusive_and_selects(self) -> None:
        geom = geometry(LANDSCAPE)
        # A flat zero logit plane sigmoids to exactly 0.5 everywhere.
        outputs = self.seg_outputs(np.zeros((PROTO, PROTO), dtype=np.float32), geom)

        inclusive = HEADS.build("yolo26_seg", mask_threshold=0.5)
        exclusive = HEADS.build("yolo26_seg", mask_threshold=0.5000001)

        assert inclusive.decode([*outputs], [geom], [_tag()])[0][0].mask.all()
        assert not exclusive.decode([*outputs], [geom], [_tag()])[0][0].mask.any()

    def test_a_frame_with_no_detections_has_no_masks_and_still_carries_its_tag(self) -> None:
        geom = geometry(LANDSCAPE)
        detections = np.zeros((1, 4, 38), dtype=np.float32)
        prototypes = np.zeros((1, 32, PROTO, PROTO), dtype=np.float32)

        result = HEADS.build("yolo26_seg").decode([detections, prototypes], [geom], [_tag()])[0]

        assert len(result) == 0
        assert result.boxes.shape == (0, 4)
        assert result.tag == _tag()

    def test_the_prototype_tensor_may_be_given_first(self) -> None:
        geom = geometry(LANDSCAPE)
        detections, prototypes = self.seg_outputs(banded_proto([CONTENT_ROWS]), geom)

        swapped = HEADS.build("yolo26_seg").decode([prototypes, detections], [geom], [_tag()])[
            0
        ]

        assert len(swapped) == 1 and swapped[0].mask.all()

    def test_a_coefficient_count_that_disagrees_with_the_basis_is_refused(self) -> None:
        """The reference warns and builds every mask from a truncated basis, which looks like a
        model that segments badly rather than like a wiring mistake."""
        geom = geometry(LANDSCAPE)
        detections = np.zeros((1, 4, 38), dtype=np.float32)
        prototypes = np.zeros((1, 16, PROTO, PROTO), dtype=np.float32)

        with pytest.raises(DimensionMismatchError, match="truncated basis"):
            HEADS.build("yolo26_seg").decode([detections, prototypes], [geom], [_tag()])

    def test_a_prototype_batch_that_covers_neither_one_frame_nor_all_is_refused(self) -> None:
        geom = geometry(LANDSCAPE)
        detections = np.zeros((3, 4, 38), dtype=np.float32)
        prototypes = np.zeros((2, 32, PROTO, PROTO), dtype=np.float32)

        with pytest.raises(DimensionMismatchError, match="prototype tensor has a batch"):
            HEADS.build("yolo26_seg").decode([detections, prototypes], [geom] * 3, [_tag()] * 3)

    def test_one_shared_prototype_serves_a_whole_batch(self) -> None:
        """``(1, 32, h/4, w/4)`` against several frames is what the readme describes."""
        geom = geometry(LANDSCAPE)
        detections = np.zeros((3, 4, 7), dtype=np.float32)
        detections[:, 0, :4] = to_network_space(self.BOX, geom)
        detections[:, 0, 4] = 0.9
        detections[:, 0, 6] = 1.0
        prototypes = banded_proto([CONTENT_ROWS])[None, None]

        results = HEADS.build("yolo26_seg").decode(
            [detections, prototypes], [geom] * 3, [_tag(i) for i in range(3)]
        )

        assert [len(r) for r in results] == [1, 1, 1]
        assert all(r[0].mask.all() for r in results)

    def test_a_bad_mask_threshold_is_refused_at_construction(self) -> None:
        with pytest.raises(ConfigurationError, match="probability"):
            HEADS.build("yolo26_seg", mask_threshold=1.5)

    def test_a_non_rank_four_second_output_is_refused(self) -> None:
        geom = geometry(LANDSCAPE)

        with pytest.raises(DimensionMismatchError, match="ranks"):
            HEADS.build("yolo26_seg").decode(
                [np.zeros((1, 4, 38), np.float32), np.zeros((1, 4, 38), np.float32)],
                [geom],
                [_tag()],
            )


def _tag(frame_id: int = 3):
    from shipvision.types import FrameTag

    return FrameTag("cam-11", frame_id)


class TestMaskGeometryUsesTheCarriedNumbers:
    """A fabricated geometry must be obeyed, not second-guessed from the two shapes."""

    def test_a_fabricated_pad_moves_the_mask(self) -> None:
        plane = banded_proto([(0, PROTO)])
        strange = LetterboxGeometry(
            scale=0.5,
            pad_left=0,
            pad_top=0,
            source_height=200,
            source_width=200,
            target_height=NETWORK[0],
            target_width=NETWORK[1],
        )

        mask = unpad_mask(plane, strange)

        # scale 0.5 on a 200-pixel source occupies 100 of the 640 canvas rows, so only the top
        # 100/640 of the (all-high) plane is used — which is still all high, and the shape must
        # follow the geometry rather than the canvas.
        assert mask.shape == (200, 200)
        assert mask.mean() > 0.99
