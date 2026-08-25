"""Every backend against the numpy oracle.

A fused kernel nobody can compare against is a fused kernel nobody can trust. These tests are
what make the numpy implementation earn its place in the repository: it is not a fallback that
happens to exist, it is the definition of the answer, and ``torch`` and ``native`` are correct
exactly insofar as they reproduce it.

The tolerances are chosen to be loose enough for float32 associativity and tight enough to
fail on a real disagreement, which is the only property that matters here. The three backends
compute the same sampling coordinates in slightly different orders — torch derives
``source / resized`` and then multiplies, this library multiplies and then divides, and the
difference shows up in the seventh significant digit. A **half-pixel** convention error, by
contrast, moves a sample onto a neighbouring pixel and changes a value by O(0.1) on the noise
images used here: five hundred times the tolerance below. There is no tolerance that hides
one and admits the other, which is the point of testing on noise rather than on a photograph.

The offline tier must pass here with neither torch nor ``shipvision._C`` installed. With
neither, the parametrisation collapses to the oracle against itself — trivially true, and
correct: there is nothing else on the machine to disagree.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipvision.imgproc import IMGPROC
from tests.imgproc.conftest import backend_params

VALUE_TOLERANCE = 1e-3
"""Absolute, on tensors normalised into ``[0, 1]``. Roughly a quarter of one 0-255 level."""

# 1080x1920 is the fleet's resolution; 1077x1920 is the same frame with three rows missing,
# which is what makes the vertical pad odd and the resized extent round rather than divide.
# 1079x1919 is there for a different reason: its byte count, 6 211 803, is odd, so a frame of
# that shape leaves every offset after it in the staging ring unaligned. Every other shape here
# is a multiple of 16 bytes, which is why the alignment claim below needs it — see
# `align_up` in csrc/shipvision/core/platform.h, and the misaligned-address error
# that comment describes, which is sticky.
SHAPES = [(1080, 1920), (1077, 1920), (1079, 1919), (720, 1280), (37, 53), (4, 4)]

CROP_BOXES = np.array(
    [
        [10.0, 20.0, 110.0, 220.0],  # ordinary
        [300.7, 200.3, 455.2, 390.9],  # sub-pixel edges, so the interpolation matters
        [-30.0, -30.0, 50.0, 50.0],  # over the top-left corner: must clamp
        [600.0, 400.0, 900.0, 700.0],  # past the bottom-right corner: must clamp
        [0.0, 0.0, 1.0, 1.0],  # one pixel
        [5.0, 5.0, 5.0, 5.0],  # no area at all
        [80.0, 90.0, 20.0, 30.0],  # inside out
    ],
    dtype=np.float32,
)


@pytest.fixture(params=backend_params())
def candidate(request):
    """One image-ops backend that this machine can actually build."""
    return IMGPROC.build("default", backend=request.param)


# ------------------------------------------------------------------------- letterbox


class TestLetterbox:
    @pytest.mark.parametrize("source_hw", SHAPES)
    @pytest.mark.parametrize("target_hw", [(640, 640), (512, 512)])
    def test_letterbox_agrees_with_the_oracle(
        self, candidate, oracle, source_hw: tuple[int, int], target_hw: tuple[int, int]
    ) -> None:
        rng = np.random.default_rng(hash(source_hw) % 2**32)
        image = rng.integers(0, 256, size=(*source_hw, 3), dtype=np.uint8)

        expected, expected_geometry = oracle.letterbox(image, target_hw)
        actual, actual_geometry = candidate.letterbox(image, target_hw)

        assert actual.shape == expected.shape
        assert actual.dtype == np.float32
        assert actual_geometry == expected_geometry
        assert np.abs(actual - expected).max() < VALUE_TOLERANCE


RAGGED_BATCH = [(1080, 1920), (1079, 1919), (720, 1280), (37, 53), (1077, 1920), (480, 640)]
"""Six cameras, and the second one's byte count is odd on purpose.

Without it every frame in the batch started at a 16-byte boundary, so the test could not see
the case its own docstring claimed — the offsets were ``0, 6220800, 8985600, ...``, all
multiples of 16. With ``1079x1919`` in second place they are ``0, 6220800, 12432603,
15197403, 15203286, ...``: mod 16 that is ``0, 0, 11, 11, 6, 12``, which is the ragged
alignment a real fleet produces."""


class TestLetterboxAgreesOnARaggedBatch:
    def test_letterbox_agrees_on_a_ragged_batch(self, candidate, oracle) -> None:
        """Fifty cameras do not agree on resolution, so a ragged batch is the normal case.

        Worth its own test because the native backend takes a very different path for it: one
        descriptor per image, uploaded as a table, so that a single kernel launch covers the whole
        batch. Getting an offset wrong in that table shows up here and nowhere else.
        """
        rng = np.random.default_rng(11)
        images = [rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8) for h, w in RAGGED_BATCH]

        expected, expected_geometry = oracle.letterbox(images, (640, 640))
        actual, actual_geometry = candidate.letterbox(images, (640, 640))

        assert actual.shape == (len(RAGGED_BATCH), 3, 640, 640)
        assert actual_geometry == expected_geometry
        assert np.abs(actual - expected).max() < VALUE_TOLERANCE

    def test_the_ragged_batch_really_does_land_unaligned(self) -> None:
        """The premise of the test above, asserted so it cannot rot.

        A batch whose frames all happen to be 16-byte multiples exercises the descriptor table
        without ever exercising an odd offset in it, and nothing about the test would say so. This
        computes the offsets the native backend will use — one frame after another, ``h * w * 3``
        bytes each — and requires that at least one is not 16-aligned.
        """
        offsets = np.cumsum([0, *(h * w * 3 for h, w in RAGGED_BATCH)])[:-1]

        assert any(offset % 16 for offset in offsets), (
            f"every frame in RAGGED_BATCH starts 16-byte aligned ({offsets % 16}), so this batch "
            f"cannot see a misaligned descriptor offset"
        )

    def test_letterbox_agrees_with_a_non_default_normalisation(
        self, candidate, oracle, bgr_image
    ) -> None:
        """ImageNet statistics, and a pad value that is not the YOLO grey.

        Separate from the default case because the mean/std must be indexed in *destination* (RGB)
        order, after the channel swap. A backend that normalised before swapping passes every
        test with ``mean=0, std=255`` and fails this one.
        """
        mean = (123.675, 116.28, 103.53)
        std = (58.395, 57.12, 57.375)

        expected, _ = oracle.letterbox(bgr_image, (256, 128), pad_value=0, mean=mean, std=std)
        actual, _ = candidate.letterbox(bgr_image, (256, 128), pad_value=0, mean=mean, std=std)

        assert np.abs(actual - expected).max() < VALUE_TOLERANCE * 2

    # ------------------------------------------------------------------------------ crops

    def test_crop_agrees_with_the_oracle(self, candidate, oracle, bgr_image) -> None:
        """Including the boxes that clamp, the one-pixel box, the zero-area box and the inside-out
        one. Those are the rows where two implementations are most likely to differ, and the ones
        a detector produces most reliably."""
        expected = oracle.crop_batch(bgr_image, CROP_BOXES, (64, 32))
        actual = candidate.crop_batch(bgr_image, CROP_BOXES, (64, 32))

        assert actual.shape == (len(CROP_BOXES), 3, 64, 32)
        assert actual.dtype == np.float32
        for index in range(len(CROP_BOXES)):
            assert (
                np.abs(actual[index] - expected[index]).max() < VALUE_TOLERANCE
            ), f"box {index} = {CROP_BOXES[index].tolist()} disagrees"

    def test_crop_agrees_with_a_non_default_normalisation(
        self, candidate, oracle, bgr_image
    ) -> None:
        """The zero-area rows come back as ``(0 - mean) / std``, which is only observable when the
        mean is not zero."""
        mean = (123.675, 116.28, 103.53)
        std = (58.395, 57.12, 57.375)

        expected = oracle.crop_batch(bgr_image, CROP_BOXES, (48, 48), mean=mean, std=std)
        actual = candidate.crop_batch(bgr_image, CROP_BOXES, (48, 48), mean=mean, std=std)

        assert np.abs(actual - expected).max() < VALUE_TOLERANCE * 2

    def test_an_empty_frame_crops_to_an_empty_batch_everywhere(
        self, candidate, bgr_image
    ) -> None:
        crops = candidate.crop_batch(bgr_image, np.zeros((0, 4), dtype=np.float32), (64, 32))

        assert crops.shape == (0, 3, 64, 32)

    # -------------------------------------------------------------------------------- nms

    @pytest.mark.parametrize("iou_threshold", [0.3, 0.5, 0.7])
    def test_classic_nms_agrees_with_the_oracle(
        self, candidate, oracle, iou_threshold: float
    ) -> None:
        """Indices, in order, exactly — not a set.

        Order is part of the answer: it is descending score, and downstream code that takes the
        first *k* survivors depends on it. Comparing sets would let a backend that sorted
        ascending pass.
        """
        boxes, scores = _proposals(seed=3, count=400)

        expected = oracle.nms(boxes, scores, iou_threshold=iou_threshold)
        actual = candidate.nms(boxes, scores, iou_threshold=iou_threshold)

        assert actual.tolist() == expected.tolist()

    def test_classic_nms_agrees_on_exact_duplicates(self, candidate, oracle) -> None:
        """Every backend must break a tie the same way, or two of them disagree on a frame where
        a detector emitted the same proposal twice — which happens on flat water constantly."""
        boxes = np.array([[0.0, 0.0, 10.0, 10.0]] * 6, dtype=np.float32)
        scores = np.array([0.5, 0.5, 0.9, 0.5, 0.9, 0.5], dtype=np.float32)

        expected = oracle.nms(boxes, scores, iou_threshold=0.5)
        actual = candidate.nms(boxes, scores, iou_threshold=0.5)

        assert actual.tolist() == expected.tolist() == [2]

    def test_classic_nms_agrees_under_a_score_threshold(self, candidate, oracle) -> None:
        boxes, scores = _proposals(seed=5, count=200)

        expected = oracle.nms(boxes, scores, iou_threshold=0.45, score_threshold=0.4)
        actual = candidate.nms(boxes, scores, iou_threshold=0.45, score_threshold=0.4)

        assert actual.tolist() == expected.tolist()

    def test_nms_agrees_on_no_boxes_at_all(self, candidate, oracle) -> None:
        empty_boxes = np.zeros((0, 4), dtype=np.float32)
        empty_scores = np.zeros(0, dtype=np.float32)

        actual = candidate.nms(empty_boxes, empty_scores, iou_threshold=0.5)

        assert actual.shape == (0,)
        assert actual.dtype == np.int64
        assert (
            actual.tolist() == oracle.nms(empty_boxes, empty_scores, iou_threshold=0.5).tolist()
        )

    @pytest.mark.parametrize("method", ["linear", "gauss", "neighborhood", "none"])
    def test_the_non_classic_methods_are_identical_across_backends(
        self, candidate, oracle, method: str
    ) -> None:
        """They are identical because they are the same code — every backend calls the shared
        numpy implementation. Asserted anyway: it is the property callers rely on, and a backend
        that "optimised" one of them would break it silently."""
        boxes, scores = _proposals(seed=9, count=120)

        expected, expected_scores = oracle.nms_with_scores(
            boxes, scores, iou_threshold=0.5, method=method, score_threshold=0.2
        )
        actual = candidate.nms(
            boxes, scores, iou_threshold=0.5, method=method, score_threshold=0.2
        )
        actual_scores = candidate.nms_with_scores(
            boxes, scores, iou_threshold=0.5, method=method, score_threshold=0.2
        )[1]

        assert actual.tolist() == expected.tolist()
        assert actual_scores.tolist() == pytest.approx(expected_scores.tolist())


# ---------------------------------------------------------------------------- helpers


def _proposals(*, seed: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    """A realistic raw detector output: overlapping boxes with distinct scores.

    Distinct scores by construction — a random draw can repeat a float32 to the bit, and a
    tie is tested deliberately elsewhere rather than by accident here.
    """
    rng = np.random.default_rng(seed)
    top_left = rng.uniform(0.0, 600.0, size=(count, 2)).astype(np.float32)
    extent = rng.uniform(10.0, 120.0, size=(count, 2)).astype(np.float32)
    boxes = np.concatenate([top_left, top_left + extent], axis=1).astype(np.float32)
    scores = (np.arange(count, dtype=np.float32) + 1.0) / (count + 1)
    rng.shuffle(scores)
    return boxes, scores
