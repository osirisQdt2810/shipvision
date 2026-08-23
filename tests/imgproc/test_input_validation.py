"""What the image ops refuse, and why refusing beats coercing.

The cases here are the inputs a decoder produces when something upstream went wrong: a frame
with no rows because the RTSP stream reconnected mid-GOP, a box array with a NaN because a
head divided by a zero width. Every one of them has to fail the same way in all three
backends, because the backends are advertised as interchangeable — and a validation hole is
worse than a crash. On the native path a zero-extent frame makes the kernel read before its
allocation, and ``cudaErrorIllegalAddress`` is **sticky**: it poisons the context for the life
of the process, so the next frame on a brand-new instance fails too and one bad camera takes
the worker down permanently. On the same path at batch index > 0 the read lands inside the
staging ring instead and comes back as the *previous camera's pixels* — a real-looking
detection on a camera where nothing happened.

So the guard lives in :func:`~shipvision.imgproc.base.validate_image`, above every backend,
and these tests assert it from the outside: what a caller sees is a typed error, before any
kernel is launched.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipvision.errors import ConfigurationError, DimensionMismatchError
from shipvision.imgproc import IMGPROC
from shipvision.imgproc.base import validate_boxes, validate_image
from shipvision.registry import PYTHON
from tests.imgproc.conftest import NATIVE_BUILT, backend_params

ZERO_EXTENT_SHAPES = [(0, 1920, 3), (1080, 0, 3), (0, 0, 3)]


@pytest.fixture(params=backend_params())
def candidate(request):
    """One image-ops backend that this machine can actually build."""
    return IMGPROC.build("default", backend=request.param)


class TestZeroExtentFrames:
    """A frame with no pixels is refused before it reaches any sampler.

    Not merely "does not crash": the refusal is a typed
    :class:`~shipvision.errors.DimensionMismatchError` in every backend, raised on the way in,
    so a caller can drop the frame and keep the worker.
    """

    @pytest.mark.parametrize("shape", ZERO_EXTENT_SHAPES)
    def test_letterbox_refuses_a_zero_extent_frame(self, candidate, shape) -> None:
        with pytest.raises(DimensionMismatchError, match="both extents must be"):
            candidate.letterbox(np.zeros(shape, dtype=np.uint8), (640, 640))

    @pytest.mark.parametrize("shape", ZERO_EXTENT_SHAPES)
    def test_crop_refuses_a_zero_extent_frame(self, candidate, shape) -> None:
        boxes = np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32)

        with pytest.raises(DimensionMismatchError, match="both extents must be"):
            candidate.crop_batch(np.zeros(shape, dtype=np.uint8), boxes, (8, 8))

    def test_a_degenerate_frame_at_batch_index_zero_is_refused(self, candidate) -> None:
        """Index 0 is the case that used to poison the CUDA context.

        The kernel samples ``src_y = -0.5`` for a zero-row image, ``sample_bilinear`` clamps
        the high tap to ``min(y0 + 1, h - 1) = -1``, and at offset 0 in the staging ring that
        read is *before* the allocation.
        """
        good = np.zeros((16, 16, 3), dtype=np.uint8)

        with pytest.raises(DimensionMismatchError, match=r"images\[0\]"):
            candidate.letterbox([np.zeros((0, 16, 3), dtype=np.uint8), good], (64, 64))

    def test_one_degenerate_frame_rejects_the_whole_ragged_batch(self, candidate) -> None:
        """And names which frame it was.

        At an index above zero the bad read lands inside the staging ring rather than outside
        it, so there is no error at all and the row comes back holding another camera's
        pixels. The batch has to be refused by name, not repaired.
        """
        frames = [
            np.zeros((16, 16, 3), dtype=np.uint8),
            np.zeros((16, 32, 3), dtype=np.uint8),
            np.zeros((0, 1920, 3), dtype=np.uint8),
        ]

        with pytest.raises(DimensionMismatchError, match=r"images\[2\]"):
            candidate.letterbox(frames, (64, 64))

    def test_the_worker_still_works_after_a_refusal(self, candidate) -> None:
        """The point of validating early: a rejected frame costs one frame, not the process.

        This is the assertion that would have failed before the fix — the native backend's
        *next* call, on any instance in the process, raised ``GpuError`` from a sticky
        ``cudaErrorIllegalAddress``.
        """
        rng = np.random.default_rng(7)
        image = rng.integers(0, 256, size=(64, 48, 3), dtype=np.uint8)

        with pytest.raises(DimensionMismatchError):
            candidate.letterbox(np.zeros((0, 48, 3), dtype=np.uint8), (32, 32))

        batch, geometries = candidate.letterbox(image, (32, 32))

        assert batch.shape == (1, 3, 32, 32)
        assert np.isfinite(batch).all()
        assert geometries[0].source_height == 64


class TestValidateImage:
    """The one guard all three backends inherit, tested directly.

    Directly as well as through the backends because it is the *only* copy: a backend that
    grew its own check would be free to disagree with the others, which is the failure mode
    this function exists to remove.
    """

    @pytest.mark.parametrize("shape", ZERO_EXTENT_SHAPES)
    def test_it_refuses_a_zero_extent_image(self, shape) -> None:
        with pytest.raises(DimensionMismatchError, match="both extents must be"):
            validate_image(np.zeros(shape, dtype=np.uint8))

    def test_it_names_the_frame_it_refused(self) -> None:
        with pytest.raises(DimensionMismatchError, match="images\\[4\\]"):
            validate_image(np.zeros((0, 8, 3), dtype=np.uint8), what="images[4]")

    def test_a_one_pixel_image_is_still_valid(self) -> None:
        """The bound is ``> 0``, not ``> 1``: a 1x1 frame is degenerate-looking but samplable,
        and the parity suite covers it."""
        assert validate_image(np.zeros((1, 1, 3), dtype=np.uint8)).shape == (1, 1, 3)

    def test_a_wrong_dtype_is_still_a_configuration_error(self) -> None:
        """The new extent check must not shadow the dtype one, which is a different fault with
        a different fix."""
        with pytest.raises(ConfigurationError, match="uint8"):
            validate_image(np.zeros((4, 4, 3), dtype=np.float32))


class TestValidateBoxes:
    """Non-finite coordinates are refused rather than passed to three different clamps.

    A NaN is not a large number: the numpy backend indexes with
    ``int(nan) == -9223372036854775808`` and raises an untyped ``IndexError`` that kills the
    whole batch, the CUDA kernel's ``fmaxf(0, NaN)`` returns 0 and produces a *plausible* crop
    of the top-left corner, and torch's clamp produces a third answer. In NMS the same NaN
    makes the kernel compute ``iou = 9.0`` and suppress a box that numpy keeps, because
    ``NaN > threshold`` is false. Three answers from three interchangeable backends is the
    defect; refusing at the boundary is the only outcome they can all agree on.
    """

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_it_refuses_a_non_finite_coordinate(self, bad: float) -> None:
        boxes = np.array([[0.0, 0.0, 10.0, 10.0], [bad, 1.0, 11.0, 11.0]], dtype=np.float32)

        with pytest.raises(ConfigurationError, match="non-finite"):
            validate_boxes(boxes)

    def test_it_says_how_many_and_where(self) -> None:
        """One bad row out of fifteen thousand is a dropped frame; fifteen thousand bad rows is
        a broken head, and the message has to tell them apart."""
        boxes = np.zeros((3, 4), dtype=np.float32)
        boxes[1, 2] = np.nan

        with pytest.raises(ConfigurationError, match="1 non-finite"):
            validate_boxes(boxes)

    def test_a_large_but_finite_coordinate_is_accepted(self) -> None:
        """The rule is finiteness, not magnitude. A detector on a 4K frame legitimately emits
        coordinates in the thousands, and inventing a bound here would reject them."""
        boxes = np.array([[0.0, 0.0, 4096.0, 2160.0]], dtype=np.float32)

        assert validate_boxes(boxes).tolist() == [[0.0, 0.0, 4096.0, 2160.0]]


class TestNonFiniteBoxesAcrossBackends:
    """Every backend refuses them, and refuses them identically."""

    def test_crop_refuses_a_nan_box(self, candidate) -> None:
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        boxes = np.array([[np.nan, 10.0, 30.0, 40.0]], dtype=np.float32)

        with pytest.raises(ConfigurationError, match="non-finite"):
            candidate.crop_batch(frame, boxes, (8, 8))

    def test_nms_refuses_a_nan_box(self, candidate) -> None:
        boxes = np.array(
            [[0.0, 0.0, 10.0, 10.0], [np.nan, 1.0, 11.0, 11.0], [0.5, 0.5, 10.5, 10.5]],
            dtype=np.float32,
        )
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

        with pytest.raises(ConfigurationError, match="non-finite"):
            candidate.nms(boxes, scores, iou_threshold=0.5)

    @pytest.mark.parametrize("method", ["classic", "linear", "gauss", "neighborhood", "none"])
    def test_every_method_refuses_a_nan_box(self, candidate, method: str) -> None:
        """Including the methods with no kernel: the divergence was between backends, but the
        untyped ``IndexError`` was in the shared numpy path, so both need the same guard."""
        boxes = np.array([[0.0, 0.0, 10.0, 10.0], [1.0, np.inf, 11.0, 11.0]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)

        with pytest.raises(ConfigurationError, match="non-finite"):
            candidate.nms(boxes, scores, iou_threshold=0.5, method=method)

    def test_a_non_finite_score_is_refused_too(self, candidate) -> None:
        """A NaN score sorts arbitrarily and an inf score wins every comparison, so the
        surviving set depends on the sort implementation rather than on the detector."""
        boxes = np.array([[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]], dtype=np.float32)
        scores = np.array([0.9, np.nan], dtype=np.float32)

        with pytest.raises(ConfigurationError, match="non-finite"):
            candidate.nms(boxes, scores, iou_threshold=0.5)


@pytest.mark.native
@pytest.mark.skipif(not NATIVE_BUILT, reason="shipvision._C is not built, or there is no GPU")
class TestTheExtensionGuardsItself:
    """The C++ entry points refuse a zero-extent frame on their own.

    A Python-only fix would leave ``shipvision._C`` unsafe for every other caller — and the
    module docstring calls ``letterbox_into`` "the production path", which no Python
    validation covers. The guard has to exist on both sides of the binding, and the cost of
    checking is two integer comparisons per frame against a permanently dead worker.
    """

    def test_letterbox_refuses_a_zero_row_frame(self) -> None:
        from shipvision import _C

        ops = _C.ImageOps(device_index=0)

        with pytest.raises(ValueError, match="positive"):
            ops.letterbox_batch(
                [np.zeros((0, 16, 3), dtype=np.uint8)],
                64,
                64,
                [0.0, 0.0, 0.0],
                [255.0, 255.0, 255.0],
                True,
                114,
                0,
            )

    def test_crop_refuses_a_zero_column_frame(self) -> None:
        from shipvision import _C

        ops = _C.ImageOps(device_index=0)

        with pytest.raises(ValueError, match="positive"):
            ops.crop_batch(
                np.zeros((16, 0, 3), dtype=np.uint8),
                np.array([[0.0, 0.0, 4.0, 4.0]], dtype=np.float32),
                8,
                8,
                [0.0, 0.0, 0.0],
                [255.0, 255.0, 255.0],
                True,
                0,
            )

    def test_the_device_is_still_usable_afterwards(self) -> None:
        """The refusal is a host-side ``std::invalid_argument``, so no kernel ran and there is
        no sticky error to inherit. Asserted on a fresh instance, which is exactly what used
        to fail."""
        from shipvision import _C

        ops = _C.ImageOps(device_index=0)
        with pytest.raises(ValueError):
            ops.letterbox_batch(
                [np.zeros((0, 16, 3), dtype=np.uint8)],
                32,
                32,
                [0.0, 0.0, 0.0],
                [255.0, 255.0, 255.0],
                True,
                114,
                0,
            )

        survivor = IMGPROC.build("default", backend="native")
        batch, _ = survivor.letterbox(np.zeros((8, 8, 3), dtype=np.uint8), (16, 16))

        assert np.isfinite(batch).all()
        assert IMGPROC.build("default", backend=PYTHON) is not None


class TestPadValue:
    """The letterbox fill is a 0-255 source-scale value, and out-of-range means out-of-range.

    Unchecked, it was cast to ``unsigned char`` in C++ and used as a float in Python, so the
    three backends disagreed on every value outside the range: ``256`` gave numpy and torch
    256.0 and the native backend **0.0**; ``-1`` gave -1.0 against **255.0**, which is white
    bars where grey ones were asked for; ``300`` gave 44.0. A config with a typo in it produced
    a different image on the GPU than in the oracle, and the parity suite never passes an
    out-of-range value so nothing said so.
    """

    @pytest.mark.parametrize("pad_value", [256, -1, 300, 1000, -255])
    def test_a_value_outside_the_source_scale_is_refused(self, candidate, pad_value) -> None:
        with pytest.raises(ConfigurationError, match="pad_value"):
            candidate.letterbox(
                np.zeros((8, 12, 3), dtype=np.uint8), (16, 16), pad_value=pad_value
            )

    @pytest.mark.parametrize("pad_value", [0, 114, 255])
    def test_the_ends_of_the_range_are_accepted(self, candidate, pad_value) -> None:
        """Inclusive at both ends: black bars and white bars are both legitimate requests, and
        a boundary that excluded them would be a different bug."""
        batch, _ = candidate.letterbox(
            np.zeros((8, 12, 3), dtype=np.uint8),
            (16, 16),
            pad_value=pad_value,
            std=(1.0, 1.0, 1.0),
        )

        assert batch[0, 0, 0, 0] == pytest.approx(float(pad_value))

    def test_a_non_integer_pad_value_is_refused(self, candidate) -> None:
        """114.5 is not a source-scale level. C++ would truncate it and numpy would not."""
        with pytest.raises(ConfigurationError, match="pad_value"):
            candidate.letterbox(np.zeros((8, 12, 3), dtype=np.uint8), (16, 16), pad_value=114.5)


class TestNormalisation:
    """``std`` must be finite and positive, and ``mean`` finite.

    ``resolve_normalisation`` rejected only ``std == 0``. Everything else went straight through:
    ``std=(-255, 255, 255)`` silently inverts the red channel — an image that looks like a
    photographic negative in one channel, which a model does not error on, it just gets worse —
    and ``std=(nan, ...)`` produces an all-NaN tensor that poisons every downstream reduction.
    Both are start-up mistakes, and this library's rule is that a bad config fails at start-up
    rather than at frame 40 000.
    """

    @pytest.mark.parametrize(
        "std",
        [
            (-255.0, 255.0, 255.0),
            (255.0, -1.0, 255.0),
            (np.nan, 255.0, 255.0),
            (255.0, np.inf, 255.0),
            (0.0, 255.0, 255.0),
        ],
    )
    def test_a_std_that_is_not_finite_and_positive_is_refused(self, candidate, std) -> None:
        with pytest.raises(ConfigurationError, match="std"):
            candidate.letterbox(np.zeros((8, 12, 3), dtype=np.uint8), (16, 16), std=std)

    @pytest.mark.parametrize("mean", [(np.nan, 0.0, 0.0), (0.0, np.inf, 0.0)])
    def test_a_non_finite_mean_is_refused(self, candidate, mean) -> None:
        """The same failure by the other route: ``(value - nan) / std`` is NaN everywhere."""
        with pytest.raises(ConfigurationError, match="mean"):
            candidate.letterbox(np.zeros((8, 12, 3), dtype=np.uint8), (16, 16), mean=mean)

    def test_the_crop_path_refuses_them_too(self, candidate) -> None:
        """Same validator, and the crop path is the one that runs 15 000 times a second."""
        boxes = np.array([[0.0, 0.0, 8.0, 8.0]], dtype=np.float32)

        with pytest.raises(ConfigurationError, match="std"):
            candidate.crop_batch(
                np.zeros((16, 16, 3), dtype=np.uint8), boxes, (8, 8), std=(-1.0, 1.0, 1.0)
            )

    def test_imagenet_statistics_still_work(self, candidate) -> None:
        """The check must not reject the values a real checkpoint publishes."""
        batch, _ = candidate.letterbox(
            np.zeros((8, 12, 3), dtype=np.uint8),
            (16, 16),
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
        )

        assert np.isfinite(batch).all()


@pytest.mark.native
@pytest.mark.skipif(not NATIVE_BUILT, reason="shipvision._C is not built, or there is no GPU")
class TestTheExtensionRefusesThemToo:
    """``_C`` is reachable without the Python layer, so it validates for itself.

    Same argument as the zero-extent frame: the entry points are public, the cast is silent,
    and a caller that reached ``letterbox_into`` directly would get white bars for ``-1`` with
    nothing to tell it why.
    """

    @pytest.mark.parametrize("pad_value", [-1, 256, 300])
    def test_an_out_of_range_pad_value_is_refused(self, pad_value: int) -> None:
        from shipvision import _C

        ops = _C.ImageOps(device_index=0)

        with pytest.raises(ValueError, match="pad_value"):
            ops.letterbox_batch(
                [np.zeros((8, 8, 3), dtype=np.uint8)],
                16,
                16,
                [0.0, 0.0, 0.0],
                [255.0, 255.0, 255.0],
                True,
                pad_value,
                0,
            )

    @pytest.mark.parametrize("std", [[-255.0, 255.0, 255.0], [float("nan"), 255.0, 255.0]])
    def test_a_std_that_is_not_finite_and_positive_is_refused(self, std) -> None:
        from shipvision import _C

        ops = _C.ImageOps(device_index=0)

        with pytest.raises(ValueError, match="std"):
            ops.letterbox_batch(
                [np.zeros((8, 8, 3), dtype=np.uint8)],
                16,
                16,
                [0.0, 0.0, 0.0],
                std,
                True,
                114,
                0,
            )
