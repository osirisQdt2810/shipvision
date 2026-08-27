"""``swap_rb`` names the destination channel order, and nothing else — convention 4.

The BGR entry points converted to RGB unconditionally and passed a literal ``True`` down to a
binding that had taken the argument all along, so the one thing the kernel could already do
could not be asked for from Python. These tests pin what asking for it means.

The images here are **constant per channel**, which is the opposite of the choice made in
:func:`tests.imgproc.conftest.bgr_image`, and for the opposite reason. Noise is what makes a
half-pixel sampling error visible; constants are what make a channel *permutation* visible,
because every pixel of a plane carries the same recognisable number and a swapped plane is
therefore off by the whole dynamic range rather than by an interpolation weight.

The load-bearing case is the one with an asymmetric ``mean``/``std``. Under the defaults —
``mean=0``, ``std=255`` for all three channels — a backend that ignored ``swap_rb`` entirely
and one that honoured it produce byte-identical tensors on any input whose channels happen to
match, and a backend that *reordered* ``mean`` and ``std`` along with the pixels is
indistinguishable from one that did not. Only a non-symmetric normalisation separates the
three, so that is what the hand-computed cases use.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipvision.errors import BackendUnavailableError
from shipvision.imgproc import IMGPROC
from shipvision.imgproc.base import DeviceBuffer
from shipvision.registry import PYTHON
from tests.imgproc.conftest import backend_params

BLUE, GREEN, RED = 10, 120, 230
"""One recognisable level per input channel, in the BGR order the input is documented in."""

MEAN = (1.0, 2.0, 3.0)
STD = (2.0, 4.0, 8.0)
"""Deliberately different per channel, and deliberately not a real checkpoint's statistics:
the numbers are chosen so every expected value below divides exactly in float32 and can be
read off the docstring rather than recomputed by the thing under test."""

SOURCE_HW = (40, 40)
TARGET_HW = (20, 20)
"""Square into square, so the letterbox has no bars at all and every output pixel comes from
the image. A bar would be ``(pad_value - mean) / std``, which is a different claim."""

CROP_TARGET_HW = (8, 8)
WHOLE_FRAME = np.array([[0.0, 0.0, 40.0, 40.0]], dtype=np.float32)

TOLERANCE = 1e-4
"""Absolute. The interpolation weights over a constant plane sum to one, so the answer is the
constant to within float32 rounding; a channel in the wrong plane is out by ~0.9."""


@pytest.fixture(params=backend_params())
def candidate(request):
    """One image-ops backend that this machine can actually build."""
    return IMGPROC.build("default", backend=request.param)


@pytest.fixture()
def flat_bgr() -> np.ndarray:
    """A frame whose blue, green and red planes are three different constants."""
    image = np.empty((*SOURCE_HW, 3), dtype=np.uint8)
    image[..., 0] = BLUE
    image[..., 1] = GREEN
    image[..., 2] = RED
    return image


def planes(tensor: np.ndarray) -> list[float]:
    """The three channel values of a single-image ``(1, 3, h, w)`` batch."""
    assert tensor.shape[0] == 1 and tensor.shape[1] == 3, tensor.shape
    return [float(tensor[0, channel].mean()) for channel in range(3)]


class TestLetterboxHonoursTheFlag:
    """The default still emits RGB; ``swap_rb=False`` leaves the input's BGR order alone."""

    def test_the_default_puts_red_in_plane_zero(self, candidate, flat_bgr) -> None:
        """The behaviour every existing caller has, asserted so the new keyword cannot change
        it by accident: an unqualified call is still BGR in, RGB out."""
        tensor, _ = candidate.letterbox(flat_bgr, TARGET_HW)

        assert planes(tensor) == pytest.approx(
            [RED / 255.0, GREEN / 255.0, BLUE / 255.0], abs=TOLERANCE
        )

    def test_swap_rb_false_puts_blue_in_plane_zero(self, candidate, flat_bgr) -> None:
        """Plane 0 is the input's plane 0. That is the whole feature."""
        tensor, _ = candidate.letterbox(flat_bgr, TARGET_HW, swap_rb=False)

        assert planes(tensor) == pytest.approx(
            [BLUE / 255.0, GREEN / 255.0, RED / 255.0], abs=TOLERANCE
        )

    def test_the_two_are_each_other_reversed_under_a_symmetric_normalisation(
        self, candidate, flat_bgr
    ) -> None:
        """Stated as a relation rather than as two lists of constants, because it is the
        relation a caller reasons with — and it holds only while ``mean`` and ``std`` are the
        same in every channel, which the next class is about."""
        swapped, _ = candidate.letterbox(flat_bgr, TARGET_HW, swap_rb=True)
        kept, _ = candidate.letterbox(flat_bgr, TARGET_HW, swap_rb=False)

        assert np.abs(swapped - kept[:, ::-1]).max() < TOLERANCE

    def test_the_geometry_is_the_same_either_way(self, candidate, flat_bgr) -> None:
        """A colour flag must not move a pixel. If it did, every box decoded through the
        returned geometry would land somewhere else on one of the two settings."""
        _, swapped = candidate.letterbox(flat_bgr, TARGET_HW, swap_rb=True)
        _, kept = candidate.letterbox(flat_bgr, TARGET_HW, swap_rb=False)

        assert swapped == kept


class TestCropBatchHonoursTheFlag:
    """The same claim on the embedding stage's hot path, which has its own sampler."""

    def test_the_default_puts_red_in_plane_zero(self, candidate, flat_bgr) -> None:
        crops = candidate.crop_batch(flat_bgr, WHOLE_FRAME, CROP_TARGET_HW)

        assert planes(crops) == pytest.approx(
            [RED / 255.0, GREEN / 255.0, BLUE / 255.0], abs=TOLERANCE
        )

    def test_swap_rb_false_puts_blue_in_plane_zero(self, candidate, flat_bgr) -> None:
        crops = candidate.crop_batch(flat_bgr, WHOLE_FRAME, CROP_TARGET_HW, swap_rb=False)

        assert planes(crops) == pytest.approx(
            [BLUE / 255.0, GREEN / 255.0, RED / 255.0], abs=TOLERANCE
        )

    def test_an_empty_box_list_is_still_an_empty_batch(self, candidate, flat_bgr) -> None:
        crops = candidate.crop_batch(
            flat_bgr, np.zeros((0, 4), dtype=np.float32), CROP_TARGET_HW, swap_rb=False
        )

        assert crops.shape == (0, 3, *CROP_TARGET_HW)


class TestMeanAndStdStayInDestinationOrder:
    """They are indexed by *output* plane and are never reordered to follow the flag.

    This is the half of convention 4 that a plausible implementation gets wrong. Reordering
    ``mean`` alongside the pixels feels symmetrical and makes the two settings agree, which is
    exactly the wrong answer: a checkpoint publishes its statistics in the order its own input
    has, so a model that wants BGR publishes them in BGR, and a library that helpfully
    reversed them would apply the red mean to the blue plane on every frame — a constant
    per-channel bias no downstream metric attributes to preprocessing.

    Both expected triples are computed by hand from ``(value - mean) / std`` and divide
    exactly in float32, so a wrong answer is wrong by a readable amount.
    """

    def test_the_swapped_output_takes_the_statistics_in_rgb_order(
        self, candidate, flat_bgr
    ) -> None:
        """Destination is ``(R, G, B) = (230, 120, 10)``, so the planes are
        ``(230-1)/2``, ``(120-2)/4`` and ``(10-3)/8``."""
        tensor, _ = candidate.letterbox(flat_bgr, TARGET_HW, mean=MEAN, std=STD, swap_rb=True)

        assert planes(tensor) == pytest.approx([114.5, 29.5, 0.875], abs=TOLERANCE)

    def test_the_unswapped_output_takes_the_same_statistics_in_bgr_order(
        self, candidate, flat_bgr
    ) -> None:
        """Destination is ``(B, G, R) = (10, 120, 230)``, so the *same* mean and std now give
        ``(10-1)/2``, ``(120-2)/4`` and ``(230-3)/8``. Nothing about ``mean`` moved."""
        tensor, _ = candidate.letterbox(flat_bgr, TARGET_HW, mean=MEAN, std=STD, swap_rb=False)

        assert planes(tensor) == pytest.approx([4.5, 29.5, 28.375], abs=TOLERANCE)

    def test_the_crop_path_agrees_with_the_letterbox_path(self, candidate, flat_bgr) -> None:
        """Two samplers, one normalisation rule. A crop of the whole frame and a letterbox of
        it are different code in every backend, and they must not disagree about which mean
        belongs to which plane."""
        crops = candidate.crop_batch(
            flat_bgr, WHOLE_FRAME, CROP_TARGET_HW, mean=MEAN, std=STD, swap_rb=False
        )

        assert planes(crops) == pytest.approx([4.5, 29.5, 28.375], abs=TOLERANCE)


class TestTheDeviceStubsCarryTheKeywordToo:
    """``letterbox_into`` and ``crop_batch_into`` take ``swap_rb`` on the base class.

    A backend without a device path must refuse with
    :class:`~shipvision.errors.BackendUnavailableError`, the same as it always did — and not
    with a ``TypeError`` about an unexpected keyword, which is what an ABC that grew the
    argument on only two of its four entry points would produce. The two failures read
    completely differently to a caller deciding whether to fall back.
    """

    def test_letterbox_into_still_refuses_typedly(self) -> None:
        ops = IMGPROC.build("default", backend=PYTHON)
        buffer = DeviceBuffer(pointer=0x1000, nbytes=1 << 20, device_index=0)

        with pytest.raises(BackendUnavailableError, match="device output"):
            ops.letterbox_into(
                [np.zeros((8, 8, 3), dtype=np.uint8)], TARGET_HW, buffer, swap_rb=False
            )

    def test_crop_batch_into_still_refuses_typedly(self) -> None:
        ops = IMGPROC.build("default", backend=PYTHON)
        buffer = DeviceBuffer(pointer=0x1000, nbytes=1 << 20, device_index=0)

        with pytest.raises(BackendUnavailableError, match="device output"):
            ops.crop_batch_into(
                np.zeros((8, 8, 3), dtype=np.uint8),
                np.zeros((1, 4), dtype=np.float32),
                CROP_TARGET_HW,
                buffer,
                swap_rb=False,
            )
