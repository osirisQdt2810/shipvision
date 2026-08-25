"""NV12 letterbox: the conventions, and the fused kernel against the numpy oracle.

A colour-conversion bug does not raise. It shifts hue, or washes the frame out, or moves
every edge by a pixel — and the detector that follows loses a point of mAP with nothing in a
log to say why. So the assertions here are about the *specific* ways NV12 goes wrong:

* nearest chroma versus bilinear chroma, which differ only along colour edges;
* limited range versus full range, which differ everywhere but only by a scale;
* the stride, which differs only when the decoder pads;
* convert-then-interpolate versus interpolate-then-convert, which differ only where a
  channel saturates.

The fixtures are **structured, not random**. A 2x2 chroma checkerboard is what tells nearest
apart from bilinear; uniform noise averages the difference away and passes either. Colour
bars at full saturation are what tell the clamp's position apart; a photograph rarely
saturates. That is the opposite reasoning from ``test_parity.py``'s noise images, and for the
opposite reason: there the risk is a half-pixel geometry error, which noise exposes and
structure hides.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipvision.errors import ConfigurationError, DimensionMismatchError
from shipvision.imgproc import IMGPROC, bgr_to_nv12, nv12_to_rgb
from shipvision.imgproc.colour import nv12_height, nv12_rows, split_nv12
from shipvision.imgproc.geometry import LetterboxGeometry
from shipvision.registry import NATIVE, PYTHON
from tests.imgproc.conftest import NATIVE_BUILT

VALUE_TOLERANCE = 1e-3
"""Absolute, on tensors normalised into ``[0, 1]``. A quarter of one 0-255 level."""


# --------------------------------------------------------------------------- fixtures


def chroma_checkerboard(height: int = 64, width: int = 96) -> tuple[np.ndarray, int]:
    """NV12 whose chroma alternates every 2x2 block, with flat luma.

    The single most diagnostic NV12 fixture there is. Luma carries no signal, so anything
    visible in the output came from chroma — and because the pattern's period is exactly the
    chroma sampling grid, nearest upsampling gives hard 2x2 colour squares while bilinear
    gives a smooth wash. The two disagree by ~40 levels in the middle of every block.
    """
    buffer = np.zeros((nv12_rows(height), width), dtype=np.uint8)
    buffer[:height, :] = 128  # mid luma: BT.601 grey, and no luma structure at all
    chroma = np.empty((height // 2, width // 2, 2), dtype=np.uint8)
    block_y = np.arange(height // 2)[:, None]
    block_x = np.arange(width // 2)[None, :]
    alternating = (block_y + block_x) % 2
    chroma[..., 0] = np.where(alternating, 240, 16)  # U: hard blue / hard yellow
    chroma[..., 1] = np.where(alternating, 16, 240)  # V: anti-correlated, so hue swings hard
    buffer[height:, :] = chroma.reshape(height // 2, width)
    return buffer, width


def colour_bars(height: int = 48, width: int = 64) -> tuple[np.ndarray, int]:
    """Eight saturated vertical bars, encoded through the reference BT.601 forward transform.

    Saturation is the point: several of these bars sit at or past the edge of the RGB cube
    after the limited-range expansion, so they exercise the clamp — the one non-linear step,
    and therefore the only place convert-then-interpolate can be told from
    interpolate-then-convert.
    """
    bars = np.array(
        [
            (255, 255, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0),
            (255, 0, 255), (0, 0, 255), (255, 0, 0), (0, 0, 0),
        ],
        dtype=np.uint8,
    )
    image = np.zeros((height, width, 3), dtype=np.uint8)
    span = max(width // len(bars), 2)
    for index, colour in enumerate(bars):
        image[:, index * span : (index + 1) * span] = colour
    return bgr_to_nv12(image)


def luma_ramp(height: int = 34, width: int = 50) -> tuple[np.ndarray, int]:
    """A diagonal luma ramp over neutral chroma: a smooth gradient with no colour.

    The geometry fixture of the three. A ramp makes a half-pixel sampling error a *constant*
    offset that shows up in every pixel, where a checkerboard would make it a local sign flip
    that a max-absolute assertion could still pass by luck.
    """
    buffer = np.full((nv12_rows(height), width), 128, dtype=np.uint8)
    ramp = (np.arange(height)[:, None] * 3 + np.arange(width)[None, :] * 2) % 220 + 16
    buffer[:height, :] = ramp.astype(np.uint8)
    return buffer, width


FIXTURES = {
    "chroma_checkerboard": chroma_checkerboard,
    "colour_bars": colour_bars,
    "luma_ramp": luma_ramp,
}


@pytest.fixture()
def oracle_ops():
    return IMGPROC.build("default", backend=PYTHON)


# ------------------------------------------------------------------- the conventions


class TestNv12Layout:
    """A packed NV12 buffer is (height * 3 // 2, stride), and the stride is not the width."""

    def test_rows_and_height_invert_each_other(self) -> None:
        for height in (2, 48, 720, 1080):
            assert nv12_height(nv12_rows(height)) == height

    def test_an_odd_height_has_no_chroma_row(self) -> None:
        with pytest.raises(DimensionMismatchError, match="even height"):
            nv12_rows(1081)

    def test_a_row_count_that_is_not_a_multiple_of_three_is_not_nv12(self) -> None:
        # 1620 is a real 1080p buffer; 1621 is one byte of somebody else's frame.
        assert nv12_height(1620) == 1080
        with pytest.raises(DimensionMismatchError, match="multiple of 3"):
            nv12_height(1621)

    def test_a_padded_stride_still_splits_into_the_visible_planes(self) -> None:
        """The case a naive reader gets wrong: 1918 visible pixels in 1920-byte rows."""
        buffer = np.zeros((nv12_rows(6), 1920), dtype=np.uint8)
        buffer[:6, :1918] = 200
        buffer[:6, 1918:] = 7  # the pad bytes, which must never be read
        luma, chroma = split_nv12(buffer, 1918)
        assert luma.shape == (6, 1918)
        assert chroma.shape == (3, 959, 2)
        assert luma.max() == 200, "the pad columns leaked into the visible plane"

    def test_a_stride_below_the_width_is_refused(self) -> None:
        with pytest.raises(DimensionMismatchError, match="sheared"):
            split_nv12(np.zeros((nv12_rows(4), 8), dtype=np.uint8), 10)

    def test_a_three_dimensional_array_is_not_nv12(self) -> None:
        with pytest.raises(DimensionMismatchError, match="2-D"):
            split_nv12(np.zeros((6, 8, 3), dtype=np.uint8), 8)

    def test_a_float_buffer_means_something_already_converted_it(self) -> None:
        with pytest.raises(ConfigurationError, match="uint8"):
            split_nv12(np.zeros((nv12_rows(4), 8), dtype=np.float32), 8)


class TestNv12Colour:
    """BT.601 limited range, nearest chroma — the two decisions nothing else can see."""

    def test_neutral_chroma_and_mid_luma_is_grey(self) -> None:
        buffer = np.full((nv12_rows(4), 4), 128, dtype=np.uint8)
        rgb = nv12_to_rgb(buffer, 4)
        # 1.164 * (128 - 16) = 130.368, and no chroma term.
        assert np.allclose(rgb, 130.368, atol=1e-3)
        assert np.ptp(rgb) == pytest.approx(0.0, abs=1e-4)

    def test_limited_range_black_is_zero_and_white_is_235(self) -> None:
        """Y=16 is black and Y=235 is white. Reading them as 0 and 255 washes the frame out."""
        black = np.full((nv12_rows(2), 2), 128, dtype=np.uint8)
        black[:2, :] = 16
        assert nv12_to_rgb(black, 2).max() == pytest.approx(0.0)

        white = np.full((nv12_rows(2), 2), 128, dtype=np.uint8)
        white[:2, :] = 235
        # 1.164 * (235 - 16) = 254.9, which is white to within a level and NOT clamped.
        assert nv12_to_rgb(white, 2).min() == pytest.approx(254.916, abs=1e-2)

    def test_luma_above_the_limited_range_clamps_rather_than_overflowing(self) -> None:
        buffer = np.full((nv12_rows(2), 2), 128, dtype=np.uint8)
        buffer[:2, :] = 255
        assert nv12_to_rgb(buffer, 2).max() == pytest.approx(255.0)

    def test_chroma_upsampling_is_nearest_so_a_2x2_block_is_one_flat_colour(self) -> None:
        """Convention 5. Bilinear chroma would put four different colours in this block."""
        buffer, width = chroma_checkerboard(height=8, width=8)
        rgb = nv12_to_rgb(buffer, width)
        for top in (0, 2, 4, 6):
            for left in (0, 2, 4, 6):
                block = rgb[top : top + 2, left : left + 2]
                assert np.ptp(block.reshape(-1, 3), axis=0).max() == pytest.approx(0.0), (
                    f"the 2x2 block at ({top}, {left}) is not flat, so the chroma upsample "
                    f"interpolated where it must replicate"
                )

    def test_neighbouring_chroma_blocks_really_do_differ(self) -> None:
        """Guards the test above: a fixture with flat chroma would satisfy it vacuously."""
        buffer, width = chroma_checkerboard(height=8, width=8)
        rgb = nv12_to_rgb(buffer, width)
        assert np.abs(rgb[0:2, 0:2].mean(axis=(0, 1)) - rgb[0:2, 2:4].mean(axis=(0, 1))).max() > 50

    def test_a_bgr_round_trip_lands_within_chroma_subsampling_error(self) -> None:
        """4:2:0 is lossy, so this bounds the loss rather than asserting equality.

        The bound is per-channel and generous on colour and tight on luma, which is the shape
        of the error 4:2:0 actually makes: luma is untouched, chroma is averaged over 2x2.
        """
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        image[:, :, 0] = np.arange(16, dtype=np.uint8)[None, :] * 15  # B ramp across
        image[:, :, 1] = np.arange(16, dtype=np.uint8)[:, None] * 15  # G ramp down
        image[:, :, 2] = 128
        buffer, width = bgr_to_nv12(image)
        recovered = nv12_to_rgb(buffer, width)[..., ::-1]  # back to BGR for comparison
        assert np.abs(recovered - image.astype(np.float32)).max() < 24


class TestNv12Stride:
    """A padded stride must change nothing about the answer."""

    def test_padding_the_stride_does_not_change_the_output(self, oracle_ops) -> None:
        """The whole risk of a stride bug: it is invisible until a camera pads."""
        packed, width = luma_ramp(height=34, width=50)
        padded = np.zeros((packed.shape[0], 64), dtype=np.uint8)
        padded[:, :width] = packed
        padded[:, width:] = 199  # junk in the pad columns; reading it would show up loudly

        tight, _ = oracle_ops.nv12_letterbox([packed], [width], (32, 32))
        loose, _ = oracle_ops.nv12_letterbox([padded], [width], (32, 32))
        assert np.abs(tight - loose).max() == pytest.approx(0.0)


class TestNv12Letterbox:
    """The NV12 path must be the BGR path in every convention except the input format."""

    @pytest.mark.parametrize("name", sorted(FIXTURES))
    @pytest.mark.parametrize("target_hw", [(64, 64), (40, 96)])
    def test_geometry_matches_the_bgr_path(self, oracle_ops, name, target_hw) -> None:
        buffer, width = FIXTURES[name]()
        height = nv12_height(buffer.shape[0])
        tensor, geometries = oracle_ops.nv12_letterbox([buffer], [width], target_hw)

        assert tensor.shape == (1, 3, *target_hw)
        assert geometries == [LetterboxGeometry.plan((height, width), target_hw)]

    def test_the_bars_carry_the_normalised_pad_value(self, oracle_ops) -> None:
        buffer, width = luma_ramp(height=34, width=50)
        tensor, geometry = oracle_ops.nv12_letterbox([buffer], [width], (64, 64), pad_value=114)
        top = geometry[0].pad_top
        assert top > 0, "this fixture must letterbox with a real bar to test anything"
        assert np.abs(tensor[0, :, :top, :] - 114.0 / 255.0).max() < 1e-6

    def test_swap_rb_names_the_destination_order(self, oracle_ops) -> None:
        """``True`` is RGB and ``False`` is BGR — the same meaning as on the BGR path."""
        buffer, width = colour_bars()
        as_rgb, _ = oracle_ops.nv12_letterbox([buffer], [width], (32, 32), swap_rb=True)
        as_bgr, _ = oracle_ops.nv12_letterbox([buffer], [width], (32, 32), swap_rb=False)
        assert np.abs(as_rgb[:, ::-1] - as_bgr).max() == pytest.approx(0.0)
        assert np.abs(as_rgb - as_bgr).max() > 0.1, "the fixture has no colour to swap"

    def test_a_ragged_batch_is_one_call(self, oracle_ops) -> None:
        """50 cameras do not agree on resolution, which is why the batch is ragged."""
        frames, widths = zip(*(FIXTURES[name]() for name in sorted(FIXTURES)))
        tensor, geometries = oracle_ops.nv12_letterbox(list(frames), list(widths), (64, 64))
        assert tensor.shape == (3, 3, 64, 64)
        assert len({(g.source_height, g.source_width) for g in geometries}) == 3

    def test_a_width_per_frame_is_required(self, oracle_ops) -> None:
        buffer, width = luma_ramp()
        with pytest.raises(ConfigurationError, match="one width per frame"):
            oracle_ops.nv12_letterbox([buffer, buffer], [width], (32, 32))

    def test_an_empty_batch_is_a_caller_bug(self, oracle_ops) -> None:
        with pytest.raises(ConfigurationError, match="at least one frame"):
            oracle_ops.nv12_letterbox([], [], (32, 32))


class TestNv12MatchesTheBgrPathOnGreyscale:
    """A frame with neutral chroma must letterbox identically through both paths.

    This is the strongest available cross-check that is not a parity test against the same
    author's oracle: with U = V = 128 the NV12 decode reduces to a luma gain, so the same
    pixels can be pushed through the *BGR* letterbox — a completely separate code path, with
    its own sampler — and the two must agree to within the one-level rounding that the uint8
    intermediate costs.
    """

    def test_a_grey_ramp_agrees_with_the_bgr_letterbox(self, oracle_ops) -> None:
        buffer, width = luma_ramp(height=34, width=50)
        height = nv12_height(buffer.shape[0])
        grey = nv12_to_rgb(buffer, width)  # exact, and already RGB

        via_nv12, _ = oracle_ops.nv12_letterbox([buffer], [width], (64, 64))
        # Feed the same values to the BGR path. It expects uint8 BGR, so round once and
        # reverse the channels; the rounding is why the tolerance below is a level and not
        # a float32 epsilon.
        as_bgr = np.clip(grey + 0.5, 0, 255).astype(np.uint8)[..., ::-1]
        via_bgr, _ = oracle_ops.letterbox(as_bgr, (64, 64))

        assert via_nv12.shape == via_bgr.shape == (1, 3, 64, 64)
        assert np.abs(via_nv12 - via_bgr).max() < 1.5 / 255.0
        assert height == 34


@pytest.mark.native
@pytest.mark.skipif(not NATIVE_BUILT, reason="shipvision._C is not built, or there is no GPU")
class TestNativeNv12Parity:
    """The fused kernel against the oracle. The reason the oracle is in the repository."""

    @pytest.fixture()
    def native_ops(self):
        return IMGPROC.build("default", backend=NATIVE)

    @pytest.mark.parametrize("name", sorted(FIXTURES))
    @pytest.mark.parametrize("target_hw", [(64, 64), (40, 96), (128, 128)])
    def test_the_kernel_agrees_with_the_oracle(
        self, native_ops, oracle_ops, name, target_hw
    ) -> None:
        buffer, width = FIXTURES[name]()
        expected, expected_geometry = oracle_ops.nv12_letterbox([buffer], [width], target_hw)
        actual, actual_geometry = native_ops.nv12_letterbox([buffer], [width], target_hw)

        assert actual.shape == expected.shape
        assert actual.dtype == np.float32
        assert actual_geometry == expected_geometry
        assert np.abs(actual - expected).max() < VALUE_TOLERANCE

    def test_a_padded_stride_agrees_too(self, native_ops, oracle_ops) -> None:
        """The kernel reads `y_stride` from the buffer's shape; a bug here shears the frame."""
        packed, width = colour_bars(height=48, width=62)
        padded = np.zeros((packed.shape[0], 64), dtype=np.uint8)
        padded[:, :width] = packed
        padded[:, width:] = 199

        expected, _ = oracle_ops.nv12_letterbox([padded], [width], (64, 64))
        actual, _ = native_ops.nv12_letterbox([padded], [width], (64, 64))
        assert np.abs(actual - expected).max() < VALUE_TOLERANCE

    def test_a_ragged_batch_agrees_frame_for_frame(self, native_ops, oracle_ops) -> None:
        frames, widths = zip(*(FIXTURES[name]() for name in sorted(FIXTURES)))
        expected, _ = oracle_ops.nv12_letterbox(list(frames), list(widths), (64, 64))
        actual, _ = native_ops.nv12_letterbox(list(frames), list(widths), (64, 64))
        assert np.abs(actual - expected).max() < VALUE_TOLERANCE

    def test_bgr_output_agrees_too(self, native_ops, oracle_ops) -> None:
        buffer, width = colour_bars()
        expected, _ = oracle_ops.nv12_letterbox([buffer], [width], (64, 64), swap_rb=False)
        actual, _ = native_ops.nv12_letterbox([buffer], [width], (64, 64), swap_rb=False)
        assert np.abs(actual - expected).max() < VALUE_TOLERANCE

    def test_normalisation_agrees(self, native_ops, oracle_ops) -> None:
        buffer, width = colour_bars()
        mean, std = (123.675, 116.28, 103.53), (58.395, 57.12, 57.375)
        expected, _ = oracle_ops.nv12_letterbox(
            [buffer], [width], (64, 64), mean=mean, std=std
        )
        actual, _ = native_ops.nv12_letterbox([buffer], [width], (64, 64), mean=mean, std=std)
        assert np.abs(actual - expected).max() < VALUE_TOLERANCE

    def test_an_odd_extent_is_refused_rather_than_read_off_the_end(self, native_ops) -> None:
        buffer = np.full((nv12_rows(8), 10), 128, dtype=np.uint8)
        with pytest.raises(DimensionMismatchError, match="even width"):
            native_ops.nv12_letterbox([buffer], [9], (32, 32))
