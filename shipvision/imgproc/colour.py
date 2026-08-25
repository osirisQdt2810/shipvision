"""NV12 and the colour conventions that decide what a decoded frame's pixels *are*.

This module exists for the same reason :mod:`shipvision.imgproc.geometry` does: none of the
decisions below can fail visibly. A frame converted with the wrong YUV range is uniformly
washed out or crushed, a frame converted with bilinear chroma has every colour edge shifted
by one luma pixel, and a frame whose chroma plane is read with the wrong stride is sheared.
All three look like a plausible photograph, and all three cost a detector accuracy that
nobody can attribute afterwards. So there is one statement of each rule, here, and the numpy
oracle and the CUDA kernel both implement *this*.

WHY NV12 AT ALL

A GPU video decoder produces NV12 — 8-bit luma followed by half-resolution interleaved
chroma, 12 bits per pixel. Every pipeline that then asks for BGR pays a colour-conversion
pass into a buffer twice the size, and if that conversion lands in system memory it pays a
device-to-host copy of the *larger* buffer plus a host-to-device copy back before inference.
At 1000 frames a second that is 6.2 GB/s each way for a result the network does not want, in
a layout the network does not want either. Reading NV12 straight from the decoder removes
the pass and halves the bytes.

CONVENTION 5 — CHROMA UPSAMPLING IS NEAREST
    The U,V pair at chroma coordinate ``(y // 2, x // 2)`` serves all four luma pixels of
    its 2x2 block::

        u = uv[(y // 2) * uv_stride + (x // 2) * 2 + 0]
        v = uv[(y // 2) * uv_stride + (x // 2) * 2 + 1]

    Bilinear chroma is arguably nicer and is *not* what a decoder's own conversion does, nor
    what the reference implementation this replaces does. Mixing the two shifts colour by
    half a chroma sample — one luma pixel — along every edge in the frame.

CONVENTION 6 — BT.601, LIMITED ("VIDEO") RANGE
    The range an H.264 camera stream is in, and the one the reference uses. Luma occupies
    16-235 and chroma 16-240, so the decode subtracts the offsets and scales back up::

        y' = Y - 16      u' = U - 128      v' = V - 128

        R = 1.164 y'              + 1.596 v'
        G = 1.164 y' - 0.391 u' - 0.813 v'
        B = 1.164 y' + 2.018 u'

    Each channel is then clamped to ``[0, 255]``. The coefficients are written out rather
    than kept as a matrix so that they can be compared line for line against
    ``csrc/shipvision/imgproc/image_ops.cu``, which is the only way two implementations of a colour
    transform stay equal.

CONVENTION 7 — CONVERT, THEN INTERPOLATE
    Bilinear resampling happens on the *converted* RGB values, with the clamp already
    applied to each tap. Interpolating in YUV and converting once is cheaper and gives a
    different answer, because the clamp is not linear — and the difference appears exactly at
    saturated colours, which is where an operator looks first.

CONVENTION 8 — THE STRIDE IS DATA, NOT ARITHMETIC
    A decoder pads rows to suit itself. GStreamer's system-memory NV12 gives a 1918-wide
    frame a 1920-byte stride; a pitched device allocator pads 1920 to 2048. So a host NV12
    frame is carried as a **2-D** ``(height * 3 // 2, stride)`` uint8 array — the row count
    gives the height, the second extent gives the stride, and the only thing a caller must
    still say is the *visible* width, which no buffer layout records.
"""

from __future__ import annotations

import numpy as np

from shipvision.errors import ConfigurationError, DimensionMismatchError

__all__ = [
    "NV12_BLUE_U",
    "NV12_CHROMA_OFFSET",
    "NV12_GREEN_U",
    "NV12_GREEN_V",
    "NV12_LUMA_GAIN",
    "NV12_LUMA_OFFSET",
    "NV12_RED_V",
    "bgr_to_nv12",
    "nv12_height",
    "nv12_rows",
    "nv12_to_rgb",
    "split_nv12",
    "validate_nv12_frame",
]

# Convention 6, as float32 so that every product below is computed in the kernel's precision.
NV12_LUMA_OFFSET = np.float32(16.0)
NV12_CHROMA_OFFSET = np.float32(128.0)
NV12_LUMA_GAIN = np.float32(1.164)
NV12_RED_V = np.float32(1.596)
NV12_GREEN_U = np.float32(0.391)
NV12_GREEN_V = np.float32(0.813)
NV12_BLUE_U = np.float32(2.018)


def nv12_rows(height: int) -> int:
    """Rows in a packed NV12 buffer for ``height`` luma rows: ``height * 3 // 2``."""
    if height <= 0 or height % 2:
        raise DimensionMismatchError(
            f"NV12 needs a positive even height, got {height}. 4:2:0 has one chroma row per "
            f"two luma rows, so there is no half row to store"
        )
    return height * 3 // 2


def nv12_height(rows: int) -> int:
    """The luma height a packed NV12 row count implies. The inverse of :func:`nv12_rows`.

    Inverted rather than passed alongside the array, so a caller has one fewer number to keep
    in step with the buffer. ``rows`` must be a multiple of 3 and the recovered height even,
    which together reject every off-by-one a ``(rows, stride)`` array could otherwise hide.
    """
    if rows <= 0 or rows % 3:
        raise DimensionMismatchError(
            f"a packed NV12 buffer has height * 3 / 2 rows, so the row count must be a "
            f"positive multiple of 3; got {rows}"
        )
    height = rows // 3 * 2
    if height % 2:
        raise DimensionMismatchError(f"{rows} rows implies an odd height of {height}")
    return height


def validate_nv12_frame(frame: np.ndarray, width: int, *, what: str = "frame") -> np.ndarray:
    """A C-contiguous ``(height * 3 // 2, stride)`` uint8 view of one NV12 frame.

    Convention 8. The width is checked against the stride rather than assumed equal to it,
    because they are equal only when the decoder happened not to pad — which is most cameras
    and not the interesting ones.

    Raises:
        DimensionMismatchError: not 2-D, an impossible row count, an odd extent, or a stride
            narrower than the visible width.
        ConfigurationError: not uint8.
    """
    array = np.asarray(frame)
    if array.ndim != 2:
        raise DimensionMismatchError(
            f"{what} must be a 2-D (height * 3 // 2, stride) uint8 NV12 buffer, got shape "
            f"{array.shape}. A 3-D array is an interleaved image, not NV12"
        )
    if array.dtype != np.uint8:
        raise ConfigurationError(
            f"{what} must be uint8, got {array.dtype}. NV12 is 8-bit by definition; a float "
            f"buffer means something upstream already converted it"
        )
    height = nv12_height(int(array.shape[0]))
    stride = int(array.shape[1])
    visible = int(width)
    if visible <= 0 or visible % 2:
        raise DimensionMismatchError(
            f"{what} needs a positive even width, got {visible}; one chroma sample serves a "
            f"2x2 luma block"
        )
    if stride < visible:
        raise DimensionMismatchError(
            f"{what} has stride {stride} but a visible width of {visible}. A stride below the "
            f"width makes every row start inside the previous one — a sheared image, with "
            f"nothing to report it"
        )
    if height <= 0:
        raise DimensionMismatchError(f"{what} has no rows")
    return np.ascontiguousarray(array)


def split_nv12(frame: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    """One NV12 buffer as ``(luma, chroma)`` views, both cropped to the visible width.

    ``luma`` is ``(height, width)`` and ``chroma`` is ``(height // 2, width // 2, 2)`` with
    U at index 0 and V at index 1. Views into ``frame``, not copies: this is called on the
    oracle path where the frame is already in host memory and a copy of a 1080p plane is
    2 MB nobody asked for.
    """
    array = validate_nv12_frame(frame, width)
    height = nv12_height(int(array.shape[0]))
    luma = array[:height, :width]
    # The chroma plane starts exactly `height` rows in, at the same stride — which is why the
    # offset is a row count and not a byte count computed from the width.
    chroma = array[height:, : width // 2 * 2].reshape(height // 2, width // 2, 2)
    return luma, chroma


def nv12_to_rgb(frame: np.ndarray, width: int) -> np.ndarray:
    """One NV12 frame as ``(height, width, 3)`` float32 RGB in ``[0, 255]``.

    Conventions 5 and 6, and the readable half of the parity test: the CUDA kernel converts
    only the four bilinear taps it needs, and this converts the whole frame — the same
    per-pixel arithmetic either way, so the two agree pixel for pixel and the kernel's
    fusion is not part of what is being trusted.

    The chroma upsample is ``np.repeat`` twice, which *is* nearest-neighbour by construction:
    every 2x2 luma block gets the one chroma sample that covers it. Writing it as an index
    expression would work equally well and would leave the convention implicit.
    """
    luma, chroma = split_nv12(frame, width)
    height = luma.shape[0]

    y = luma.astype(np.float32) - NV12_LUMA_OFFSET
    upsampled = np.repeat(np.repeat(chroma, 2, axis=0), 2, axis=1)[:height, : luma.shape[1]]
    u = upsampled[..., 0].astype(np.float32) - NV12_CHROMA_OFFSET
    v = upsampled[..., 1].astype(np.float32) - NV12_CHROMA_OFFSET

    gain = NV12_LUMA_GAIN * y
    # Left-to-right, one term at a time, because that is the association order the kernel's
    # `1.164f * luma - 0.391f * u - 0.813f * v` compiles to. Grouping the two chroma terms
    # first is algebraically identical and differs in the last float32 digit.
    red = gain + NV12_RED_V * v
    green = gain - NV12_GREEN_U * u
    green = green - NV12_GREEN_V * v
    blue = gain + NV12_BLUE_U * u

    out = np.stack((red, green, blue), axis=-1)
    return np.clip(out, np.float32(0.0), np.float32(255.0)).astype(np.float32)


# --------------------------------------------------------------------------- the encoder
#
# The forward direction is not on any production path — a decoder emits NV12 and nothing
# here ever has to produce it. It is in this module because the *tests* need real NV12 and a
# separate encoder written next to them would drift from the conventions above, which is
# exactly the failure this file exists to prevent.

_BGR_TO_Y = np.array([0.098, 0.504, 0.257], dtype=np.float32)
_BGR_TO_U = np.array([0.439, -0.291, -0.148], dtype=np.float32)
_BGR_TO_V = np.array([-0.071, -0.368, 0.439], dtype=np.float32)


def bgr_to_nv12(image: np.ndarray, *, stride: int | None = None) -> tuple[np.ndarray, int]:
    """``(h, w, 3)`` uint8 BGR -> a packed NV12 buffer and its visible width.

    The inverse of :func:`nv12_to_rgb` under the same BT.601 limited-range convention, with
    chroma taken as the mean of each 2x2 block — which is what a decoder's 4:2:0 subsampling
    does and the inverse of the nearest-neighbour upsample on the way back.

    Args:
        image: an even-sized BGR frame. Odd extents are refused rather than cropped, because
            a silently cropped frame shifts every box by half a pixel.
        stride: bytes per row in the result, defaulting to the width. Pass something larger
            to reproduce a padding decoder — which is the case a naive reader gets wrong.

    Returns:
        ``(buffer, width)`` where ``buffer`` is ``(h * 3 // 2, stride)`` uint8.
    """
    frame = np.asarray(image)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise DimensionMismatchError(f"expected (h, w, 3) BGR, got {frame.shape}")
    if frame.dtype != np.uint8:
        raise ConfigurationError(f"expected uint8 BGR, got {frame.dtype}")
    height, width = int(frame.shape[0]), int(frame.shape[1])
    if height % 2 or width % 2:
        raise DimensionMismatchError(
            f"NV12 is 4:2:0, so both extents must be even; got {height}x{width}"
        )
    row = width if stride is None else int(stride)
    if row < width:
        raise ConfigurationError(f"stride {row} is narrower than the width {width}")

    values = frame.astype(np.float32)
    luma = values @ _BGR_TO_Y + np.float32(16.0)
    blocks = values.reshape(height // 2, 2, width // 2, 2, 3).mean(axis=(1, 3))
    u = blocks @ _BGR_TO_U + np.float32(128.0)
    v = blocks @ _BGR_TO_V + np.float32(128.0)

    buffer = np.zeros((nv12_rows(height), row), dtype=np.uint8)
    # `+ 0.5` then truncate, matching the reference encoder's `static_cast<uchar>(x + 0.5f)`.
    buffer[:height, :width] = np.clip(luma + np.float32(0.5), 0, 255).astype(np.uint8)
    chroma = np.empty((height // 2, width // 2, 2), dtype=np.uint8)
    chroma[..., 0] = np.clip(u + np.float32(0.5), 0, 255).astype(np.uint8)
    chroma[..., 1] = np.clip(v + np.float32(0.5), 0, 255).astype(np.uint8)
    buffer[height:, :width] = chroma.reshape(height // 2, width)
    return buffer, width
