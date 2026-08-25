"""Pixel geometry: how an image is fitted into a network input, and how to undo it.

Everything in this module is a *convention* rather than an algorithm, and the conventions are
written down here because none of them can fail visibly. A letterboxed image with the pad on
the wrong side looks exactly like a correct one. An image sampled half a pixel off looks
exactly like a correct one. In both cases every box the detector produces is shifted by the
same amount, on every camera, for as long as nobody checks — and the reference implementations
this library replaces disagree with each other on all three of the rules below.

So there is one statement of each rule, and one implementation of it, and all three backends
call it.

CONVENTION 1 — SAMPLING CENTRES ARE HALF-PIXEL, ``align_corners=False``
    Output pixel ``i`` reads the source at::

        src = (i + 0.5) * source_extent / resized_extent - 0.5

    This is what OpenCV's ``INTER_LINEAR``, ``torch.nn.functional.interpolate(...,
    align_corners=False)`` and the CUDA kernel in ``csrc/shipvision/imgproc/image_ops.cu`` all do.
    Note the ratio is ``source_extent / resized_extent`` — the *achieved* ratio, after the
    resized extent was rounded to whole pixels — and **not** ``1 / scale``. Those differ by up
    to half a pixel over the whole image, and the kernel uses the achieved ratio, so numpy and
    torch do too. See :func:`resize_centres`.

    Taps outside the source are handled by clamping the *index*, not the coordinate:
    ``x0 = floor(src)``, ``x1 = min(x0 + 1, extent - 1)``, and the low tap is read at
    ``max(x0, 0)`` while the interpolation weight still uses the unclamped ``x0``. For the
    coordinate range this code can produce — ``src`` is never below ``-0.5`` nor above
    ``extent - 1`` — that is bit-identical to torch's "clamp the coordinate to 0" and to
    ``grid_sample(padding_mode="border")``. There is no antialiasing on the downscale path, in
    any backend, because the kernel has none.

CONVENTION 2 — THE RESIZED EXTENT IS ROUNDED HALF UP
    ``scale = min(target_h / source_h, target_w / source_w)`` computed in **float32**, then
    ``resized = max(1, floor(source * scale + 0.5))``. Round half up rather than numpy's
    default round-half-to-even, because the kernel's ``lroundf`` rounds half away from zero and
    the two rules genuinely disagree — a 5-row source scaled by 0.5 is 3 rows here and 2 rows
    under ``numpy.round``. float32 for the same reason: the kernel has no float64, and a scale
    that differs in the seventh digit is enough to round one extent differently.

CONVENTION 3 — ODD PADDING GOES TO THE BOTTOM AND THE RIGHT
    ``pad_top = (target_h - resized_h) // 2``, ``pad_left = (target_w - resized_w) // 2``.
    Floor division, so a total pad of 281 becomes 140 on top and 141 on the bottom, and the
    image sits one pixel above centre. That is what ``(dst_h - out_h) / 2`` in integer C++
    already did; making the Python match it is cheaper than making three backends agree on a
    nicer rule.

Convention 4, colour order and normalisation, belongs to :mod:`shipvision.imgproc.base`
alongside the ``mean``/``std`` arguments it governs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from shipvision.errors import ConfigurationError, DimensionMismatchError

__all__ = [
    "LetterboxGeometry",
    "clamp_boxes_to_frame",
    "crop_centres",
    "resize_centres",
    "validate_target_hw",
]


@dataclass(slots=True, frozen=True)
class LetterboxGeometry:
    """How one image was fitted into the network input.

    Returned by :meth:`~shipvision.imgproc.base.ImageOps.letterbox` and carried to
    post-processing, never recomputed there. Recomputing means re-deriving ``scale`` and the
    pads from the two shapes, which works right up to the first camera whose resolution rounds
    differently — and then boxes drift by a pixel or two on that camera only, which is the
    hardest class of bug to attribute. Passing the numbers that were actually used removes the
    possibility.

    It lives in its own module, above the backends rather than inside one, because it is what
    a *consumer* needs: a detector inverts its boxes through this object and has no reason to
    know which backend produced the tensor.

    Attributes:
        scale: the single factor applied to both axes, so aspect ratio is preserved.
        pad_left: destination x of the resized image's left edge.
        pad_top: destination y of the resized image's top edge.
        source_height: the original image's height, so the inverse can clip to it.
        source_width: the original image's width.
        target_height: the network input's height.
        target_width: the network input's width.
    """

    scale: float
    pad_left: int
    pad_top: int
    source_height: int
    source_width: int
    target_height: int
    target_width: int

    def __post_init__(self) -> None:
        if self.scale <= 0.0:
            raise ConfigurationError(f"letterbox scale must be positive, got {self.scale}")
        if self.source_height <= 0 or self.source_width <= 0:
            raise ConfigurationError(
                f"source extent must be positive, got {self.source_height}x{self.source_width}"
            )

    # -- construction -----------------------------------------------------------------

    @classmethod
    def plan(cls, source_hw: tuple[int, int], target_hw: tuple[int, int]) -> LetterboxGeometry:
        """The one implementation of conventions 2 and 3.

        Every backend calls this — including the native one, which computes the same numbers
        in C++ and then has them checked against these. Two copies of a rounding rule is one
        copy too many.
        """
        source_h, source_w = int(source_hw[0]), int(source_hw[1])
        target_h, target_w = validate_target_hw(target_hw)
        if source_h <= 0 or source_w <= 0:
            raise DimensionMismatchError(
                f"cannot letterbox a {source_h}x{source_w} image; both extents must be > 0"
            )

        # float32 throughout: see convention 2. The C++ side has no float64.
        scale = float(
            min(
                np.float32(target_h) / np.float32(source_h),
                np.float32(target_w) / np.float32(source_w),
            )
        )
        resized_h = _round_half_up(source_h, scale)
        resized_w = _round_half_up(source_w, scale)
        return cls(
            scale=scale,
            pad_left=(target_w - resized_w) // 2,
            pad_top=(target_h - resized_h) // 2,
            source_height=source_h,
            source_width=source_w,
            target_height=target_h,
            target_width=target_w,
        )

    # -- derived extents --------------------------------------------------------------

    @property
    def resized_height(self) -> int:
        """The height the source occupies inside the canvas, bars excluded."""
        return _round_half_up(self.source_height, self.scale)

    @property
    def resized_width(self) -> int:
        return _round_half_up(self.source_width, self.scale)

    @property
    def pad_bottom(self) -> int:
        """One more than :attr:`pad_top` when the total vertical pad is odd."""
        return self.target_height - self.pad_top - self.resized_height

    @property
    def pad_right(self) -> int:
        return self.target_width - self.pad_left - self.resized_width

    # -- inversion --------------------------------------------------------------------

    def invert_boxes(self, boxes: np.ndarray) -> np.ndarray:
        """``(n, 4)`` xyxy in network space -> ``(n, 4)`` xyxy in original-image pixels.

        The algebraic inverse of the forward map ``dst = src * scale + pad``: subtract the pad,
        divide by the scale, clip to the source extent. Dividing rather than multiplying by a
        precomputed reciprocal, because the reciprocal of a float32 scale is not exact and the
        error is then multiplied by up to 1920.

        Clipping is to ``[0, source_width]`` and ``[0, source_height]`` — the continuous
        extent, not ``extent - 1`` — because an xyxy box is a pair of continuous edges, and a
        detection that genuinely touches the right edge of the frame has ``x2 == width``.

        A detector never sees the source grid, so the recovered box carries the resize's
        rounding: the residual is bounded by ``0.5 / scale`` source pixels, which is sub-pixel
        at every sizing this library targets and exactly zero whenever ``source * scale`` is
        already whole.
        """
        array = np.asarray(boxes, dtype=np.float32)
        if array.size == 0:
            return np.zeros((0, 4), dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 4:
            raise DimensionMismatchError(
                f"boxes must be (n, 4) xyxy, got {array.shape}. A (4,) box is not a batch — "
                f"reshape it, so the caller decides rather than a broadcast"
            )

        scale = np.float32(self.scale)
        out = np.empty_like(array)
        out[:, 0::2] = (array[:, 0::2] - np.float32(self.pad_left)) / scale
        out[:, 1::2] = (array[:, 1::2] - np.float32(self.pad_top)) / scale
        np.clip(out[:, 0::2], 0.0, self.source_width, out=out[:, 0::2])
        np.clip(out[:, 1::2], 0.0, self.source_height, out=out[:, 1::2])
        return out

    def invert_points(self, points: np.ndarray) -> np.ndarray:
        """``(n, 2)`` or ``(n, k, 2)`` in network space -> original-image pixels.

        Any trailing columns beyond the first two are copied through untouched, so a
        ``(n, k, 3)`` keypoint array keeps its confidence channel. Shape is preserved.
        """
        array = np.asarray(points, dtype=np.float32)
        if array.size == 0:
            return array.astype(np.float32, copy=True)
        if array.ndim < 2 or array.shape[-1] < 2:
            raise DimensionMismatchError(
                f"points must have at least two trailing coordinate columns, got {array.shape}"
            )

        scale = np.float32(self.scale)
        out = array.astype(np.float32, copy=True)
        out[..., 0] = np.clip(
            (array[..., 0] - np.float32(self.pad_left)) / scale, 0.0, self.source_width
        )
        out[..., 1] = np.clip(
            (array[..., 1] - np.float32(self.pad_top)) / scale, 0.0, self.source_height
        )
        return out


def _round_half_up(extent: int, scale: float) -> int:
    """``floor(extent * scale + 0.5)`` in float32, floored at 1. Convention 2."""
    scaled = float(np.float32(extent) * np.float32(scale))
    return max(1, math.floor(scaled + 0.5))


# --------------------------------------------------------------------------- sampling

# Convention 1 lives in these two functions, and every backend calls them. The torch backend
# turns the coordinates into a `grid_sample` grid and the numpy backend gathers at them
# directly; neither re-derives them, because a second copy of a half-pixel rule is how two
# backends end up half a pixel apart.


def resize_centres(source_extent: int, resized_extent: int) -> np.ndarray:
    """Half-pixel sampling centres for a resize, in the CUDA kernel's evaluation order.

    ``((i + 0.5) * source) / resized - 0.5`` — multiplying *before* dividing, exactly as
    ``imgproc_image_ops.cu`` does. Computing ``source / resized`` first and then multiplying,
    which is what torch does internally, differs in the last float32 digit; this function is
    the thing the kernel is compared against, so it matches the kernel.
    """
    index = np.arange(resized_extent, dtype=np.float32) + np.float32(0.5)
    return (index * np.float32(source_extent)) / np.float32(resized_extent) - np.float32(0.5)


def crop_centres(low: float, high: float, target_extent: int) -> np.ndarray:
    """Half-pixel centres inside a continuous ``[low, high]`` sub-region of the source.

    The same convention as :func:`resize_centres`, with the sub-region's origin added before
    the half-pixel shift — which is what makes a crop of the whole frame identical to a resize
    of it.
    """
    index = np.arange(target_extent, dtype=np.float32) + np.float32(0.5)
    span = np.float32(high) - np.float32(low)
    return np.float32(low) + (index * span) / np.float32(target_extent) - np.float32(0.5)


def clamp_boxes_to_frame(boxes: np.ndarray, height: int, width: int) -> np.ndarray:
    """``(n, 4)`` xyxy clamped to the last addressable pixel, ``[0, extent - 1]``.

    Sampling coordinates, not box edges, so the bound is ``extent - 1`` and not ``extent`` —
    the opposite of :meth:`LetterboxGeometry.invert_boxes`, which clips continuous edges. The
    CUDA crop kernel clamps to ``extent - 1`` for the same reason, and the two must agree or a
    crop that touches the frame border comes out one row short in one backend.
    """
    clamped = np.empty_like(boxes)
    np.clip(boxes[:, 0::2], 0.0, float(width - 1), out=clamped[:, 0::2])
    np.clip(boxes[:, 1::2], 0.0, float(height - 1), out=clamped[:, 1::2])
    return clamped


# ------------------------------------------------------------------------- validation


def validate_target_hw(target_hw: Sequence[int]) -> tuple[int, int]:
    """Two positive ints, in ``(height, width)`` order."""
    values = tuple(int(v) for v in target_hw)
    if len(values) != 2:
        raise ConfigurationError(f"target_hw must be (height, width), got {target_hw!r}")
    if values[0] <= 0 or values[1] <= 0:
        raise ConfigurationError(f"target_hw must be positive, got {values}")
    return values[0], values[1]
