"""Mask reconstruction: 32 prototype planes and 32 coefficients become one instance mask.

The arithmetic is a gemm and two resizes, and the only thing that is easy to get wrong is the
*order* of the two resizes — so that order is stated here once and tested directly.

THE MASK PIPELINE, IN ORDER
    1. ``coefficients @ prototypes`` — one gemm per frame, giving a ``(ph, pw)`` logit plane
       per detection. ``ph, pw`` is the proto resolution, a quarter of the network input.
    2. **Sigmoid, in proto space.** Before the upsample, not after
       (``Yolo26SegPostProcessor.cpp:36-42``). A sigmoid is monotone but not linear, so
       ``resize(sigmoid(x))`` and ``sigmoid(resize(x))`` differ — most visibly at a mask's
       edge, which is the only part of a mask anybody looks at.
    3. Upsample to the **network input** extent, letterbox bars included.
    4. **Crop the bars away** (``:47-52``), leaving the region the source image actually
       occupies.
    5. Resize *that* to the source extent (``:55``).

STEPS 4 AND 5 DO NOT COMMUTE, AND GETTING THEM BACKWARDS IS INVISIBLE
    Crop-then-resize maps the un-padded region onto the whole source image. Resize-then-crop
    stretches the padded canvas — bars and all — up to the source extent and then cuts a
    rectangle out of the middle of it, so every mask is scaled by ``target / resized`` on the
    letterboxed axis and shifted by the pad. For a 1080x1920 frame in a 640x640 input that is
    a 1.78x vertical stretch on every mask in the fleet. Both orders produce a plausible mask
    of exactly the right shape, and no smoke test tells them apart.

WHY THE RESIZE IS HERE AND NOT BORROWED
    :mod:`shipvision.imgproc` deliberately refuses non-uint8 input — it owns the *decoder*
    boundary, where coercing a dtype would make two backends disagree — and its ``crop_batch``
    both normalises and swaps colour channels. Neither is what a float mask plane needs.
    OpenCV's ``resize`` would do, but ``opencv-python`` is an optional extra and the offline
    test tier must not need it. So the sampling here is a dozen lines of numpy that calls
    :func:`shipvision.imgproc.geometry.resize_centres` for the coordinates: the half-pixel
    rule is imported rather than restated, which is the part that would otherwise drift.
"""

from __future__ import annotations

import numpy as np

from shipvision.errors import DimensionMismatchError
from shipvision.imgproc.geometry import LetterboxGeometry, resize_centres

__all__ = [
    "EDGE_TOLERANCE",
    "bilinear_resize",
    "box_crop_bounds",
    "fuse_mask_logits",
    "unpad_mask",
]


def fuse_mask_logits(coefficients: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    """``(k, c)`` coefficients and ``(c, ph, pw)`` prototypes to ``(k, ph, pw)`` logits.

    One matrix product per frame rather than one per detection: ``k`` is at most a few dozen
    and ``c`` is 32, so this is a tiny gemm, and BLAS does it in less time than the Python
    loop over detections that the reference uses (``Yolo26SegPostProcessor.cpp:137-143``)
    spends on its first iteration.
    """
    coeffs = np.asarray(coefficients, dtype=np.float32)
    protos = np.asarray(prototypes, dtype=np.float32)
    if coeffs.ndim != 2 or protos.ndim != 3:
        raise DimensionMismatchError(
            f"expected (k, c) coefficients and (c, ph, pw) prototypes, got {coeffs.shape} "
            f"and {protos.shape}"
        )
    if coeffs.shape[1] != protos.shape[0]:
        raise DimensionMismatchError(
            f"{coeffs.shape[1]} mask coefficients against {protos.shape[0]} prototype "
            f"planes. The coefficient count is the detection output's width minus six and "
            f"the plane count comes from the proto tensor; the two disagreeing means the "
            f"engine's two outputs are not from the same export"
        )
    channels, height, width = protos.shape
    return (coeffs @ protos.reshape(channels, height * width)).reshape(-1, height, width)


def bilinear_resize(
    plane: np.ndarray,
    out_height: int,
    out_width: int,
    *,
    window: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """``(h, w)`` float32 to ``(out_height, out_width)``, half-pixel centres, no antialiasing.

    Convention 1 of :mod:`shipvision.imgproc.geometry`, sampled with that module's own
    :func:`~shipvision.imgproc.geometry.resize_centres` so the coordinates are not restated
    here. Taps are clamped by *index* while the weight keeps the unclamped position, which is
    what the CUDA kernel and ``grid_sample(padding_mode="border")`` both do.

    An extent that already matches is returned as a copy rather than resampled: resampling at
    scale 1.0 is a no-op in exact arithmetic and is not in float32, and a mask that changes
    when nothing was asked of it makes every downstream tolerance guesswork.

    Args:
        plane: the source, ``(h, w)``.
        out_height: the full output height. Note *full*, even when ``window`` is given.
        out_width: the full output width.
        window: ``(y0, y1, x0, x1)`` half-open in **output** coordinates. Only that rectangle
            is computed, and the result is bit-identical to slicing the full output — the
            sampling centres for an axis depend on the two extents alone, so taking a
            sub-range of them is the same as taking a sub-range of the answer. That identity
            is what makes it safe to use, and it is what stops an instance mask from
            allocating a full 1080x1920 float plane per detection to then keep a 40x90 crop
            of it.
    """
    array = np.asarray(plane, dtype=np.float32)
    if array.ndim != 2:
        raise DimensionMismatchError(f"bilinear_resize takes one (h, w) plane, got {array.shape}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise DimensionMismatchError(f"cannot resize an empty plane {array.shape}")
    if out_height <= 0 or out_width <= 0:
        raise DimensionMismatchError(
            f"resize target must be positive, got {out_height}x{out_width}"
        )
    y0, y1, x0, x1 = _validated_window(window, out_height, out_width)
    if array.shape == (out_height, out_width):
        return array[y0:y1, x0:x1].copy()

    row_low, row_high, row_weight = (t[y0:y1] for t in _axis_taps(array.shape[0], out_height))
    col_low, col_high, col_weight = (t[x0:x1] for t in _axis_taps(array.shape[1], out_width))

    left = array[np.ix_(row_low, col_low)]
    right = array[np.ix_(row_low, col_high)]
    top = left + (right - left) * col_weight[None, :]

    left = array[np.ix_(row_high, col_low)]
    right = array[np.ix_(row_high, col_high)]
    bottom = left + (right - left) * col_weight[None, :]

    return (top + (bottom - top) * row_weight[:, None]).astype(np.float32)


def _validated_window(
    window: tuple[int, int, int, int] | None, out_height: int, out_width: int
) -> tuple[int, int, int, int]:
    if window is None:
        return 0, out_height, 0, out_width
    y0, y1, x0, x1 = (int(v) for v in window)
    if not (0 <= y0 < y1 <= out_height and 0 <= x0 < x1 <= out_width):
        raise DimensionMismatchError(
            f"window {(y0, y1, x0, x1)} is not a non-empty half-open rectangle inside a "
            f"{out_height}x{out_width} output"
        )
    return y0, y1, x0, x1


def _axis_taps(source_extent: int, out_extent: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(low_index, high_index, weight)`` for one axis of a bilinear resize."""
    centres = resize_centres(source_extent, out_extent)
    low = np.floor(centres).astype(np.int64)
    weight = (centres - low).astype(np.float32)
    high = np.minimum(low + 1, source_extent - 1)
    return np.clip(low, 0, source_extent - 1), high, weight


def unpad_mask(
    logits: np.ndarray,
    geometry: LetterboxGeometry,
    *,
    window: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """One ``(ph, pw)`` logit plane to a ``(source_h, source_w)`` probability map.

    Sigmoid, upsample to the network input, **crop the letterbox bars, then** resize to the
    source. See the module docstring for why that order is not interchangeable.

    Args:
        logits: the fused mask plane, in proto space. Logits, not probabilities — the sigmoid
            happens here so that it happens before the upsample.
        geometry: the letterbox that produced the network input this mask came out of. The
            pads and the resized extent are read from it, never recomputed:
            ``Yolo26SegPostProcessor.cpp:30-33`` re-derives them with a ``- 0.1`` fudge to
            bias its rounding, which is a second rounding rule for the same quantity.
        window: ``(y0, y1, x0, x1)`` in source-image coordinates. Only that rectangle of the
            final map is computed — see :func:`bilinear_resize`. A caller that only wants the
            mask inside a detection's box passes the box's bounds and never materialises the
            full frame.

    Returns:
        float32 in ``[0, 1]``, aligned with the original image: ``(source_h, source_w)``, or
        the ``window`` rectangle of it.
    """
    probability = _sigmoid(logits)
    canvas = bilinear_resize(probability, geometry.target_height, geometry.target_width)

    top = max(0, min(geometry.pad_top, geometry.target_height - 1))
    left = max(0, min(geometry.pad_left, geometry.target_width - 1))
    inner = canvas[
        top : top + max(1, min(geometry.resized_height, geometry.target_height - top)),
        left : left + max(1, min(geometry.resized_width, geometry.target_width - left)),
    ]
    return bilinear_resize(
        inner, geometry.source_height, geometry.source_width, window=window
    )


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    """``1 / (1 + exp(-x))``, written so a large negative logit cannot overflow.

    ``exp(-x)`` for ``x = -800`` is an overflow warning and an inf on the way to the right
    answer of 0.0, and background logits are routinely that large. The branchless form below
    evaluates the numerically stable half for each sign.
    """
    array = np.asarray(logits, dtype=np.float32)
    positive = array >= 0.0
    out = np.empty_like(array)
    exp_negative = np.exp(-np.abs(array))
    out[positive] = 1.0 / (1.0 + exp_negative[positive])
    out[~positive] = exp_negative[~positive] / (1.0 + exp_negative[~positive])
    return out


#: Sub-pixel slack applied before the outward rounding in :func:`box_crop_bounds`. A thousandth
#: of a pixel: large enough to absorb the float32 residual of a letterbox round trip, small
#: enough that no real box edge is within it.
EDGE_TOLERANCE = 1e-3


def box_crop_bounds(box: np.ndarray, height: int, width: int) -> tuple[int, int, int, int]:
    """Integer ``(y0, y1, x0, x1)`` covering a float xyxy box, clipped to the frame.

    Outward: ``floor`` on the low edge and ``ceil`` on the high one, so the window contains
    every pixel the box touches rather than losing the partially-covered border row.

    Both edges are nudged inward by :data:`EDGE_TOLERANCE` first, and that is not cosmetic. A
    box that was 400.0 in source pixels comes back from the letterbox inverse as 400.00003 —
    float32 division by a float32 scale — and a bare ``ceil`` turns that into 401, so the mask
    of an integer-aligned box would be one pixel wider than the box for no reason anybody could
    find. A thousandth of a pixel of slack removes the whole class of surprise.

    Guaranteed to be at least 1x1 even for a degenerate box, because a ``(0, 0)`` mask is a
    shape every downstream consumer has to special-case and a one-pixel mask is not.
    """
    tolerance = np.float32(EDGE_TOLERANCE)
    x0 = int(np.clip(np.floor(box[0] + tolerance), 0, max(width - 1, 0)))
    y0 = int(np.clip(np.floor(box[1] + tolerance), 0, max(height - 1, 0)))
    x1 = int(np.clip(np.ceil(box[2] - tolerance), x0 + 1, width))
    y1 = int(np.clip(np.ceil(box[3] - tolerance), y0 + 1, height))
    return y0, y1, x0, x1
