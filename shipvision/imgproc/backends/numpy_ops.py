"""The oracle: letterbox, crop and NMS in numpy, written to be read.

Every other backend in this family is checked against this file, so it optimises for one
thing — being obviously the same arithmetic as ``csrc/shipvision/imgproc/image_ops.cu``, expression
by expression, in the same float32. Where the kernel writes ``(y + 0.5f) * h / out_h -
0.5f``, so does this; where it clamps a tap index rather than a coordinate, so does this. A
tidier formulation that agreed to five decimal places instead of eight would make the parity
test's tolerance the thing under test rather than the kernel.

It is not slow for what it is: the sampling is four vectorised gathers per image rather than
a Python loop over pixels. It is not the production path either — at 1000 fps the frame
should never reach host memory at all (that is what the native backend is for) — but it is
fast enough to preprocess a validation set, which is the other job this backend has.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from shipvision.errors import ConfigurationError
from shipvision.imgproc.base import (
    DEFAULT_PAD_VALUE,
    ImageOps,
    as_image_batch,
    resolve_normalisation,
    validate_boxes,
    validate_image,
    validate_pad_value,
)
from shipvision.imgproc.colour import nv12_to_rgb
from shipvision.imgproc.geometry import (
    LetterboxGeometry,
    clamp_boxes_to_frame,
    crop_centres,
    resize_centres,
    validate_target_hw,
)
from shipvision.imgproc.nms import CLASSIC, suppress
from shipvision.imgproc.registry import IMGPROC
from shipvision.registry import PYTHON

__all__ = ["NumpyImageOps"]


@IMGPROC.register("default", backend=PYTHON, aliases=("letterbox",))
class NumpyImageOps(ImageOps):
    """Pure numpy. Always available, and the reference the compiled backends must match."""

    def letterbox(
        self,
        images: Sequence[np.ndarray] | np.ndarray,
        target_hw: tuple[int, int],
        *,
        pad_value: int = DEFAULT_PAD_VALUE,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
    ) -> tuple[np.ndarray, list[LetterboxGeometry]]:
        """See :meth:`ImageOps.letterbox`."""
        frames = as_image_batch(images)
        target_h, target_w = validate_target_hw(target_hw)
        mean_array, std_array = resolve_normalisation(mean, std)
        pad_value = validate_pad_value(pad_value)

        out = np.empty((len(frames), 3, target_h, target_w), dtype=np.float32)
        # The bars are written before the resized image is pasted over the middle, so the
        # fill is one vectorised store per channel instead of four rectangles per image.
        bars = (np.float32(pad_value) - mean_array) / std_array
        out[:] = bars[None, :, None, None]

        geometries: list[LetterboxGeometry] = []
        for index, frame in enumerate(frames):
            geometry = LetterboxGeometry.plan(frame.shape[:2], (target_h, target_w))
            geometries.append(geometry)
            resized = _resize_bilinear(frame, geometry.resized_height, geometry.resized_width)
            out[
                index,
                :,
                geometry.pad_top : geometry.pad_top + geometry.resized_height,
                geometry.pad_left : geometry.pad_left + geometry.resized_width,
            ] = _to_nchw_rgb(resized, mean_array, std_array)
        return out, geometries

    @property
    def supports_nv12(self) -> bool:
        """Always: this is the NV12 oracle the kernel is checked against."""
        return True

    def nv12_letterbox(
        self,
        frames: Sequence[np.ndarray],
        widths: Sequence[int],
        target_hw: tuple[int, int],
        *,
        pad_value: int = DEFAULT_PAD_VALUE,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        swap_rb: bool = True,
    ) -> tuple[np.ndarray, list[LetterboxGeometry]]:
        """See :meth:`ImageOps.nv12_letterbox`.

        Structured as convert-the-whole-frame then gather, where the kernel converts only the
        four taps it needs. Convention 7 says the conversion happens before the blend, and
        conversion is per-pixel, so the two are identical arithmetic in a different order —
        which is the property that makes this an oracle rather than a second opinion.
        """
        if len(frames) != len(widths):
            raise ConfigurationError(
                f"got {len(frames)} NV12 frames and {len(widths)} widths; a decoder's stride "
                f"is not its visible width, so one width per frame is required"
            )
        if not frames:
            raise ConfigurationError(
                "nv12 letterbox needs at least one frame; an empty batch is a caller bug"
            )
        target_h, target_w = validate_target_hw(target_hw)
        mean_array, std_array = resolve_normalisation(mean, std)
        pad_value = validate_pad_value(pad_value)

        out = np.empty((len(frames), 3, target_h, target_w), dtype=np.float32)
        bars = (np.float32(pad_value) - mean_array) / std_array
        out[:] = bars[None, :, None, None]

        geometries: list[LetterboxGeometry] = []
        for index, (frame, width) in enumerate(zip(frames, widths, strict=True)):
            rgb = nv12_to_rgb(frame, width)
            geometry = LetterboxGeometry.plan(rgb.shape[:2], (target_h, target_w))
            geometries.append(geometry)
            resized = _gather_bilinear(
                rgb,
                resize_centres(rgb.shape[0], geometry.resized_height),
                resize_centres(rgb.shape[1], geometry.resized_width),
            )
            out[
                index,
                :,
                geometry.pad_top : geometry.pad_top + geometry.resized_height,
                geometry.pad_left : geometry.pad_left + geometry.resized_width,
            ] = _normalise_planar(resized, mean_array, std_array, swap_rb=swap_rb)
        return out, geometries

    def crop_batch(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        target_hw: tuple[int, int],
        *,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
    ) -> np.ndarray:
        """See :meth:`ImageOps.crop_batch`."""
        frame = validate_image(image)
        box_array = validate_boxes(boxes)
        target_h, target_w = validate_target_hw(target_hw)
        mean_array, std_array = resolve_normalisation(mean, std)

        height, width = frame.shape[:2]
        clamped = clamp_boxes_to_frame(box_array, height, width)
        # Source-scale values first, normalised in one pass at the end, so a degenerate box
        # takes the same code path as a real one and comes out as (0 - mean) / std rather
        # than as a hole in the tensor.
        raw = np.zeros((box_array.shape[0], target_h, target_w, 3), dtype=np.float32)

        for index, box in enumerate(clamped):
            x1, y1, x2, y2 = (float(v) for v in box)
            if x2 <= x1 or y2 <= y1:
                continue
            raw[index] = _gather_bilinear(
                frame, crop_centres(y1, y2, target_h), crop_centres(x1, x2, target_w)
            )

        return _to_nchw_rgb(raw, mean_array, std_array)

    def nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        *,
        iou_threshold: float,
        method: str = CLASSIC,
        sigma: float = 0.5,
        score_threshold: float = 0.0,
        min_neighbors: int = 0,
        min_score_sum: float = 0.0,
    ) -> np.ndarray:
        """See :meth:`ImageOps.nms`."""
        return suppress(
            boxes,
            scores,
            iou_threshold=iou_threshold,
            method=method,
            sigma=sigma,
            score_threshold=score_threshold,
            min_neighbors=min_neighbors,
            min_score_sum=min_score_sum,
        )[0]


# ---------------------------------------------------------------------------- sampling


def _resize_bilinear(frame: np.ndarray, resized_h: int, resized_w: int) -> np.ndarray:
    """``(h, w, 3)`` uint8 -> ``(resized_h, resized_w, 3)`` float32, source channel order."""
    height, width = frame.shape[:2]
    return _gather_bilinear(
        frame, resize_centres(height, resized_h), resize_centres(width, resized_w)
    )


def _gather_bilinear(source: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Bilinear gather at the outer product of ``ys`` and ``xs``. Border-clamped.

    The clamping is deliberately the kernel's: the *tap indices* are clamped while the
    interpolation weight keeps using the unclamped ``floor``. For a coordinate of -0.3 both taps
    land on pixel 0, so the weight cancels and the result is pixel 0 — identical to torch's
    "clamp the coordinate to zero" over the only coordinate range this code produces, which is
    ``[-0.5, extent - 1]``.

    ``source`` is read as uint8 and each of the four gathered taps is cast, rather than casting
    the frame once up front. The taps are the size of the *output* and the frame is the size of
    the input, so on the crop path — a 6 MB frame in, a few kilobytes out, fifteen thousand
    times a second — that is the difference between one allocation the size of the answer and
    four the size of the question.
    """
    height, width = source.shape[:2]
    y_floor = np.floor(ys)
    x_floor = np.floor(xs)
    weight_y = (ys - y_floor).astype(np.float32)[:, None, None]
    weight_x = (xs - x_floor).astype(np.float32)[None, :, None]

    y_low = np.clip(y_floor.astype(np.int64), 0, None)
    x_low = np.clip(x_floor.astype(np.int64), 0, None)
    y_high = np.minimum(y_floor.astype(np.int64) + 1, height - 1)
    x_high = np.minimum(x_floor.astype(np.int64) + 1, width - 1)

    rows_low, rows_high = y_low[:, None], y_high[:, None]
    cols_low, cols_high = x_low[None, :], x_high[None, :]
    top_left = source[rows_low, cols_low].astype(np.float32)
    top_right = source[rows_low, cols_high].astype(np.float32)
    bottom_left = source[rows_high, cols_low].astype(np.float32)
    bottom_right = source[rows_high, cols_high].astype(np.float32)

    top = top_left * (np.float32(1.0) - weight_x) + top_right * weight_x
    bottom = bottom_left * (np.float32(1.0) - weight_x) + bottom_right * weight_x
    return (top * (np.float32(1.0) - weight_y) + bottom * weight_y).astype(np.float32)


def _normalise_planar(
    values: np.ndarray, mean: np.ndarray, std: np.ndarray, *, swap_rb: bool
) -> np.ndarray:
    """``(..., h, w, 3)`` **RGB** source values -> ``(..., 3, h, w)`` normalised.

    The counterpart of :func:`_to_nchw_rgb` for a path whose source is already RGB: the NV12
    decode produces RGB, so ``swap_rb=True`` is the identity here and ``False`` is what asks
    for BGR. ``mean`` and ``std`` are indexed by *destination* plane either way, which is the
    one thing both paths must agree on — the kernel indexes them the same way, so a BGR
    request with an RGB mean is wrong identically in both.
    """
    ordered = values if swap_rb else values[..., ::-1]
    planar = np.moveaxis(ordered, -1, -3)
    return ((planar - mean[:, None, None]) / std[:, None, None]).astype(np.float32)


def _to_nchw_rgb(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """``(..., h, w, 3)`` BGR source values -> ``(..., 3, h, w)`` normalised RGB.

    Convention 4: the channel swap happens first, so ``mean`` and ``std`` are indexed in
    destination (RGB) order — the order a checkpoint's published statistics are written in.
    """
    swapped = values[..., ::-1]
    planar = np.moveaxis(swapped, -1, -3)
    return ((planar - mean[:, None, None]) / std[:, None, None]).astype(np.float32)
