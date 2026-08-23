"""The image-ops contract: what a backend must implement, and what it may be handed.

Read :mod:`shipvision.imgproc.geometry` first. It states the three conventions that decide the
*numbers* — half-pixel sampling centres, how the resized extent rounds, which side gets the odd
pad pixel — and this module only restates the first of them in one line, because a half-pixel
disagreement between two backends is invisible in a smoke test and shifts every box in
production::

    src = (i + 0.5) * source_extent / resized_extent - 0.5

CONVENTION 4 — COLOUR AND NORMALISATION, which this module owns
    Input is HWC uint8 **BGR**, because that is what every decoder in the fleet emits. Output
    is NCHW float32 **RGB**, always: the swap is not optional, since a model trained on RGB
    and fed BGR loses a few points of mAP and never errors. ``mean`` and ``std`` are in the
    *source* 0-255 scale and in *destination* (RGB) channel order — the order a checkpoint's
    published statistics are written in — and are applied as ``(value - mean) / std``. The
    defaults, ``mean=0`` and ``std=255``, give ``[0, 1]``. Letterbox bars are filled with
    ``pad_value`` *before* normalisation, so a bar comes out as ``(pad_value - mean) / std``
    per channel.

Input is validated rather than coerced. uint8 is required and not cast: the pybind layer would
happily ``forcecast`` a float64 image, truncating towards zero, and the numpy backend would
not — so a caller who passed the wrong dtype would get two different answers from two backends
that are supposed to be interchangeable. Refusing is the only way they stay comparable.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence

import numpy as np

from shipvision.errors import ConfigurationError, DimensionMismatchError
from shipvision.imgproc.geometry import LetterboxGeometry
from shipvision.imgproc.nms import CLASSIC, METHODS, SOFT_METHODS, suppress
from shipvision.imgproc.validation import reject_non_finite

__all__ = [
    "DEFAULT_MEAN",
    "DEFAULT_PAD_VALUE",
    "DEFAULT_STD",
    "ImageOps",
    "as_image_batch",
    "resolve_normalisation",
    "validate_boxes",
    "validate_image",
]

DEFAULT_MEAN: tuple[float, float, float] = (0.0, 0.0, 0.0)
DEFAULT_STD: tuple[float, float, float] = (255.0, 255.0, 255.0)
"""Scale-only normalisation to ``[0, 1]``, matching ``NormalizeParams``' defaults in C++."""

DEFAULT_PAD_VALUE = 114
"""The YOLO letterbox grey. Kept as the default so a detector config that omits it matches the
preprocessing its weights were trained with."""


# ------------------------------------------------------------------------- validation


def validate_image(image: np.ndarray, *, what: str = "image") -> np.ndarray:
    """An ``(h, w, 3)`` uint8 C-contiguous view of ``image``. See the module docstring.

    Both spatial extents must be positive, and that check belongs *here* rather than in
    :meth:`~shipvision.imgproc.geometry.LetterboxGeometry.plan` alone: the native backend
    launches its kernel before it builds the geometries, so a zero-row frame used to reach the
    sampler, which clamps its high tap to ``min(y0 + 1, h - 1) = -1`` and indexes *before* the
    allocation. That raises ``cudaErrorIllegalAddress``, which is **sticky** — it poisons the
    context for the life of the process, so one reconnecting camera killed the worker
    permanently. At a batch index above zero the same read lands inside the staging ring
    instead and returns the previous camera's pixels with no error at all, which is worse.
    """
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise DimensionMismatchError(f"{what} must be (h, w, 3) HWC BGR, got {array.shape}")
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise DimensionMismatchError(
            f"{what} is {array.shape[0]}x{array.shape[1]}; both extents must be > 0. An empty "
            f"frame is a decoder failure, not a frame with no objects in it"
        )
    if array.dtype != np.uint8:
        raise ConfigurationError(
            f"{what} must be uint8 in 0-255, got {array.dtype}. Scale and cast at the decoder "
            f"boundary, not here — the native backend would silently truncate"
        )
    return np.ascontiguousarray(array)


def validate_boxes(boxes: np.ndarray) -> np.ndarray:
    """An ``(n, 4)`` float32 C-contiguous xyxy array. ``(0, 4)`` for an empty frame.

    Non-finite coordinates are refused rather than clamped — see
    :func:`~shipvision.imgproc.validation.reject_non_finite` for why a NaN cannot be given a
    sensible value here: the three backends already gave it three different ones.
    """
    array = np.asarray(boxes, dtype=np.float32)
    if array.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 4:
        raise DimensionMismatchError(f"boxes must be (n, 4) xyxy, got {array.shape}")
    reject_non_finite(array, "boxes")
    return np.ascontiguousarray(array)


def as_image_batch(images: Sequence[np.ndarray] | np.ndarray) -> list[np.ndarray]:
    """One image or a ragged sequence of them, validated, as a list.

    Shared by all three backends so that they agree on the two edge cases. A single
    ``(h, w, 3)`` array is a batch of one, because a caller with one frame should not have to
    wrap it. An **empty** batch raises: a frame with no objects is normal input and gets an
    empty crop tensor, but a batch with no frames is a scheduler bug, and the native entry
    point already refuses it — so all three refuse it, identically.
    """
    if isinstance(images, np.ndarray) and images.ndim == 3:
        return [validate_image(images)]
    frames = [validate_image(image, what=f"images[{i}]") for i, image in enumerate(images)]
    if not frames:
        raise ConfigurationError(
            "letterbox needs at least one image; an empty batch is a caller bug, not an empty "
            "frame"
        )
    return frames


def resolve_normalisation(
    mean: Sequence[float] | None, std: Sequence[float] | None
) -> tuple[np.ndarray, np.ndarray]:
    """``(3,)`` float32 mean and std in the 0-255 source scale, RGB order. Convention 4."""
    mean_array = np.asarray(DEFAULT_MEAN if mean is None else mean, dtype=np.float32)
    std_array = np.asarray(DEFAULT_STD if std is None else std, dtype=np.float32)
    if mean_array.shape != (3,) or std_array.shape != (3,):
        raise ConfigurationError(
            f"mean and std must each have three entries, got {mean_array.shape} and "
            f"{std_array.shape}"
        )
    if np.any(std_array == 0.0):
        raise ConfigurationError(f"normalisation std must be non-zero, got {std_array}")
    return mean_array, std_array


# ------------------------------------------------------------------------------- base


class ImageOps(abc.ABC):
    """Fused pre-processing and NMS. Registered in ``IMGPROC`` under ``(name, backend)``.

    Three operations, chosen because each one is several memory-bound passes that a caller
    would otherwise run separately over a 1080p frame. Everything else a pipeline needs from a
    GPU — allocation, streams, gemm, a plain resize — belongs to torch or to TensorRT.

    Implementations are stateful (a device index, persistent scratch) but their methods are
    pure functions of their arguments, and one instance is expected to serve one worker thread
    for the life of the process. Sharing an instance across threads is not supported: the
    native backend's staging ring is per-instance by design.
    """

    name: str
    backend: str

    # -- pre-processing ---------------------------------------------------------------

    @abc.abstractmethod
    def letterbox(
        self,
        images: Sequence[np.ndarray] | np.ndarray,
        target_hw: tuple[int, int],
        *,
        pad_value: int = DEFAULT_PAD_VALUE,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
    ) -> tuple[np.ndarray, list[LetterboxGeometry]]:
        """Resize-with-aspect-preserved, pad, BGR->RGB, normalise, HWC->NCHW, in one pass.

        Args:
            images: one ``(h, w, 3)`` uint8 BGR image, or a sequence of them. The sequence may
                be ragged — fifty cameras do not agree on resolution, which is the whole reason
                the native kernel takes a descriptor table.
            target_hw: the network input extent, ``(height, width)``.
            pad_value: fill for the letterbox bars, in the 0-255 source scale.
            mean: per-channel mean in the 0-255 source scale, RGB order. ``None`` -> zeros.
            std: per-channel divisor, same scale and order. ``None`` -> 255.

        Returns:
            ``(n, 3, target_h, target_w)`` float32, and one
            :class:`~shipvision.imgproc.geometry.LetterboxGeometry` per image, in input order.
            The geometry is *returned* rather than left to be recomputed: boxes must be
            un-mapped with the same numbers that mapped them.

        Raises:
            ConfigurationError: empty batch, non-uint8 input, or a bad target/normalisation.
            DimensionMismatchError: an image that is not ``(h, w, 3)``.
        """

    @abc.abstractmethod
    def crop_batch(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        target_hw: tuple[int, int],
        *,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
    ) -> np.ndarray:
        """Crop each xyxy box out of one frame and resize it to ``target_hw``.

        The embedding stage's hot path: the frame is megabytes and the crops are kilobytes, so
        this is where a pipeline stops moving whole frames around.

        Boxes are **clamped** to the frame, not rejected: a detector on the edge of the image
        emits coordinates a pixel or two outside it, and dropping those detections loses
        exactly the objects that are entering or leaving the scene. A box with no area after
        clamping yields a crop of source value zero — normalised like any other pixel, so the
        tensor stays finite and the batch survives one bad box.

        Args:
            image: one ``(h, w, 3)`` uint8 BGR frame.
            boxes: ``(n, 4)`` xyxy in that frame's pixels. An empty array is normal input, not
                an error — most frames have no objects on most cameras.
            target_hw: the crop extent, ``(height, width)``.
            mean: per-channel mean, 0-255 scale, RGB order. ``None`` -> zeros.
            std: per-channel divisor. ``None`` -> 255.

        Returns:
            ``(n, 3, target_h, target_w)`` float32, in box order.
        """

    # -- post-processing --------------------------------------------------------------

    @abc.abstractmethod
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
        """Class-agnostic suppression. Returns surviving indices, descending score first.

        The rules — strict ``iou >``, inclusive admission, the departure condition, the tie
        order — are stated in :mod:`shipvision.imgproc.nms` and implemented once there.

        ``method`` selects how an overlapping box is punished:

        ``"classic"``
            removed outright. Standard greedy NMS, and the only method with a device kernel.
        ``"linear"``, ``"gauss"``
            soft-NMS: the score is multiplied by ``1 - iou`` or by ``exp(-iou^2 / sigma)`` and
            the box stays in the pool. **Soft methods suppress by lowering scores, not by
            removing boxes**, so with the default ``score_threshold=0.0`` every index comes
            back, merely re-ranked. For these two methods — and only these two — the indices
            alone are not the answer: use :meth:`nms_with_scores` and threshold what it
            returns. For every other method :meth:`nms_with_scores` is this method plus a
            gather, so neither call loses the accelerated path.
        ``"neighborhood"``
            greedy, but a survivor must have at least ``min_neighbors`` suppressed neighbours
            whose scores sum to at least ``min_score_sum``. With the defaults ``(0, 0.0)`` —
            the values the C++ reference hard-coded — it is exactly ``"classic"``.
        ``"none"``
            no suppression, only the score threshold. There so a benchmark can measure what
            NMS is worth instead of assuming.

        Args:
            boxes: ``(n, 4)`` xyxy float32.
            scores: ``(n,)`` float32.
            iou_threshold: overlap above which a box is punished.
            method: one of :data:`~shipvision.imgproc.nms.METHODS`.
            sigma: the gaussian's width. Read by ``"gauss"`` only.
            score_threshold: a box below this never enters the pool, and leaves it if decay
                takes it under. Inclusive: ``score >= score_threshold`` stays.
            min_neighbors: ``"neighborhood"`` only.
            min_score_sum: ``"neighborhood"`` only.

        Returns:
            ``(k,)`` int64 indices into the input, ordered by descending score — decayed
            score, for a soft method.
        """

    def nms_with_scores(
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
    ) -> tuple[np.ndarray, np.ndarray]:
        """:meth:`nms`, but it also returns the score each survivor kept.

        Soft-NMS's whole output is a re-weighted score; returning only indices throws away the
        half of the answer the caller needs. Rather than change :meth:`nms`'s signature — the
        detector lane codes against it — the score-carrying variant is its own method.

        **It is a wrapper around :meth:`nms`, not a second implementation.** Only the soft
        methods take the shared sequential loop, and only because their answer *is* the decayed
        score: the decay is order-dependent, so there is no bitmask formulation and nothing for
        a GPU to do over the few dozen boxes that clear a score threshold. For ``"classic"``,
        ``"neighborhood"`` and ``"none"`` the kept scores are the scores that came in, so this
        gathers them at the indices :meth:`nms` returned and the accelerated path is preserved.

        Routing every method through the shared loop — which is what this used to do — cost a
        measured 150x on the native backend at 25 000 proposals (77 ms against 11.6 s at
        ``iou_threshold=0.5``) while returning identical indices. Identical answers by a
        different route is the failure a test has to be written for on purpose, because nothing
        about the output looks wrong.

        Returns:
            ``(indices, scores)``, both ``(k,)``, aligned and in descending score order.
            ``scores`` holds decayed values for ``"linear"``/``"gauss"`` and the original
            values for every other method.
        """
        if method in SOFT_METHODS:
            return suppress(
                boxes,
                scores,
                iou_threshold=iou_threshold,
                method=method,
                sigma=sigma,
                score_threshold=score_threshold,
                min_neighbors=min_neighbors,
                min_score_sum=min_score_sum,
            )

        indices = self.nms(
            boxes,
            scores,
            iou_threshold=iou_threshold,
            method=method,
            sigma=sigma,
            score_threshold=score_threshold,
            min_neighbors=min_neighbors,
            min_score_sum=min_score_sum,
        )
        # Re-read the scores through numpy rather than trusting the caller's container: `nms`
        # accepts anything array-like, and a list would not take an index array.
        score_array = np.asarray(scores, dtype=np.float32).reshape(-1)
        return indices, score_array[indices]

    # -- introspection ----------------------------------------------------------------

    @property
    def methods(self) -> tuple[str, ...]:
        """The NMS methods this backend accepts. The same set for all of them."""
        return METHODS

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} backend={self.backend!r}>"
