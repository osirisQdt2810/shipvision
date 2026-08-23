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
from dataclasses import dataclass

import numpy as np

from shipvision.errors import (
    BackendUnavailableError,
    ConfigurationError,
    DimensionMismatchError,
)
from shipvision.imgproc.geometry import LetterboxGeometry
from shipvision.imgproc.nms import CLASSIC, METHODS, SOFT_METHODS, suppress
from shipvision.imgproc.validation import reject_non_finite

__all__ = [
    "DEFAULT_MEAN",
    "DEFAULT_PAD_VALUE",
    "DEFAULT_STD",
    "DeviceBuffer",
    "ImageOps",
    "as_image_batch",
    "nchw_nbytes",
    "resolve_normalisation",
    "validate_boxes",
    "validate_image",
    "validate_pad_value",
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


def validate_pad_value(pad_value: int) -> int:
    """A whole letterbox fill level in ``[0, 255]``. Shared by all three backends.

    Unvalidated, this was the one argument on which the backends openly disagreed: C++ does
    ``static_cast<unsigned char>``, so ``256`` became 0 and ``-1`` became 255 — white bars
    instead of grey — while numpy and torch used the number as given. Nothing errored, the
    parity suite never passes an out-of-range value, and the difference is a band around every
    frame on the GPU path only.

    Raises:
        ConfigurationError: not an integer, or outside the 0-255 source scale.
    """
    if isinstance(pad_value, bool) or int(pad_value) != pad_value:
        raise ConfigurationError(
            f"pad_value must be a whole 0-255 source-scale level, got {pad_value!r}. The "
            f"native backend truncates towards zero and the numpy one does not"
        )
    value = int(pad_value)
    if not 0 <= value <= 255:
        raise ConfigurationError(
            f"pad_value must be in [0, 255], got {value}. The native backend casts it to "
            f"unsigned char, so 256 would fill with 0 and -1 with 255 — white bars, silently"
        )
    return value


def resolve_normalisation(
    mean: Sequence[float] | None, std: Sequence[float] | None
) -> tuple[np.ndarray, np.ndarray]:
    """``(3,)`` float32 mean and std in the 0-255 source scale, RGB order. Convention 4.

    ``std`` must be finite and **positive**, and ``mean`` finite. Only ``std == 0`` used to be
    refused, which left the two failures that do not announce themselves: a negative divisor
    inverts that channel — a photographic negative in one plane, which a model does not error
    on, it merely gets worse — and a non-finite entry produces an all-NaN tensor that poisons
    every reduction downstream of it. Both are configuration mistakes, so they fail here, at
    start-up, rather than on frame 40 000.
    """
    mean_array = np.asarray(DEFAULT_MEAN if mean is None else mean, dtype=np.float32)
    std_array = np.asarray(DEFAULT_STD if std is None else std, dtype=np.float32)
    if mean_array.shape != (3,) or std_array.shape != (3,):
        raise ConfigurationError(
            f"mean and std must each have three entries, got {mean_array.shape} and "
            f"{std_array.shape}"
        )
    if not np.all(np.isfinite(mean_array)):
        raise ConfigurationError(
            f"normalisation mean must be finite, got {mean_array}. A non-finite mean makes "
            f"every pixel NaN, and a NaN propagates through every reduction after it"
        )
    if not np.all(np.isfinite(std_array)) or np.any(std_array <= 0.0):
        raise ConfigurationError(
            f"normalisation std must be finite and positive, got {std_array}. A negative "
            f"divisor inverts that channel and a non-finite one makes the whole tensor NaN; "
            f"neither raises anywhere downstream"
        )
    return mean_array, std_array


# --------------------------------------------------------------- device output seam


@dataclass(frozen=True, slots=True)
class DeviceBuffer:
    """A caller-owned device allocation for a backend to write a batch into.

    The seam that lets pre-processing land where the engine wants it. At 1000 frames a second
    a preprocessed batch that goes device -> host -> device gives back most of what the fused
    kernel saved: on this box a batch of eight 1080p frames into 640x640 costs 44.7 ms through
    the numpy-returning path and 8.6 ms written straight to the device, and fifteen crops cost
    2.17 ms against 0.76 ms. The copy and the ``gpuStreamSynchronize`` behind it are the whole
    difference.

    A *descriptor*, not an allocator. This package does not own device memory — that belongs
    to torch or to TensorRT (ADR-003) — so a consumer passes in the binding it already has and
    this carries what a kernel needs to write into it safely.

    ``owner`` is the object the descriptor was built from, when there was one, and it is what
    makes the seam implementable by more than one backend: the native backend writes through
    ``pointer``, while the torch backend cannot build a tensor over a foreign device pointer in
    pure Python and writes through ``owner`` instead. A consumer that builds its buffer with
    :meth:`from_tensor` therefore works with either backend and does not have to know which
    one it holds.

    Attributes:
        pointer: the device address, as an integer — ``torch.Tensor.data_ptr()`` or a TensorRT
            binding address.
        nbytes: the allocation's capacity. Checked against the batch, because an overrun
            inside a kernel is an illegal access, and that error is sticky.
        device_index: which GPU the allocation lives on. Checked against the backend's own
            device: on a 16-GPU box "the engine is on 5 and the image ops are on 0" is a
            mistake that happens, and a cross-device write either faults or corrupts.
        owner: the tensor-like object, kept alive for the duration and used by backends that
            write through an object rather than an address.
    """

    pointer: int
    nbytes: int
    device_index: int
    owner: object | None = None

    def __post_init__(self) -> None:
        if self.pointer <= 0:
            raise ConfigurationError(
                f"device buffer pointer must be a positive address, got {self.pointer}. A "
                f"null pointer here would be written to by a kernel"
            )
        if self.nbytes <= 0:
            raise ConfigurationError(
                f"device buffer capacity must be positive, got {self.nbytes}"
            )
        if self.device_index < 0:
            raise ConfigurationError(
                f"device_index must be non-negative, got {self.device_index}"
            )

    @classmethod
    def from_tensor(cls, tensor: object) -> DeviceBuffer:
        """Describe a torch tensor — or anything that quacks like one — as a device buffer.

        Duck-typed rather than typed on ``torch.Tensor``, because this package must not import
        torch (the offline tier is a second long and torch costs about that on its own), and
        because a TensorRT binding wrapped in a small adapter is not a torch tensor either.
        Four things are required, and each one refused here is a failure that would otherwise
        be silent: a host allocation whose pointer a kernel cannot write, a strided tensor that
        would be filled as if it were dense, a non-float32 buffer that would come back as
        noise at the right byte count, and an object that is not a tensor at all.

        Raises:
            ConfigurationError: the object does not expose a device allocation this can write.
        """
        for attribute in ("data_ptr", "numel", "element_size"):
            if not callable(getattr(tensor, attribute, None)):
                raise ConfigurationError(
                    f"a device buffer needs an object exposing data_ptr(), numel() and "
                    f"element_size(); {type(tensor).__name__} has no {attribute}()"
                )
        device = getattr(tensor, "device", None)
        device_type = str(getattr(device, "type", "cuda"))
        if device_type not in ("cuda", "hip", "xpu"):
            raise ConfigurationError(
                f"a device buffer must live on an accelerator, got a {device_type!r} tensor. "
                f"Its pointer is host memory, and a kernel writing to it faults"
            )
        is_contiguous = getattr(tensor, "is_contiguous", None)
        if callable(is_contiguous) and not is_contiguous():
            raise ConfigurationError(
                "a device buffer must be contiguous: the kernels write one dense NCHW block, "
                "so a strided destination would be filled in the wrong order"
            )
        dtype = getattr(tensor, "dtype", None)
        if dtype is not None and "float32" not in str(dtype):
            raise ConfigurationError(
                f"a device buffer must be float32, got {dtype}. A half-precision binding of "
                f"the right byte count would be written successfully and read as noise"
            )
        index = getattr(device, "index", None)
        return cls(
            pointer=int(tensor.data_ptr()),  # type: ignore[attr-defined]
            nbytes=int(tensor.numel()) * int(tensor.element_size()),  # type: ignore[attr-defined]
            device_index=0 if index is None else int(index),
            owner=tensor,
        )

    def require(self, nbytes: int, device_index: int, *, what: str = "output") -> None:
        """Refuse now if this buffer cannot hold that batch, or is on the wrong device.

        Before the launch rather than during it: a kernel that runs off the end of an
        allocation raises ``cudaErrorIllegalAddress``, which poisons the context for the life
        of the process, so one mis-sized batch would take the worker down permanently.

        Raises:
            ConfigurationError: too small, or bound to a different device.
        """
        if device_index != self.device_index:
            raise ConfigurationError(
                f"the {what} buffer is on device {self.device_index} but these image ops are "
                f"bound to device {device_index}. One instance serves one device for its whole "
                f"life, so the buffer is the thing to move"
            )
        if self.nbytes < nbytes:
            raise ConfigurationError(
                f"the {what} buffer is too small: {self.nbytes} bytes for a batch that needs "
                f"{nbytes}. Size it with nchw_nbytes()"
            )


def nchw_nbytes(count: int, target_hw: Sequence[int]) -> int:
    """Bytes an ``(count, 3, h, w)`` float32 batch needs, so a caller can size its buffer.

    Here rather than in each consumer because the layout is this package's decision: a
    consumer that computed ``n * 3 * h * w * 4`` itself would be re-deriving the output
    convention, and would keep the old number if it ever changed.
    """
    if count < 0:
        raise ConfigurationError(f"batch count cannot be negative, got {count}")
    height, width = target_hw[0], target_hw[1]
    return int(count) * 3 * int(height) * int(width) * 4


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

    # -- pre-processing, straight to the device ---------------------------------------

    # Optional, and asked about rather than assumed. `supports_device_output` is what lets a
    # consumer prefer the fast path without holding a list of backend names: the detection
    # package asks the object it was handed by the registry, and the two routes produce the
    # same tensor. Not abstract, because the numpy backend genuinely cannot do it and a
    # backend should not have to write a raising stub to say so.

    @property
    def supports_device_output(self) -> bool:
        """Whether :meth:`letterbox_into` and :meth:`crop_batch_into` work on this instance.

        An instance property, not a class one: the torch backend can do it on ``cuda`` and not
        on ``cpu``, and that is decided at construction.
        """
        return False

    def letterbox_into(
        self,
        images: Sequence[np.ndarray] | np.ndarray,
        target_hw: tuple[int, int],
        out: DeviceBuffer,
        *,
        pad_value: int = DEFAULT_PAD_VALUE,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
    ) -> list[LetterboxGeometry]:
        """:meth:`letterbox`, written straight into ``out`` instead of into a numpy array.

        The production path: pre-processing feeds an engine on the same device, so the tensor
        should never reach host memory. The host-returning form pays a device-to-host copy and
        a stream synchronise per call, which on this box is 44.7 ms against 8.6 ms for a batch
        of eight 1080p frames into 640x640.

        Args:
            images: as :meth:`letterbox`.
            target_hw: as :meth:`letterbox`.
            out: a caller-owned device allocation of at least
                ``nchw_nbytes(len(images), target_hw)`` bytes, on this instance's device.
            pad_value: as :meth:`letterbox`.
            mean: as :meth:`letterbox`.
            std: as :meth:`letterbox`.

        Returns:
            One :class:`~shipvision.imgproc.geometry.LetterboxGeometry` per image, in input
            order. There is no tensor to return — that is the point — but the geometry is
            still the thing post-processing must invert with.

        Raises:
            BackendUnavailableError: this backend has no device output path.
            ConfigurationError: ``out`` is too small or on another device.
        """
        raise BackendUnavailableError(
            f"the {getattr(self, 'backend', type(self).__name__)!r} image-ops backend has no "
            f"device output path; ask supports_device_output before calling this, or build the "
            f"native backend"
        )

    def crop_batch_into(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        target_hw: tuple[int, int],
        out: DeviceBuffer,
        *,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
    ) -> None:
        """:meth:`crop_batch`, written straight into ``out``. See :meth:`letterbox_into`.

        Args:
            out: at least ``nchw_nbytes(len(boxes), target_hw)`` bytes, on this instance's
                device.

        Raises:
            BackendUnavailableError: this backend has no device output path.
            ConfigurationError: ``out`` is too small or on another device.
        """
        raise BackendUnavailableError(
            f"the {getattr(self, 'backend', type(self).__name__)!r} image-ops backend has no "
            f"device output path; ask supports_device_output before calling this, or build the "
            f"native backend"
        )

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
