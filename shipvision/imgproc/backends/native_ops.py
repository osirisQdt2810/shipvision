"""The native backend: the fused CUDA/HIP kernels in ``shipvision._C``.

This module is a translator, not an algorithm. All three operations already exist in
``csrc/`` — letterbox as one pass that resizes, pads, converts, normalises and transposes
without touching the frame four times; crop as one launch over every box; NMS as a device
bitmask with a sequential host sweep — and what is left in Python is argument marshalling
plus one guard.

The guard is the interesting part. The C++ side computes the letterbox scale, the pads and the
resized extent itself, in float32, and hands all three back; this module computes the same
numbers through :meth:`~shipvision.imgproc.geometry.LetterboxGeometry.plan` and refuses to
continue if they disagree. Two implementations of a rounding rule will eventually drift — a
``lroundf`` becomes a ``roundf``, a ``/ 2`` becomes a ``* 0.5f`` — and the symptom is boxes off
by a pixel on the cameras whose resolution happens to land on a boundary, which is close to
undebuggable after the fact. Checking costs a handful of comparisons per frame and turns that
into a loud failure on the first frame.

The **resized extent** has to be one of the compared numbers, and it was the one missing. It is
what the kernel divides by, so it decides the sampling ratio, and a one-pixel disagreement in
it hides behind a scale and a pad that both still match — ``(T - r) // 2`` is the same for
``r`` and ``r + 1`` whenever ``T - r`` is even. See :func:`_assert_geometry_agrees`.

Importing this module never fails, even with no build and no device. Only construction does,
with :class:`~shipvision.errors.BackendUnavailableError`, which is what lets
``shipvision.imgproc`` register the backend unconditionally and
:func:`~shipvision.imgproc.build_image_ops` fall back to numpy without a try/import dance at
every call site.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence

import numpy as np

from shipvision.errors import BackendUnavailableError, ConfigurationError, InferenceError
from shipvision.imgproc.base import (
    DEFAULT_PAD_VALUE,
    DeviceBuffer,
    ImageOps,
    as_image_batch,
    nchw_nbytes,
    resolve_normalisation,
    validate_boxes,
    validate_image,
    validate_pad_value,
)
from shipvision.imgproc.colour import nv12_height, validate_nv12_frame
from shipvision.imgproc.geometry import LetterboxGeometry, validate_target_hw
from shipvision.imgproc.nms import CLASSIC, prepare, suppress
from shipvision.imgproc.registry import IMGPROC
from shipvision.registry import NATIVE

__all__ = ["NativeImageOps", "native_available"]

try:  # pragma: no cover - depends on whether the extension was built, not on a branch
    from shipvision._native import load_extension

    _C, _IMPORT_ERROR = load_extension()
except ImportError as exc:  # pragma: no cover
    _C = None  # type: ignore[assignment]
    _IMPORT_ERROR = str(exc)


def native_available() -> bool:
    """True when ``shipvision._C`` imported and reports a visible device.

    Cheap enough to call from a test's skip condition, which is the main reason it exists —
    ``_C`` importing is not the same as a GPU being present, and a build made on a machine
    with a driver can land on one without.
    """
    if _C is None:
        return False
    try:
        return bool(_C.cuda_available())
    except Exception:  # pragma: no cover - a broken build should not crash collection
        return False


# Half the pixel pitch of the smallest input this library supports: any real drift in the
# rounding rule is a whole pixel, so this only tolerates float32 representation noise.
_GEOMETRY_EPSILON = 1e-3


@IMGPROC.register("default", backend=NATIVE)
class NativeImageOps(ImageOps):
    """Fused kernels bound to one device, with persistent staging scratch.

    One instance owns one device index and one staging ring, so it belongs to one worker
    thread for the life of the process. Sharing it between threads races the ring, and the
    result is plausible-looking output from the wrong frame — see ``csrc/shipvision/core/buffers.h``.

    That last sentence used to be the whole enforcement. It is now checked: the first thread
    to *use* an instance claims it, and a call from any other raises
    :class:`~shipvision.errors.InferenceError`. Measured before the check, two threads sharing
    one instance over 300 frames each either returned the other camera's pixels — 11 and 242
    frames of 300 in the review that found this — or, on the box this was fixed on, failed every
    call with a sticky ``GpuError`` that ends the process. Both are unacceptable and neither is
    attributable after the fact: a mis-tagged detection looks exactly like a real one.

    Ownership is claimed on first use rather than in ``__init__`` on purpose. The ring is only
    touched by a call, so first use is when the invariant starts to matter, and claiming at
    construction would refuse the ordinary pattern of assembling a pipeline on one thread and
    running it on a worker — a false failure on every frame, in exchange for nothing.
    """

    def __init__(self, *, device_index: int = 0, stream: int = 0) -> None:
        """
        Args:
            device_index: which GPU. Set once here, not per call, because the extension
                calls ``gpuSetDevice`` on construction and again on every launch.
            stream: a raw stream handle as an integer — ``torch.cuda.Stream.cuda_stream``,
                typically. ``0`` is the default stream. Passing the stream the engine will
                run on is what lets an upload overlap the previous batch's compute.

        Raises:
            BackendUnavailableError: the extension is not built, or there is no device.
        """
        if _C is None:
            raise BackendUnavailableError(
                f"shipvision._C is not built: {_IMPORT_ERROR}. Build it with "
                f"`cmake -S . -B build && cmake --build build -j`, or use backend='python'"
            )
        if device_index < 0:
            raise ConfigurationError(f"device_index must be non-negative, got {device_index}")
        try:
            self._ops = _C.ImageOps(device_index=device_index)
        except RuntimeError as exc:  # GpuError is registered onto RuntimeError
            raise BackendUnavailableError(
                f"cannot bind the native image ops to device {device_index}: {exc}"
            ) from exc
        self._device_index = device_index
        self._stream = int(stream)
        self._owner_thread: int | None = None
        self._owner_name = ""
        self._claim_lock = threading.Lock()

    @property
    def device_index(self) -> int:
        """The GPU this instance is bound to, for the life of the instance."""
        return self._device_index

    def _claim_thread(self) -> None:
        """Bind this instance to the calling thread, or refuse a foreign caller.

        Raises:
            InferenceError: another thread already owns this instance. Not
                :class:`~shipvision.errors.ConfigurationError`, because it is not the
                configuration that is wrong — the work reached the wrong object at run time,
                and the frame has to be dropped rather than the process taken down.
        """
        current = threading.get_ident()
        if self._owner_thread is None:
            with self._claim_lock:
                if self._owner_thread is None:
                    self._owner_thread = current
                    self._owner_name = threading.current_thread().name
                    return
        if self._owner_thread != current:
            raise InferenceError(
                f"these native image ops belong to thread {self._owner_name!r} and were called "
                f"from {threading.current_thread().name!r}. One instance serves one thread for "
                f"the life of the process: the staging ring is per-instance, so two threads "
                f"sharing it either race a live DMA — a sticky illegal access that ends the "
                f"worker — or return the previous camera's pixels with no error at all. Build "
                f"one instance per worker thread; construction is cheap"
            )

    def scratch_bytes(self) -> dict[str, int]:
        """Persistent device and pinned-host scratch this instance holds, in bytes.

        Exposed because it is the number that answers "why is this process using 4 GB": the
        scratch grows to the high-water mark of the largest batch ever submitted and is never
        handed back.
        """
        return dict(self._ops.scratch_bytes())

    # -- pre-processing ---------------------------------------------------------------

    def letterbox(
        self,
        images: Sequence[np.ndarray] | np.ndarray,
        target_hw: tuple[int, int],
        *,
        pad_value: int = DEFAULT_PAD_VALUE,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        swap_rb: bool = True,
    ) -> tuple[np.ndarray, list[LetterboxGeometry]]:
        """See :meth:`ImageOps.letterbox`."""
        self._claim_thread()
        frames = as_image_batch(images)
        target_h, target_w = validate_target_hw(target_hw)
        mean_array, std_array = resolve_normalisation(mean, std)
        pad_value = validate_pad_value(pad_value)

        tensor, scales, pads, extents = self._ops.letterbox_batch(
            frames,
            target_h,
            target_w,
            mean_array.tolist(),
            std_array.tolist(),
            bool(swap_rb),
            int(pad_value),
            self._stream,
        )
        geometries = [
            LetterboxGeometry.plan(frame.shape[:2], (target_h, target_w)) for frame in frames
        ]
        _assert_geometry_agrees(geometries, scales, pads, extents)
        return np.asarray(tensor, dtype=np.float32), geometries

    def crop_batch(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        target_hw: tuple[int, int],
        *,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        swap_rb: bool = True,
    ) -> np.ndarray:
        """See :meth:`ImageOps.crop_batch`."""
        self._claim_thread()
        frame = validate_image(image)
        box_array = validate_boxes(boxes)
        target_h, target_w = validate_target_hw(target_hw)
        mean_array, std_array = resolve_normalisation(mean, std)

        crops = self._ops.crop_batch(
            frame,
            box_array,
            target_h,
            target_w,
            mean_array.tolist(),
            std_array.tolist(),
            bool(swap_rb),
            self._stream,
        )
        return np.asarray(crops, dtype=np.float32)

    # -- pre-processing, straight to the device ---------------------------------------

    @property
    def supports_device_output(self) -> bool:
        """Always, for this backend: it is why the kernels exist.

        ``csrc/bindings/module.cpp`` calls ``letterbox_into`` the production path, and until
        this seam existed nothing in Python could reach it — every consumer went through
        :meth:`letterbox`, paying a device-to-host copy and a stream synchronise to hand the
        result back to something that wanted it on the device anyway.
        """
        return True

    def letterbox_into(
        self,
        images: Sequence[np.ndarray] | np.ndarray,
        target_hw: tuple[int, int],
        out: DeviceBuffer,
        *,
        pad_value: int = DEFAULT_PAD_VALUE,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        swap_rb: bool = True,
    ) -> list[LetterboxGeometry]:
        """See :meth:`ImageOps.letterbox_into`. 8.6 ms where :meth:`letterbox` costs 44.7 ms
        for a batch of eight 1080p frames into 640x640 on this box."""
        self._claim_thread()
        frames = as_image_batch(images)
        target_h, target_w = validate_target_hw(target_hw)
        mean_array, std_array = resolve_normalisation(mean, std)
        pad_value = validate_pad_value(pad_value)
        out.require(nchw_nbytes(len(frames), (target_h, target_w)), self._device_index)

        scales, pads, extents = self._ops.letterbox_into(
            frames,
            out.pointer,
            out.nbytes,
            target_h,
            target_w,
            mean_array.tolist(),
            std_array.tolist(),
            bool(swap_rb),
            int(pad_value),
            self._stream,
        )
        geometries = [
            LetterboxGeometry.plan(frame.shape[:2], (target_h, target_w)) for frame in frames
        ]
        _assert_geometry_agrees(geometries, scales, pads, extents)
        return geometries

    # -- letterbox from a decoder's NV12 ----------------------------------------------

    @property
    def supports_nv12(self) -> bool:
        """Always, for this backend: the fused NV12 kernel is why it exists."""
        return True

    @property
    def supports_nv12_device_input(self) -> bool:
        """Always. :meth:`nv12_letterbox_device_into` is the only zero-PCIe path here."""
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
        """See :meth:`ImageOps.nv12_letterbox`."""
        self._claim_thread()
        buffers, visible = _validate_nv12_batch(frames, widths)
        target_h, target_w = validate_target_hw(target_hw)
        mean_array, std_array = resolve_normalisation(mean, std)
        pad_value = validate_pad_value(pad_value)

        tensor, scales, pads, extents = self._ops.nv12_letterbox_batch(
            buffers,
            visible,
            target_h,
            target_w,
            mean_array.tolist(),
            std_array.tolist(),
            bool(swap_rb),
            int(pad_value),
            self._stream,
        )
        geometries = _nv12_geometries(buffers, visible, (target_h, target_w))
        _assert_geometry_agrees(geometries, scales, pads, extents)
        return np.asarray(tensor, dtype=np.float32), geometries

    def nv12_letterbox_into(
        self,
        frames: Sequence[np.ndarray],
        widths: Sequence[int],
        target_hw: tuple[int, int],
        out: DeviceBuffer,
        *,
        pad_value: int = DEFAULT_PAD_VALUE,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        swap_rb: bool = True,
    ) -> list[LetterboxGeometry]:
        """See :meth:`ImageOps.nv12_letterbox_into`."""
        self._claim_thread()
        buffers, visible = _validate_nv12_batch(frames, widths)
        target_h, target_w = validate_target_hw(target_hw)
        mean_array, std_array = resolve_normalisation(mean, std)
        pad_value = validate_pad_value(pad_value)
        out.require(nchw_nbytes(len(buffers), (target_h, target_w)), self._device_index)

        scales, pads, extents = self._ops.nv12_letterbox_into(
            buffers,
            visible,
            out.pointer,
            out.nbytes,
            target_h,
            target_w,
            mean_array.tolist(),
            std_array.tolist(),
            bool(swap_rb),
            int(pad_value),
            self._stream,
        )
        geometries = _nv12_geometries(buffers, visible, (target_h, target_w))
        _assert_geometry_agrees(geometries, scales, pads, extents)
        return geometries

    def nv12_letterbox_device_into(
        self,
        descriptors: np.ndarray,
        target_hw: tuple[int, int],
        out: DeviceBuffer,
        *,
        pad_value: int = DEFAULT_PAD_VALUE,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        swap_rb: bool = True,
    ) -> list[LetterboxGeometry]:
        """See :meth:`ImageOps.nv12_letterbox_device_into`.

        The descriptors are *not* validated against the device — a pointer from another
        context is indistinguishable from one from this context, and a wrong one is a sticky
        illegal access that ends the worker. What is checked is everything that can be: the
        shape, positive even extents, strides at least the width, and non-null pointers.
        """
        self._claim_thread()
        table = np.ascontiguousarray(np.asarray(descriptors, dtype=np.int64))
        if table.ndim != 2 or table.shape[1] != 6:
            raise ConfigurationError(
                f"descriptors must be (n, 6) — [y_ptr, uv_ptr, height, width, y_stride, "
                f"uv_stride] — got {table.shape}"
            )
        target_h, target_w = validate_target_hw(target_hw)
        mean_array, std_array = resolve_normalisation(mean, std)
        pad_value = validate_pad_value(pad_value)
        out.require(nchw_nbytes(table.shape[0], (target_h, target_w)), self._device_index)

        scales, pads, extents = self._ops.nv12_letterbox_device_into(
            table,
            out.pointer,
            out.nbytes,
            target_h,
            target_w,
            mean_array.tolist(),
            std_array.tolist(),
            bool(swap_rb),
            int(pad_value),
            self._stream,
        )
        geometries = [
            LetterboxGeometry.plan((int(row[2]), int(row[3])), (target_h, target_w))
            for row in table
        ]
        _assert_geometry_agrees(geometries, scales, pads, extents)
        return geometries

    def crop_batch_into(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        target_hw: tuple[int, int],
        out: DeviceBuffer,
        *,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        swap_rb: bool = True,
    ) -> None:
        """See :meth:`ImageOps.crop_batch_into`."""
        self._claim_thread()
        frame = validate_image(image)
        box_array = validate_boxes(boxes)
        target_h, target_w = validate_target_hw(target_hw)
        mean_array, std_array = resolve_normalisation(mean, std)
        if box_array.shape[0] == 0:
            # A frame with no objects is normal input, and the C++ side returns early for it.
            # Refusing to size-check an empty batch keeps a caller from having to special-case
            # the quiet cameras.
            return
        out.require(nchw_nbytes(box_array.shape[0], (target_h, target_w)), self._device_index)

        self._ops.crop_into(
            frame,
            box_array,
            out.pointer,
            out.nbytes,
            target_h,
            target_w,
            mean_array.tolist(),
            std_array.tolist(),
            bool(swap_rb),
            self._stream,
        )

    # -- post-processing --------------------------------------------------------------

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
        max_output: int | None = None,
    ) -> np.ndarray:
        """See :meth:`ImageOps.nms`. ``"classic"`` runs on the device.

        Only ``"classic"`` has a kernel, and deliberately: the device formulation computes
        the whole ``(box, box)`` bitmask in parallel and sweeps it once, which is worth doing
        for 25 000 proposals. Soft-NMS's decay is order-dependent — box j's score depends on
        which boxes were picked before it — so it cannot be that bitmask, and over the few
        dozen boxes that survive a score threshold there is nothing left to parallelise.
        Those methods run the shared numpy implementation, so every backend gives the same
        answer for them.

        ``max_output`` is pushed all the way down to the kernel rather than applied to what it
        returns. The C++ sweep already takes the argument — the Python side used to hand it the
        box count, which is the same as no cap — and its stopping rule is
        ``keep.size() < max_output`` over survivors visited in descending score, which is
        exactly this method's contract. Pushing it down is also the only version that saves
        anything: the sweep stops early instead of finishing 25 000 boxes and throwing the tail
        away.
        """
        self._claim_thread()
        if method != CLASSIC:
            return suppress(
                boxes,
                scores,
                iou_threshold=iou_threshold,
                method=method,
                sigma=sigma,
                score_threshold=score_threshold,
                min_neighbors=min_neighbors,
                min_score_sum=min_score_sum,
                max_output=max_output,
            )[0]

        # prepare() validates and would raise for a bad method or threshold; the kernel does
        # its own score filtering and sorting, so its output is used, not `order`.
        box_array, score_array, _ = prepare(
            boxes,
            scores,
            iou_threshold=iou_threshold,
            method=method,
            sigma=sigma,
            score_threshold=score_threshold,
            max_output=max_output,
        )
        if box_array.shape[0] == 0:
            return np.zeros(0, dtype=np.int64)
        # No cap is expressed as "as many as there are boxes", because the binding's argument
        # is a plain int with no null: the sweep can never keep more survivors than candidates,
        # so the box count is the identity budget.
        budget = box_array.shape[0] if max_output is None else max_output
        kept = self._ops.nms(
            box_array,
            score_array,
            float(iou_threshold),
            float(score_threshold),
            int(budget),
            self._stream,
        )
        return np.asarray(kept, dtype=np.int64)


def _validate_nv12_batch(
    frames: Sequence[np.ndarray], widths: Sequence[int]
) -> tuple[list[np.ndarray], list[int]]:
    """Validate a ragged NV12 batch here, in Python, before any pointer reaches the kernel.

    The C++ side checks the same things — it has to, since its entry points are reachable
    without this — but doing it here first is what turns a caller's mistake into a readable
    Python exception instead of an ``invalid_argument`` from a binding.
    """
    if len(frames) != len(widths):
        raise ConfigurationError(
            f"got {len(frames)} NV12 frames and {len(widths)} widths; a decoder's stride is "
            f"not its visible width, so one width per frame is required"
        )
    if not frames:
        raise ConfigurationError(
            "nv12 letterbox needs at least one frame; an empty batch is a caller bug, not a "
            "quiet camera"
        )
    buffers = [
        validate_nv12_frame(frame, width, what=f"frames[{index}]")
        for index, (frame, width) in enumerate(zip(frames, widths, strict=True))
    ]
    return buffers, [int(width) for width in widths]


def _nv12_geometries(
    buffers: Sequence[np.ndarray], widths: Sequence[int], target_hw: tuple[int, int]
) -> list[LetterboxGeometry]:
    """One geometry per NV12 frame, planned from the *visible* extent.

    The height comes from the row count and not from the buffer's shape directly, because a
    packed NV12 buffer has 1.5x the luma rows — planning against ``shape[0]`` would letterbox
    a 1620-row image and every box would be scaled by 2/3.
    """
    return [
        LetterboxGeometry.plan((nv12_height(int(buffer.shape[0])), int(width)), target_hw)
        for buffer, width in zip(buffers, widths, strict=True)
    ]


def _assert_geometry_agrees(
    geometries: Sequence[LetterboxGeometry],
    scales: np.ndarray,
    pads: np.ndarray,
    extents: np.ndarray,
) -> None:
    """Fail loudly if C++ and Python rounded the letterbox differently.

    The **resized extent** is compared as well as the scale and the pads, and it is the one
    that matters most: the kernel's sampling ratio is ``view.height / view.out_h``, so ``out_h``
    is the number that decides where every row is read from, and Python re-derives it from the
    scale rather than being told. Those two derivations can differ by a pixel while the scale
    and the pad still match to the bit — ``pad = (T - r) // 2`` is the same for ``r`` and
    ``r + 1`` whenever ``T - r`` is even, so a 7-row source at scale 0.5 into a 100-row canvas
    pads to 48 whether the extent is 3 or 4 — and the earlier version of this guard compared
    only the numbers that agreed.

    Raises:
        InferenceError: the two implementations of conventions 2 and 3 have drifted. This is
            a build problem, not a data problem, and it must not be recoverable — the
            alternative is a pipeline that quietly reports boxes a pixel off on a subset of
            cameras.
    """
    for index, geometry in enumerate(geometries):
        kernel_extent = (int(extents[index][0]), int(extents[index][1]))
        python_extent = (geometry.resized_height, geometry.resized_width)
        if kernel_extent != python_extent:
            raise InferenceError(
                f"native letterbox resized extent disagrees with the Python convention for "
                f"image {index}: kernel resized to {kernel_extent[0]}x{kernel_extent[1]}, "
                f"Python to {python_extent[0]}x{python_extent[1]}. The kernel samples at "
                f"height / out_h, so every row of that image is read from the wrong ratio — "
                f"and the pad can still match, which is why the extent is compared"
            )
        if (
            abs(float(scales[index]) - geometry.scale) > _GEOMETRY_EPSILON
            or abs(float(pads[index][0]) - geometry.pad_left) > _GEOMETRY_EPSILON
            or abs(float(pads[index][1]) - geometry.pad_top) > _GEOMETRY_EPSILON
        ):
            raise InferenceError(
                f"native letterbox geometry disagrees with the Python convention for image "
                f"{index}: kernel gave scale={float(scales[index])!r} "
                f"pad=({float(pads[index][0])!r}, {float(pads[index][1])!r}), Python gave "
                f"scale={geometry.scale!r} pad=({geometry.pad_left}, {geometry.pad_top}). "
                f"One of the two rounding rules has drifted; every box would be shifted"
            )
