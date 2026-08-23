"""The native backend: the fused CUDA/HIP kernels in ``shipvision._C``.

This module is a translator, not an algorithm. All three operations already exist in
``csrc/`` — letterbox as one pass that resizes, pads, converts, normalises and transposes
without touching the frame four times; crop as one launch over every box; NMS as a device
bitmask with a sequential host sweep — and what is left in Python is argument marshalling
plus one guard.

The guard is the interesting part. The C++ side computes the letterbox scale and pads itself,
in float32, and hands them back; this module computes the same numbers through
:meth:`~shipvision.imgproc.base.LetterboxGeometry.plan` and refuses to continue if they
disagree. Two implementations of a rounding rule will eventually drift — a ``lroundf``
becomes a ``roundf``, a ``/ 2`` becomes a ``* 0.5f`` — and the symptom is boxes off by a
pixel on the cameras whose resolution happens to land on a boundary, which is close to
undebuggable after the fact. Checking costs two float comparisons per frame and turns that
into a loud failure on the first frame.

Importing this module never fails, even with no build and no device. Only construction does,
with :class:`~shipvision.errors.BackendUnavailableError`, which is what lets
``shipvision.imgproc`` register the backend unconditionally and
:func:`~shipvision.imgproc.build_image_ops` fall back to numpy without a try/import dance at
every call site.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from shipvision.errors import BackendUnavailableError, ConfigurationError, InferenceError
from shipvision.imgproc.base import (
    DEFAULT_PAD_VALUE,
    ImageOps,
    as_image_batch,
    resolve_normalisation,
    validate_boxes,
    validate_image,
)
from shipvision.imgproc.geometry import LetterboxGeometry, validate_target_hw
from shipvision.imgproc.nms import CLASSIC, prepare, suppress
from shipvision.imgproc.registry import IMGPROC
from shipvision.registry import NATIVE

__all__ = ["NativeImageOps", "native_available"]

try:  # pragma: no cover - depends on whether the extension was built, not on a branch
    from shipvision import _C

    _IMPORT_ERROR: str | None = None
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
    result is plausible-looking output from the wrong frame — see ``csrc/core/buffers.hpp``.
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

    @property
    def device_index(self) -> int:
        """The GPU this instance is bound to, for the life of the instance."""
        return self._device_index

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
    ) -> tuple[np.ndarray, list[LetterboxGeometry]]:
        """See :meth:`ImageOps.letterbox`."""
        frames = as_image_batch(images)
        target_h, target_w = validate_target_hw(target_hw)
        mean_array, std_array = resolve_normalisation(mean, std)

        tensor, scales, pads = self._ops.letterbox_batch(
            frames,
            target_h,
            target_w,
            mean_array.tolist(),
            std_array.tolist(),
            True,
            int(pad_value),
            self._stream,
        )
        geometries = [
            LetterboxGeometry.plan(frame.shape[:2], (target_h, target_w)) for frame in frames
        ]
        _assert_geometry_agrees(geometries, scales, pads)
        return np.asarray(tensor, dtype=np.float32), geometries

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

        crops = self._ops.crop_batch(
            frame,
            box_array,
            target_h,
            target_w,
            mean_array.tolist(),
            std_array.tolist(),
            True,
            self._stream,
        )
        return np.asarray(crops, dtype=np.float32)

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
    ) -> np.ndarray:
        """See :meth:`ImageOps.nms`. ``"classic"`` runs on the device.

        Only ``"classic"`` has a kernel, and deliberately: the device formulation computes
        the whole ``(box, box)`` bitmask in parallel and sweeps it once, which is worth doing
        for 25 000 proposals. Soft-NMS's decay is order-dependent — box j's score depends on
        which boxes were picked before it — so it cannot be that bitmask, and over the few
        dozen boxes that survive a score threshold there is nothing left to parallelise.
        Those methods run the shared numpy implementation, so every backend gives the same
        answer for them.
        """
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
        )
        if box_array.shape[0] == 0:
            return np.zeros(0, dtype=np.int64)
        kept = self._ops.nms(
            box_array,
            score_array,
            float(iou_threshold),
            float(score_threshold),
            int(box_array.shape[0]),
            self._stream,
        )
        return np.asarray(kept, dtype=np.int64)


def _assert_geometry_agrees(
    geometries: Sequence[LetterboxGeometry], scales: np.ndarray, pads: np.ndarray
) -> None:
    """Fail loudly if C++ and Python rounded the letterbox differently.

    Raises:
        InferenceError: the two implementations of conventions 2 and 3 have drifted. This is
            a build problem, not a data problem, and it must not be recoverable — the
            alternative is a pipeline that quietly reports boxes a pixel off on a subset of
            cameras.
    """
    for index, geometry in enumerate(geometries):
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
