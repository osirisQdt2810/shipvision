"""INT8 calibration: the piece every reference left commented out.

``pytools/onnx2trt.py:75-76`` in the reference has

    # use INT8
    # config.set_flag(trt.BuilderFlag.INT8)

and there is no calibrator class anywhere in that repository, so the flag could not have been
enabled — TensorRT refuses an INT8 build with no calibrator and no explicit quantisation. This
module is that missing half.

WHAT CALIBRATION IS
    An INT8 engine needs a scale per activation tensor: the value that maps that tensor's real
    dynamic range onto ``[-127, 127]``. TensorRT derives those scales by running the network in
    float over a few hundred representative images and choosing, per tensor, the clipping
    threshold that loses the least information — the KL-divergence criterion, hence
    ``IInt8EntropyCalibrator2``. So a calibrator is nothing more than a batch source, plus
    somewhere to cache the answer.

THE TRAP, STATED PLAINLY
    **The calibration data must go through exactly the same preprocessing as inference.** Same
    letterbox, same pad value, same BGR->RGB swap, same mean and std. A calibrator fed raw
    0-255 images while inference feeds ``[0, 1]`` produces scales 255x too wide: the engine
    builds without complaint, runs at full INT8 speed, and every activation lands in the bottom
    half-percent of its quantised range. The output is not garbage — it is *plausible*, with
    perhaps ten points of mAP missing — which is why this is worth a paragraph rather than a
    line. The only reliable way to avoid it is to produce calibration batches with the same
    :class:`~shipvision.imgproc.base.ImageOps` call the detector will use, and
    :class:`CalibrationBatchFeeder` accepts an optional ``value_range`` so that a mistake this
    expensive can at least be made loud.

WHY THE CACHE MATTERS
    Calibration costs one forward pass per batch over the whole calibration set, and it happens
    *inside* the build. Without a cache, changing an unrelated builder flag re-runs all of it.
    The cache is TensorRT's own opaque blob of per-tensor scales; it is keyed to the network,
    so it must be invalidated when the ONNX changes — which is why :class:`CalibrationCache`
    writes to a path the caller chose rather than to a temporary directory it invented.

WHAT IS TESTABLE WITHOUT TENSORRT, AND WHY IT IS SPLIT THIS WAY
    Subclassing ``trt.IInt8EntropyCalibrator2`` requires tensorrt at *class definition* time,
    so the subclass is created inside :func:`build_int8_calibrator` and this module still
    imports on a laptop. Everything with a decision in it — batch validation, the padding of a
    short final batch, cache read/write and invalidation — lives in the two plain classes above
    it, which are exercised in the offline tier.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from shipvision.errors import (
    BackendUnavailableError,
    ConfigurationError,
    DimensionMismatchError,
)

__all__ = ["CalibrationBatchFeeder", "CalibrationCache", "build_int8_calibrator"]


class CalibrationCache:
    """TensorRT's per-tensor scales on disk, so a rebuild does not recalibrate.

    Args:
        path: where the blob lives. `None` disables caching entirely — which is the right
            choice exactly once, when you are checking whether a stale cache is the reason an
            engine regressed.

    Attributes:
        hits: how many times TensorRT read the cache. Zero after a build means calibration
            actually ran; one means it did not. Recorded because "did it use the cache" is
            otherwise invisible, and a stale cache is the first thing to suspect when an INT8
            engine's accuracy changes without its ONNX changing.
    """

    def __init__(self, path: str | Path | None) -> None:
        self.path = None if path is None else Path(path)
        self.hits = 0
        self.writes = 0

    def read(self) -> bytes | None:
        """The cached blob, or `None` — which is what tells TensorRT to calibrate."""
        if self.path is None or not self.path.is_file():
            return None
        blob = self.path.read_bytes()
        if not blob:
            # An empty file is a build that was interrupted while writing. Treating it as a
            # cache miss recalibrates; treating it as a hit would build an engine with no
            # scales at all.
            return None
        self.hits += 1
        return blob

    def write(self, blob: bytes) -> None:
        """Persist the blob, creating the parent directory if need be."""
        if self.path is None or not blob:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(bytes(blob))
        self.writes += 1

    def invalidate(self) -> None:
        """Delete the cache. Call this when the ONNX changes — the scales belong to a network."""
        if self.path is not None:
            self.path.unlink(missing_ok=True)

    def __repr__(self) -> str:
        return f"<CalibrationCache path={self.path} hits={self.hits} writes={self.writes}>"


class CalibrationBatchFeeder:
    """Hands out validated, uniformly-shaped batches from a caller-supplied iterable.

    The state machine TensorRT drives during a build: it asks for a batch, gets one, and stops
    when it gets `None`. Everything the calibrator needs to be *correct* is here rather than in
    the tensorrt subclass, because none of it needs a device.

    Args:
        batches: an iterable of already-preprocessed ``(n, c, h, w)`` float32 arrays. An
            iterable rather than a list so a calibration set larger than memory can be a
            generator over files — a few hundred 640x640 frames is a gigabyte.
        batch_shape: the ``(n, c, h, w)`` the engine's profile will be calibrated at. Every
            batch is checked against ``(c, h, w)``; ``n`` is what a short final batch is padded
            up to.
        value_range: optional ``(low, high)``. When given, a batch whose values fall outside it
            raises — the cheapest available defence against the preprocessing mismatch in the
            module docstring. For the default ``[0, 1]`` normalisation, pass ``(0.0, 1.0)``
            with a little slack, or ``(-3.0, 3.0)`` for mean/std normalisation.
        limit: stop after this many batches. Calibration accuracy plateaus after a few hundred
            images and the build time does not, so a cap is a real knob rather than a guard.
    """

    def __init__(
        self,
        batches: Iterable[np.ndarray],
        *,
        batch_shape: Sequence[int],
        value_range: tuple[float, float] | None = None,
        limit: int | None = None,
    ) -> None:
        shape = tuple(int(v) for v in batch_shape)
        if len(shape) != 4 or any(v <= 0 for v in shape):
            raise ConfigurationError(
                f"batch_shape must be (n, c, h, w) with positive values, got {batch_shape!r}"
            )
        if value_range is not None:
            low, high = (float(v) for v in value_range)
            if low >= high:
                raise ConfigurationError(
                    f"value_range must be (low, high) with low < high, got {value_range!r}"
                )
            value_range = (low, high)
        if limit is not None and limit <= 0:
            raise ConfigurationError(f"limit must be positive or None, got {limit}")

        self.batch_shape = shape
        self.value_range = value_range
        self.limit = limit
        self.served = 0
        self._source: Iterator[np.ndarray] = iter(batches)

    @property
    def batch_size(self) -> int:
        return self.batch_shape[0]

    def next_batch(self) -> np.ndarray | None:
        """The next full batch, or `None` when the source is exhausted.

        A short final batch is **padded by repeating its own rows** rather than with zeros.
        TensorRT calibrates at a fixed batch size and a partial batch has to be filled with
        something; zeros are not a plausible image, and a whole batch of them drags the
        activation histograms towards zero exactly where the entropy criterion is choosing a
        clipping threshold. Repeating real rows biases the histogram by counting a few images
        twice, which is the smaller of the two errors by a wide margin.
        """
        if self.limit is not None and self.served >= self.limit:
            return None
        batch = next(self._source, None)
        if batch is None:
            return None

        array = self._validated(batch)
        rows = array.shape[0]
        if rows < self.batch_size:
            repeats = np.arange(self.batch_size) % rows
            array = array[repeats]
        elif rows > self.batch_size:
            array = array[: self.batch_size]
        self.served += 1
        return np.ascontiguousarray(array, dtype=np.float32)

    def _validated(self, batch: np.ndarray) -> np.ndarray:
        array = np.asarray(batch)
        if array.dtype != np.float32:
            raise ConfigurationError(
                f"calibration batch {self.served} is {array.dtype}; it must be float32 and "
                f"already preprocessed. A uint8 batch here is the failure the module docstring "
                f"describes: the engine builds, runs, and is quietly wrong"
            )
        if array.ndim != 4 or array.shape[1:] != self.batch_shape[1:]:
            raise DimensionMismatchError(
                f"calibration batch {self.served} has shape {array.shape}; the engine is "
                f"calibrated at (n, {', '.join(str(v) for v in self.batch_shape[1:])}). "
                f"Calibration data must go through the same letterbox as inference"
            )
        if array.shape[0] == 0:
            raise DimensionMismatchError(
                f"calibration batch {self.served} is empty; an empty batch calibrates nothing "
                f"and TensorRT would read it as the end of the data"
            )
        if self.value_range is not None:
            low, high = self.value_range
            lowest, highest = float(array.min()), float(array.max())
            if lowest < low or highest > high:
                raise ConfigurationError(
                    f"calibration batch {self.served} spans [{lowest:.4g}, {highest:.4g}], "
                    f"outside the declared value_range [{low:.4g}, {high:.4g}]. This is almost "
                    f"always preprocessing that does not match inference — raw 0-255 pixels "
                    f"against a model fed [0, 1] — and it produces an engine that is fast and "
                    f"quietly inaccurate"
                )
        return array

    def __repr__(self) -> str:
        return (
            f"<CalibrationBatchFeeder shape={self.batch_shape} served={self.served} "
            f"limit={self.limit}>"
        )


def build_int8_calibrator(
    trt: Any,
    feeder: CalibrationBatchFeeder,
    *,
    cache: CalibrationCache | None = None,
) -> Any:
    """An ``IInt8EntropyCalibrator2`` over ``feeder``, caching to ``cache``.

    The class is defined *inside* this function because subclassing a tensorrt type needs
    tensorrt imported, and this module must stay importable on a machine without it. That is
    the same reason the calibrator's decisions live in :class:`CalibrationBatchFeeder` and
    :class:`CalibrationCache`, which are plain Python and are tested offline.

    Device memory comes from torch, for the reason it always does in this repository: torch's
    caching allocator is already the best-tested one available from Python, and one persistent
    buffer sized from ``feeder.batch_shape`` means calibration allocates once rather than once
    per batch.

    Args:
        trt: the imported ``tensorrt`` module.
        feeder: the batch source.
        cache: where to persist the scales. `None` means calibrate every time.

    Returns:
        A calibrator to hand to :meth:`IBuilderConfig.int8_calibrator`. It also carries
        ``feeder`` and ``cache`` as attributes so a caller can assert afterwards whether
        calibration actually ran.

    Raises:
        BackendUnavailableError: torch is not installed, so there is nowhere to stage batches.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - needs a machine with trt and no torch
        raise BackendUnavailableError(
            "INT8 calibration stages batches in device memory through torch; install the "
            "'torch' extra alongside tensorrt"
        ) from exc

    resolved_cache = cache if cache is not None else CalibrationCache(None)

    class _EntropyCalibrator(trt.IInt8EntropyCalibrator2):  # type: ignore[misc, name-defined]
        """Feeds :class:`CalibrationBatchFeeder` to TensorRT, one device pointer at a time."""

        def __init__(self) -> None:
            super().__init__()
            self.feeder = feeder
            self.cache = resolved_cache
            self._buffer = torch.empty(
                tuple(feeder.batch_shape), dtype=torch.float32, device="cuda"
            )

        def get_batch_size(self) -> int:
            return self.feeder.batch_size

        def get_batch(self, names: Sequence[str], *_: Any) -> list[int] | None:
            """One device pointer per input name, or `None` to end calibration.

            The same buffer every time. TensorRT consumes a batch before asking for the next
            one, so overwriting is safe and is what keeps calibration from allocating a
            gigabyte of device memory for a calibration set that is a gigabyte on disk.
            """
            batch = self.feeder.next_batch()
            if batch is None:
                return None
            self._buffer.copy_(torch.from_numpy(batch))
            return [int(self._buffer.data_ptr())] * max(len(names), 1)

        def read_calibration_cache(self) -> bytes | None:
            return self.cache.read()

        def write_calibration_cache(self, blob: bytes) -> None:
            self.cache.write(blob)

    return _EntropyCalibrator()
