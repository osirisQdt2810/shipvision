"""The shared three-layer detector: letterbox, execute, decode.

Everything that is the same for every runtime lives here, so that a backend is only the part
that is genuinely different — how a tensor gets to a device and back. That split is the
reference's (``TRTDetector.h``: a pre-processor, a detector and a post-processor) and it earns
its keep twice over: one head decodes the output of TensorRT and of TorchScript, and one
runtime feeds a detection head or a segmentation head, without either knowing about the other.

The pre-processing is :mod:`shipvision.imgproc`'s and nothing here re-implements a step of it.
The letterbox returns the geometry it used and that object is carried to the decode, which is
what makes the box inverse exact rather than re-derived — see
:class:`~shipvision.imgproc.geometry.LetterboxGeometry`.

**Batches are chunked here, not in the backend.** A caller may hand over any number of frames;
a backend's device buffers are sized once at construction for ``max_batch`` and never
per-request, which is the allocation that a thousand frames a second cannot afford.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence

import numpy as np

from shipvision.detection.base import Detector, frame_image
from shipvision.detection.heads.base import DetectionHead
from shipvision.errors import ConfigurationError
from shipvision.imgproc import DEFAULT_PAD_VALUE, ImageOps, build_image_ops
from shipvision.types import Detections, Frame, FrameTag

__all__ = ["ArtefactDetector"]


class ArtefactDetector(Detector):
    """A detector that runs a trained artefact. Subclasses implement :meth:`_execute` only.

    Args:
        input_hw: the network input, ``(height, width)``, **as read from the artefact**. A
            subclass discovers it and passes it here; this class never guesses and never
            accepts a caller's value directly, which is what keeps the promise in
            :attr:`~shipvision.detection.base.Detector.input_hw` structural.
        head: the decode. Usually produced by
            :func:`shipvision.detection.heads.resolve_head` from the artefact's output shapes.
        max_batch: frames per execution. A larger batch than this is split; the device buffers
            are sized for this and nothing on the frame path allocates.
        image_ops: the pre-processing backend. `None` resolves the fastest available one, with
            numpy as the floor.
        image_ops_backend: pin that resolution by name — ``"python"``, ``"torch"``,
            ``"native"``. Ignored when ``image_ops`` is given.
        pad_value: letterbox fill, 0-255. 114 is the YOLO grey and the value these weights
            were trained with; changing it changes what the model sees in the bars.
        mean: per-channel mean in the 0-255 source scale, RGB order. `None` gives zeros.
        std: per-channel divisor, same scale and order. `None` gives 255, i.e. ``[0, 1]`` —
            which is what a YOLO export expects.
    """

    def __init__(
        self,
        *,
        input_hw: tuple[int, int],
        head: DetectionHead,
        max_batch: int = 1,
        image_ops: ImageOps | None = None,
        image_ops_backend: str | None = None,
        pad_value: int = DEFAULT_PAD_VALUE,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
    ) -> None:
        height, width = (int(v) for v in input_hw)
        if height <= 0 or width <= 0:
            raise ConfigurationError(f"input_hw must be positive, got {input_hw!r}")
        if max_batch <= 0:
            raise ConfigurationError(f"max_batch must be positive, got {max_batch}")
        if not 0 <= int(pad_value) <= 255:
            raise ConfigurationError(
                f"pad_value is a uint8 source-scale value and must be in [0, 255], got "
                f"{pad_value}"
            )

        self._input_hw: tuple[int, int] = (height, width)
        self._head = head
        self.max_batch = int(max_batch)
        self.pad_value = int(pad_value)
        self.mean = None if mean is None else tuple(float(v) for v in mean)
        self.std = None if std is None else tuple(float(v) for v in std)
        self._ops = (
            image_ops if image_ops is not None else build_image_ops(backend=image_ops_backend)
        )

    # -- introspection ----------------------------------------------------------------

    @property
    def input_hw(self) -> tuple[int, int]:
        return self._input_hw

    @property
    def head(self) -> DetectionHead:
        """The decode this detector was wired to. Exposed so a caller can retune a threshold
        without rebuilding the engine — the confidence threshold is the one detector
        parameter an operator changes at 3 a.m."""
        return self._head

    @property
    def image_ops(self) -> ImageOps:
        return self._ops

    # -- the frame path ---------------------------------------------------------------

    def detect(self, frames: Sequence[Frame]) -> list[Detections]:
        """See :meth:`~shipvision.detection.base.Detector.detect`."""
        if len(frames) == 0:
            return []
        results: list[Detections] = []
        for start in range(0, len(frames), self.max_batch):
            results.extend(self._detect_chunk(frames[start : start + self.max_batch]))
        return results

    def _detect_chunk(self, frames: Sequence[Frame]) -> list[Detections]:
        tags = [frame.tag for frame in frames]
        batch, geometries = self._ops.letterbox(
            [frame_image(frame) for frame in frames],
            self._input_hw,
            pad_value=self.pad_value,
            mean=self.mean,
            std=self.std,
        )
        outputs = self._execute(batch, tags)
        return self._head.decode(outputs, geometries, tags)

    @abc.abstractmethod
    def _execute(self, batch: np.ndarray, tags: Sequence[FrameTag]) -> list[np.ndarray]:
        """Run the artefact over one ``(n, 3, h, w)`` float32 batch.

        Args:
            batch: pre-processed and already the network's input extent. At most
                ``max_batch`` rows.
            tags: the frames in the batch, in row order. Passed in so a failure can name the
                frame it happened on — a
                :class:`~shipvision.detection.base.DetectionError` carries the tag, and a
                backend is the only layer that knows which row was being executed when the
                driver returned an error.

        Returns:
            The artefact's outputs, in its own order, as host arrays with the frame axis
            leading. One row per input row.

        Raises:
            DetectionError: the artefact ran and failed.
        """
