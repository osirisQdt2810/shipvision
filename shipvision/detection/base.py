"""The detector contract: frames in, tagged detections out, in original image pixels.

Two parts of this interface are load-bearing rather than convenient, and both exist because
the reference implementation this package replaces got them wrong in a way that has no
symptom.

**``input_hw`` is discovered from the artefact, never configured.** A TensorRT engine states
its input extent in its input binding and the reference C++ reads it from there
(``BaseDetector::loadInputOutputInformation``) — and then a YAML file states it again, and
nothing checks that the two agree. A configured 640x640 against a 512x512 engine does not
crash: the letterbox produces a tensor of the configured size, TensorRT refuses it or, with a
dynamic profile, accepts it and every box comes back scaled by 640/512. So implementations
read the number from the artefact and raise :class:`~shipvision.errors.ModelLoadError` when a
caller passes one that disagrees.

**Boxes come back in original image pixels, un-letterboxed with the geometry that mapped
them.** Not network space, not normalised. The inverse is
:meth:`~shipvision.imgproc.geometry.LetterboxGeometry.invert_boxes` and nothing here
re-derives it — the reference recomputes ``gain = min(scaleH, scaleW)`` and the two pads by
hand in every post-processor (``Yolo26PostProcessor.cpp:118-125`` and again at
``Yolo26SegPostProcessor.cpp:117-119``), which is three copies of one rounding rule and
exactly the arithmetic that drifts on the one camera whose resolution rounds differently.

The three-layer shape — preprocess, execute, postprocess — is the reference's
(``TRTDetector.h``) and is worth keeping: it is what lets one head decode the output of two
runtimes, and one runtime feed two heads.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence

import numpy as np

from shipvision.errors import ConfigurationError, DimensionMismatchError, InferenceError
from shipvision.registry import PYTHON, Registry
from shipvision.types import Detections, Frame, FrameTag

__all__ = [
    "DETECTORS",
    "DetectionFailure",
    "Detector",
    "empty_detections",
    "frame_hw",
    "frame_image",
]


class DetectionFailure(InferenceError):
    """A detection failed, and it says *which frame* it failed on.

    A subclass rather than a bare :class:`~shipvision.errors.InferenceError` because the tag
    has to survive the error path too. ``(camera_id, frame_id)`` reaching the caller is what
    lets a server decrement the right camera's counter, decide whether one stream is sick or
    the GPU is, and avoid attributing the gap to a camera that was fine — and a message a
    human has to read with a regular expression is not that. The message still names the tag,
    so a log line is self-contained either way.

    Attributes:
        tag: the frame being processed when the failure happened, or `None` if the failure was
            not attributable to one frame (an engine-wide fault, say).
    """

    def __init__(self, message: str, *, tag: FrameTag | None = None) -> None:
        super().__init__(f"{message} (frame {tag})" if tag is not None else message)
        self.tag = tag


class Detector(abc.ABC):
    """Frames to detections. Registered in :data:`DETECTORS` under ``(name, backend)``.

    Implementations are stateful — an engine, a stream, a device — and one instance serves one
    worker thread for the life of the process. Sharing an instance between threads is not
    supported: the device buffers are per-instance by design, exactly as they are in
    :class:`~shipvision.imgproc.base.ImageOps`.
    """

    name: str = "detector"
    backend: str = PYTHON

    @property
    @abc.abstractmethod
    def input_hw(self) -> tuple[int, int]:
        """``(height, width)`` of the network input, discovered from the artefact.

        See the module docstring: a configured value that disagrees with the artefact is a
        silent correctness bug, so an implementation with an artefact reads it from there and
        fails at load when a caller's value contradicts it.
        """

    @abc.abstractmethod
    def detect(self, frames: Sequence[Frame]) -> list[Detections]:
        """One :class:`~shipvision.types.Detections` per :class:`~shipvision.types.Frame`.

        The result is positionally aligned with ``frames`` and each element carries that
        frame's own tag. A frame with nothing in it yields an empty ``Detections``, never
        `None` and never a silently shortened list — a list the caller has to re-align by
        guessing is how a detection ends up on the wrong camera.

        Args:
            frames: the batch. May be empty, which returns ``[]``.

        Returns:
            ``len(frames)`` results, boxes in **original image pixels**, xyxy float32.

        Raises:
            DetectionFailure: the model ran and failed. The frame it failed on is on the
                exception.
            ConfigurationError: a frame this implementation cannot read (no pixels and no
                declared extent).
        """

    def detect_one(self, frame: Frame) -> Detections:
        """One frame. For tests and single-shot calls, not the frame path.

        Batching is most of what makes an accelerator worth having — a thousand frames a
        second arrive as batches and must stay that way — so this is a convenience rather
        than the interface anything hot goes through.
        """
        return self.detect([frame])[0]

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} name={self.name!r} backend={self.backend!r} "
            f"input_hw={self.input_hw}>"
        )


#: The detector family. ``mock`` is deterministic and hardware-free; the artefact-driven
#: implementations share the name of the *model family* whose output layout they decode,
#: because that layout is what differs — ``yolo26`` on TensorRT and ``yolo26`` on TorchScript
#: are one algorithm at two speeds, which is the comparison a registry exists to allow.
#:
#: There is deliberately no "build the best available detector" helper here. ``mock`` would be
#: the fallback, and a missing engine silently becoming a mock is a deployment that reports a
#: successful start-up and detects nothing real. A missing engine must be an error.
DETECTORS: Registry[Detector] = Registry("detector")


# ------------------------------------------------------------------------ frame accessors


def frame_hw(frame: Frame) -> tuple[int, int]:
    """``(height, width)`` of a frame's *source* image, from whichever field has it.

    :class:`~shipvision.types.Frame` carries both an opaque ``image`` and an optional declared
    extent, because the point of hardware decode is that the pixels never reach host memory.
    So the extent is taken from the declared fields when they are set, and from the array
    otherwise — which is what lets a tracking or MTMC test drive the mock detector with
    ``Frame(tag, image=None, height=1080, width=1920)`` and no pixels at all.

    When both are present they must agree. A frame that says 1080x1920 while holding a
    540x960 array has been resized by something that did not update the tag, and every box
    derived from it is off by a factor of two with no other symptom.

    Raises:
        ConfigurationError: neither source is usable, or the two disagree.
    """
    declared = (int(frame.height), int(frame.width))
    array = frame.image if isinstance(frame.image, np.ndarray) else None
    actual = (int(array.shape[0]), int(array.shape[1])) if array is not None and array.ndim >= 2 else None

    if declared[0] > 0 and declared[1] > 0:
        if actual is not None and actual != declared:
            raise ConfigurationError(
                f"frame {frame.tag} declares {declared[0]}x{declared[1]} but holds a "
                f"{actual[0]}x{actual[1]} image; something resized the pixels without "
                f"updating the frame, and every box derived from it is scaled wrongly"
            )
        return declared
    if actual is not None:
        return actual
    raise ConfigurationError(
        f"frame {frame.tag} has neither pixels nor a declared height and width, so there is "
        f"no coordinate space to return boxes in"
    )


def frame_image(frame: Frame) -> np.ndarray:
    """The frame's pixels as an ``(h, w, 3)`` uint8 array, or a typed refusal.

    Validation is left to :mod:`shipvision.imgproc`, which owns the dtype and layout rule and
    states why it refuses rather than coerces. This only turns "the image is not host memory"
    into a sentence naming the frame, because a ``DimensionMismatchError`` from three layers
    down does not say which camera sent it.
    """
    if not isinstance(frame.image, np.ndarray):
        raise ConfigurationError(
            f"frame {frame.tag} holds {type(frame.image).__name__}, not a numpy array. This "
            f"detector preprocesses on the host; a device-resident frame needs a backend "
            f"that takes a device pointer"
        )
    return frame.image


def empty_detections(tag: FrameTag, height: int, width: int) -> Detections:
    """The answer for a frame with nothing in it. Never `None`, never an exception.

    An empty frame is ordinary input — most cameras are empty most of the time — and it still
    carries its tag and its frame extent, so a downstream tracker can age its tracks on a
    quiet frame instead of skipping it.
    """
    return Detections(tag=tag, items=[], height=int(height), width=int(width))


def validate_alignment(
    count: int, geometry_count: int, tag_count: int, *, what: str = "decode"
) -> None:
    """Refuse a batch whose outputs, geometries and tags do not describe the same frames.

    The failure this prevents is the worst one in the library: a result list one element
    shorter than the frame list re-aligns every subsequent tag by one, so every detection is
    attributed to the previous camera and all of them look real.
    """
    if not (count == geometry_count == tag_count):
        raise DimensionMismatchError(
            f"{what} got {count} output rows, {geometry_count} letterbox geometries and "
            f"{tag_count} tags. These must be the same length: a mismatch shifts every "
            f"result onto the wrong frame, and a shifted detection looks exactly like a real "
            f"one"
        )
