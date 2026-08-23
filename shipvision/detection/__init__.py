"""Object detection: frames in, tagged boxes in original image pixels out.

The package is four separable pieces, and the separation is what makes the risky parts
testable with no hardware:

:mod:`~shipvision.detection.base`
    The :class:`~shipvision.detection.base.Detector` contract and :data:`DETECTORS`. Two rules
    live here — ``input_hw`` is discovered from the artefact, and the tag survives every path
    including the error path (:class:`~shipvision.detection.base.DetectionError` carries it).
:mod:`~shipvision.detection.heads`
    Output tensor to :class:`~shipvision.types.Detections`, one head per model family. Pure
    numpy, so a synthesised output whose right answer you chose is a complete test of a decode
    — which is where the invisible bugs are: the confidence boundary, the class rounding, the
    letterbox inverse.
:mod:`~shipvision.detection.artefact`
    The shared three-layer path — letterbox, execute, decode — so a backend is only the part
    that is genuinely different.
:mod:`~shipvision.detection.backends`
    ``mock`` (deterministic, hardware-free), ``torch`` and ``tensorrt``.

Plus :mod:`~shipvision.detection.engine_build`, which turns an ONNX into an engine **in
process** — no filename parsing and no subprocess, unlike every reference this replaces.

    from shipvision.detection import DETECTORS

    detector = DETECTORS.build("yolo26", backend="tensorrt", path="yolo26n.engine")
    for detections in detector.detect(frames):
        print(detections.tag, len(detections), detections.boxes)

The mock is what lets the tracking, MTMC and pipeline lanes test end to end with no model at
all: it synthesises a per-camera scene with smooth motion, keyed on ``(camera_id, frame_id)``
so the same frame always gives the same detections::

    detector = DETECTORS.build("mock", objects=(2, 6), jitter=1.5, fail_every=None)

Note what is deliberately *not* here: a "build the best available detector" helper. ``mock``
would be its fallback, and a missing engine quietly becoming a mock is a deployment that
reports a successful start-up and detects nothing. Ask for what you want.
"""

from __future__ import annotations

from shipvision.detection.artefact import ArtefactDetector
from shipvision.detection.backends import MockDetector
from shipvision.detection.base import (
    DETECTORS,
    DetectionError,
    Detector,
    empty_detections,
    frame_hw,
    frame_image,
)
from shipvision.detection.engine_build import OptimisationProfile, build_engine
from shipvision.detection.heads import (
    HEADS,
    Candidates,
    DetectionHead,
    Yolo26Head,
    Yolo26SegHead,
    bilinear_resize,
    box_crop_bounds,
    build_detections,
    fuse_mask_logits,
    resolve_head,
    round_class_ids,
    unpad_mask,
)

__all__ = [
    "DETECTORS",
    "HEADS",
    "ArtefactDetector",
    "Candidates",
    "DetectionError",
    "DetectionHead",
    "Detector",
    "MockDetector",
    "OptimisationProfile",
    "TensorRTDetector",
    "TorchDetector",
    "Yolo26Head",
    "Yolo26SegHead",
    "bilinear_resize",
    "box_crop_bounds",
    "build_detections",
    "build_engine",
    "empty_detections",
    "frame_hw",
    "frame_image",
    "fuse_mask_logits",
    "resolve_head",
    "round_class_ids",
    "unpad_mask",
]


def __getattr__(attribute: str) -> object:
    """Resolve the artefact backends without importing torch or tensorrt at import time.

    PEP 562, for the same reason ``register_lazy`` is used below: ``import shipvision.detection``
    must not cost a second of torch import in a process that only ever builds the mock. It goes
    through the registry rather than importing the module directly, because resolving a lazy
    target is also what stamps ``name`` and ``backend`` onto the class.
    """
    lazy = {"TensorRTDetector": "tensorrt", "TorchDetector": "torch"}
    if attribute in lazy:
        return DETECTORS.get("yolo26", lazy[attribute])
    raise AttributeError(f"module {__name__!r} has no attribute {attribute!r}")
