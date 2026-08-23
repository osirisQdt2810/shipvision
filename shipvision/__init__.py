"""Maritime computer-vision algorithms: detection, re-identification, tracking, MTMC.

The algorithm half of `ShipInfer <https://github.com/osirisQdt2810/shipinfer>`_. ShipInfer
owns the *system* — reading fifty RTSP cameras, scheduling work across sixteen GPUs, serving
and observing. This library owns the *algorithms*, and it imports nothing from ShipInfer.

That split is not organisational. An algorithm is judged by HOTA, IDF1, rank-1 and mAP on
recorded footage, and that measurement has to be runnable in seconds with no GPU, no model
repository and no engine to load — or it will not be run, and the algorithm that shipped
first will win by default instead of by evidence.

**Every algorithm exists at least twice.** A compiled C++/CUDA/HIP backend for production,
and a readable numpy backend that the compiled one is checked against. Both register in the
same registry under the same name, so a deployment picks by config and a test can compare
them::

    from shipvision import TRACKERS

    fast = TRACKERS.build("bytetrack", backend="native")
    reference = TRACKERS.build("bytetrack", backend="python")

Omit ``backend`` and the fastest available one is chosen — falling back to numpy, which is
always there. A fused kernel nobody can compare against is a fused kernel nobody can trust.
"""

from __future__ import annotations

from shipvision.errors import (
    BackendUnavailableError,
    ConfigurationError,
    DimensionMismatchError,
    InferenceError,
    ModelLoadError,
    ShipVisionError,
    TrackingError,
)
from shipvision.registry import NATIVE, PYTHON, TENSORRT, TORCH, Registry
from shipvision.types import (
    Detection,
    Detections,
    Embedding,
    Frame,
    FrameTag,
    GlobalTrack,
    Track,
    TrackState,
    cxcyah_to_xyxy,
    cxcywh_to_xyxy,
    iou_matrix,
    xyxy_to_cxcyah,
    xyxy_to_cxcywh,
)

__version__ = "0.1.0"

__all__ = [
    "NATIVE",
    "PYTHON",
    "TENSORRT",
    "TORCH",
    "BackendUnavailableError",
    "ConfigurationError",
    "Detection",
    "Detections",
    "DimensionMismatchError",
    "Embedding",
    "Frame",
    "FrameTag",
    "GlobalTrack",
    "InferenceError",
    "ModelLoadError",
    "Registry",
    "ShipVisionError",
    "Track",
    "TrackState",
    "TrackingError",
    "__version__",
    "cxcyah_to_xyxy",
    "cxcywh_to_xyxy",
    "iou_matrix",
    "xyxy_to_cxcyah",
    "xyxy_to_cxcywh",
]
