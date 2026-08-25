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

#: The registries, reachable from the top level but resolved on first access.
#:
#: `from shipvision import TRACKERS` has to work — it is the documented entry point — but
#: importing the families eagerly would defeat the point of the library importing on a
#: laptop with no GPU: `detection` reaches for TensorRT, `mtmc` for scipy and cv2, `imgproc`
#: for torch. Each of those is lazy *within* its family, but merely importing the family to
#: read its registry would run those lazy registrations' module bodies.
#:
#: PEP 562 module `__getattr__` defers the import to the attribute access that needs it, so
#: `import shipvision` stays free and `shipvision.TRACKERS` costs exactly the one family.
#: Every entry must actually resolve — `tests/test_package.py` reads this table rather than
#: keeping its own list, so declaring a family before it exists fails the suite instead of
#: waiting to fail a user.
_REGISTRY_HOMES: dict[str, str] = {
    "IMGPROC": "shipvision.imgproc",
    # Deliberately empty here. Every entry must resolve — `tests/test_package.py` reads this
    # table rather than keeping its own list — so a family declared before it exists fails the
    # suite instead of waiting to fail a user. Each package therefore adds its own line when
    # it lands, which is also why a package's pull request touches this file: registering at
    # the top level is genuinely part of shipping the family, not incidental bookkeeping.
}


def __getattr__(name: str) -> object:
    """Resolve a registry on first access. See :data:`_REGISTRY_HOMES`."""
    home = _REGISTRY_HOMES.get(name)
    if home is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    try:
        module = import_module(home)
    except ImportError as error:  # a family whose own dependencies are absent
        raise AttributeError(
            f"{name} lives in {home}, which could not be imported: {error}"
        ) from error
    value = getattr(module, name)
    globals()[name] = value  # cache, so the second access is a plain global lookup
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_REGISTRY_HOMES))


__all__ = [
    "IMGPROC",
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
