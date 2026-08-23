"""Fused image pre-processing and NMS: letterbox, crop, colour, normalise, suppress.

Every stage in this library starts by fitting an image into a network input and ends by
suppressing overlapping boxes, so these are the two operations whose conventions everything
else inherits. Read :mod:`shipvision.imgproc.geometry` before using them — it states the
half-pixel sampling rule, the rounding of the resized extent and which side gets the odd pad
pixel; :mod:`shipvision.imgproc.base` adds the colour and normalisation rule and the contract
itself. All three backends implement exactly those.

    from shipvision.imgproc import build_image_ops

    ops = build_image_ops()                         # fastest available, numpy as the floor
    batch, geometry = ops.letterbox(frames, (640, 640))
    keep = ops.nms(boxes, scores, iou_threshold=0.45)
    boxes = geometry[0].invert_boxes(boxes[keep])    # back to source pixels

Layout::

    imgproc/
    ├── geometry.py     LetterboxGeometry and the sampling conventions — what a *consumer*
    │                   needs, so it sits above the backends rather than inside one
    ├── base.py         the ImageOps contract, input validation, colour and normalisation
    ├── registry.py     IMGPROC
    ├── nms/            five suppression methods and the rules they share
    └── backends/       numpy (the oracle), torch, native

Three backends, one algorithm name:

``python``
    :class:`~shipvision.imgproc.backends.numpy_ops.NumpyImageOps`. The oracle. Always
    importable, needs no build and no device, and is what the parity tests compare against.
``torch``
    :class:`~shipvision.imgproc.backends.torch_ops.TorchImageOps`. ``F.interpolate``,
    ``F.grid_sample`` and ``torchvision.ops.nms`` — no hand-rolled resize, no hand-rolled
    suppression. Registered lazily, so torch is only imported if something asks for it.
``native``
    :class:`~shipvision.imgproc.backends.native_ops.NativeImageOps`. The fused CUDA/HIP kernels
    in ``shipvision._C``, which touch a 1080p frame once instead of four times. Registered
    unconditionally; construction is what fails when there is no build.
"""

from __future__ import annotations

from typing import Any

from shipvision.errors import BackendUnavailableError, ConfigurationError
from shipvision.imgproc.backends.native_ops import NativeImageOps, native_available
from shipvision.imgproc.backends.numpy_ops import NumpyImageOps
from shipvision.imgproc.base import (
    DEFAULT_MEAN,
    DEFAULT_PAD_VALUE,
    DEFAULT_STD,
    ImageOps,
)
from shipvision.imgproc.geometry import LetterboxGeometry
from shipvision.imgproc.nms import METHODS, SOFT_METHODS, suppress
from shipvision.imgproc.registry import IMGPROC
from shipvision.registry import TORCH

# Two eager imports above, one lazy registration below — and that asymmetry is the point.
# `numpy_ops` and `native_ops` import cleanly with no build and no device, so importing them
# here is free and their registration is unconditional. Importing torch costs about a second
# and several hundred megabytes of address space, which the offline test tier and every
# numpy-only process must not pay; `register_lazy` claims the (name, backend) key now and
# imports the module the first time somebody actually asks for that backend.
IMGPROC.register_lazy(
    "default", "shipvision.imgproc.backends.torch_ops:TorchImageOps", backend=TORCH
)

__all__ = [
    "DEFAULT_MEAN",
    "DEFAULT_PAD_VALUE",
    "DEFAULT_STD",
    "IMGPROC",
    "METHODS",
    "SOFT_METHODS",
    "ImageOps",
    "LetterboxGeometry",
    "NativeImageOps",
    "NumpyImageOps",
    "TorchImageOps",
    "build_image_ops",
    "native_available",
    "suppress",
]


def build_image_ops(
    name: str = "default", *, backend: str | None = None, **kwargs: Any
) -> ImageOps:
    """Build an :class:`ImageOps`, falling back down the preference order if need be.

    ``IMGPROC.build(name)`` picks the fastest *registered* backend, but registration is not
    availability: the native backend registers on a laptop with no CUDA toolchain, and only its
    constructor knows that. So this walks the preference order and treats
    :class:`~shipvision.errors.BackendUnavailableError` as "try the next one" — which is what
    makes "fastest available, numpy as the floor" true at run time rather than just at import
    time.

    Naming a ``backend`` explicitly disables the fallback. A deployment that asked for
    ``native`` and silently got numpy would be a large throughput regression reported as a
    successful start-up, so being told is the only acceptable behaviour.

    Args:
        name: the algorithm name. Only ``"default"`` exists today.
        backend: pin a backend, or ``None`` to resolve one.
        **kwargs: forwarded to the constructor. Backend-specific, so pass them only together
            with an explicit ``backend`` — a stray ``device_index`` reaching the numpy backend
            raises ``TypeError``, loudly, rather than being dropped.

    Raises:
        ConfigurationError: no such algorithm, or no such backend for it.
        BackendUnavailableError: nothing registered under ``name`` could actually be built.
    """
    if backend is not None:
        return IMGPROC.build(name, backend=backend, **kwargs)

    candidates = IMGPROC.backends(name)
    if not candidates:
        raise ConfigurationError(f"unknown image ops {name!r}; available: {IMGPROC.names()}")

    failures: list[str] = []
    for candidate in candidates:
        try:
            return IMGPROC.build(name, backend=candidate, **kwargs)
        except BackendUnavailableError as exc:
            failures.append(f"{candidate}: {exc}")
    raise BackendUnavailableError(
        f"no usable backend for image ops {name!r}. Tried " + "; ".join(failures)
    )


def __getattr__(attribute: str) -> Any:
    """Resolve ``TorchImageOps`` without importing torch at package import time.

    PEP 562, for the same reason as ``register_lazy`` above. It goes through the registry
    rather than importing the module directly, because resolving a lazy target is also what
    stamps ``name`` and ``backend`` onto the class — a direct import would hand back a class
    that cannot say what it is.
    """
    if attribute == "TorchImageOps":
        return IMGPROC.get("default", TORCH)
    raise AttributeError(f"module {__name__!r} has no attribute {attribute!r}")
