"""The typed failure vocabulary.

Every error raised out of this library descends from :class:`ShipVisionError`, so a caller
can tell "the library refused this input" from "numpy raised" from "the GPU is gone". A
server that cannot make that distinction cannot decide whether to retry, drop the frame, or
take the worker out of rotation — and those are three different operational events.
"""

from __future__ import annotations

__all__ = [
    "BackendUnavailableError",
    "ConfigurationError",
    "DimensionMismatchError",
    "InferenceError",
    "ModelLoadError",
    "ShipVisionError",
    "TrackingError",
]


class ShipVisionError(Exception):
    """Base of every error this library raises."""


class ConfigurationError(ShipVisionError):
    """A component was built with arguments that cannot work.

    Raised at construction, never at frame 40 000. A typo in a config file must stop the
    process at start-up; discovering it mid-stream costs a camera's worth of footage.
    """


class DimensionMismatchError(ShipVisionError):
    """Tensors that must agree on a dimension do not.

    Its own class because it is the failure that actually happens in production: two models
    in one pipeline, one emitting 512-d embeddings and one 1280-d, where the only other
    symptom is a similarity matrix that cannot be formed — or worse, one that can, because
    a broadcast quietly succeeded.
    """


class BackendUnavailableError(ShipVisionError):
    """The requested backend's runtime is not installed on this machine.

    Distinct from :class:`ModelLoadError`: "there is no TensorRT here" is a deployment
    problem, "this engine is for a different GPU" is an artefact problem, and the operator
    fixes them in different places.
    """


class ModelLoadError(ShipVisionError):
    """The artefact exists but cannot be used as claimed."""


class InferenceError(ShipVisionError):
    """A model ran and failed. Never signalled by returning empty outputs."""


class TrackingError(ShipVisionError):
    """A tracker reached a state it cannot continue from."""
