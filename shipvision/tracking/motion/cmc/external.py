"""Camera motion supplied by the caller rather than recovered from pixels."""

from __future__ import annotations

import numpy as np

from shipvision.errors import ConfigurationError, TrackingError
from shipvision.tracking.motion.cmc.base import (
    CAMERA_MOTION,
    IDENTITY_AFFINE,
    CameraMotionEstimator,
)

__all__ = ["ExternalCameraMotion"]


@CAMERA_MOTION.register("external", aliases=("telemetry", "ptz"))
class ExternalCameraMotion(CameraMotionEstimator):
    """Uses the affine the caller pushed in, once, then falls back to identity.

    Not a test double: it is the *best* estimator available on a real PTZ installation. The
    head knows its own pan and tilt to a fraction of a degree and publishes them at frame
    rate, and no optical-flow estimate over a scene of moving water will ever match that. The
    same applies to a hull-mounted camera with an IMU. Recovering from pixels what the
    hardware already reports is work done twice, badly.

    Each pushed affine is consumed by exactly one frame. That is deliberate: a stale affine
    silently applied to a later frame moves every prediction by a motion that did not happen,
    and identity loss from over-compensation looks exactly like identity loss from
    under-compensation.

    Args:
        strict: raise when a frame arrives with no affine pushed for it, instead of assuming
            the camera was still. Turn it on when telemetry is supposed to be present on
            every frame, so a dropped telemetry packet is an incident rather than a slow
            degradation nobody notices.
    """

    def __init__(self, *, strict: bool = False) -> None:
        self._pending: np.ndarray | None = None
        self._strict = strict

    def push(self, affine: np.ndarray) -> None:
        """Supply the motion for the next frame, as a ``(2, 3)`` previous-to-current affine."""
        matrix = np.asarray(affine, dtype=np.float32)
        if matrix.shape != (2, 3):
            raise ConfigurationError(
                f"a camera-motion affine must be (2, 3), got {matrix.shape}"
            )
        self._pending = matrix

    def estimate(self, image: np.ndarray | None) -> np.ndarray:
        pending, self._pending = self._pending, None
        if pending is None:
            if self._strict:
                raise TrackingError(
                    "no camera motion was pushed for this frame; push() before update() or "
                    "build this estimator with strict=False to treat silence as no motion"
                )
            return IDENTITY_AFFINE.copy()
        return pending

    def reset(self) -> None:
        self._pending = None
