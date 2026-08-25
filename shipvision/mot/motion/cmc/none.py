"""The default: assume the camera is bolted down."""

from __future__ import annotations

import numpy as np

from shipvision.mot.motion.cmc.base import CAMERA_MOTION, IDENTITY_AFFINE, CameraMotionEstimator

__all__ = ["NoCameraMotion"]


@CAMERA_MOTION.register("none", aliases=("off", "static", "identity"))
class NoCameraMotion(CameraMotionEstimator):
    """Always identity.

    The default rather than an omission. Most of the fifty cameras in the target deployment
    are fixed, estimating motion from their pixels costs a pyramidal optical flow per frame,
    and the answer would be a small nonzero number produced by waves and rain — which is
    worse than zero, because it moves every prediction by noise.
    """

    def estimate(self, image: np.ndarray | None) -> np.ndarray:
        return IDENTITY_AFFINE.copy()
