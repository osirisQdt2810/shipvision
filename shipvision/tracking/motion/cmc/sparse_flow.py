"""Camera motion from sparse optical flow. The image-based estimator BoT-SORT describes.

Track a few hundred corners from the previous frame into this one with a pyramidal
Lucas-Kanade solver, then fit the partial affine — rotation, uniform scale, translation —
that best explains where they went. Four degrees of freedom rather than six because a camera
rotating and zooming produces a similarity transform; allowing shear lets the fit absorb the
motion of the *objects* into the "camera" motion, which then moves every prediction by
whatever the largest moving thing in frame did.

Both halves are OpenCV's. ``goodFeaturesToTrack``, ``calcOpticalFlowPyrLK`` and
``estimateAffinePartial2D`` are decades-tuned, SIMD, and RANSAC-hardened; a hand-written
corner tracker here would be slower, longer and worse, and would need its own tests to say
anything about a problem that is not this library's problem.

OpenCV is an optional dependency. This module imports without it and the class refuses to
construct without it, which is how the offline test tier stays installable from ``numpy``
alone.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from shipvision.errors import BackendUnavailableError, ConfigurationError, TrackingError
from shipvision.tracking.motion.cmc.base import (
    CAMERA_MOTION,
    IDENTITY_AFFINE,
    CameraMotionEstimator,
)

__all__ = ["SparseOpticalFlowCameraMotion"]


def _load_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - exercised only where cv2 is absent
        raise BackendUnavailableError(
            "sparse-flow camera-motion compensation needs OpenCV; install "
            "shipvision[vision], or use the 'none' or 'external' estimator"
        ) from exc
    return cv2


@CAMERA_MOTION.register("sparse_flow", aliases=("optical_flow", "lk", "cmc"))
class SparseOpticalFlowCameraMotion(CameraMotionEstimator):
    """Lucas-Kanade flow on shi-Tomasi corners, fitted to a partial affine.

    Args:
        downscale: factor to shrink the frame by before tracking. The default of two is not
            a quality compromise: camera motion is a global, low-frequency signal, halving
            each side quarters the work, and the recovered translation is scaled back up
            exactly. At 1000 frames a second the estimator has a budget of microseconds.
        max_corners: how many features to follow. A few hundred is plenty — the fit has four
            parameters and RANSAC needs outliers to be a minority, not to be rare.
        quality: ``goodFeaturesToTrack`` quality level, relative to the strongest corner.
        min_distance: minimum spacing between corners, in downscaled pixels. Spreading them
            out matters more than their individual strength: a hundred corners on one moving
            ship describe that ship's motion, not the camera's.
        min_inliers: refuse to report a motion supported by fewer than this many inliers.
            Below it the honest answer is "unknown", and for a camera-motion term "unknown"
            must mean identity — over-compensating on a bad fit loses every identity at once,
            which is strictly worse than not compensating at all.

    Known limitation: corners found on the tracked objects themselves describe *their* motion,
    not the camera's. RANSAC discards them as long as the background is the majority of the
    frame, which it is for the maritime scenes this is aimed at; a harbour view filled edge to
    edge with one moving hull would need the detections masked out before corner selection,
    and that needs the boxes plumbed into :meth:`estimate`, which the contract does not carry
    today.
    """

    def __init__(
        self,
        *,
        downscale: int = 2,
        max_corners: int = 300,
        quality: float = 0.01,
        min_distance: int = 8,
        min_inliers: int = 8,
    ) -> None:
        self._cv2 = _load_cv2()
        if downscale < 1:
            raise ConfigurationError(f"downscale must be >= 1, got {downscale}")
        self._downscale = downscale
        self._max_corners = max_corners
        self._quality = quality
        self._min_distance = min_distance
        self._min_inliers = min_inliers
        self._previous: np.ndarray | None = None
        self._previous_corners: np.ndarray | None = None

    def estimate(self, image: np.ndarray | None) -> np.ndarray:
        if image is None:
            raise TrackingError(
                "the sparse-flow estimator needs the frame; pass image= to update(), or "
                "build the tracker with cmc='none' if pixels are not available"
            )
        cv2 = self._cv2
        frame = self._prepare(image)
        previous, previous_corners = self._previous, self._previous_corners
        self._previous = frame
        self._previous_corners = self._corners(frame)

        too_few = previous_corners is None or len(previous_corners) < self._min_inliers
        if previous is None or too_few:
            return IDENTITY_AFFINE.copy()

        moved, status, _error = cv2.calcOpticalFlowPyrLK(
            previous, frame, previous_corners, None
        )
        if moved is None or status is None:
            return IDENTITY_AFFINE.copy()
        kept = status.reshape(-1).astype(bool)
        if int(kept.sum()) < self._min_inliers:
            return IDENTITY_AFFINE.copy()

        affine, inliers = cv2.estimateAffinePartial2D(
            previous_corners[kept], moved[kept], method=cv2.RANSAC
        )
        if affine is None or inliers is None or int(inliers.sum()) < self._min_inliers:
            return IDENTITY_AFFINE.copy()

        result = np.asarray(affine, dtype=np.float32)
        # The rotation and scale block is dimensionless, so it needs no correction; only the
        # translation was measured in downscaled pixels.
        result[:, 2] *= self._downscale
        return result

    def _prepare(self, image: np.ndarray) -> np.ndarray:
        cv2 = self._cv2
        frame = np.asarray(image)
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        if self._downscale > 1:
            frame = cv2.resize(
                frame,
                (frame.shape[1] // self._downscale, frame.shape[0] // self._downscale),
                interpolation=cv2.INTER_AREA,
            )
        return frame

    def _corners(self, frame: np.ndarray) -> np.ndarray | None:
        corners = self._cv2.goodFeaturesToTrack(
            frame,
            maxCorners=self._max_corners,
            qualityLevel=self._quality,
            minDistance=self._min_distance,
        )
        return None if corners is None else np.asarray(corners, dtype=np.float32)

    def reset(self) -> None:
        self._previous = None
        self._previous_corners = None
