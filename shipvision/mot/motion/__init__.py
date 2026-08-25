"""Motion: how a track moves, and how the camera moves under it.

Two different things that a tracker must not confuse, which is exactly why they share a
directory. The Kalman filter models the *object's* motion in image coordinates; camera-motion
compensation removes the part of that apparent motion which the camera caused. A tracker that
folds the second into the first — by widening the filter's noise until a pan looks like
plausible object motion — buys a few frames of survival at the cost of a filter that can no
longer tell an object from a wobble.
"""

from shipvision.mot.motion.cmc import (
    CAMERA_MOTION,
    IDENTITY_AFFINE,
    CameraMotionEstimator,
    ExternalCameraMotion,
    NoCameraMotion,
    SparseOpticalFlowCameraMotion,
)
from shipvision.mot.motion.kalman import CHI2_INV_95_4DOF, KalmanFilter

__all__ = [
    "CAMERA_MOTION",
    "CHI2_INV_95_4DOF",
    "IDENTITY_AFFINE",
    "CameraMotionEstimator",
    "ExternalCameraMotion",
    "KalmanFilter",
    "NoCameraMotion",
    "SparseOpticalFlowCameraMotion",
]
