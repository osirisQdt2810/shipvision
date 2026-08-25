"""Camera-motion compensation, one estimator per file, selected from the registry.

Importing this package registers every built-in estimator. The optical-flow one registers too
— registration does not import OpenCV, only construction does, so a machine without it can
still list what exists and gets a typed
:class:`~shipvision.errors.BackendUnavailableError` if it asks for it.
"""

from shipvision.mot.motion.cmc.base import CAMERA_MOTION, IDENTITY_AFFINE, CameraMotionEstimator
from shipvision.mot.motion.cmc.external import ExternalCameraMotion
from shipvision.mot.motion.cmc.none import NoCameraMotion
from shipvision.mot.motion.cmc.sparse_flow import SparseOpticalFlowCameraMotion

__all__ = [
    "CAMERA_MOTION",
    "IDENTITY_AFFINE",
    "CameraMotionEstimator",
    "ExternalCameraMotion",
    "NoCameraMotion",
    "SparseOpticalFlowCameraMotion",
]
