"""Camera-motion compensation: the contract, and why it is a separate family.

BoT-SORT's headline change over ByteTrack is that it stops assuming the camera is bolted
down. On a fixed camera a track's prediction is wrong only by however much the object
deviated from constant velocity; on a panning one it is wrong by the pan, for *every* track
at once, so the whole association fails on the same frame and the tracker re-births the
entire scene. That is not a subtle accuracy loss, it is a total identity loss, and it is the
normal case for a PTZ head or a camera on a moving vessel.

How the motion is *measured* is a genuinely separate question from how it is *used*, and it
has several right answers depending on the deployment: sparse optical flow from the pixels, a
homography from a known ground plane, or simply reading the pan-tilt encoder, which a real
PTZ installation already publishes and which no image-based estimator will ever beat. So this
is a registry, not a function, and the default is a no-op — a tracker that silently invented
a camera motion it could not measure would be worse than one that assumed none.

The convention, stated once: an estimator returns a ``(2, 3)`` affine that maps a point in
the **previous** frame to where it appears in the **current** one. That direction is the
useful one, because what needs warping is last frame's prediction into this frame's
coordinates.
"""

from __future__ import annotations

import abc
from typing import ClassVar

import numpy as np

from shipvision.registry import Registry

__all__ = ["CAMERA_MOTION", "IDENTITY_AFFINE", "CameraMotionEstimator"]

IDENTITY_AFFINE: np.ndarray = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
"""No motion. Returned as a fresh copy by every estimator, never shared — a caller that
scaled the module-level array in place would silently change what "no motion" means."""


class CameraMotionEstimator(abc.ABC):
    """Estimates the frame-to-frame image motion of the camera itself."""

    name: ClassVar[str] = "abstract"
    backend: ClassVar[str] = "python"

    @abc.abstractmethod
    def estimate(self, image: np.ndarray | None) -> np.ndarray:
        """The ``(2, 3)`` affine from the previous frame's coordinates to this frame's.

        Args:
            image: the current frame, HWC or HW. `None` is allowed and means "no pixels this
                time" — an estimator that needs them must raise rather than return identity,
                because silently reporting "the camera did not move" is indistinguishable
                from a correct answer and is wrong on exactly the frames that matter.

        Returns:
            ``(2, 3)`` float32. Identity on the first frame, since there is nothing to
            compare against yet.
        """

    def reset(self) -> None:
        """Forget the reference frame. Called when the stream is discontinuous."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"


CAMERA_MOTION: Registry[CameraMotionEstimator] = Registry("camera-motion")
