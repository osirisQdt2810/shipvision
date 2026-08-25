"""Using a camera-to-ground-plane homography: the value type, and the projection.

Two cameras watching the same quay from opposite ends produce boxes that share no pixel
coordinates at all. What they do share is the ground: project both boxes' foot points onto
one map and two views of the same person land in the same place, while two different people
do not. That projection is a homography, and it is the only thing that makes a spatial gate
possible.

**This half is pure numpy and runs per instant; fitting is a different job.** Applying a
homography is a 3x3 matrix product over a few hundred points, on the frame path, on every
deployment. Fitting one needs OpenCV, happens once when somebody clicks calibration points,
and is allowed to be slow and to fail loudly — see
:mod:`shipvision.mtmc.topology.calibration`. Keeping them in separate modules is what lets a
deployment that receives its matrices already calibrated never import cv2 at all.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from shipvision.errors import ConfigurationError

__all__ = ["GroundPlane", "Homography", "project"]


def project(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """``(n, 2)`` points through a ``(3, 3)`` homography, back to ``(n, 2)``.

    Vectorised over the whole set rather than looped per point. This runs once per
    synchronised group over every track in flight, and a Python loop here would cost more
    than the clustering it feeds.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if pts.shape[-1] != 2:
        raise ConfigurationError(f"points must be (n, 2), got shape {pts.shape}")
    homogeneous = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    projected = homogeneous @ np.asarray(matrix, dtype=np.float64).T
    scale = projected[:, 2:3]
    # A zero third component means the point maps to infinity — the horizon line of this
    # homography. Clamping rather than dividing by zero keeps the result finite and very
    # far away, which is what "above the horizon" should mean to a spatial gate: never
    # close to anything. NaN would instead poison every comparison it touches.
    safe = np.where(np.abs(scale) < 1e-12, np.sign(scale) * 1e-12 + 1e-12, scale)
    return (projected[:, :2] / safe).astype(np.float32)


@dataclass(slots=True, frozen=True)
class Homography:
    """One camera's mapping onto the shared ground plane, plus its calibration domain.

    ``camera_width``/``camera_height`` are the frame size the matrix was *calibrated* at, and
    they are why this is a class rather than a bare 3x3 array. A homography fitted on 1080p
    stills does not apply to the 720p stream the same camera serves at night: the pixel
    coordinates differ by a factor of 1.5 and the projection lands somewhere else on the map,
    silently. Keeping the calibration size next to the matrix lets the projection rescale
    into it, so a resolution change stops being a correctness bug.

    ``max_error`` is what :func:`~shipvision.mtmc.topology.calculate_homography` measured,
    carried along so that a consumer can refuse a camera whose calibration is worse than the
    threshold it is about to gate with. `None` means nobody measured — which is different from
    "measured and fine".
    """

    matrix: np.ndarray
    camera_width: int = 0
    camera_height: int = 0
    map_width: int = 0
    map_height: int = 0
    max_error: float | None = None

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ConfigurationError(f"a homography is a 3x3 matrix, got shape {matrix.shape}")
        if not np.all(np.isfinite(matrix)):
            raise ConfigurationError("a homography must be finite; got NaN or inf")
        if abs(float(np.linalg.det(matrix))) < 1e-12:
            raise ConfigurationError(
                "this homography is singular — it collapses the image onto a line or a "
                "point, so every track on this camera would project to the same place"
            )
        object.__setattr__(self, "matrix", matrix)

    def to_calibration_domain(
        self, points: np.ndarray, *, frame_width: int, frame_height: int
    ) -> np.ndarray:
        """Rescale image points from the live frame size to the calibrated one.

        A no-op when the calibration size was not recorded, on the assumption that the caller
        is already handing over points in the matrix's own domain.
        """
        if self.camera_width <= 0 or self.camera_height <= 0:
            return np.asarray(points, dtype=np.float64)
        scale = np.array(
            [self.camera_width / frame_width, self.camera_height / frame_height],
            dtype=np.float64,
        )
        return np.asarray(points, dtype=np.float64) * scale

    def project(
        self, points: np.ndarray, *, frame_width: int = 0, frame_height: int = 0
    ) -> np.ndarray:
        """``(n, 2)`` image points to ``(n, 2)`` ground-plane points."""
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        if frame_width > 0 and frame_height > 0:
            pts = self.to_calibration_domain(
                pts, frame_width=frame_width, frame_height=frame_height
            )
        return project(pts, self.matrix)

    def __str__(self) -> str:
        error = "unmeasured" if self.max_error is None else f"{self.max_error:.2f}"
        return (
            f"<Homography camera={self.camera_width}x{self.camera_height} "
            f"map={self.map_width}x{self.map_height} max_error={error}>"
        )


class GroundPlane:
    """The homographies for a camera group, and the cameras that have none.

    A camera without a homography is the normal case, not an error: a new camera goes live
    before anyone has clicked its calibration points, and a PTZ camera invalidates its own
    the moment it moves. So this answers :meth:`has` rather than raising, and the spatial gate
    above it treats "unknown" as "no spatial evidence" and falls back to appearance. The
    alternative — excluding an uncalibrated camera from the group — is worse and quieter: that
    camera's identities simply never merge with anyone, and nothing in the metrics says so.
    """

    def __init__(self, homographies: Mapping[str, Homography] | None = None) -> None:
        self._homographies: dict[str, Homography] = dict(homographies or {})
        for camera_id, homography in self._homographies.items():
            if not isinstance(homography, Homography):
                raise ConfigurationError(
                    f"camera {camera_id!r} maps to {type(homography).__name__}, not a "
                    f"Homography; wrap the raw 3x3 so its calibration domain travels with it"
                )

    def has(self, camera_id: str) -> bool:
        return camera_id in self._homographies

    def get(self, camera_id: str) -> Homography | None:
        return self._homographies.get(camera_id)

    def add(self, camera_id: str, homography: Homography) -> None:
        """Register or replace one camera's homography — a PTZ camera recalibrating."""
        if not isinstance(homography, Homography):
            raise ConfigurationError(f"expected a Homography, got {type(homography).__name__}")
        self._homographies[camera_id] = homography

    @property
    def cameras(self) -> tuple[str, ...]:
        return tuple(sorted(self._homographies))

    def __len__(self) -> int:
        return len(self._homographies)

    def __contains__(self, camera_id: object) -> bool:
        return isinstance(camera_id, str) and camera_id in self._homographies

    def __repr__(self) -> str:
        return f"<GroundPlane cameras={list(self.cameras)}>"
