"""Camera-to-ground-plane geometry: the homography, and how much to trust it.

Two cameras watching the same quay from opposite ends produce boxes that share no pixel
coordinates at all. What they do share is the ground: project both boxes' foot points onto
one map and two views of the same person land in the same place, while two different people
do not. That projection is a homography, and it is the only thing that makes a spatial gate
possible.

**The error estimate is returned, not discarded.** :func:`calculate_homography` gives back
``(matrix, max_error)``, and the second half is the part most implementations drop. A
homography fitted to four hand-clicked points is exact *at those four points* by
construction, so the residual on the calibration set is always near zero and says nothing
about the matrix. What matters operationally is how wrong the projection is at a point that
was **not** used to fit it — because that is every point the tracker will ever project — and
the only cheap way to measure that is to hold each correspondence out in turn. The number
that comes back is in map units, so it can be compared directly against the spatial
threshold it is about to gate: an error of 30 with a threshold of 280 is a usable
calibration, an error of 400 is not, and without the estimate the second one looks exactly
like the first until identities start swapping.

**OpenCV is optional.** Fitting a homography needs ``cv2``; *using* one is a 3x3 matrix
product, which is numpy. So the import is lazy, and a deployment that receives its matrices
already calibrated — from a file, from a config service — never needs OpenCV installed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from shipvision.errors import BackendUnavailableError, ConfigurationError

__all__ = ["GroundPlane", "Homography", "calculate_homography", "project"]

#: Below this the DLT design matrix is rank-deficient in a direction it should not be, which
#: means the correspondences do not determine a homography — the classic case being points
#: that all lie on one line. ``cv2.findHomography`` does not report this: it happily returns
#: a matrix that maps the line correctly and is arbitrary everywhere off it, and the
#: reprojection error on the calibration points is *zero*. So the check is here.
_DEGENERACY_RATIO = 1e-6


def _load_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on the host
        raise BackendUnavailableError(
            "fitting a homography needs OpenCV; install the 'vision' extra "
            "(pip install 'shipvision[vision]'). Projecting with an already-fitted matrix "
            "does not need it"
        ) from exc
    return cv2


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

    ``max_error`` is what :func:`calculate_homography` measured, carried along so that a
    consumer can refuse a camera whose calibration is worse than the threshold it is about to
    gate with. `None` means nobody measured — which is different from "measured and fine".
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


# ------------------------------------------------------------------------------ fitting


def _design_matrix(camera_points: np.ndarray, map_points: np.ndarray) -> np.ndarray:
    """The ``(2n, 9)`` DLT matrix on Hartley-normalised points.

    Normalised because the un-normalised version's columns differ in magnitude by the square
    of the image size, which makes its singular values say more about the units than about
    the geometry — and the singular values are exactly what the degeneracy check reads.
    """

    def hartley(points: np.ndarray) -> np.ndarray:
        centre = points.mean(axis=0)
        centred = points - centre
        spread = float(np.sqrt((centred**2).sum(axis=1)).mean())
        return centred * (np.sqrt(2.0) / max(spread, 1e-12))

    src = hartley(camera_points)
    dst = hartley(map_points)
    rows = np.zeros((2 * len(src), 9), dtype=np.float64)
    x, y = src[:, 0], src[:, 1]
    u, v = dst[:, 0], dst[:, 1]
    rows[0::2, 0] = -x
    rows[0::2, 1] = -y
    rows[0::2, 2] = -1.0
    rows[0::2, 6] = u * x
    rows[0::2, 7] = u * y
    rows[0::2, 8] = u
    rows[1::2, 3] = -x
    rows[1::2, 4] = -y
    rows[1::2, 5] = -1.0
    rows[1::2, 6] = v * x
    rows[1::2, 7] = v * y
    rows[1::2, 8] = v
    return rows


def _fit(camera_points: np.ndarray, map_points: np.ndarray, cv2: Any) -> np.ndarray:
    matrix, _ = cv2.findHomography(camera_points, map_points, 0)
    if matrix is None:
        raise ConfigurationError(
            "cv2.findHomography could not fit these correspondences; they do not determine "
            "a plane-to-plane mapping"
        )
    return np.asarray(matrix, dtype=np.float64)


def _reprojection_errors(
    camera_points: np.ndarray, map_points: np.ndarray, cv2: Any
) -> np.ndarray:
    """Per-point error in map units, each point held out of its own fit where possible.

    Held out because the in-sample residual of an exactly-determined fit is zero by
    construction and therefore measures nothing. With only four correspondences there is
    nothing to hold out — four points determine the homography exactly — so the in-sample
    residual is returned and it will be ~0; that is honest rather than useful, and it is why
    :func:`calculate_homography` says four points cannot be validated.
    """
    count = len(camera_points)
    if count <= 4:
        projected = project(camera_points, _fit(camera_points, map_points, cv2))
        return np.linalg.norm(projected - map_points, axis=1)

    errors = np.zeros(count, dtype=np.float64)
    for index in range(count):
        keep = np.ones(count, dtype=bool)
        keep[index] = False
        matrix = _fit(camera_points[keep], map_points[keep], cv2)
        predicted = project(camera_points[index : index + 1], matrix)[0]
        errors[index] = float(np.linalg.norm(predicted - map_points[index]))
    return errors


def calculate_homography(
    camera_points: np.ndarray,
    map_points: np.ndarray,
    *,
    camera_width: int = 0,
    camera_height: int = 0,
    map_width: int = 0,
    map_height: int = 0,
) -> tuple[Homography, float]:
    """Fit a camera-to-ground-plane homography and estimate how wrong it is.

    Args:
        camera_points: ``(n, 2)`` points in the camera image, ``n >= 4``.
        map_points: ``(n, 2)`` the same points on the ground-plane map.
        camera_width: frame width these points were clicked at. Recorded on the result so a
            later resolution change rescales instead of silently mis-projecting.
        camera_height: frame height these points were clicked at.
        map_width: ground-plane map width, carried for consumers that normalise.
        map_height: ground-plane map height.

    Returns:
        The :class:`Homography`, and the **maximum leave-one-out reprojection error in map
        units**. The same number is stored on the returned homography's ``max_error``.

    Raises:
        BackendUnavailableError: OpenCV is not installed.
        ConfigurationError: fewer than four correspondences, mismatched lengths, or a
            degenerate configuration — collinear points being the common one. A degenerate
            set is refused rather than scored, because its reprojection error is *low*: the
            fitted matrix maps the line exactly and is arbitrary off it, so the number that
            would normally protect the operator is the number that would reassure them.
    """
    cv2 = _load_cv2()
    camera = np.atleast_2d(np.asarray(camera_points, dtype=np.float64))
    ground = np.atleast_2d(np.asarray(map_points, dtype=np.float64))
    if camera.shape[-1] != 2 or ground.shape[-1] != 2:
        raise ConfigurationError(
            f"point sets must be (n, 2); got {camera.shape} and {ground.shape}"
        )
    if camera.shape[0] != ground.shape[0]:
        raise ConfigurationError(
            f"{camera.shape[0]} camera points against {ground.shape[0]} map points; a "
            f"homography is fitted to correspondences, so the two sets must pair up"
        )
    if camera.shape[0] < 4:
        raise ConfigurationError(
            f"a homography needs at least 4 correspondences, got {camera.shape[0]}"
        )

    singular = np.linalg.svd(_design_matrix(camera, ground), compute_uv=False)
    # A well-posed configuration leaves exactly a one-dimensional null space: the eight
    # degrees of freedom of the homography are determined, so s[7] is comfortably positive
    # and only s[8] is ~0. Collinear points leave four or more directions undetermined,
    # which shows up as s[7] collapsing too.
    if singular[0] <= 0.0 or singular[7] / singular[0] < _DEGENERACY_RATIO:
        raise ConfigurationError(
            f"these correspondences are degenerate (conditioning "
            f"{singular[7] / max(singular[0], 1e-30):.2e}); the classic cause is points that "
            f"all lie on one line, which fixes the mapping along the line and leaves it "
            f"arbitrary everywhere else. cv2.findHomography returns a matrix for this and "
            f"its calibration-set error is zero, so it has to be caught here"
        )

    matrix = _fit(camera, ground, cv2)
    max_error = float(np.max(_reprojection_errors(camera, ground, cv2)))
    homography = Homography(
        matrix=matrix,
        camera_width=camera_width,
        camera_height=camera_height,
        map_width=map_width,
        map_height=map_height,
        max_error=max_error,
    )
    return homography, max_error
