"""Fitting a camera-to-ground-plane homography, and estimating how wrong it is.

The other half of :mod:`shipvision.mtmc.topology`, and a genuinely different job: this runs
once, when somebody clicks four or more correspondences on a still and on a map, and it needs
OpenCV. :mod:`shipvision.mtmc.topology.homography` runs on every synchronised instant and
needs only numpy. Separating them is what lets a deployment that receives its matrices
already calibrated — from a file, from a config service — never install cv2, and it keeps the
frame path free of an import that can fail.

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
"""

from __future__ import annotations

from typing import Any

import numpy as np

from shipvision.errors import BackendUnavailableError, ConfigurationError
from shipvision.mtmc.topology.homography import Homography, project

__all__ = ["calculate_homography"]

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
