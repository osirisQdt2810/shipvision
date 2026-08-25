"""The ground plane: projecting with a homography, and fitting one honestly.

Projection is numpy and always available. *Fitting* needs OpenCV, so those tests skip where it
is absent — which is also the point being tested: a deployment that receives its matrices
already calibrated must not need OpenCV installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipvision.errors import ConfigurationError
from shipvision.mtmc import GroundPlane, Homography, calculate_homography, project

#: A well-conditioned set of image points: spread across the frame, nothing collinear.
CAMERA_POINTS = np.array(
    [
        [100.0, 900.0],
        [1800.0, 950.0],
        [1500.0, 400.0],
        [400.0, 380.0],
        [950.0, 600.0],
        [700.0, 820.0],
    ]
)

#: The homography the map points below are generated *through*, so the correspondence is
#: exactly projectively consistent and the fit has a right answer to recover. Hand-writing a
#: plausible-looking pair of point sets instead does not give a well-conditioned problem: the
#: first attempt at this file did that, and the leave-one-out error came back at 69 map units
#: — correctly, because no homography maps those six points onto those six others.
TRUE_MATRIX = np.array(
    [
        [0.021, 0.004, -2.1],
        [0.0015, 0.030, -10.5],
        [1e-5, 4e-4, 1.0],
    ]
)
MAP_POINTS = np.asarray(project(CAMERA_POINTS, TRUE_MATRIX), dtype=np.float64)

#: Points somebody clicked by eye: no homography maps these onto MAP_POINTS' scale, which is
#: the everyday calibration failure the error estimate exists to report.
INCONSISTENT_MAP_POINTS = np.array(
    [
        [0.0, 0.0],
        [40.0, 0.0],
        [40.0, 30.0],
        [0.0, 30.0],
        [20.0, 15.0],
        [10.0, 5.0],
    ]
)


class TestProjection:
    """A 3x3 matrix product. No OpenCV, no exceptions on the horizon."""

    def test_it_maps_points_through_the_matrix(self) -> None:
        matrix = np.array([[2.0, 0.0, 5.0], [0.0, 3.0, -1.0], [0.0, 0.0, 1.0]])

        assert project(np.array([[1.0, 2.0]]), matrix)[0] == pytest.approx([7.0, 5.0])

    def test_a_perspective_divide_is_applied(self) -> None:
        matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])

        assert project(np.array([[4.0, 6.0]]), matrix)[0] == pytest.approx([2.0, 3.0])

    def test_a_point_on_the_horizon_stays_finite_instead_of_becoming_nan(self) -> None:
        """A zero third component means the point maps to infinity. Clamping keeps it very far
        away — which is what "above the horizon" should mean to a spatial gate — where NaN
        would poison every comparison it touches."""
        matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])

        result = project(np.array([[1.0, 0.0]]), matrix)

        assert np.all(np.isfinite(result))
        assert abs(result[0][0]) > 1e6

    def test_it_is_vectorised_over_the_whole_set(self) -> None:
        matrix = np.eye(3)
        points = np.random.default_rng(0).normal(size=(64, 2))

        assert project(points, matrix).shape == (64, 2)

    def test_a_wrongly_shaped_point_set_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match=r"\(n, 2\)"):
            project(np.zeros((4, 3)), np.eye(3))


class TestHomographyValidation:
    """A matrix that cannot work is refused when it is handed over, not when it is used."""

    def test_a_non_3x3_matrix_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="3x3"):
            Homography(matrix=np.eye(4))

    def test_a_singular_matrix_is_refused(self) -> None:
        """It collapses the image onto a line, so every track on that camera would project to
        the same place — and every pair would then look like the same object."""
        with pytest.raises(ConfigurationError, match="singular"):
            Homography(matrix=np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 0.0, 1.0]]))

    def test_a_non_finite_matrix_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="finite"):
            Homography(matrix=np.array([[np.nan, 0, 0], [0, 1, 0], [0, 0, 1]]))

    def test_the_calibration_domain_rescales_a_changed_resolution(self) -> None:
        """A homography fitted on 1080p stills does not apply to the 720p night stream."""
        homography = Homography(matrix=np.eye(3), camera_width=1920, camera_height=1080)

        rescaled = homography.to_calibration_domain(
            np.array([[480.0, 270.0]]), frame_width=960, frame_height=540
        )

        assert rescaled[0] == pytest.approx([960.0, 540.0])

    def test_an_unrecorded_calibration_domain_is_a_no_op(self) -> None:
        """The caller is assumed to be handing over points in the matrix's own domain."""
        homography = Homography(matrix=np.eye(3))

        unchanged = homography.to_calibration_domain(
            np.array([[480.0, 270.0]]), frame_width=960, frame_height=540
        )

        assert unchanged[0] == pytest.approx([480.0, 270.0])


class TestGroundPlane:
    """A camera without a homography is the normal case, not an error."""

    def test_it_answers_rather_than_raising_for_an_unknown_camera(self) -> None:
        plane = GroundPlane({"cam-a": Homography(matrix=np.eye(3))})

        assert plane.has("cam-a")
        assert not plane.has("cam-b")
        assert plane.get("cam-b") is None
        assert "cam-a" in plane

    def test_a_camera_can_be_recalibrated(self) -> None:
        """A PTZ camera invalidates its own homography the moment it moves."""
        plane = GroundPlane({"cam-a": Homography(matrix=np.eye(3))})

        plane.add("cam-a", Homography(matrix=np.eye(3) * 2.0))

        assert plane.get("cam-a").matrix[0, 0] == 2.0
        assert len(plane) == 1

    def test_a_raw_matrix_is_refused_so_the_calibration_domain_cannot_be_lost(self) -> None:
        with pytest.raises(ConfigurationError, match="not a Homography"):
            GroundPlane({"cam-a": np.eye(3)})


class TestHomographyFitting:
    """Fitting, and the error estimate most implementations throw away."""

    @staticmethod
    def require_cv2() -> None:
        pytest.importorskip("cv2", reason="fitting a homography is OpenCV's job, not ours")

    def test_a_well_conditioned_point_set_gives_a_low_error(self) -> None:
        self.require_cv2()

        homography, error = calculate_homography(
            CAMERA_POINTS, MAP_POINTS, camera_width=1920, camera_height=1080
        )

        # The map spans tens of units, so a millionth of one is "this calibration is exact".
        assert error < 1e-4
        assert homography.max_error == error
        assert homography.camera_width == 1920

    def test_it_recovers_the_matrix_the_points_were_generated_through(self) -> None:
        """The strongest statement available: not "the residual is small" but "this is the
        same projective map", up to the scale a homography is only defined to."""
        self.require_cv2()

        homography, _ = calculate_homography(CAMERA_POINTS, MAP_POINTS)

        recovered = homography.matrix / homography.matrix[2, 2]
        assert recovered == pytest.approx(TRUE_MATRIX / TRUE_MATRIX[2, 2], rel=1e-3)

    def test_the_fitted_matrix_maps_the_calibration_points(self) -> None:
        self.require_cv2()

        homography, _ = calculate_homography(CAMERA_POINTS, MAP_POINTS)

        assert homography.project(CAMERA_POINTS) == pytest.approx(MAP_POINTS, abs=1e-3)

    def test_a_noisy_point_set_reports_a_far_larger_error(self) -> None:
        """The number is only useful if it moves."""
        self.require_cv2()
        noise = np.array(
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 4.0], [0.0, -4.0]]
        )

        _, clean = calculate_homography(CAMERA_POINTS, MAP_POINTS)
        _, noisy = calculate_homography(CAMERA_POINTS, MAP_POINTS + noise)

        assert clean < 1e-4
        assert noisy > 1.0

    def test_points_clicked_by_eye_are_reported_as_a_bad_calibration(self) -> None:
        """The everyday failure, and the reason to hold points out rather than measure the
        residual. cv2 returns a matrix without complaint; its worst *in-sample* residual is
        about 2.6 map units, which looks survivable — while the held-out error is an order of
        magnitude larger, because the fit is absorbing the inconsistency rather than
        describing the scene."""
        self.require_cv2()

        homography, error = calculate_homography(CAMERA_POINTS, INCONSISTENT_MAP_POINTS)

        in_sample = np.linalg.norm(
            homography.project(CAMERA_POINTS) - INCONSISTENT_MAP_POINTS, axis=1
        ).max()

        assert in_sample < 5.0
        assert error > 10.0 * in_sample

    def test_collinear_points_are_a_typed_failure_rather_than_a_flattering_number(self) -> None:
        """cv2.findHomography returns a matrix for a collinear set — one that maps the line
        exactly and is arbitrary off it — and its reprojection error is *zero*. So the number
        that would normally protect the operator is the number that would reassure them, and
        the degeneracy has to be caught by conditioning instead."""
        self.require_cv2()
        line = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]])

        with pytest.raises(ConfigurationError, match="degenerate"):
            calculate_homography(line, line * 0.1)

    def test_fewer_than_four_correspondences_is_refused(self) -> None:
        self.require_cv2()

        with pytest.raises(ConfigurationError, match="at least 4"):
            calculate_homography(CAMERA_POINTS[:3], MAP_POINTS[:3])

    def test_mismatched_set_sizes_are_refused(self) -> None:
        self.require_cv2()

        with pytest.raises(ConfigurationError, match="camera points against"):
            calculate_homography(CAMERA_POINTS, MAP_POINTS[:5])

    def test_four_points_fit_but_cannot_be_validated(self) -> None:
        """Four points determine a homography exactly, so there is nothing to hold out and the
        residual is ~0 by construction. Reporting that honestly is better than pretending it
        is a measurement — and it is why a calibration UI should ask for more than four."""
        self.require_cv2()

        _, error = calculate_homography(CAMERA_POINTS[:4], INCONSISTENT_MAP_POINTS[:4])

        assert error < 1e-6
