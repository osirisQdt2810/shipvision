"""Camera-motion estimators: the registry, the default, telemetry, and optical flow.

The end-to-end proof that compensation saves identities lives in
``tests/tracking/trackers/test_botsort.py``. What is here is the contract each estimator has
to keep on its own, and one property in particular: an estimator that cannot answer must say
so rather than return identity, because "the camera did not move" is indistinguishable from a
correct answer and is wrong on exactly the frames that matter.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipvision.errors import ConfigurationError, TrackingError
from shipvision.tracking import CAMERA_MOTION, IDENTITY_AFFINE


class TestTheRegistry:
    def test_every_built_in_estimator_is_registered_and_buildable(self) -> None:
        assert set(CAMERA_MOTION.names()) == {"none", "external", "sparse_flow"}
        for name in ("none", "external"):
            assert CAMERA_MOTION.build(name).name == name

    def test_aliases_resolve(self) -> None:
        for alias, name in (("off", "none"), ("ptz", "external"), ("cmc", "sparse_flow")):
            assert CAMERA_MOTION.get(alias) is CAMERA_MOTION.get(name)

    def test_registering_the_flow_estimator_does_not_import_opencv(self) -> None:
        """Listing what exists must work on a machine with no OpenCV. Only *construction*
        needs it, and it raises a typed error there."""
        assert "sparse_flow" in CAMERA_MOTION.names()

    def test_an_unknown_estimator_names_the_alternatives(self) -> None:
        with pytest.raises(ConfigurationError, match="available:"):
            CAMERA_MOTION.build("magnetometer")


class TestNoCameraMotion:
    """The default: assume the camera is bolted down."""

    def test_it_always_returns_identity(self) -> None:
        estimator = CAMERA_MOTION.build("none")
        np.testing.assert_allclose(estimator.estimate(None), IDENTITY_AFFINE)
        np.testing.assert_allclose(
            estimator.estimate(np.zeros((8, 8), np.uint8)), IDENTITY_AFFINE
        )

    def test_it_hands_out_a_copy_not_the_shared_constant(self) -> None:
        """A caller that scaled the returned array in place would silently change what "no
        motion" means for every estimator in the process."""
        estimator = CAMERA_MOTION.build("none")
        first = estimator.estimate(None)
        first[0, 2] = 999.0
        np.testing.assert_allclose(estimator.estimate(None), IDENTITY_AFFINE)


class TestExternalCameraMotion:
    """Telemetry: the best estimate available on a PTZ head, and no image needed."""

    def test_a_pushed_affine_is_returned_once_and_then_forgotten(self) -> None:
        """A stale affine applied to a later frame moves every prediction by a motion that
        did not happen. Over-compensating loses identities exactly as under-compensating
        does, so each push is consumed by one frame."""
        estimator = CAMERA_MOTION.build("external")
        affine = np.array([[1.0, 0.0, -45.0], [0.0, 1.0, 3.0]], np.float32)
        estimator.push(affine)
        np.testing.assert_allclose(estimator.estimate(None), affine)
        np.testing.assert_allclose(estimator.estimate(None), IDENTITY_AFFINE)

    def test_strict_mode_treats_missing_telemetry_as_an_incident(self) -> None:
        """Where telemetry is supposed to be on every frame, silence is a dropped packet, not
        a still camera — and a slow degradation nobody notices is the worse outcome."""
        estimator = CAMERA_MOTION.build("external", strict=True)
        with pytest.raises(TrackingError, match="no camera motion was pushed"):
            estimator.estimate(None)

    def test_a_misshaped_affine_is_refused_at_the_push(self) -> None:
        estimator = CAMERA_MOTION.build("external")
        with pytest.raises(ConfigurationError, match=r"\(2, 3\)"):
            estimator.push(np.eye(3, dtype=np.float32))

    def test_reset_drops_a_pending_affine(self) -> None:
        estimator = CAMERA_MOTION.build("external")
        estimator.push(np.array([[1.0, 0.0, -45.0], [0.0, 1.0, 0.0]], np.float32))
        estimator.reset()
        np.testing.assert_allclose(estimator.estimate(None), IDENTITY_AFFINE)


class TestSparseOpticalFlow:
    """Lucas-Kanade on shi-Tomasi corners, fitted to a partial affine."""

    @staticmethod
    def _backdrop() -> np.ndarray:
        """A textured quay wall. Block-upsampled noise gives plenty of findable corners and
        is deterministic, which uniform per-pixel noise is not for a corner detector."""
        rng = np.random.default_rng(7)
        coarse = (rng.random((150, 400)) * 255).astype(np.uint8)
        return np.kron(coarse, np.ones((8, 8), np.uint8))

    def test_the_first_frame_has_nothing_to_compare_against(self) -> None:
        pytest.importorskip("cv2")
        estimator = CAMERA_MOTION.build("sparse_flow")
        np.testing.assert_allclose(
            estimator.estimate(self._backdrop()[:600, :800]), IDENTITY_AFFINE
        )

    def test_it_recovers_a_pure_translation_to_within_half_a_pixel(self) -> None:
        pytest.importorskip("cv2")
        backdrop = self._backdrop()
        estimator = CAMERA_MOTION.build("sparse_flow")
        estimator.estimate(backdrop[:600, 0:800])
        for step in range(1, 6):
            affine = estimator.estimate(backdrop[:600, step * 30 : step * 30 + 800])
            assert affine[0, 2] == pytest.approx(-30.0, abs=0.5)
            assert affine[1, 2] == pytest.approx(0.0, abs=0.5)

    def test_downscaling_does_not_change_the_answer(self) -> None:
        """The translation is measured in downscaled pixels and scaled back; the rotation and
        scale block is dimensionless and must not be touched. Getting that wrong halves or
        doubles every compensated prediction."""
        pytest.importorskip("cv2")
        backdrop = self._backdrop()
        results = []
        for downscale in (1, 2):
            estimator = CAMERA_MOTION.build("sparse_flow", downscale=downscale)
            estimator.estimate(backdrop[:600, 0:800])
            results.append(estimator.estimate(backdrop[:600, 40:840])[0, 2])
        assert results[0] == pytest.approx(-40.0, abs=0.6)
        assert results[1] == pytest.approx(-40.0, abs=0.6)

    def test_a_colour_frame_and_a_grey_one_agree(self) -> None:
        pytest.importorskip("cv2")
        backdrop = self._backdrop()

        def run(colour: bool) -> float:
            estimator = CAMERA_MOTION.build("sparse_flow")
            for shift in (0, 25):
                view = backdrop[:600, shift : shift + 800]
                frame = np.repeat(view[:, :, None], 3, axis=2) if colour else view
                affine = estimator.estimate(frame)
            return float(affine[0, 2])

        assert run(colour=True) == pytest.approx(run(colour=False), abs=0.6)

    def test_a_featureless_frame_reports_no_motion_rather_than_a_guess(self) -> None:
        """Fog, or a lens cap. With too few inliers the honest answer is "unknown", and for a
        camera-motion term unknown must mean identity: over-compensating on a bad fit loses
        every identity at once, which is strictly worse than not compensating."""
        pytest.importorskip("cv2")
        estimator = CAMERA_MOTION.build("sparse_flow")
        flat = np.full((600, 800), 128, np.uint8)
        estimator.estimate(flat)
        np.testing.assert_allclose(estimator.estimate(flat), IDENTITY_AFFINE)

    def test_it_refuses_to_answer_without_pixels(self) -> None:
        pytest.importorskip("cv2")
        estimator = CAMERA_MOTION.build("sparse_flow")
        with pytest.raises(TrackingError, match="needs the frame"):
            estimator.estimate(None)

    def test_reset_forgets_the_reference_frame(self) -> None:
        """A reconnected camera has no continuity with the frame before the break, so a kept
        reference would produce one large spurious motion on the first frame back."""
        pytest.importorskip("cv2")
        backdrop = self._backdrop()
        estimator = CAMERA_MOTION.build("sparse_flow")
        estimator.estimate(backdrop[:600, 0:800])
        estimator.reset()
        np.testing.assert_allclose(
            estimator.estimate(backdrop[:600, 200:1000]), IDENTITY_AFFINE
        )

    def test_a_downscale_below_one_is_refused(self) -> None:
        pytest.importorskip("cv2")
        with pytest.raises(ConfigurationError, match="downscale"):
            CAMERA_MOTION.build("sparse_flow", downscale=0)
