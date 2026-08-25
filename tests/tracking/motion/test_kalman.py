"""The motion model. If this drifts, every tracker built on it drifts with it."""

from __future__ import annotations

import numpy as np

from shipvision.tracking.motion.kalman import CHI2_INV_95_4DOF, KalmanFilter
from shipvision.types import xyxy_to_cxcyah


def _walk(steps: int, dx: float = 6.0) -> list[np.ndarray]:
    """A box moving right at a constant speed — the case the model is built for."""
    return [
        xyxy_to_cxcyah(np.array([[100 + i * dx, 200, 150 + i * dx, 320]], np.float32))[0]
        for i in range(steps)
    ]


class TestConstantVelocity:
    """The model itself: it exists so a track survives a frame the detector missed."""

    def test_the_filter_learns_a_constant_velocity(self) -> None:
        """After a few observations the prediction should land on the next one, which is the
        only reason the filter is here: it is what carries a track through a missed frame."""
        kf = KalmanFilter()
        walk = _walk(10)
        mean, cov = kf.initiate(walk[0])
        means, covs = mean[None, :], cov[None, ...]

        for measurement in walk[1:8]:
            means, covs = kf.predict(means, covs)
            means, covs = kf.update(means, covs, measurement[None, :])

        predicted, _ = kf.predict(means, covs)
        # Within a couple of pixels of the true next centre.
        assert abs(float(predicted[0, 0]) - float(walk[8][0])) < 3.0

    def test_velocity_starts_at_zero_but_not_confidently(self) -> None:
        """One frame tells you where something is and nothing about where it is going.

        Note what is NOT asserted: that the velocity variance exceeds the position variance.
        Those are pixels against pixels-per-frame and comparing them is meaningless — an
        earlier version of this test did exactly that and failed against a correct filter.
        What matters is that the velocity covariance is substantial relative to the object's
        scale, so a new track can follow something moving fast from its first frames instead of
        insisting it is stationary.
        """
        kf = KalmanFilter()
        mean, cov = kf.initiate(_walk(1)[0])

        assert np.allclose(mean[4:], 0.0), "velocity must start at zero, not at a guess"
        height = float(mean[3])
        velocity_std = float(np.sqrt(np.mean(np.diag(cov)[4:6])))
        # A track may be moving at a good fraction of its own height per frame — a person
        # crossing the frame does exactly that — and the filter must admit the possibility.
        assert velocity_std > height / 40


class TestNoiseScaling:
    """Perspective. One absolute noise term is wrong for either the near object or the far one."""

    def test_noise_scales_with_object_height(self) -> None:
        """Perspective: a near object's centre moves tens of pixels per frame and a far one's
        moves two. A single absolute noise term is wrong for one of them."""
        kf = KalmanFilter()
        near = kf.initiate(xyxy_to_cxcyah(np.array([[0, 0, 200, 400]], np.float32))[0])[1]
        far = kf.initiate(xyxy_to_cxcyah(np.array([[0, 0, 20, 40]], np.float32))[0])[1]
        assert np.diag(near)[0] > np.diag(far)[0] * 10


class TestGating:
    """Forbidding an association the motion model calls impossible, before it is priced."""

    def test_gating_rejects_the_implausible_and_admits_the_expected(self) -> None:
        kf = KalmanFilter()
        walk = _walk(6)
        mean, cov = kf.initiate(walk[0])
        means, covs = mean[None, :], cov[None, ...]
        for measurement in walk[1:5]:
            means, covs = kf.predict(means, covs)
            means, covs = kf.update(means, covs, measurement[None, :])
        means, covs = kf.predict(means, covs)

        plausible = walk[5]
        far_away = xyxy_to_cxcyah(np.array([[2000, 1500, 2050, 1620]], np.float32))[0]
        distances = kf.gating_distance(means, covs, np.stack([plausible, far_away]))

        assert distances[0, 0] < CHI2_INV_95_4DOF
        assert distances[0, 1] > CHI2_INV_95_4DOF


class TestEmptyInput:
    """A camera can legitimately see nothing, and the filter must age gracefully through it."""

    def test_empty_input_is_not_an_error(self) -> None:
        """A camera can legitimately see nothing; the filter must age gracefully."""
        kf = KalmanFilter()
        means = np.zeros((0, 8), np.float32)
        covs = np.zeros((0, 8, 8), np.float32)
        assert kf.predict(means, covs)[0].shape == (0, 8)
        assert kf.gating_distance(means, covs, np.zeros((3, 4), np.float32)).shape == (0, 3)
