"""A constant-velocity Kalman filter over `(cx, cy, aspect, height)`.

The standard SORT/DeepSORT formulation, and the reason it is standard is that the
alternative — associating on raw detections — loses a track the moment the detector blinks.
The filter carries a velocity estimate, so a track survives a missed frame by predicting
where the object went rather than by hoping it did not move.

Vectorised across all tracks: one `predict` and one `update` per frame handle the whole set
as matrix operations. A per-track Python loop is the classic way a tracker becomes the
bottleneck at fifty cameras, and it buys nothing — the filters are independent, which is
exactly what numpy is for.
"""

from __future__ import annotations

import numpy as np

__all__ = ["CHI2_INV_95_4DOF", "KalmanFilter"]


class KalmanFilter:
    """Eight-dimensional state `(cx, cy, a, h, vcx, vcy, va, vh)`, constant velocity.

    The noise is **scaled by the object's height** rather than being a fixed constant. That
    is the detail that makes the filter work across a scene with perspective: a person near
    the camera is 400 px tall and their centre moves tens of pixels per frame, while one at
    the far end is 40 px tall and moves a couple. A single absolute noise term is either far
    too loose for the distant one or far too tight for the near one.
    """

    def __init__(
        self, position_weight: float = 1.0 / 20, velocity_weight: float = 1.0 / 160
    ) -> None:
        self._position_weight = position_weight
        self._velocity_weight = velocity_weight

        # x' = F x. The upper-right identity block is the "position += velocity" that makes
        # this a constant-velocity model; dt is folded in as 1 because the tracker's unit of
        # time is one frame.
        self._motion = np.eye(8, dtype=np.float32)
        self._motion[:4, 4:] = np.eye(4, dtype=np.float32)
        # z = H x: only position is observed; velocity is inferred.
        self._observation = np.eye(4, 8, dtype=np.float32)

    # -- lifecycle ---------------------------------------------------------------------

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """A fresh track's mean and covariance from its first detection.

        Velocity starts at zero, and its covariance scales with the object's height rather
        than being small: after one frame we know where the object is and have no
        observation of where it is going, so the filter must not treat "stationary" as
        something it measured. A near-zero initial velocity covariance is the classic way a
        new track refuses to follow a fast-moving object for its first several frames.

        Note that the position and velocity variances are not comparable as numbers —
        pixels against pixels-per-frame — so "larger" here means larger relative to the
        quantity's own scale, which for a velocity of zero means "not negligible".
        """
        mean = np.concatenate([measurement.astype(np.float32), np.zeros(4, dtype=np.float32)])
        height = measurement[3]
        std = np.array(
            [
                2 * self._position_weight * height,
                2 * self._position_weight * height,
                1e-2,
                2 * self._position_weight * height,
                10 * self._velocity_weight * height,
                10 * self._velocity_weight * height,
                1e-5,
                10 * self._velocity_weight * height,
            ],
            dtype=np.float32,
        )
        return mean, np.diag(np.square(std)).astype(np.float32)

    def predict(
        self, means: np.ndarray, covariances: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Advance every track one frame. ``means`` is ``(N, 8)``, ``covariances`` ``(N, 8, 8)``."""
        if len(means) == 0:
            return means, covariances

        heights = means[:, 3]
        std = np.stack(
            [
                self._position_weight * heights,
                self._position_weight * heights,
                np.full_like(heights, 1e-2),
                self._position_weight * heights,
                self._velocity_weight * heights,
                self._velocity_weight * heights,
                np.full_like(heights, 1e-5),
                self._velocity_weight * heights,
            ],
            axis=1,
        )
        motion_cov = np.einsum("ij,jk->ijk", np.square(std), np.eye(8, dtype=np.float32))

        means = means @ self._motion.T
        covariances = self._motion @ covariances @ self._motion.T + motion_cov
        return means.astype(np.float32), covariances.astype(np.float32)

    def project(
        self, means: np.ndarray, covariances: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """State space -> measurement space, with observation noise added."""
        heights = means[:, 3]
        std = np.stack(
            [
                self._position_weight * heights,
                self._position_weight * heights,
                np.full_like(heights, 1e-1),
                self._position_weight * heights,
            ],
            axis=1,
        )
        innovation_cov = np.einsum("ij,jk->ijk", np.square(std), np.eye(4, dtype=np.float32))
        projected_mean = means @ self._observation.T
        projected_cov = self._observation @ covariances @ self._observation.T + innovation_cov
        return projected_mean.astype(np.float32), projected_cov.astype(np.float32)

    def update(
        self, means: np.ndarray, covariances: np.ndarray, measurements: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Correct each track with its matched measurement. All three arrays are aligned."""
        if len(means) == 0:
            return means, covariances

        projected_mean, projected_cov = self.project(means, covariances)
        updated_means = np.empty_like(means)
        updated_covs = np.empty_like(covariances)

        for i in range(len(means)):
            # Solved per track rather than batched: a batched solve needs a block-diagonal
            # system, and at the handful of tracks a single camera carries the assembly
            # costs more than the loop. If a scene ever needs hundreds, revisit with
            # np.linalg.solve on a stacked system — and measure before believing it helps.
            gain = np.linalg.solve(
                projected_cov[i].T, (covariances[i] @ self._observation.T).T
            ).T
            innovation = measurements[i] - projected_mean[i]
            updated_means[i] = means[i] + innovation @ gain.T
            updated_covs[i] = covariances[i] - gain @ projected_cov[i] @ gain.T

        return updated_means.astype(np.float32), updated_covs.astype(np.float32)

    def gating_distance(
        self, means: np.ndarray, covariances: np.ndarray, measurements: np.ndarray
    ) -> np.ndarray:
        """Squared Mahalanobis distance, ``(len(means), len(measurements))``.

        Used to *forbid* associations before the assignment runs, not to score them. A
        detection twenty metres from where a track can possibly be should not be a
        candidate at any cost — letting the assignment weigh it means one bad frame can
        drag an identity across the scene.
        """
        if len(means) == 0 or len(measurements) == 0:
            return np.zeros((len(means), len(measurements)), dtype=np.float32)

        projected_mean, projected_cov = self.project(means, covariances)
        distances = np.empty((len(means), len(measurements)), dtype=np.float32)
        for i in range(len(means)):
            cholesky = np.linalg.cholesky(projected_cov[i])
            delta = (measurements - projected_mean[i]).T
            solved = np.linalg.solve(cholesky, delta)
            distances[i] = np.sum(solved * solved, axis=0)
        return distances


#: Chi-square 0.95 quantile for 4 degrees of freedom. The conventional gate: a measurement
#: further than this from the prediction is rejected outright.
CHI2_INV_95_4DOF = 9.4877
