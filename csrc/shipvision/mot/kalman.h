// The constant-velocity Kalman filter, in the same formulation as the numpy one.
//
// This is a PORT, not a second design. `shipvision/mot/motion/kalman.py` is the
// specification: the same 8-d state `(cx, cy, a, h, vcx, vcy, va, vh)`, the same
// height-scaled noise, the same order of operations. Anywhere the two could differ they
// would differ *silently* — a filter with slightly different process noise still tracks, it
// just associates differently in a crowd — so the parity tests compare the trackers built on
// top of these, and this header exists to make the two readable side by side.
//
// ONE TRACK AT A TIME, deliberately, where the numpy version is batched over all tracks. The
// batching there buys nothing mathematically: the filters are independent, and it exists only
// because a Python loop over fifteen tracks costs more than the arithmetic. In C++ that
// pressure is gone, and the per-track form is what lets `predict` and `update` be read against
// the equations.

#pragma once

#include <array>

namespace shipvision::mot {

    /// One track's filter state: the mean and its covariance, row-major.
    ///
    /// Plain arrays rather than a matrix type because the sizes are fixed at 8 and there is no
    /// linear-algebra dependency in this library. Eigen would be a better vehicle for the
    /// arithmetic below and a worse one for a header a reader is meant to check against the
    /// Python — and it would be a dependency the parent has to satisfy to build kernels it
    /// does not otherwise need.
    struct KalmanState {
            std::array<float, 8> mean{};
            std::array<float, 64> covariance{};  ///< 8x8, row-major
    };

    /// Eight-dimensional constant-velocity filter over `(cx, cy, a, h)` measurements.
    ///
    /// The noise is scaled by the object's HEIGHT, which is the detail that makes one filter
    /// work across a scene with perspective: a person near the camera is 400 px tall and their
    /// centre moves tens of pixels per frame, one at the far end is 40 px tall and moves a
    /// couple. A single absolute noise term is either far too loose for the near one or far
    /// too tight for the far one.
    class KalmanFilter {
        public:
            explicit KalmanFilter(float position_weight = 1.f / 20.f,
                                  float velocity_weight = 1.f / 160.f)
                : position_weight_(position_weight), velocity_weight_(velocity_weight) {}

            /// A fresh track's state from its first measurement.
            ///
            /// The velocity covariance is large rather than small: after one frame we know
            /// where the object is and have *no* observation of where it is going, and a
            /// near-zero initial velocity covariance is the classic way a new track refuses to
            /// follow a fast-moving object for its first several frames.
            KalmanState initiate(const float measurement[4]) const;

            /// Advance one frame: `x' = F x`, `P' = F P F^T + Q`.
            void predict(KalmanState& state) const;

            /// State space to measurement space, with observation noise added.
            ///
            /// @param mean_out  4 floats
            /// @param cov_out   4x4 row-major, 16 floats
            void project(const KalmanState& state, float mean_out[4], float cov_out[16]) const;

            /// Correct with one measurement. Mirrors the numpy `update` exactly, including the
            /// order `P - K S K^T` — the algebraically equivalent Joseph form is more stable
            /// and would put the two implementations a few ULP apart on every frame for no
            /// benefit anybody can observe through a 1e-3 px tolerance.
            void update(KalmanState& state, const float measurement[4]) const;

            /// Squared Mahalanobis distance between this state and a measurement.
            ///
            /// Used to FORBID associations before the assignment runs, never to score them: a
            /// detection twenty metres from where a track can possibly be should not be a
            /// candidate at any cost, because letting the solver weigh it means one bad frame
            /// can drag an identity across the scene.
            float gating_distance(const KalmanState& state, const float measurement[4]) const;

        private:
            float position_weight_;
            float velocity_weight_;
    };

    /// `xyxy` pixels to the filter's `(cx, cy, aspect, height)` measurement.
    ///
    /// Next to the filter rather than in a geometry header because the convention belongs to
    /// the state: aspect and **height**, in that order. The bug this pairing exists to stop is
    /// documented in the Python and was found in a reference implementation — a converter that
    /// writes width where height belongs tracks square objects perfectly and falls apart on a
    /// ship, because the height-scaled process noise is then driven by the wrong extent.
    void xyxy_to_cxcyah(const float box[4], float measurement[4]);

    /// Inverse of `xyxy_to_cxcyah`.
    void cxcyah_to_xyxy(const float measurement[4], float box[4]);

    /// Chi-square 0.95 quantile for 4 degrees of freedom — the conventional gate.
    ///
    /// Duplicated from `CHI2_INV_95_4DOF` in the Python, and it has to be: it belongs to the
    /// filter that produced the distances, and a caller passing its own would eventually pass
    /// the 2-DOF one. The parity tests are what keep the two numbers equal.
    constexpr float kChi2Inv95_4Dof = 9.4877f;

}  // namespace shipvision::mot
