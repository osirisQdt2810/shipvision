#include "shipvision/mot/kalman.h"

#include <cmath>
#include <stdexcept>

namespace shipvision::mot {

    namespace {

        /// `C = A B` for row-major matrices of the given shape.
        void matmul(const float* a, const float* b, float* c, int rows, int inner, int cols) {
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    float sum = 0.f;
                    for (int k = 0; k < inner; ++k)
                        sum += a[i * inner + k] * b[k * cols + j];
                    c[i * cols + j] = sum;
                }
            }
        }

        /// Lower-triangular Cholesky factor of a symmetric positive-definite 4x4.
        ///
        /// Throws rather than returning a flag. A projected covariance that is not positive
        /// definite means the filter has diverged, and the states that follow are numbers
        /// rather than estimates — a gate computed from them admits everything, so the tracker
        /// keeps running and quietly hands identities to the wrong objects.
        void cholesky4(const float source[16], float factor[16]) {
            for (int i = 0; i < 16; ++i)
                factor[i] = 0.f;
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j <= i; ++j) {
                    float sum = source[i * 4 + j];
                    for (int k = 0; k < j; ++k)
                        sum -= factor[i * 4 + k] * factor[j * 4 + k];
                    if (i == j) {
                        if (!(sum > 0.f)) {
                            throw std::runtime_error(
                                "the projected covariance is not positive definite; the Kalman "
                                "filter has diverged and every gate computed from it would "
                                "admit anything");
                        }
                        factor[i * 4 + j] = std::sqrt(sum);
                    } else {
                        factor[i * 4 + j] = sum / factor[j * 4 + j];
                    }
                }
            }
        }

        /// Solve `L y = b` for a lower-triangular L, in place over `y`.
        void forward_substitute(const float factor[16], const float b[4], float y[4]) {
            for (int i = 0; i < 4; ++i) {
                float sum = b[i];
                for (int k = 0; k < i; ++k)
                    sum -= factor[i * 4 + k] * y[k];
                y[i] = sum / factor[i * 4 + i];
            }
        }

        /// Solve `L^T x = y` for a lower-triangular L, in place over `x`.
        void back_substitute(const float factor[16], const float y[4], float x[4]) {
            for (int i = 3; i >= 0; --i) {
                float sum = y[i];
                for (int k = i + 1; k < 4; ++k)
                    sum -= factor[k * 4 + i] * x[k];
                x[i] = sum / factor[i * 4 + i];
            }
        }

    }  // namespace

    void xyxy_to_cxcyah(const float box[4], float measurement[4]) {
        const float width = box[2] - box[0];
        const float height = box[3] - box[1];
        // The floor mirrors the numpy `np.maximum(heights, 1e-6)`: a zero-height box is legal
        // input (a detector can emit one) and dividing by it would make the aspect infinite,
        // which propagates into the state and then into every gate the track is part of.
        const float safe = height > 1e-6f ? height : 1e-6f;
        measurement[0] = (box[0] + box[2]) * 0.5f;
        measurement[1] = (box[1] + box[3]) * 0.5f;
        measurement[2] = width / safe;
        measurement[3] = height;
    }

    void cxcyah_to_xyxy(const float measurement[4], float box[4]) {
        const float height = measurement[3];
        const float width = measurement[2] * height;
        box[0] = measurement[0] - width * 0.5f;
        box[1] = measurement[1] - height * 0.5f;
        box[2] = measurement[0] + width * 0.5f;
        box[3] = measurement[1] + height * 0.5f;
    }

    KalmanState KalmanFilter::initiate(const float measurement[4]) const {
        KalmanState state;
        for (int i = 0; i < 4; ++i) {
            state.mean[i] = measurement[i];
            state.mean[i + 4] = 0.f;
        }
        const float height = measurement[3];
        const float std[8] = {2.f * position_weight_ * height,
                              2.f * position_weight_ * height,
                              1e-2f,
                              2.f * position_weight_ * height,
                              10.f * velocity_weight_ * height,
                              10.f * velocity_weight_ * height,
                              1e-5f,
                              10.f * velocity_weight_ * height};
        for (int i = 0; i < 8; ++i)
            state.covariance[i * 8 + i] = std[i] * std[i];
        return state;
    }

    void KalmanFilter::predict(KalmanState& state) const {
        const float height = state.mean[3];
        const float std[8] = {
            position_weight_ * height, position_weight_ * height, 1e-2f, position_weight_ * height,
            velocity_weight_ * height, velocity_weight_ * height, 1e-5f, velocity_weight_ * height};

        // x' = F x, with F the identity plus a `position += velocity` block. dt is folded in
        // as 1 because the tracker's unit of time is one frame.
        for (int i = 0; i < 4; ++i)
            state.mean[i] += state.mean[i + 4];

        // P' = F P F^T + Q. Written as two multiplies in the same order as the numpy version
        // rather than as the closed-form block update, so the two round identically.
        float motion[64] = {0.f};
        for (int i = 0; i < 8; ++i)
            motion[i * 8 + i] = 1.f;
        for (int i = 0; i < 4; ++i)
            motion[i * 8 + (i + 4)] = 1.f;

        float scratch[64];
        float updated[64];
        matmul(motion, state.covariance.data(), scratch, 8, 8, 8);
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                float sum = 0.f;
                for (int k = 0; k < 8; ++k)
                    sum += scratch[i * 8 + k] * motion[j * 8 + k];  // scratch @ F^T
                updated[i * 8 + j] = sum;
            }
        }
        for (int i = 0; i < 8; ++i)
            updated[i * 8 + i] += std[i] * std[i];
        for (int i = 0; i < 64; ++i)
            state.covariance[i] = updated[i];
    }

    void KalmanFilter::project(const KalmanState& state, float mean_out[4],
                               float cov_out[16]) const {
        const float height = state.mean[3];
        const float std[4] = {position_weight_ * height, position_weight_ * height, 1e-1f,
                              position_weight_ * height};
        for (int i = 0; i < 4; ++i) {
            mean_out[i] = state.mean[i];
            for (int j = 0; j < 4; ++j)
                cov_out[i * 4 + j] = state.covariance[i * 8 + j];
            cov_out[i * 4 + i] += std[i] * std[i];
        }
    }

    void KalmanFilter::update(KalmanState& state, const float measurement[4]) const {
        float projected_mean[4];
        float projected_cov[16];
        project(state, projected_mean, projected_cov);

        // K = B S^-1, where B is the first four columns of P (that is `P H^T`) and S is the
        // projected covariance. Solved column by column through one Cholesky factor rather
        // than by forming an inverse: an explicit 4x4 inverse of a near-singular S is where a
        // filter stops being a filter.
        float factor[16];
        cholesky4(projected_cov, factor);

        float gain[32];  // 8x4, row-major
        for (int column = 0; column < 4; ++column) {
            float rhs[4];
            for (int i = 0; i < 4; ++i)
                rhs[i] = 0.f;
            rhs[column] = 1.f;
            float y[4];
            float inverse_column[4];
            forward_substitute(factor, rhs, y);
            back_substitute(factor, y, inverse_column);
            for (int row = 0; row < 8; ++row) {
                float sum = 0.f;
                for (int k = 0; k < 4; ++k)
                    sum += state.covariance[row * 8 + k] * inverse_column[k];
                gain[row * 4 + column] = sum;
            }
        }

        float innovation[4];
        for (int i = 0; i < 4; ++i)
            innovation[i] = measurement[i] - projected_mean[i];
        for (int row = 0; row < 8; ++row) {
            float sum = 0.f;
            for (int k = 0; k < 4; ++k)
                sum += gain[row * 4 + k] * innovation[k];
            state.mean[row] += sum;
        }

        // P <- P - K S K^T.
        float scratch[32];  // 8x4
        matmul(gain, projected_cov, scratch, 8, 4, 4);
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                float sum = 0.f;
                for (int k = 0; k < 4; ++k)
                    sum += scratch[i * 4 + k] * gain[j * 4 + k];  // scratch @ K^T
                state.covariance[i * 8 + j] -= sum;
            }
        }
    }

    float KalmanFilter::gating_distance(const KalmanState& state,
                                        const float measurement[4]) const {
        float projected_mean[4];
        float projected_cov[16];
        project(state, projected_mean, projected_cov);

        float factor[16];
        cholesky4(projected_cov, factor);

        float delta[4];
        for (int i = 0; i < 4; ++i)
            delta[i] = measurement[i] - projected_mean[i];
        float solved[4];
        forward_substitute(factor, delta, solved);

        float distance = 0.f;
        for (int i = 0; i < 4; ++i)
            distance += solved[i] * solved[i];
        return distance;
    }

}  // namespace shipvision::mot
