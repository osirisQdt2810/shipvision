#include "shipvision/mtmc/topology/homography.h"

#include <cmath>

namespace shipvision::mtmc {

    namespace {

        /// numpy's `np.sign` for a double: 0 at zero, so the clamp below lands on +1e-12.
        double sign_of(double value) {
            return static_cast<double>(value > 0.0) - static_cast<double>(value < 0.0);
        }

    }  // namespace

    void project(const double* points, int n, const Homography& homography, int frame_width,
                 int frame_height, float* out) {
        // The rescale is skipped when either size is unknown, on the assumption stated in
        // `to_calibration_domain`: the caller is already handing over points in the matrix's own
        // domain. Guessing a scale from one known side would be worse than not scaling.
        const bool rescale = homography.camera_width > 0 && homography.camera_height > 0 &&
                             frame_width > 0 && frame_height > 0;
        const double scale_x =
            rescale ? static_cast<double>(homography.camera_width) / frame_width : 1.0;
        const double scale_y =
            rescale ? static_cast<double>(homography.camera_height) / frame_height : 1.0;

        const double* m = homography.matrix;
        for (int index = 0; index < n; ++index) {
            const double x = points[index * 2] * scale_x;
            const double y = points[index * 2 + 1] * scale_y;
            const double u = m[0] * x + m[1] * y + m[2];
            const double v = m[3] * x + m[4] * y + m[5];
            const double w = m[6] * x + m[7] * y + m[8];
            const double safe = std::fabs(w) < 1e-12 ? sign_of(w) * 1e-12 + 1e-12 : w;
            out[index * 2] = static_cast<float>(u / safe);
            out[index * 2 + 1] = static_cast<float>(v / safe);
        }
    }

    GroundPlane::GroundPlane(std::vector<Homography> homographies,
                             std::vector<unsigned char> calibrated)
        : homographies_(std::move(homographies)), calibrated_(std::move(calibrated)) {
        // A flag per matrix, checked here rather than trusted: a shorter flag vector would make
        // the cameras past its end silently uncalibrated, which reads as a tuning problem
        // ("why does camera 7 never merge with anyone") rather than as a wiring bug.
        calibrated_.resize(homographies_.size(), 0);
    }

    bool GroundPlane::has(int camera_code) const {
        return camera_code >= 0 && static_cast<size_t>(camera_code) < calibrated_.size() &&
               calibrated_[static_cast<size_t>(camera_code)] != 0;
    }

    const Homography* GroundPlane::get(int camera_code) const {
        return has(camera_code) ? &homographies_[static_cast<size_t>(camera_code)] : nullptr;
    }

    size_t GroundPlane::size() const {
        size_t count = 0;
        for (unsigned char flag : calibrated_)
            count += flag != 0 ? 1u : 0u;
        return count;
    }

}  // namespace shipvision::mtmc
