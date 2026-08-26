// Using a camera-to-ground-plane homography: the value type, and the projection.
//
// The C++ twin of the *applying* half of `shipvision/mtmc/topology/homography.py`. Fitting one
// is not here and never will be: it happens once when somebody clicks calibration points, it
// needs OpenCV's `findHomography`, and it is allowed to be slow and to fail loudly. Applying
// one runs per synchronised instant, on every deployment, and is a 3x3 product over a few
// hundred points.
//
// The calibration domain travels WITH the matrix, which is why this is a struct rather than a
// bare nine doubles. A homography fitted on 1080p stills does not apply to the 720p stream the
// same camera serves at night: the pixel coordinates differ by a factor of 1.5 and every track
// on that camera projects to the wrong place on the map, silently and plausibly.

#pragma once

#include <cstddef>
#include <vector>

namespace shipvision::mtmc {

    /// One camera's mapping onto the shared ground plane, plus the frame size it was fitted at.
    struct Homography {
            /// Row-major 3x3. Identity by default, which means "the image plane IS the map".
            double matrix[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
            int camera_width = 0;   ///< 0 means unrecorded: the caller's points are already in
            int camera_height = 0;  ///< this matrix's own domain, so the rescale is skipped
    };

    /// `(n, 2)` image points through a homography, written as `(n, 2)` float32 map points.
    ///
    /// `frame_width` / `frame_height` are the size the points were MEASURED at; pass 0 for
    /// either to skip the rescale into the calibration domain.
    ///
    /// A zero third component means the point maps to the horizon line of this homography.
    /// Clamped rather than divided by zero, which keeps the result finite and very far away —
    /// what "above the horizon" should mean to a spatial gate. NaN would instead poison every
    /// comparison it touches, and a gate that compares NaN falls open on everything.
    void project(const double* points, int n, const Homography& homography, int frame_width,
                 int frame_height, float* out);

    /// The homographies for a camera group, indexed by the camera code the tracks carry.
    ///
    /// A camera without one is the normal case, not an error: a new camera goes live before
    /// anyone has clicked its calibration points, and a PTZ camera invalidates its own the
    /// moment it moves. So this answers `has()` rather than throwing, and the spatial gate above
    /// it treats "unknown" as "no spatial evidence" and falls back to appearance. Excluding an
    /// uncalibrated camera instead is worse and quieter: its identities simply never merge with
    /// anyone, and nothing in the metrics says so.
    class GroundPlane {
        public:
            GroundPlane() = default;

            /// @param homographies one per camera code, indexed by it
            /// @param calibrated one flag per camera code; false means "no matrix for this one"
            GroundPlane(std::vector<Homography> homographies,
                        std::vector<unsigned char> calibrated);

            bool has(int camera_code) const;

            /// The camera's matrix, or `nullptr` when it has none. Never a default identity:
            /// an identity homography is a real mapping — the image plane as the map — and
            /// handing one back for an uncalibrated camera would put every one of its tracks at
            /// a plausible place instead of at no place.
            const Homography* get(int camera_code) const;

            /// How many cameras have a matrix. The number `SpatialMatcher`'s repr reports.
            size_t size() const;

        private:
            std::vector<Homography> homographies_;
            std::vector<unsigned char> calibrated_;
    };

}  // namespace shipvision::mtmc
