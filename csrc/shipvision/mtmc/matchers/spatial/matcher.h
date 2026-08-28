// Ground-plane geometry: where each track is standing, and how far apart two are.
//
// The C++ twin of `shipvision/mtmc/matchers/spatial/matcher.py`. Appearance says two crops look
// alike. Geometry says whether they *can* be the same object, and it is much harder to fool:
// two crew members in identical overalls score high on appearance from any model, and are forty
// metres apart on the quay.
//
// Usable on its own only where every camera is calibrated and the scene is sparse enough that
// position alone identifies an object. Its real job is to be the gate inside `GatedMatcher` —
// kept a separate class so the projection has a test of its own, because a gate whose geometry
// is wrong and whose appearance is right produces output that looks fine until two people walk
// past each other.

#pragma once

#include <vector>

#include "shipvision/mtmc/frames.h"
#include "shipvision/mtmc/topology/homography.h"

namespace shipvision::mtmc {

    class SpatialMatcher {
        public:
            /// The same four knobs as the Python constructor, with the same defaults.
            ///
            /// A struct rather than three loose floats: `foot_ratio` and `aspect_ratio` are both
            /// small positive numbers about the shape of a person, and a positional signature
            /// would make them silently swappable — a transposition that shifts every foot point
            /// in the near field without failing anywhere.
            struct Options {
                    /// How far apart, in ground-plane units, two projections may be and still be
                    /// the same object. 280 map pixels is the reference's production value.
                    float spatial_threshold = 280.0f;
                    double foot_ratio = 1.0;
                    double aspect_ratio = 0.25;
            };

            SpatialMatcher() = default;

            /// @param plane the camera-to-map homographies. An empty one is legal and means
            ///        nothing can be judged spatially, which the gate handles by falling back to
            ///        appearance — that is what makes `gated` safe as a default.
            SpatialMatcher(const Options& options, GroundPlane plane);

            /// `(n, 2)` ground points and `(n,)` "this one is calibrated", both caller-allocated.
            ///
            /// Cameras without a homography get `(NaN, NaN)` and a false flag rather than
            /// `(0, 0)` and a side-list of invalid indices. The origin is a real place on the
            /// map: with the reference's arrangement, one forgotten check leaves every
            /// uncalibrated camera's tracks coincident with each other.
            void ground_positions(const Observation* observations, int n, float* points,
                                  unsigned char* known) const;

            /// `(n, n)` euclidean distance on the ground plane; infinity where unknowable.
            ///
            /// Infinity is the honest value for a pair where at least one camera is
            /// uncalibrated: not "far apart" and not "close", but "this matcher has nothing to
            /// say". The two consumers decide differently what that means — `build` refuses the
            /// pair, `gate` lets it through — which is why the decision is not taken here. It is
            /// an internal primitive, so the non-finite value never reaches a clusterer.
            std::vector<double> ground_distances(const Observation* observations, int n) const;

            /// `(n, n)` similarity in [0, 1] from ground separation, for the shared `build`.
            std::vector<float> similarities(const Observation* observations, int n) const;

            /// `(n, n)` "geometry does not object", one byte per pair.
            ///
            /// True within the threshold **and** where the pair cannot be judged at all. Falling
            /// open on "cannot judge" is what lets an uncalibrated camera keep taking part in
            /// cross-camera tracking on appearance alone, instead of quietly never merging with
            /// anyone.
            std::vector<unsigned char> gate(const Observation* observations, int n) const;

            /// `(n, n)` clusterable distance from position alone.
            ///
            /// An unknowable pair becomes `kNeverMerge`, the opposite of what `gate` does with
            /// it, and both are right: with no other evidence in play "I cannot tell" must not
            /// become "merge them", whereas the gate still has appearance to fall back on.
            std::vector<float> build(const Observation* observations, int n) const;

            const Options& options() const { return options_; }

            const GroundPlane& ground_plane() const { return plane_; }

        private:
            Options options_;
            GroundPlane plane_;
    };

}  // namespace shipvision::mtmc
