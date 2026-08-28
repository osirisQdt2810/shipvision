// Appearance, vetoed by geometry. The production matcher.
//
// The C++ twin of `shipvision/mtmc/matchers/gated/matcher.py`. Ten lines of logic, and they are the
// ten lines that make cross-camera tracking work on a real site: take the appearance
// similarity, and zero it wherever the two tracks project to ground positions further apart
// than they could possibly be for one object. Appearance decides *which* of several candidates;
// geometry decides *whether* any of them is possible.
//
// The composition matters more than the arithmetic. Both halves already exist as matchers with
// their own tests, so this class owns no distance function, no mask and no threshold logic — it
// owns a decision about how two independent pieces of evidence combine. The reference
// implemented the same idea by multiple-inheriting from both builders and calling protected
// methods across the hierarchy; composing instances instead means the gate can be tested with a
// hand-built appearance matrix, and either half can be replaced without touching this file.
//
// The Python package's `utils.py` — `veto` — has no counterpart under this directory because it
// already exists in `mtmc/matcher.h`, where the other fused (n, n) passes live. A second copy
// here would be one more place for "a vetoed pair is EXACTLY zero" to drift, and that is the one
// property the whole idea rests on: `to_distance` is what turns zero into `kNeverMerge`, so a
// veto that merely scaled the similarity down would leave an impossible pair expensive rather
// than forbidden — which average linkage buys the moment somebody loosens a threshold.

#pragma once

#include <vector>

#include "shipvision/mtmc/frames.h"
#include "shipvision/mtmc/matchers/appearance/matcher.h"
#include "shipvision/mtmc/matchers/spatial/matcher.h"

namespace shipvision::mtmc {

    class GatedMatcher {
        public:
            GatedMatcher() = default;

            /// Both halves, pre-built. That is how one side of the gate gets A/B'd against a
            /// recorded stream without rebuilding the other, and it is why there is no
            /// constructor here that takes five loose thresholds: this class's whole content is
            /// the composition, so the thresholds belong to the components that own them.
            GatedMatcher(AppearanceMatcher appearance, SpatialMatcher spatial);

            /// `(n, n)` appearance similarity with geometrically impossible pairs zeroed.
            std::vector<float> similarities(const float* gram, const Observation* observations,
                                            int n) const;

            /// `(n, n)` clusterable distance: appearance, the geometric veto, then the shared
            /// conversion — one crossing of the boundary instead of the four the same sequence
            /// costs when each pass is called from numpy in turn.
            std::vector<float> build(const float* gram, const Observation* observations,
                                     int n) const;

            const AppearanceMatcher& appearance() const { return appearance_; }

            const SpatialMatcher& spatial() const { return spatial_; }

        private:
            AppearanceMatcher appearance_;
            SpatialMatcher spatial_;
    };

}  // namespace shipvision::mtmc
