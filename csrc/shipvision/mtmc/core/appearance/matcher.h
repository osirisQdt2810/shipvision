// Cosine appearance similarity, hard-thresholded, with same-camera pairs excluded.
//
// The C++ twin of `shipvision/mtmc/core/appearance/matcher.py`, and the baseline matcher: the
// only one that works on an uncalibrated site, and the right thing to compose a geometric gate
// on top of rather than a competitor to it.
//
// THE GEMM IS NOT HERE, AND THAT IS THE PONYTAIL PRINCIPLE APPLIED RATHER THAN IGNORED.
// `features @ features.T` is what BLAS is for — multithreaded, blocked for the cache — and a
// triple loop in this file would be slower than the thing it replaced while looking like an
// optimisation. So this class takes the gram matrix the caller already has and owns everything
// *around* it: the threshold, the same-camera exclusion and the conversion to a clusterable
// distance, which in numpy are four separate (n, n) temporaries over the same 560 000 entries.
//
// The Python package's `utils.py` has no counterpart here for the same reason. Its whole content
// is a refusal — every track needs an embedding, and two cameras must not be running different
// re-ID models — plus the L2 normalisation that has to happen BEFORE the gemm. Both belong on
// the numpy side of the boundary, and duplicating the refusal here would mean two subtly
// different opinions about what input is acceptable.

#pragma once

#include <vector>

#include "shipvision/mtmc/frames.h"

namespace shipvision::mtmc {

    class AppearanceMatcher {
        public:
            /// @param appearance_threshold minimum cosine similarity for a pair to be considered
            ///        at all. 0.86 is the reference implementation's production value.
            ///
            /// The hard threshold is not a tuning nicety. Without it average-linkage clustering
            /// is free to chain: A resembles B a little, B resembles C a little, and a threshold
            /// on the *average* groups all three even though A and C are strangers. Zeroing weak
            /// evidence means a chain has to be built out of links that each stand on their own.
            ///
            /// Cosine only, with no metric switch. On L2-normalised vectors euclidean distance is
            /// a monotone function of cosine distance and therefore ranks identically, so a
            /// metric option would change nothing except the scale the threshold is expressed in.
            /// The reference had that switch and shipped two configurations whose thresholds
            /// (0.55 and 0.86) were not comparable — a way to misconfigure a system rather than a
            /// way to improve one.
            explicit AppearanceMatcher(float appearance_threshold = 0.86f);

            /// `(n, n)` thresholded cosine similarity. Zero means "no appearance evidence".
            ///
            /// Deliberately NOT camera-masked: the mask belongs to exactly one place,
            /// `to_distance`, and having it applied twice is how it ends up applied zero times
            /// after a refactor. `GatedMatcher` composes this, because it needs the raw
            /// appearance evidence before deciding whether geometry vetoes it.
            ///
            /// @param gram `(n, n)` row-major cosine similarity, from the caller's BLAS
            std::vector<float> similarities(const float* gram, int n) const;

            /// `(n, n)` clusterable distance: threshold, then the shared conversion.
            ///
            /// Takes the camera codes directly rather than a whole instant, because that is
            /// genuinely all the appearance half reads. Asking a caller with no geometry for
            /// boxes and frame sizes would be asking it to invent them, and an invented frame
            /// height is what makes a foot point land somewhere plausible and wrong.
            std::vector<float> build(const float* gram, const int* camera_codes, int n) const;

            /// The same, for a caller that already has the instant. `GatedMatcher` composes this.
            std::vector<float> build(const float* gram, const Observation* observations,
                                     int n) const;

            float appearance_threshold() const { return appearance_threshold_; }

        private:
            float appearance_threshold_ = 0.86f;
    };

}  // namespace shipvision::mtmc
