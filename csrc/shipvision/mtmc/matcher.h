// Cross-camera similarity: the (n, n) passes that turn evidence into a clusterable distance.
//
// WHAT IS HERE AND WHAT IS DELIBERATELY NOT
//
// The cosine similarity itself is **not** here, and that is the ponytail principle applied
// rather than ignored. `features @ features.T` is a gemm; numpy hands it to BLAS, which is
// multithreaded and blocked for the cache, and a naive triple loop in this file would be
// slower than the thing it replaced while looking like an optimisation. The Python matcher
// keeps that call.
//
// What is here is everything *around* it, which in numpy is five full (n, n) temporaries —
// threshold, gate, veto, symmetrise, mask — each of them a separate pass over a matrix that at
// fifty cameras and fifteen tracks each is 560 000 entries, rebuilt once per synchronised
// instant. Fusing them into one pass is the same argument the letterbox kernel makes: several
// memory-bound passes over one buffer, run as one.
//
// The ground-plane distance is here for a second reason as well. The reference implementation
// computed the same-camera exclusion with a nested string comparison, which is 560 000 string
// compares per instant — on its own more expensive than the clustering it feeds. Camera
// identity crosses this boundary as an integer code, assigned once on the Python side.
//
// Every function mirrors one function in `shipvision/mtmc/`, and the parity tests compare them
// one for one. Matrices are row-major `(n, n)`.

#pragma once

#include <vector>

namespace shipvision::mtmc {

    /// The distance between two tracks that must not be grouped. Finite on purpose.
    ///
    /// Hierarchical clustering on a precomputed matrix cannot take a non-finite input:
    /// `scipy.spatial.distance.squareform` rejects it outright, and an average-linkage update
    /// that got past that would compute `inf - inf` and produce NaN, which poisons the rest of
    /// the dendrogram instead of failing. A finite sentinel that is simply enormous next to a
    /// threshold of ~0.15 gives the arithmetic something to work with while keeping the
    /// meaning. Must equal `NEVER_MERGE` in `shipvision/mtmc/base.py`.
    constexpr float kNeverMerge = 1e5f;

    /// Zero every similarity at or below `threshold`, in place.
    ///
    /// The hard threshold is not a tuning nicety. Without it, average-linkage clustering is
    /// free to chain: A resembles B a little, B resembles C a little, and a threshold on the
    /// *average* groups all three even though A and C are strangers. Zeroing weak evidence
    /// means a chain has to be built out of links that each stand on their own.
    void threshold_similarity(std::vector<float>& similarity, int n, float threshold);

    /// `(n, n)` euclidean distance between ground-plane points; infinity where unknowable.
    ///
    /// Infinity is the honest value for a pair where at least one camera is uncalibrated: not
    /// "far apart" and not "close", but "this matcher has nothing to say". The two consumers
    /// decide differently what that means — `spatial_similarity` refuses the pair,
    /// `spatial_gate` lets it through — which is why the decision is not taken here.
    ///
    /// Double rather than float to match the numpy version's dtype exactly: a map coordinate
    /// can be tens of thousands of units from the origin, and squaring a difference of those in
    /// float32 loses the last digits the threshold comparison is made of.
    std::vector<double> ground_distances(const float* points, const unsigned char* known, int n);

    /// `(n, n)` similarity in `[0, 1]`: 1 at zero separation, falling linearly to 0 at the
    /// threshold, and exactly 0 beyond it or where the pair cannot be judged. Diagonal 1.
    std::vector<float> spatial_similarity(const double* distances, int n, float threshold);

    /// `(n, n)` "geometry does not object", as one byte per pair.
    ///
    /// True when the two projections are within `threshold` **and** when the pair cannot be
    /// judged at all. Falling open on "cannot judge" is what lets an uncalibrated camera keep
    /// taking part in cross-camera tracking on appearance alone, instead of quietly never
    /// merging with anyone.
    std::vector<unsigned char> spatial_gate(const double* distances, int n, float threshold);

    /// Set every similarity the gate refuses to EXACTLY zero, in place.
    ///
    /// Exactly zero because `to_distance` is what turns zero into `kNeverMerge`. Scale the
    /// similarity down instead, or subtract a penalty, and a pair the geometry ruled impossible
    /// is merely expensive — which average linkage will happily buy the moment somebody loosens
    /// a threshold, and no test of either half would notice.
    void veto(std::vector<float>& similarity, const unsigned char* allowed, int n);

    /// Similarities to clusterable distances, with the same-camera exclusion folded in.
    ///
    /// Zero similarity becomes `kNeverMerge` rather than a distance of 1. That distinction is
    /// the whole point of thresholding earlier: "these two scored 0.2, which is below the bar"
    /// and "these two are in the same camera" both mean *do not group*, and expressing both as
    /// 1.0 would let average linkage merge them anyway once a threshold moved.
    ///
    /// Two tracks in one camera can never merge, and this is the one place that says so. If
    /// they were the same object the single-camera tracker upstream had one job and failed at
    /// it; merge them anyway and MTMC quietly becomes a within-camera deduplicator — every
    /// count drops, every metric improves, and the system is worse.
    ///
    /// The result is symmetric with a zero diagonal, because that is what the clusterer
    /// requires: `squareform` has no tolerance for asymmetry, it silently reads the upper
    /// triangle. Symmetrised explicitly rather than assumed — BLAS does not promise bitwise
    /// symmetry for `A @ A.T`.
    ///
    /// @param camera_codes one integer per track, equal iff two tracks share a camera
    std::vector<float> to_distance(const float* similarity, const int* camera_codes, int n);

}  // namespace shipvision::mtmc
