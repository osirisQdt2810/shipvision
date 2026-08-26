// Average-linkage agglomerative clustering with a distance cut.
//
// The C++ twin of `shipvision/mtmc/clustering/agglomerative.py`, which is a fifteen-line call
// into scipy. Four words carry the whole algorithm and each of them is a decision:
//
// AGGLOMERATIVE, because the number of identities in front of a camera group at one instant is
// exactly what is being asked and cannot be supplied. Nothing here takes a `k`; the reference
// threaded an unused `n_clusters` through four layers on the way to a function that ignored it.
//
// AVERAGE LINKAGE, because single linkage chains — A near B, B near C, so A, B and C are one
// person even though A and C are strangers — and complete linkage refuses a third view of an
// identity whose worst pairing is mediocre, which is what a partially-occluded crop always is.
// Average is also what makes the `kNeverMerge` sentinel work: one forbidden pair drags a
// candidate merge's mean distance to ~50 000, so the group is never formed.
//
// A DISTANCE CUT, not a cluster count.
//
// ON A PRECOMPUTED MATRIX, because the evidence combined in it — cosine appearance, a
// ground-plane veto, a same-camera exclusion — is not a metric embedding of anything and there
// are no coordinates to hand a clusterer instead.
//
// WHY THIS EXISTS IN C++ AT ALL, given the ponytail principle says to call scipy. Because there
// is no scipy on this side of the boundary, and the alternative is not "call the library", it is
// "cross back into Python between the matcher and the clusterer" — which is the crossing the
// native matcher exists to remove. What is written here is the Lance-Williams update itself, the
// eight lines the textbook gives; the reference implementation vendors 2 400 lines of
// hand-written reciprocal-nearest-neighbour clustering to do the same thing, and that is the
// shape of code the principle is actually pointed at.

#pragma once

#include <vector>

namespace shipvision::mtmc {

    class AgglomerativeClusterer {
        public:
            /// @param distance_threshold the cut, in the matrix's own units. With the appearance
            ///        matcher that is `1 - cosine_similarity` — 0.14 meaning "group things that
            ///        are at least 0.86 alike", the reference's production value and
            ///        deliberately the same number as its appearance threshold. The two doing
            ///        the same work is not redundant: the appearance threshold removes weak
            ///        *pairs* before linkage, and this one bounds the *average* over a group, so
            ///        without the first a group of mediocre pairs still averages under the cut.
            explicit AgglomerativeClusterer(double distance_threshold = 0.14);

            /// `(n, n)` distances to `(n,)` labels. Only equality between labels carries meaning.
            ///
            /// Numbered by first appearance — the cluster containing track 0 is label 0, the
            /// first track not in it is label 1 — which scipy's numbering is not. That is
            /// deliberate and it is what makes a parity test a plain array comparison instead of
            /// a partition isomorphism check: the two implementations must agree on *which
            /// tracks are together*, and canonical numbering is the only way to say that in one
            /// `==`.
            ///
            /// `n == 0` returns an empty vector and `n == 1` returns `{0}`. One visible track is
            /// the normal state of a quiet site, not an edge case — and it is also the input
            /// scipy refuses outright, which is why it is answered before any linkage happens.
            ///
            /// Ties are broken by the lowest pair index, where scipy breaks them by its own merge
            /// order. Average linkage is monotone, so the *cut* is what a tie can move and only
            /// when two candidate merges sit at exactly the same distance to the last bit.
            ///
            /// @throws std::invalid_argument the matrix is not square, or contains inf or NaN.
            ///         The non-finite check is the one worth having: inf is the obvious way to
            ///         say "never merge these" and it is wrong, because averaging it computes
            ///         `inf - inf` and gets NaN, which does not fail — it produces a dendrogram
            ///         whose merges are arbitrary. That is why `kNeverMerge` is a large finite
            ///         number, and this is where a builder that ignored it finds out.
            std::vector<int> fit_predict(const double* distances, int n) const;

            double distance_threshold() const { return distance_threshold_; }

        private:
            double distance_threshold_ = 0.14;
    };

}  // namespace shipvision::mtmc
