#include "shipvision/mtmc/clustering/agglomerative.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace shipvision::mtmc {

    AgglomerativeClusterer::AgglomerativeClusterer(double distance_threshold)
        : distance_threshold_(distance_threshold) {
        if (!(distance_threshold > 0.0)) {
            throw std::invalid_argument("distance_threshold must be positive, got " +
                                        std::to_string(distance_threshold));
        }
    }

    std::vector<int> AgglomerativeClusterer::fit_predict(const double* distances, int n) const {
        if (n <= 0)
            return {};
        if (n == 1)
            return {0};

        const size_t side = static_cast<size_t>(n);
        std::vector<double> working(side * side);
        for (size_t i = 0; i < side; ++i) {
            for (size_t j = 0; j < side; ++j) {
                const double value = distances[i * side + j];
                if (!std::isfinite(value)) {
                    throw std::invalid_argument(
                        "the distance matrix contains inf or NaN. Use the finite NEVER_MERGE "
                        "sentinel for pairs that must not be grouped: hierarchical clustering "
                        "cannot consume non-finite distances, and average linkage turns them "
                        "into NaN rather than into a refusal");
                }
                working[i * side + j] = value;
            }
        }
        // Symmetrise and zero the diagonal before anything reads a pair. Both are what the
        // Python side does before handing scipy a condensed matrix, and for the same reason:
        // `squareform`'s tolerance for asymmetry is zero, and with its checks off it silently
        // keeps the upper triangle — so half a matrix gets clustered and the result is plausible.
        for (size_t i = 0; i < side; ++i) {
            for (size_t j = i + 1; j < side; ++j) {
                const double mean = 0.5 * (working[i * side + j] + working[j * side + i]);
                working[i * side + j] = mean;
                working[j * side + i] = mean;
            }
            working[i * side + i] = 0.0;
        }

        std::vector<int> parent(side);
        std::vector<double> weight(side, 1.0);
        std::vector<unsigned char> active(side, 1);
        for (size_t index = 0; index < side; ++index)
            parent[index] = static_cast<int>(index);

        for (int merges = 0; merges + 1 < n; ++merges) {
            double best = std::numeric_limits<double>::infinity();
            int left = -1;
            int right = -1;
            for (int i = 0; i < n; ++i) {
                if (!active[static_cast<size_t>(i)])
                    continue;
                for (int j = i + 1; j < n; ++j) {
                    if (!active[static_cast<size_t>(j)])
                        continue;
                    const double value = working[static_cast<size_t>(i) * side + j];
                    if (value < best) {
                        best = value;
                        left = i;
                        right = j;
                    }
                }
            }
            // `<=`, not `<`. scipy's `fcluster(criterion="distance")` keeps every merge whose
            // cophenetic distance is at most t, and average linkage is monotone, so stopping the
            // greedy sequence at the first merge above t reproduces exactly that cut. A strict
            // comparison here would split a group whose members sit exactly on the threshold —
            // which is what a threshold copied from a tuning run is made of.
            if (left < 0 || !(best <= distance_threshold_))
                break;

            // Lance-Williams for average linkage (UPGMA): the merged cluster's distance to a
            // third is the size-weighted mean of its parts'. Weighted by cluster SIZE, not by a
            // plain average of the two — the unweighted variant (WPGMA) is a different algorithm
            // that lets a singleton outvote a group of five, and both are called "average".
            const double left_weight = weight[static_cast<size_t>(left)];
            const double right_weight = weight[static_cast<size_t>(right)];
            const double total = left_weight + right_weight;
            for (int k = 0; k < n; ++k) {
                if (k == left || k == right || !active[static_cast<size_t>(k)])
                    continue;
                const double merged =
                    (left_weight * working[static_cast<size_t>(left) * side + k] +
                     right_weight * working[static_cast<size_t>(right) * side + k]) /
                    total;
                working[static_cast<size_t>(left) * side + k] = merged;
                working[static_cast<size_t>(k) * side + left] = merged;
            }
            weight[static_cast<size_t>(left)] = total;
            active[static_cast<size_t>(right)] = 0;
            parent[static_cast<size_t>(right)] = left;
        }

        std::vector<int> labels(side, -1);
        int next_label = 0;
        for (int index = 0; index < n; ++index) {
            int root = index;
            while (parent[static_cast<size_t>(root)] != root)
                root = parent[static_cast<size_t>(root)];
            // Numbered by first appearance, so that "these two implementations agree" is one
            // array comparison rather than a partition isomorphism.
            if (labels[static_cast<size_t>(root)] < 0)
                labels[static_cast<size_t>(root)] = next_label++;
            labels[static_cast<size_t>(index)] = labels[static_cast<size_t>(root)];
        }
        return labels;
    }

}  // namespace shipvision::mtmc
