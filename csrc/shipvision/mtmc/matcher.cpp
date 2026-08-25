#include "shipvision/mtmc/matcher.h"

#include <cmath>
#include <limits>

namespace shipvision::mtmc {

    void threshold_similarity(std::vector<float>& similarity, int n, float threshold) {
        const size_t count = static_cast<size_t>(n) * static_cast<size_t>(n);
        for (size_t index = 0; index < count && index < similarity.size(); ++index) {
            if (!(similarity[index] > threshold))
                similarity[index] = 0.f;
        }
    }

    std::vector<double> ground_distances(const float* points, const unsigned char* known, int n) {
        const size_t side = static_cast<size_t>(n);
        std::vector<double> distances(side * side, 0.0);
        const double unknown = std::numeric_limits<double>::infinity();
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (!known[i] || !known[j]) {
                    distances[static_cast<size_t>(i) * side + j] = unknown;
                    continue;
                }
                const double dx = static_cast<double>(points[i * 2]) - points[j * 2];
                const double dy = static_cast<double>(points[i * 2 + 1]) - points[j * 2 + 1];
                distances[static_cast<size_t>(i) * side + j] = std::sqrt(dx * dx + dy * dy);
            }
        }
        return distances;
    }

    std::vector<float> spatial_similarity(const double* distances, int n, float threshold) {
        const size_t side = static_cast<size_t>(n);
        std::vector<float> similarity(side * side, 0.f);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                const size_t index = static_cast<size_t>(i) * side + j;
                const double distance = distances[index];
                if (std::isfinite(distance) && distance < threshold)
                    similarity[index] = static_cast<float>(1.0 - distance / threshold);
            }
            // Set after the row so a zero-length threshold cannot leave a track dissimilar to
            // itself, which is what the numpy `np.fill_diagonal(similarity, 1.0)` guarantees.
            similarity[static_cast<size_t>(i) * side + i] = 1.f;
        }
        return similarity;
    }

    std::vector<unsigned char> spatial_gate(const double* distances, int n, float threshold) {
        const size_t side = static_cast<size_t>(n);
        std::vector<unsigned char> allowed(side * side, 0);
        for (size_t index = 0; index < side * side; ++index) {
            const double distance = distances[index];
            allowed[index] = (!std::isfinite(distance) || distance < threshold) ? 1 : 0;
        }
        return allowed;
    }

    void veto(std::vector<float>& similarity, const unsigned char* allowed, int n) {
        const size_t count = static_cast<size_t>(n) * static_cast<size_t>(n);
        for (size_t index = 0; index < count && index < similarity.size(); ++index) {
            if (!allowed[index])
                similarity[index] = 0.f;
        }
    }

    std::vector<float> to_distance(const float* similarity, const int* camera_codes, int n) {
        const size_t side = static_cast<size_t>(n);
        std::vector<float> distance(side * side, 0.f);
        // One pass, both triangles at once: the mask, the zero-means-never rule and the
        // symmetrisation are three numpy temporaries over the same 560 000 entries, and the
        // only reason they are three is that numpy has no way to express them as one.
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                const size_t upper = static_cast<size_t>(i) * side + j;
                const size_t lower = static_cast<size_t>(j) * side + i;
                // Each direction is converted BEFORE the two are averaged, which is the
                // order the numpy version uses and is not interchangeable with averaging the
                // similarities first: a pair whose two halves straddle zero — which BLAS can
                // produce, since `A @ A.T` is not promised to be bitwise symmetric — would
                // otherwise come out as ordinary evidence instead of as "never merge".
                const bool mergeable = camera_codes[i] != camera_codes[j];
                const float forward =
                    mergeable && similarity[upper] > 0.f ? 1.f - similarity[upper] : kNeverMerge;
                const float backward =
                    mergeable && similarity[lower] > 0.f ? 1.f - similarity[lower] : kNeverMerge;
                const float value = 0.5f * (forward + backward);
                distance[upper] = value;
                distance[lower] = value;
            }
            distance[static_cast<size_t>(i) * side + i] = 0.f;
        }
        return distance;
    }

}  // namespace shipvision::mtmc
