#include "shipvision/mtmc/matchers/appearance/matcher.h"

#include <cstring>
#include <stdexcept>
#include <string>

#include "shipvision/mtmc/matcher.h"

namespace shipvision::mtmc {

    AppearanceMatcher::AppearanceMatcher(float appearance_threshold)
        : appearance_threshold_(appearance_threshold) {
        // Refused at construction rather than at frame 40 000. A threshold outside [-1, 1] is
        // not a strict or a lenient matcher, it is one that admits everything or nothing, and
        // both look like a tuning problem from the outside.
        if (!(appearance_threshold >= -1.0f && appearance_threshold <= 1.0f)) {
            throw std::invalid_argument(
                "appearance_threshold is a cosine similarity and must be in [-1, 1], got " +
                std::to_string(appearance_threshold));
        }
    }

    std::vector<float> AppearanceMatcher::similarities(const float* gram, int n) const {
        const size_t count = static_cast<size_t>(n) * static_cast<size_t>(n);
        std::vector<float> similarity(count);
        if (count != 0)
            std::memcpy(similarity.data(), gram, count * sizeof(float));
        threshold_similarity(similarity, n, appearance_threshold_);
        return similarity;
    }

    std::vector<float> AppearanceMatcher::build(const float* gram, const int* codes, int n) const {
        const std::vector<float> similarity = similarities(gram, n);
        return to_distance(similarity.data(), codes, n);
    }

    std::vector<float> AppearanceMatcher::build(const float* gram, const Observation* observations,
                                                int n) const {
        const std::vector<int> codes = camera_codes(observations, n);
        return build(gram, codes.data(), n);
    }

}  // namespace shipvision::mtmc
