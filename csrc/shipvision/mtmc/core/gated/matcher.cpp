#include "shipvision/mtmc/core/gated/matcher.h"

#include "shipvision/mtmc/matcher.h"

namespace shipvision::mtmc {

    GatedMatcher::GatedMatcher(AppearanceMatcher appearance, SpatialMatcher spatial)
        : appearance_(appearance), spatial_(std::move(spatial)) {}

    std::vector<float> GatedMatcher::similarities(const float* gram,
                                                  const Observation* observations, int n) const {
        std::vector<float> similarity = appearance_.similarities(gram, n);
        if (similarity.empty())
            return similarity;
        const std::vector<unsigned char> allowed = spatial_.gate(observations, n);
        veto(similarity, allowed.data(), n);
        return similarity;
    }

    std::vector<float> GatedMatcher::build(const float* gram, const Observation* observations,
                                           int n) const {
        const std::vector<float> similarity = similarities(gram, observations, n);
        const std::vector<int> codes = camera_codes(observations, n);
        return to_distance(similarity.data(), codes.data(), n);
    }

}  // namespace shipvision::mtmc
