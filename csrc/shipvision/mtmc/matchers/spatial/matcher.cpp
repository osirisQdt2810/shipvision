#include "shipvision/mtmc/matchers/spatial/matcher.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "shipvision/mtmc/matcher.h"
#include "shipvision/mtmc/matchers/spatial/utils.h"

namespace shipvision::mtmc {

    SpatialMatcher::SpatialMatcher(const Options& options, GroundPlane plane)
        : options_(options), plane_(std::move(plane)) {
        // All three at construction. A spatial threshold of zero is not a strict gate, it is one
        // that refuses every pair including a track against itself, and it would present as
        // "cross-camera tracking stopped working" on a site that was calibrated yesterday.
        if (!(options.spatial_threshold > 0.0f)) {
            throw std::invalid_argument("spatial_threshold must be positive, got " +
                                        std::to_string(options.spatial_threshold));
        }
        if (!(options.aspect_ratio > 0.0 && options.aspect_ratio <= 4.0)) {
            throw std::invalid_argument(
                "aspect_ratio is width over height for a whole object and must be in (0, 4], "
                "got " +
                std::to_string(options.aspect_ratio));
        }
        if (!(options.foot_ratio > 0.0 && options.foot_ratio <= 2.0)) {
            throw std::invalid_argument("foot_ratio must be in (0, 2], got " +
                                        std::to_string(options.foot_ratio));
        }
    }

    void SpatialMatcher::ground_positions(const Observation* observations, int n, float* points,
                                          unsigned char* known) const {
        const float nowhere = std::numeric_limits<float>::quiet_NaN();
        for (int index = 0; index < n; ++index) {
            points[index * 2] = nowhere;
            points[index * 2 + 1] = nowhere;
            known[index] = 0;
        }
        if (n <= 0)
            return;

        // Grouped by camera because a homography is per camera, not per track: fifty 3x3
        // products instead of one per track, and the rescale into the calibration domain needs
        // one frame size for the whole group anyway.
        std::unordered_map<int, std::vector<int>> by_camera;
        for (int index = 0; index < n; ++index)
            by_camera[observations[index].camera_code].push_back(index);

        std::vector<float> boxes;
        std::vector<double> heights;
        std::vector<double> image;
        std::vector<float> projected;
        for (const auto& entry : by_camera) {
            const Homography* homography = plane_.get(entry.first);
            if (homography == nullptr)
                continue;  // uncalibrated: NaN and false, already written above

            const std::vector<int>& indices = entry.second;
            const int count = static_cast<int>(indices.size());
            boxes.resize(static_cast<size_t>(count) * 4);
            heights.resize(static_cast<size_t>(count));
            image.resize(static_cast<size_t>(count) * 2);
            projected.resize(static_cast<size_t>(count) * 2);
            for (int slot = 0; slot < count; ++slot) {
                const Observation& observation = observations[indices[static_cast<size_t>(slot)]];
                for (int axis = 0; axis < 4; ++axis)
                    boxes[static_cast<size_t>(slot) * 4 + axis] = observation.box[axis];
                heights[static_cast<size_t>(slot)] = static_cast<double>(observation.frame_height);
            }
            foot_points(boxes.data(), heights.data(), count, options_.foot_ratio,
                        options_.aspect_ratio, image.data());
            // Every observation in this group shares a camera, so one frame size applies.
            const Observation& first = observations[indices[0]];
            project(image.data(), count, *homography, first.frame_width, first.frame_height,
                    projected.data());
            for (int slot = 0; slot < count; ++slot) {
                const int index = indices[static_cast<size_t>(slot)];
                points[index * 2] = projected[static_cast<size_t>(slot) * 2];
                points[index * 2 + 1] = projected[static_cast<size_t>(slot) * 2 + 1];
                known[index] = 1;
            }
        }
    }

    std::vector<double> SpatialMatcher::ground_distances(const Observation* observations,
                                                         int n) const {
        if (n <= 0)
            return {};
        std::vector<float> points(static_cast<size_t>(n) * 2);
        std::vector<unsigned char> known(static_cast<size_t>(n));
        ground_positions(observations, n, points.data(), known.data());
        // The pairwise pass itself is `mtmc/matcher.h`'s, not a second copy of it: this class
        // owns where a track is standing, and that file owns what to do with n^2 of them.
        return mtmc::ground_distances(points.data(), known.data(), n);
    }

    std::vector<float> SpatialMatcher::similarities(const Observation* observations, int n) const {
        if (n <= 0)
            return {};
        const std::vector<double> ground = ground_distances(observations, n);
        return spatial_similarity(ground.data(), n, options_.spatial_threshold);
    }

    std::vector<unsigned char> SpatialMatcher::gate(const Observation* observations, int n) const {
        if (n <= 0)
            return {};
        const std::vector<double> ground = ground_distances(observations, n);
        return spatial_gate(ground.data(), n, options_.spatial_threshold);
    }

    std::vector<float> SpatialMatcher::build(const Observation* observations, int n) const {
        if (n <= 0)
            return {};
        const std::vector<float> similarity = similarities(observations, n);
        const std::vector<int> codes = camera_codes(observations, n);
        return to_distance(similarity.data(), codes.data(), n);
    }

}  // namespace shipvision::mtmc
