#include "shipvision/mot/trackers/sort/tracker.h"

#include <numeric>
#include <stdexcept>

#include "shipvision/mot/association.h"

namespace shipvision::mot {

    SortTracker::SortTracker(const Options& options)
        : options_(options),
          max_cost_(1.f - options.iou_threshold),
          pool_(options.max_age, options.min_hits) {
        // Validated here as well as in the Python wrapper, because this constructor is
        // reachable from the bindings without it. A bad configuration must stop the process at
        // start-up; discovering it at frame 40 000 costs a camera's worth of footage.
        if (!(options.det_threshold >= 0.f && options.det_threshold <= 1.f))
            throw std::invalid_argument("det_threshold must be in [0, 1]");
        if (!(options.iou_threshold > 0.f && options.iou_threshold <= 1.f))
            throw std::invalid_argument("iou_threshold must be in (0, 1]");
    }

    std::vector<Track> SortTracker::update(const std::vector<Detection>& detections) {
        pool_.predict();

        // The kept detections, as indices into the frame's own list. Indices rather than a
        // second vector of detections, so every index this function hands to the pool is one a
        // caller can use — see `Track::last_match`.
        std::vector<int> kept;
        kept.reserve(detections.size());
        for (size_t index = 0; index < detections.size(); ++index) {
            if (detections[index].score >= options_.det_threshold)
                kept.push_back(static_cast<int>(index));
        }

        std::vector<int> rows(pool_.size());
        std::iota(rows.begin(), rows.end(), 0);

        Association result;
        if (!rows.empty() && !kept.empty()) {
            const int n = static_cast<int>(rows.size());
            const int m = static_cast<int>(kept.size());
            const std::vector<float> track_boxes = pool_.boxes();
            const std::vector<float> detection_boxes =
                gather_boxes(pack_boxes(detections).data(), kept);

            std::vector<float> cost = iou_cost(track_boxes.data(), n, detection_boxes.data(), m);
            if (options_.gate) {
                const std::vector<float> distances =
                    pool_.gating_distance(rows, detection_boxes.data(), m);
                gate_cost(cost, n, m, distances.data(), kChi2Inv95_4Dof);
            }
            result = associate_subset(cost, rows, kept, max_cost_);
        } else {
            result.unmatched_rows = rows;
            result.unmatched_columns = kept;
        }

        pool_.apply_matches(result.matches, detections);
        pool_.mark_missed(result.unmatched_rows);
        pool_.spawn(detections, result.unmatched_columns);
        pool_.sweep();
        return pool_.output();
    }

}  // namespace shipvision::mot
