#include "shipvision/mot/trackers/botsort/tracker.h"

#include <stdexcept>

namespace shipvision::mot {

    BotSortTracker::BotSortTracker(const Options& options)
        : ByteTrackTracker(options.byte),
          appearance_gate_(options.appearance_gate),
          appearance_weight_(options.appearance_weight) {
        if (!(options.appearance_gate > 0.f && options.appearance_gate <= 2.f)) {
            throw std::invalid_argument(
                "appearance_gate is a cosine distance and must be in (0, 2]");
        }
        if (!(options.appearance_weight > 0.f && options.appearance_weight <= 1.f))
            throw std::invalid_argument("appearance_weight must be in (0, 1]");
    }

    void BotSortTracker::compensate(const FrameContext& context) {
        pool_.apply_camera_motion(context.affine);
    }

    std::vector<float> BotSortTracker::first_cost(const std::vector<int>& rows,
                                                  const std::vector<int>& columns,
                                                  const std::vector<Detection>& detections,
                                                  const FrameContext& context) const {
        const int n = static_cast<int>(rows.size());
        const int m = static_cast<int>(columns.size());
        const std::vector<float> track_boxes = gather_boxes(pool_.boxes().data(), rows);
        const std::vector<float> detection_boxes =
            gather_boxes(pack_boxes(detections).data(), columns);

        std::vector<float> cost = iou_cost(track_boxes.data(), n, detection_boxes.data(), m);
        if (!context.appearance.empty()) {
            const std::vector<float> appearance = gather_submatrix(
                context.appearance.data(), static_cast<int>(detections.size()), rows, columns);
            min_fuse(cost, appearance.data(), n, m, max_cost_, appearance_gate_,
                     appearance_weight_);
        }
        if (options_.gate) {
            const std::vector<float> distances =
                pool_.gating_distance(rows, detection_boxes.data(), m);
            gate_cost(cost, n, m, distances.data(), kChi2Inv95_4Dof);
        }
        return cost;
    }

}  // namespace shipvision::mot
