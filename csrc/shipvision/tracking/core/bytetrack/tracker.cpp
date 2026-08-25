#include "shipvision/tracking/core/bytetrack/tracker.h"

#include <numeric>
#include <stdexcept>

namespace shipvision::tracking {

    ByteTrackTracker::ByteTrackTracker(const Options& options)
        : options_(options),
          max_cost_(1.f - options.match_threshold),
          second_max_cost_(1.f - options.second_match_threshold),
          pool_(options.max_age, options.min_hits) {
        if (!(options.low_threshold < options.track_threshold && options.track_threshold <= 1.f)) {
            throw std::invalid_argument(
                "need low_threshold < track_threshold <= 1; with the two the other way round "
                "the high tier is empty and no track is ever born");
        }
    }

    void ByteTrackTracker::compensate(const FrameContext& context) {
        // ByteTrack assumes the camera is bolted down. See `BotSortTracker`, which is this one
        // line plus a different stage-one cost.
        (void)context;
    }

    std::vector<float> ByteTrackTracker::gated_iou(const std::vector<int>& rows,
                                                   const std::vector<int>& columns,
                                                   const std::vector<Detection>& detections) const {
        const int n = static_cast<int>(rows.size());
        const int m = static_cast<int>(columns.size());
        const std::vector<float> track_boxes = gather_boxes(pool_.boxes().data(), rows);
        const std::vector<float> detection_boxes =
            gather_boxes(pack_boxes(detections).data(), columns);

        std::vector<float> cost = iou_cost(track_boxes.data(), n, detection_boxes.data(), m);
        if (options_.gate) {
            const std::vector<float> distances =
                pool_.gating_distance(rows, detection_boxes.data(), m);
            gate_cost(cost, n, m, distances.data(), kChi2Inv95_4Dof);
        }
        return cost;
    }

    std::vector<float> ByteTrackTracker::first_cost(const std::vector<int>& rows,
                                                    const std::vector<int>& columns,
                                                    const std::vector<Detection>& detections,
                                                    const FrameContext& context) const {
        // ByteTrack's stage one never consults appearance; the parameter is here because the
        // subclass that does needs the same signature.
        (void)context;
        const int n = static_cast<int>(rows.size());
        const int m = static_cast<int>(columns.size());
        const std::vector<float> track_boxes = gather_boxes(pool_.boxes().data(), rows);
        const std::vector<float> detection_boxes =
            gather_boxes(pack_boxes(detections).data(), columns);
        std::vector<float> scores(columns.size());
        for (size_t index = 0; index < columns.size(); ++index)
            scores[index] = detections[static_cast<size_t>(columns[index])].score;

        std::vector<float> cost = iou_cost(track_boxes.data(), n, detection_boxes.data(), m);
        fuse_score(cost, n, m, scores.data());
        if (options_.gate) {
            const std::vector<float> distances =
                pool_.gating_distance(rows, detection_boxes.data(), m);
            gate_cost(cost, n, m, distances.data(), kChi2Inv95_4Dof);
        }
        return cost;
    }

    std::vector<Track> ByteTrackTracker::update(const std::vector<Detection>& detections,
                                                const FrameContext& context) {
        check_appearance(context, pool_.size(), detections.size());
        pool_.predict();
        compensate(context);

        // The split IS the algorithm — everything else in the paper follows from having two
        // tiers. Both tiers are lists of indices into the frame's own detection list, so every
        // index that reaches the pool is one a caller can use; see `Track::last_match`.
        std::vector<int> high;
        std::vector<int> low;
        for (size_t index = 0; index < detections.size(); ++index) {
            if (detections[index].score >= options_.track_threshold)
                high.push_back(static_cast<int>(index));
            else if (detections[index].score >= options_.low_threshold)
                low.push_back(static_cast<int>(index));
        }

        std::vector<int> rows(pool_.size());
        std::iota(rows.begin(), rows.end(), 0);

        Association first;
        if (!rows.empty() && !high.empty()) {
            first = associate_subset(first_cost(rows, high, detections, context), rows, high,
                                     max_cost_);
        } else {
            first.unmatched_rows = rows;
            first.unmatched_columns = high;
        }
        pool_.apply_matches(first.matches, detections);

        // Stage two, over CONFIRMED and LOST tracks only. A tentative track rescued by a
        // 0.3-confidence box is two weak pieces of evidence agreeing with each other, which is
        // not evidence — it is how a noise track becomes a published identity.
        std::vector<int> eligible;
        std::vector<int> ineligible;
        for (int row : first.unmatched_rows) {
            const TrackState state = pool_.tracks()[static_cast<size_t>(row)].state;
            if (state == TrackState::Confirmed || state == TrackState::Lost)
                eligible.push_back(row);
            else
                ineligible.push_back(row);
        }

        Association second;
        if (!eligible.empty() && !low.empty()) {
            second = associate_subset(gated_iou(eligible, low, detections), eligible, low,
                                      second_max_cost_);
        } else {
            second.unmatched_rows = eligible;
            second.unmatched_columns = low;
        }
        pool_.apply_matches(second.matches, detections);

        std::vector<int> missed = second.unmatched_rows;
        missed.insert(missed.end(), ineligible.begin(), ineligible.end());
        pool_.mark_missed(missed);

        // Only high-score detections may start a track. This is the asymmetry that keeps a
        // low-confidence false positive from ever becoming an identity.
        pool_.spawn(detections, first.unmatched_columns);
        pool_.sweep();
        return pool_.output();
    }

}  // namespace shipvision::tracking
