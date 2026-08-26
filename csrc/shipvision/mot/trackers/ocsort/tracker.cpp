#include "shipvision/mot/trackers/ocsort/tracker.h"

#include <numeric>
#include <stdexcept>

#include "shipvision/mot/association.h"

namespace shipvision::mot {

    OcSortTracker::OcSortTracker(const Options& options)
        : options_(options),
          max_cost_(1.f - options.iou_threshold),
          recovery_max_cost_(1.f - options.recovery_iou_threshold),
          // The ring is `delta_t + 1` long because that is the smallest history that can
          // measure a heading over `delta_t` frames — and it is bounded, because this process
          // runs for weeks and an unbounded per-track history is a slow leak with no symptom.
          pool_(options.max_age, options.min_hits, options.delta_t + 1, options.re_update) {
        // Validated here as well as in the Python wrapper, because this constructor is
        // reachable from the bindings without it. A bad configuration must stop the process at
        // start-up; discovering it at frame 40 000 costs a camera's worth of footage.
        if (!(options.det_threshold >= 0.f && options.det_threshold <= 1.f))
            throw std::invalid_argument("det_threshold must be in [0, 1]");
        if (!(options.iou_threshold > 0.f && options.iou_threshold <= 1.f))
            throw std::invalid_argument("iou_threshold must be in (0, 1]");
        if (!(options.recovery_iou_threshold > 0.f && options.recovery_iou_threshold <= 1.f))
            throw std::invalid_argument("recovery_iou_threshold must be in (0, 1]");
        if (options.delta_t < 1)
            throw std::invalid_argument("delta_t must be >= 1");
        if (!(options.momentum_weight >= 0.f && options.momentum_weight <= 1.f))
            throw std::invalid_argument("momentum_weight must be in [0, 1]");
    }

    std::vector<float> OcSortTracker::primary_cost(
        const std::vector<int>& rows, const std::vector<int>& columns,
        const std::vector<float>& detection_boxes) const {
        const int n = static_cast<int>(rows.size());
        const int m = static_cast<int>(columns.size());
        const std::vector<float> track_boxes = gather_boxes(pool_.boxes().data(), rows);
        const std::vector<float> boxes = gather_boxes(detection_boxes.data(), columns);

        std::vector<float> cost = iou_cost(track_boxes.data(), n, boxes.data(), m);
        // The ring and the last-observation array are read only when the momentum term is on,
        // so a tracker with momentum_weight = 0 pays nothing for state it still maintains.
        if (options_.momentum_weight > 0.f) {
            const std::vector<float> all_headings = pool_.directions(options_.delta_t);
            std::vector<float> headings(rows.size() * 2);
            for (size_t index = 0; index < rows.size(); ++index) {
                headings[index * 2 + 0] = all_headings[static_cast<size_t>(rows[index]) * 2 + 0];
                headings[index * 2 + 1] = all_headings[static_cast<size_t>(rows[index]) * 2 + 1];
            }
            const std::vector<float> origins = gather_boxes(pool_.observed_boxes().data(), rows);
            const std::vector<float> direction =
                direction_cost(headings.data(), origins.data(), n, boxes.data(), m);
            for (size_t index = 0; index < cost.size(); ++index)
                cost[index] += options_.momentum_weight * direction[index];
        }
        if (options_.gate) {
            const std::vector<float> distances = pool_.gating_distance(rows, boxes.data(), m);
            gate_cost(cost, n, m, distances.data(), kChi2Inv95_4Dof);
        }
        return cost;
    }

    std::vector<float> OcSortTracker::recovery_cost(
        const std::vector<int>& rows, const std::vector<int>& columns,
        const std::vector<float>& detection_boxes) const {
        const std::vector<float> observed = gather_boxes(pool_.observed_boxes().data(), rows);
        const std::vector<float> boxes = gather_boxes(detection_boxes.data(), columns);
        return iou_cost(observed.data(), static_cast<int>(rows.size()), boxes.data(),
                        static_cast<int>(columns.size()));
    }

    std::vector<Track> OcSortTracker::update(const std::vector<Detection>& detections) {
        pool_.predict();

        // OC-SORT is a single-threshold tracker; the low-score second pass is ByteTrack's idea
        // and lives there. The kept list is indices into the frame's own detection list, so
        // every index that reaches the pool is one a caller can use — see `Track::last_match`.
        std::vector<int> kept;
        for (size_t index = 0; index < detections.size(); ++index) {
            if (detections[index].score >= options_.det_threshold)
                kept.push_back(static_cast<int>(index));
        }
        const std::vector<float> detection_boxes = pack_boxes(detections);

        std::vector<int> rows(pool_.size());
        std::iota(rows.begin(), rows.end(), 0);

        Association primary;
        if (!rows.empty() && !kept.empty()) {
            primary =
                associate_subset(primary_cost(rows, kept, detection_boxes), rows, kept, max_cost_);
        } else {
            primary.unmatched_rows = rows;
            primary.unmatched_columns = kept;
        }
        pool_.apply_matches(primary.matches, detections);

        std::vector<int> unmatched_rows = primary.unmatched_rows;
        std::vector<int> unmatched_columns = primary.unmatched_columns;
        if (options_.recover) {
            // OCR sees only tracks that have earned trust. Offering a tentative track a
            // stale-box match is the same "two weak signals agreeing" mistake ByteTrack's
            // second stage avoids, and here it would additionally resurrect noise.
            std::vector<int> eligible;
            std::vector<int> ineligible;
            for (int row : unmatched_rows) {
                const TrackState state = pool_.tracks()[static_cast<size_t>(row)].state;
                if (state == TrackState::Confirmed || state == TrackState::Lost)
                    eligible.push_back(row);
                else
                    ineligible.push_back(row);
            }

            Association recovered;
            if (!eligible.empty() && !unmatched_columns.empty()) {
                recovered =
                    associate_subset(recovery_cost(eligible, unmatched_columns, detection_boxes),
                                     eligible, unmatched_columns, recovery_max_cost_);
            } else {
                recovered.unmatched_rows = eligible;
                recovered.unmatched_columns = unmatched_columns;
            }
            pool_.apply_matches(recovered.matches, detections);

            unmatched_rows = recovered.unmatched_rows;
            unmatched_rows.insert(unmatched_rows.end(), ineligible.begin(), ineligible.end());
            unmatched_columns = recovered.unmatched_columns;
        }

        pool_.mark_missed(unmatched_rows);
        pool_.spawn(detections, unmatched_columns);
        pool_.sweep();
        return pool_.output();
    }

}  // namespace shipvision::mot
