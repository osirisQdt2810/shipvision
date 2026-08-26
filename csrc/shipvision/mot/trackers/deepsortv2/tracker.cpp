#include "shipvision/mot/trackers/deepsortv2/tracker.h"

#include <algorithm>
#include <stdexcept>

#include "shipvision/mot/association.h"

namespace shipvision::mot {

    namespace {

        /// The members of `rows` that are not in `taken`, keeping `rows` order.
        std::vector<int> difference(const std::vector<int>& rows, const std::vector<int>& taken) {
            std::vector<int> remainder;
            for (int row : rows) {
                if (std::find(taken.begin(), taken.end(), row) == taken.end())
                    remainder.push_back(row);
            }
            return remainder;
        }

    }  // namespace

    DeepSortV2Tracker::DeepSortV2Tracker(const Options& options)
        : options_(options),
          // The only one of the five pools that asks for the appearance EMA *and* the
          // observation-centric re-update — and that combination is what "DeepSORTv2" names.
          // The EMA itself lives in Python; what the pool has to provide is ORU.
          pool_(options.max_age, options.min_hits, 0, options.re_update) {
        // Validated here as well as in the Python wrapper, because this constructor is
        // reachable from the bindings without it. A bad configuration must stop the process at
        // start-up; discovering it at frame 40 000 costs a camera's worth of footage.
        if (!(options.det_threshold >= 0.f && options.det_threshold <= 1.f))
            throw std::invalid_argument("det_threshold must be in [0, 1]");
        if (!(options.appearance_weight >= 0.f && options.appearance_weight <= 1.f))
            throw std::invalid_argument("appearance_weight must be in [0, 1]");
        if (options.cascade_stride < 1)
            throw std::invalid_argument("cascade_stride must be >= 1");
        if (!(options.border_fraction >= 0.f && options.border_fraction < 0.5f))
            throw std::invalid_argument("border_fraction must be in [0, 0.5)");
    }

    std::vector<float> DeepSortV2Tracker::stage_a_cost(const std::vector<int>& rows,
                                                       const std::vector<int>& columns,
                                                       const std::vector<float>& detection_boxes,
                                                       const FrameContext& context) const {
        const int n = static_cast<int>(rows.size());
        const int m = static_cast<int>(columns.size());
        const std::vector<float> track_boxes = gather_boxes(pool_.boxes().data(), rows);
        const std::vector<float> boxes = gather_boxes(detection_boxes.data(), columns);

        std::vector<float> geometry = giou_cost(track_boxes.data(), n, boxes.data(), m);
        // Stage A's Mahalanobis gate is required, not optional: this is where a confirmed track
        // gets first refusal on the best evidence in the frame, and an ungated first refusal is
        // how one crowded frame swaps two identities.
        const std::vector<float> distances = pool_.gating_distance(rows, boxes.data(), m);

        if (context.appearance.empty()) {
            gate_cost(geometry, n, m, distances.data(), kChi2Inv95_4Dof);
            return geometry;
        }

        const std::vector<float> appearance = gather_submatrix(
            context.appearance.data(), static_cast<int>(detection_boxes.size() / 4), rows, columns);
        std::vector<float> fused(geometry.size());
        for (size_t index = 0; index < fused.size(); ++index) {
            const bool forbidden = geometry[index] > options_.giou_gate ||
                                   appearance[index] > options_.appearance_gate;
            fused[index] = forbidden ? kInfeasible
                                     : options_.appearance_weight * appearance[index] +
                                           (1.f - options_.appearance_weight) * geometry[index];
        }
        gate_cost(fused, n, m, distances.data(), kChi2Inv95_4Dof);
        return fused;
    }

    std::vector<float> DeepSortV2Tracker::stage_b_cost(const std::vector<int>& rows,
                                                       const std::vector<int>& columns,
                                                       const std::vector<float>& detection_boxes,
                                                       const FrameContext& context) const {
        const int n = static_cast<int>(rows.size());
        const int m = static_cast<int>(columns.size());
        const std::vector<float> track_boxes = gather_boxes(pool_.boxes().data(), rows);
        const std::vector<float> boxes = gather_boxes(detection_boxes.data(), columns);

        std::vector<float> cost = iou_cost(track_boxes.data(), n, boxes.data(), m);
        if (!context.appearance.empty()) {
            const std::vector<float> appearance =
                gather_submatrix(context.appearance.data(),
                                 static_cast<int>(detection_boxes.size() / 4), rows, columns);
            for (size_t index = 0; index < cost.size(); ++index) {
                if (appearance[index] > options_.appearance_gate)
                    cost[index] = kInfeasible;
            }
        }
        const std::vector<float> distances = pool_.gating_distance(rows, boxes.data(), m);
        gate_cost(cost, n, m, distances.data(), kChi2Inv95_4Dof);
        return cost;
    }

    std::vector<float> DeepSortV2Tracker::stage_c_cost(
        const std::vector<int>& rows, const std::vector<int>& columns,
        const std::vector<float>& detection_boxes) const {
        const std::vector<float> observed = gather_boxes(pool_.observed_boxes().data(), rows);
        const std::vector<float> boxes = gather_boxes(detection_boxes.data(), columns);
        return iou_cost(observed.data(), static_cast<int>(rows.size()), boxes.data(),
                        static_cast<int>(columns.size()));
    }

    std::vector<float> DeepSortV2Tracker::stage_d_cost(
        const std::vector<int>& rows, const std::vector<int>& columns,
        const std::vector<float>& detection_boxes) const {
        const std::vector<float> track_boxes = gather_boxes(pool_.boxes().data(), rows);
        const std::vector<float> boxes = gather_boxes(detection_boxes.data(), columns);
        return iou_cost(track_boxes.data(), static_cast<int>(rows.size()), boxes.data(),
                        static_cast<int>(columns.size()));
    }

    std::vector<int> DeepSortV2Tracker::recoverable(const std::vector<int>& rows,
                                                    const FrameContext& context) const {
        std::vector<int> eligible;
        for (int row : rows) {
            const TrackState state = pool_.tracks()[static_cast<size_t>(row)].state;
            if (state == TrackState::Confirmed || state == TrackState::Lost)
                eligible.push_back(row);
        }
        // The empty check short-circuits before touching the pool: `observed_boxes()` builds
        // the whole array, and stage C runs once per frame per camera on the path that is
        // already the one under measurement.
        if (!options_.skip_border_recovery || eligible.empty() || context.height <= 0 ||
            context.width <= 0) {
            return eligible;
        }

        const std::vector<float> observed = pool_.observed_boxes();
        const float margin =
            options_.border_fraction * static_cast<float>(std::min(context.height, context.width));
        std::vector<int> away_from_the_edge;
        for (int row : eligible) {
            const float* box = observed.data() + static_cast<size_t>(row) * 4;
            const bool at_edge = box[0] < margin || box[1] < margin ||
                                 static_cast<float>(context.width) - box[2] < margin ||
                                 static_cast<float>(context.height) - box[3] < margin;
            if (!at_edge)
                away_from_the_edge.push_back(row);
        }
        return away_from_the_edge;
    }

    std::vector<Track> DeepSortV2Tracker::update(const std::vector<Detection>& detections,
                                                 const FrameContext& context) {
        check_appearance(context, pool_.size(), detections.size());
        pool_.predict();

        std::vector<int> columns;
        for (size_t index = 0; index < detections.size(); ++index) {
            if (detections[index].score >= options_.det_threshold)
                columns.push_back(static_cast<int>(index));
        }
        const std::vector<float> detection_boxes = pack_boxes(detections);

        std::vector<int> tentative;
        std::vector<int> established;
        for (size_t row = 0; row < pool_.size(); ++row) {
            if (pool_.tracks()[row].state == TrackState::Tentative)
                tentative.push_back(static_cast<int>(row));
            else
                established.push_back(static_cast<int>(row));
        }
        // Read once, before anything is matched: every stage bands on the age the track had
        // when the frame opened, and `apply_matches` runs only after all four have chosen.
        const std::vector<int> ages = pool_.ages();

        // -- A: the cascade, on fused appearance and geometry -------------------------------
        const Association a = cascade_associate(
            [&](const std::vector<int>& r, const std::vector<int>& c) {
                return stage_a_cost(r, c, detection_boxes, context);
            },
            options_.stage_a_max_cost, established, columns, ages, options_.cascade_stride,
            options_.max_age + 1);
        columns = a.unmatched_columns;

        // -- B: geometry alone, for tracks whose prediction is still worth something ---------
        std::vector<int> recent;
        std::vector<int> stale;
        for (int row : a.unmatched_rows) {
            if (ages[static_cast<size_t>(row)] <= options_.stage_b_max_age)
                recent.push_back(row);
            else
                stale.push_back(row);
        }
        Association b;
        if (!recent.empty() && !columns.empty()) {
            b = associate_subset(stage_b_cost(recent, columns, detection_boxes, context), recent,
                                 columns, options_.stage_b_max_cost);
        } else {
            b.unmatched_rows = recent;
            b.unmatched_columns = columns;
        }
        columns = b.unmatched_columns;

        // -- C: OCR, against the last observation instead of the prediction -----------------
        std::vector<int> leftovers = stale;
        leftovers.insert(leftovers.end(), b.unmatched_rows.begin(), b.unmatched_rows.end());
        const std::vector<int> candidates =
            options_.recover ? recoverable(leftovers, context) : std::vector<int>{};
        Association c;
        if (!candidates.empty() && !columns.empty()) {
            c = associate_subset(stage_c_cost(candidates, columns, detection_boxes), candidates,
                                 columns, options_.stage_c_max_cost);
        } else {
            c.unmatched_rows = candidates;
            c.unmatched_columns = columns;
        }
        columns = c.unmatched_columns;

        // -- D: tentative tracks, last and on the weakest evidence --------------------------
        Association d;
        if (!tentative.empty() && !columns.empty()) {
            d = associate_subset(stage_d_cost(tentative, columns, detection_boxes), tentative,
                                 columns, options_.stage_d_max_cost);
        } else {
            d.unmatched_rows = tentative;
            d.unmatched_columns = columns;
        }
        columns = d.unmatched_columns;

        std::vector<std::pair<int, int>> matches = a.matches;
        matches.insert(matches.end(), b.matches.begin(), b.matches.end());
        matches.insert(matches.end(), c.matches.begin(), c.matches.end());
        matches.insert(matches.end(), d.matches.begin(), d.matches.end());
        pool_.apply_matches(matches, detections);

        // The leftovers stage C was not allowed to try are missed too. They never entered an
        // association, so nothing else would ever age them, and a track that is never aged is
        // a track that never dies.
        std::vector<int> missed = c.unmatched_rows;
        missed.insert(missed.end(), d.unmatched_rows.begin(), d.unmatched_rows.end());
        const std::vector<int> excluded = difference(leftovers, candidates);
        missed.insert(missed.end(), excluded.begin(), excluded.end());
        pool_.mark_missed(missed);

        pool_.spawn(detections, columns);
        pool_.sweep();
        return pool_.output();
    }

}  // namespace shipvision::mot
