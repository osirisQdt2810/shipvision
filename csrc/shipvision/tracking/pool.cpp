#include "shipvision/tracking/pool.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

namespace shipvision::tracking {

    namespace {

        /// `(cx, cy, a, h)` through a similarity transform, mirroring the numpy
        /// `_warp_measurements`. The centre rotates and translates, the aspect ratio is
        /// invariant, and the height scales — which is exact for a similarity and is the
        /// reason the state is parameterised as `(cx, cy, a, h)` in the first place.
        void warp_measurement(std::array<float, 4>& measurement, const float rotation[4],
                              const float translation[2], float scale) {
            const float cx = measurement[0];
            const float cy = measurement[1];
            measurement[0] = rotation[0] * cx + rotation[1] * cy + translation[0];
            measurement[1] = rotation[2] * cx + rotation[3] * cy + translation[1];
            measurement[3] = scale * measurement[3];
        }

    }  // namespace

    std::vector<float> pack_boxes(const std::vector<Detection>& detections) {
        std::vector<float> boxes(detections.size() * 4);
        for (size_t index = 0; index < detections.size(); ++index) {
            for (int i = 0; i < 4; ++i)
                boxes[index * 4 + i] = detections[index].box[i];
        }
        return boxes;
    }

    void check_appearance(const FrameContext& context, size_t rows, size_t detections) {
        if (context.appearance.empty())
            return;
        if (context.appearance.size() != rows * detections) {
            throw std::invalid_argument(
                "the appearance matrix must be (live tracks, detections); one that is not "
                "names different tracks than the pool holds and would associate the wrong "
                "objects while looking entirely plausible");
        }
    }

    TrackPool::TrackPool(int max_age, int min_hits, int observation_history, bool re_update)
        : max_age_(max_age),
          min_hits_(min_hits),
          // The floor of one mirrors the numpy `deque(maxlen=max(history, 1))`: even a tracker
          // that asks for no history needs the most recent measurement, because that is what
          // `observed_boxes()` is.
          history_(std::max(observation_history, 1)),
          re_update_(re_update) {
        if (max_age < 1 || min_hits < 1) {
            throw std::invalid_argument("max_age and min_hits must both be >= 1");
        }
        if (observation_history < 0) {
            throw std::invalid_argument("observation_history must be >= 0");
        }
    }

    void TrackPool::predict() {
        for (size_t index = 0; index < tracks_.size(); ++index) {
            tracks_[index].age += 1;
            // Cleared on the way in rather than on the way out: `last_match` answers "which
            // detection of the frame now being processed", and a stale value would have the
            // wrapper blend last frame's appearance into a track nothing matched.
            tracks_[index].last_match = -1;
            filter_.predict(states_[index]);
            cxcyah_to_xyxy(states_[index].mean.data(), tracks_[index].box);
        }
    }

    void TrackPool::apply_camera_motion(const float affine[6]) {
        if (tracks_.empty())
            return;

        const float rotation[4] = {affine[0], affine[1], affine[3], affine[4]};
        const float translation[2] = {affine[2], affine[5]};
        const float determinant = rotation[0] * rotation[3] - rotation[1] * rotation[2];
        const float scale = std::sqrt(std::fabs(determinant));

        // The same 4x4 measurement transform applies to a state and to a velocity, which is
        // why the 8x8 is two copies of it — except that a constant offset shifts where a thing
        // is, not how fast it is going, so only the position half is translated.
        float transform[64] = {0.f};
        for (int block = 0; block < 2; ++block) {
            const int offset = block * 4;
            transform[(offset + 0) * 8 + offset + 0] = rotation[0];
            transform[(offset + 0) * 8 + offset + 1] = rotation[1];
            transform[(offset + 1) * 8 + offset + 0] = rotation[2];
            transform[(offset + 1) * 8 + offset + 1] = rotation[3];
            transform[(offset + 2) * 8 + offset + 2] = 1.f;
            transform[(offset + 3) * 8 + offset + 3] = scale;
        }

        auto warp_state = [&](KalmanState& state) {
            std::array<float, 8> mean{};
            for (int i = 0; i < 8; ++i) {
                float sum = 0.f;
                for (int k = 0; k < 8; ++k)
                    sum += transform[i * 8 + k] * state.mean[k];
                mean[i] = sum;
            }
            mean[0] += translation[0];
            mean[1] += translation[1];
            state.mean = mean;

            float scratch[64];
            for (int i = 0; i < 8; ++i) {
                for (int j = 0; j < 8; ++j) {
                    float sum = 0.f;
                    for (int k = 0; k < 8; ++k)
                        sum += transform[i * 8 + k] * state.covariance[k * 8 + j];
                    scratch[i * 8 + j] = sum;
                }
            }
            for (int i = 0; i < 8; ++i) {
                for (int j = 0; j < 8; ++j) {
                    float sum = 0.f;
                    for (int k = 0; k < 8; ++k)
                        sum += scratch[i * 8 + k] * transform[j * 8 + k];  // scratch @ T^T
                    state.covariance[i * 8 + j] = sum;
                }
            }
        };

        for (size_t index = 0; index < tracks_.size(); ++index) {
            warp_state(states_[index]);
            warp_state(observed_states_[index]);
            warp_measurement(observed_[index], rotation, translation, scale);
            for (auto& entry : observations_[index])
                warp_measurement(entry.second, rotation, translation, scale);
            cxcyah_to_xyxy(states_[index].mean.data(), tracks_[index].box);
        }
    }

    void TrackPool::apply_matches(const std::vector<std::pair<int, int>>& matches,
                                  const std::vector<Detection>& detections) {
        std::set<int> seen;
        for (const auto& [row, column] : matches) {
            if (row < 0 || static_cast<size_t>(row) >= tracks_.size())
                throw std::invalid_argument("a match names a track row that does not exist");
            if (column < 0 || static_cast<size_t>(column) >= detections.size())
                throw std::invalid_argument("a match names a detection that does not exist");
            // A track matched to two detections would be corrected twice in one frame, which
            // reads as an implausibly confident filter rather than as a bug.
            if (!seen.insert(row).second)
                throw std::invalid_argument("a track may match at most one detection per frame");

            const Detection& detection = detections[static_cast<size_t>(column)];
            float measurement[4];
            xyxy_to_cxcyah(detection.box, measurement);
            // Before the correction, and only for a track that actually coasted: the re-update
            // replaces the state the single distant measurement would otherwise be applied to.
            if (re_update_)
                re_update_gap(static_cast<size_t>(row), measurement);
            filter_.update(states_[static_cast<size_t>(row)], measurement);

            Track& track = tracks_[static_cast<size_t>(row)];
            cxcyah_to_xyxy(states_[static_cast<size_t>(row)].mean.data(), track.box);
            track.score = detection.score;
            track.class_id = detection.class_id;
            track.hits += 1;
            track.time_since_update = 0;
            track.last_match = column;

            observed_states_[static_cast<size_t>(row)] = states_[static_cast<size_t>(row)];
            std::array<float, 4> observation{measurement[0], measurement[1], measurement[2],
                                             measurement[3]};
            observed_[static_cast<size_t>(row)] = observation;
            auto& history = observations_[static_cast<size_t>(row)];
            history.emplace_back(track.age, observation);
            while (static_cast<int>(history.size()) > history_)
                history.pop_front();

            if (track.state == TrackState::Lost) {
                track.state = TrackState::Confirmed;
            } else {
                promote_if_earned(track);
            }
        }
    }

    void TrackPool::re_update_gap(size_t row, const float measurement[4]) {
        const int gap = tracks_[row].time_since_update;
        if (gap < 1)
            return;

        KalmanState state = observed_states_[row];
        const std::array<float, 4>& start = observed_[row];
        for (int step = 1; step <= gap; ++step) {
            const float fraction = static_cast<float>(step) / static_cast<float>(gap + 1);
            float virtual_measurement[4];
            for (int i = 0; i < 4; ++i)
                virtual_measurement[i] = start[i] + (measurement[i] - start[i]) * fraction;
            filter_.predict(state);
            filter_.update(state, virtual_measurement);
        }
        // One more predict, because the caller is about to correct a state that must be this
        // frame's prediction rather than the last virtual frame's correction.
        filter_.predict(state);
        states_[row] = state;
    }

    void TrackPool::mark_missed(const std::vector<int>& rows) {
        for (int row : rows) {
            if (row < 0 || static_cast<size_t>(row) >= tracks_.size())
                throw std::invalid_argument("mark_missed names a track row that does not exist");
            Track& track = tracks_[static_cast<size_t>(row)];
            track.time_since_update += 1;
            if (track.state == TrackState::Tentative) {
                // An unconfirmed track that misses even once was probably a false positive.
                // Keeping it alive costs an identity slot and invites a wrong association.
                track.state = TrackState::Removed;
            } else if (track.time_since_update > max_age_) {
                track.state = TrackState::Removed;
            } else if (track.state == TrackState::Confirmed) {
                track.state = TrackState::Lost;
            }
        }
    }

    void TrackPool::spawn(const std::vector<Detection>& detections,
                          const std::vector<int>& columns) {
        for (int column : columns) {
            if (column < 0 || static_cast<size_t>(column) >= detections.size())
                throw std::invalid_argument("spawn names a detection that does not exist");
            const Detection& detection = detections[static_cast<size_t>(column)];
            float measurement[4];
            xyxy_to_cxcyah(detection.box, measurement);
            const KalmanState state = filter_.initiate(measurement);
            states_.push_back(state);
            observed_states_.push_back(state);
            const std::array<float, 4> observation{measurement[0], measurement[1], measurement[2],
                                                   measurement[3]};
            observed_.push_back(observation);
            observations_.push_back({{1, observation}});

            Track track;
            track.track_id = ++last_track_id_;
            for (int i = 0; i < 4; ++i)
                track.box[i] = detection.box[i];
            track.state = TrackState::Tentative;
            track.score = detection.score;
            track.class_id = detection.class_id;
            track.age = 1;
            track.hits = 1;
            track.time_since_update = 0;
            // A birth is a match too, as far as anything reading `last_match` is concerned.
            // Without this the Python wrapper's appearance EMA starts one frame late — it
            // would seed from the track's SECOND crop — and a chain of averages that begins
            // in the wrong place stays wrong for the life of the track while looking
            // perfectly reasonable.
            track.last_match = column;
            // Checked on birth as well as on match: with min_hits == 1 a brand-new track has
            // already met the bar, and a caller who asked for immediate publication and got
            // silence would reasonably call that a bug rather than a policy.
            promote_if_earned(track);
            tracks_.push_back(track);
        }
    }

    void TrackPool::sweep() {
        size_t kept = 0;
        for (size_t index = 0; index < tracks_.size(); ++index) {
            if (tracks_[index].state == TrackState::Removed)
                continue;
            if (kept != index) {
                tracks_[kept] = tracks_[index];
                states_[kept] = states_[index];
                observed_states_[kept] = observed_states_[index];
                observed_[kept] = observed_[index];
                observations_[kept] = std::move(observations_[index]);
            }
            ++kept;
        }
        tracks_.resize(kept);
        states_.resize(kept);
        observed_states_.resize(kept);
        observed_.resize(kept);
        observations_.resize(kept);
    }

    void TrackPool::reset() {
        tracks_.clear();
        states_.clear();
        observed_states_.clear();
        observed_.clear();
        observations_.clear();
        // The id counter is NOT reset. A caller resets because continuity is broken, and
        // handing the next track the id a dead one had is how a downstream consumer stitches
        // two different objects into one history.
    }

    std::vector<float> TrackPool::boxes() const {
        std::vector<float> result(tracks_.size() * 4);
        for (size_t index = 0; index < tracks_.size(); ++index) {
            for (int i = 0; i < 4; ++i)
                result[index * 4 + i] = tracks_[index].box[i];
        }
        return result;
    }

    std::vector<float> TrackPool::observed_boxes() const {
        std::vector<float> result(tracks_.size() * 4);
        for (size_t index = 0; index < tracks_.size(); ++index)
            cxcyah_to_xyxy(observed_[index].data(), result.data() + index * 4);
        return result;
    }

    std::vector<int> TrackPool::ages() const {
        std::vector<int> result(tracks_.size());
        for (size_t index = 0; index < tracks_.size(); ++index)
            result[index] = tracks_[index].time_since_update;
        return result;
    }

    std::vector<float> TrackPool::directions(int delta_t) const {
        std::vector<float> result(tracks_.size() * 2, 0.f);
        for (size_t row = 0; row < observations_.size(); ++row) {
            const auto& history = observations_[row];
            if (history.size() < 2)
                continue;
            const int latest_age = history.back().first;
            const std::array<float, 4>& latest = history.back().second;
            // Falls back to the OLDEST entry when nothing in the ring is `delta_t` frames
            // back, which is what the numpy `previous = history[0][1]` seed does: a heading
            // over a shorter span is still a measurement, while no heading at all would make a
            // young track invisible to the momentum term for its first `delta_t` frames.
            const std::array<float, 4>* previous = &history.front().second;
            for (auto entry = history.rbegin(); entry != history.rend(); ++entry) {
                if (latest_age - entry->first >= delta_t) {
                    previous = &entry->second;
                    break;
                }
            }
            const float offset_x = latest[0] - (*previous)[0];
            const float offset_y = latest[1] - (*previous)[1];
            const float norm = std::sqrt(offset_x * offset_x + offset_y * offset_y);
            if (norm > 1e-6f) {
                result[row * 2 + 0] = offset_x / norm;
                result[row * 2 + 1] = offset_y / norm;
            }
        }
        return result;
    }

    std::vector<float> TrackPool::gating_distance(const std::vector<int>& rows, const float* boxes,
                                                  int m) const {
        std::vector<float> distances(rows.size() * static_cast<size_t>(std::max(m, 0)), 0.f);
        for (size_t r = 0; r < rows.size(); ++r) {
            const int row = rows[r];
            if (row < 0 || static_cast<size_t>(row) >= tracks_.size())
                throw std::invalid_argument(
                    "gating_distance names a track row that does not exist");
            for (int j = 0; j < m; ++j) {
                float measurement[4];
                xyxy_to_cxcyah(boxes + j * 4, measurement);
                distances[r * static_cast<size_t>(m) + j] =
                    filter_.gating_distance(states_[static_cast<size_t>(row)], measurement);
            }
        }
        return distances;
    }

    std::vector<Track> TrackPool::output() const {
        std::vector<Track> published;
        for (const Track& track : tracks_) {
            if (track.is_publishable())
                published.push_back(track);
        }
        return published;
    }

    void TrackPool::promote_if_earned(Track& track) const {
        if (track.state == TrackState::Tentative && track.hits >= min_hits_)
            track.state = TrackState::Confirmed;
    }

}  // namespace shipvision::tracking
