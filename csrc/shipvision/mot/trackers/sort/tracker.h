// SORT: Kalman prediction, IoU cost, one assignment per frame.
//
// The baseline every other tracker is measured against, and the C++ twin of
// `shipvision/mot/trackers/sort/tracker.py`. It is a hundred lines and it is remarkably hard
// to beat when the detector is good and the frame rate is high.
//
// Where it fails is instructive, because it is where every simple tracker fails: a detection
// that drops below the confidence threshold for a few frames — a person walking behind a
// pillar — takes its track with it. That is exactly what ByteTrack addresses, and this class
// exists partly so the claim can be tested rather than asserted.

#pragma once

#include <vector>

#include "shipvision/mot/pool.h"

namespace shipvision::mot {

    class SortTracker {
        public:
            /// The same five knobs as the Python constructor, with the same defaults.
            ///
            /// A struct rather than five arguments because the Python side is keyword-only and
            /// a positional C++ signature would make `iou_threshold` and `det_threshold`
            /// silently swappable — two floats in [0, 1] whose transposition degrades tracking
            /// without failing anywhere.
            struct Options {
                    float det_threshold = 0.5f;  ///< below this a detection is discarded outright
                    float iou_threshold = 0.3f;  ///< an association needs at least this overlap
                    int max_age = 30;
                    int min_hits = 3;
                    bool gate = true;  ///< let the motion model veto impossible associations
            };

            explicit SortTracker(const Options& options);

            /// Advance one frame. Returns the tracks that are confirmed and were seen.
            ///
            /// An empty detection list is information, not a reason to skip: tracks still age
            /// and eventually die, and a tracker that treats an empty frame as a no-op keeps
            /// dead objects alive forever.
            std::vector<Track> update(const std::vector<Detection>& detections);

            void reset() { pool_.reset(); }

            const std::vector<Track>& tracks() const { return pool_.tracks(); }

            size_t size() const { return pool_.size(); }

        private:
            Options options_;
            float max_cost_;  ///< 1 - iou_threshold, precomputed because it is what is compared
            TrackPool pool_;
    };

}  // namespace shipvision::mot
