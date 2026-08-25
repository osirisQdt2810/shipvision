// OC-SORT: stop trusting the filter's extrapolation, trust the last thing you saw.
//
// Cao et al., "Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking",
// CVPR 2023. The C++ twin of `shipvision/tracking/core/ocsort/tracker.py`, written from the
// same reading of the paper.
//
// SORT's Kalman filter is *estimation-centric*: while a track is unobserved it keeps producing
// state from its own previous state, and every frame of that compounds. Three consequences,
// and OC-SORT is three named fixes for them:
//
// **ORU** — observation-centric re-update. When a gapped track is re-found, the single distant
// measurement corrects the *position* but also drives an enormous *velocity* correction,
// because the covariance has been inflating for the whole gap. Rewind to the last real
// observation, interpolate the measurements the detector would have produced, and run the
// filter through them instead. It is a property of the state, so it lives in `TrackPool` —
// exactly where the numpy version puts it.
//
// **OCR** — observation-centric recovery. A second association that ignores the prediction
// entirely and matches unmatched tracks against their *last observation*. This is what catches
// the object that stopped moving while it was hidden: the filter carried the old velocity and
// its prediction has walked off, while the object is still standing where it was last seen.
//
// **OCM** — observation-centric momentum. A direction-consistency term in the cost, measured
// between two real observations rather than read off the filter. Displacement between two
// detections is a measurement; a filter's velocity after a gap is a guess conditioned on its
// own earlier guesses.
//
// Deliberately not implemented here, matching the Python: the paper's optional BYTE-style
// low-score second association (that is what `ByteTrackTracker` is for, and combining the two
// is a fourth tracker rather than a flag), and the "OC-SORT + appearance" variants from later
// papers.

#pragma once

#include <vector>

#include "shipvision/tracking/pool.h"

namespace shipvision::tracking {

    class OcSortTracker {
        public:
            /// The same knobs as the Python constructor, with the same defaults.
            ///
            /// Each of the three fixes can be switched off independently. That is not
            /// configurability for its own sake: it is the only way to say which of them is
            /// earning its keep on a given camera, and a feature nobody can switch off is a
            /// feature nobody can show is worth its cost.
            struct Options {
                    float det_threshold = 0.5f;  ///< below this a detection is discarded
                    float iou_threshold = 0.3f;  ///< minimum overlap for the primary stage
                    /// Minimum overlap for OCR. Stricter than the primary threshold by
                    /// default, because OCR is deliberately matching against a *stale* box and
                    /// the geometry has to be convincing to make up for it.
                    float recovery_iou_threshold = 0.5f;
                    /// How many frames back the momentum term measures heading over. One frame
                    /// of displacement at 20 fps is mostly detector jitter, so the paper
                    /// measures over a span; three is its default.
                    int delta_t = 3;
                    /// How much the direction term counts against IoU. Small on purpose — a
                    /// tie-breaker between geometrically plausible candidates, not a cost in
                    /// its own right. High makes the tracker refuse to follow anything that
                    /// changes direction.
                    float momentum_weight = 0.2f;
                    /// OC-SORT's whole point is that this can be generous.
                    int max_age = 30;
                    int min_hits = 3;
                    /// Gate the primary association. Never OCR — the filter's opinion is
                    /// exactly what OCR is overruling.
                    bool gate = true;
                    bool re_update = true;  ///< enable ORU
                    bool recover = true;    ///< enable OCR
            };

            explicit OcSortTracker(const Options& options);

            /// Advance one frame. Returns the tracks that are confirmed and were seen.
            std::vector<Track> update(const std::vector<Detection>& detections);

            void reset() { pool_.reset(); }

            const std::vector<Track>& tracks() const { return pool_.tracks(); }

            size_t size() const { return pool_.size(); }

        private:
            /// IoU against the prediction, nudged by whether the candidate is *ahead* (OCM).
            ///
            /// The momentum term is added, not fused: it is a tie-breaker between
            /// geometrically plausible candidates rather than a cost in its own right, which is
            /// why its weight is small by default. Two objects passing each other are
            /// geometrically interchangeable at the moment they overlap and are *not*
            /// interchangeable in heading.
            std::vector<float> primary_cost(const std::vector<int>& rows,
                                            const std::vector<int>& columns,
                                            const std::vector<float>& detection_boxes) const;

            /// IoU against the **last observation**, with no motion model and no gate (OCR).
            ///
            /// Both omissions are the point. The prediction is what already failed in the
            /// primary stage, so reusing it would just fail again; and gating on a filter whose
            /// covariance grew through the gap either admits everything or vetoes the one
            /// honest candidate.
            std::vector<float> recovery_cost(const std::vector<int>& rows,
                                             const std::vector<int>& columns,
                                             const std::vector<float>& detection_boxes) const;

            Options options_;
            float max_cost_;
            float recovery_max_cost_;
            TrackPool pool_;
    };

}  // namespace shipvision::tracking
