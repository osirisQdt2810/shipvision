// DeepSORTv2: the four-stage cascade from the internal C++ tracker, ported.
//
// The C++ twin of `shipvision/tracking/core/deepsortv2/tracker.py`, which is itself a port of
// `gitea-generic-multi-object-tracking-cpp` (`src/tracker/models/deepsortv2/`). Both sources
// are first-party. Where they disagreed with themselves, the *paper* each stage comes from
// decided it — the four defects the Python found in the reference and did not port are named
// in that file, and this one inherits those decisions rather than re-litigating them.
//
// It is DeepSORT (Wojke et al., 2017) with three additions the reference had already made:
// OC-SORT's ORU and OCR, a dynamic appearance EMA (which stays on the Python side, because the
// vector never crosses the binding), and a four-stage cascade that separates "how confident
// are we in this track" from "how good is the evidence".
//
// The four stages, in order, each consuming what the last one could not match:
//
//   A  confirmed and lost, banded by age   GIoU fused with appearance, both gated
//   B  stage-A leftovers seen recently     IoU, gated by appearance
//   C  everything still unmatched          IoU against the LAST OBSERVATION (OCR)
//   D  tentative                           IoU, nothing else
//
// The ordering is the design. Stage A gives well-supported tracks first refusal on the good
// evidence. Stage B relaxes to geometry for tracks whose appearance has gone stale but which
// were seen recently enough for the prediction to be worth something. Stage C throws away the
// prediction entirely, which is the only thing that recovers an object that stopped moving
// while hidden. Stage D runs last because a tentative track is the weakest claim in the pool
// and must never outbid a confirmed one.

#pragma once

#include <vector>

#include "shipvision/tracking/pool.h"

namespace shipvision::tracking {

    class DeepSortV2Tracker {
        public:
            /// The same knobs as the Python constructor, with the same defaults.
            ///
            /// `appearance_momentum` and `dynamic_appearance` are absent: they govern how a
            /// track's appearance vector is *averaged*, and that vector never crosses the
            /// binding. The Python wrapper computes the per-detection rate and applies it, and
            /// hands this class the finished cosine distances in `FrameContext`.
            struct Options {
                    float det_threshold = 0.5f;
                    /// How much of stage A's fused cost is appearance rather than GIoU. The
                    /// reference uses 0.9, which reads as extreme until you remember the gates:
                    /// GIoU has already vetoed anything geometrically impossible, so what is
                    /// left for the cost to decide *is* an appearance question.
                    float appearance_weight = 0.9f;
                    /// Cosine distance above which a stage-A or stage-B pair is forbidden.
                    float appearance_gate = 0.15f;
                    /// GIoU cost above which a stage-A pair is forbidden. The range is [0, 2],
                    /// so 1.2 admits pairs that do not overlap at all but are close.
                    float giou_gate = 1.2f;
                    float stage_a_max_cost = 0.45f;
                    /// Band width for the cascade. One is DeepSORT's original formulation; the
                    /// reference uses five, trading a little precedence for a fifth of the
                    /// solves.
                    int cascade_stride = 5;
                    float stage_b_max_cost = 0.55f;
                    /// A track older than this does not get a stage-B chance: its prediction
                    /// has been extrapolating too long for IoU against it to mean anything, and
                    /// stage C is where it belongs.
                    int stage_b_max_age = 6;
                    float stage_c_max_cost = 0.65f;
                    bool recover = true;  ///< run stage C at all
                    /// Loosest of the four, because a tentative track has no history to be
                    /// judged against and the cost of getting it wrong is one frame of a track
                    /// nobody has published yet.
                    float stage_d_max_cost = 0.8f;
                    /// How close to the frame edge counts as "near the border", as a fraction
                    /// of the smaller frame dimension.
                    float border_fraction = 0.05f;
                    /// Exclude near-border tracks from stage C. An object leaving the frame is
                    /// half out of it, so its last observation is a truncated box that overlaps
                    /// whatever else is at the edge, and recovering on that evidence swaps
                    /// identities between everything entering and leaving.
                    bool skip_border_recovery = true;
                    int max_age = 30;
                    int min_hits = 3;
                    bool re_update = true;  ///< enable ORU
            };

            explicit DeepSortV2Tracker(const Options& options);

            /// Advance one frame. Returns the tracks that are confirmed and were seen.
            std::vector<Track> update(const std::vector<Detection>& detections,
                                      const FrameContext& context = FrameContext{});

            void reset() { pool_.reset(); }

            const std::vector<Track>& tracks() const { return pool_.tracks(); }

            size_t size() const { return pool_.size(); }

        private:
            /// Stage A: GIoU blended with appearance, then gated on **both** independently.
            ///
            /// The conjunction is deliberate and is where the C++ reference contradicts itself:
            /// its loop path requires both gates and its vectorised path requires either. A
            /// cost matrix whose gates can each be satisfied by ignoring the other is not
            /// gated.
            ///
            /// GIoU rather than IoU because IoU is flat at zero for every non-overlapping pair,
            /// so it cannot rank two equally-unmatched candidates — and ranking them is exactly
            /// what a cascade band is for.
            std::vector<float> stage_a_cost(const std::vector<int>& rows,
                                            const std::vector<int>& columns,
                                            const std::vector<float>& detection_boxes,
                                            const FrameContext& context) const;

            /// Stage B: IoU against the prediction, with appearance demoted to a veto.
            ///
            /// The demotion is the difference from stage A and the reason both exist: these
            /// tracks lost stage A, so their appearance has gone stale and blending it into the
            /// cost would rank them by how out-of-date their gallery vector is. It is still
            /// worth a *veto* — a stale vector that is wildly wrong is still evidence of the
            /// wrong object.
            std::vector<float> stage_b_cost(const std::vector<int>& rows,
                                            const std::vector<int>& columns,
                                            const std::vector<float>& detection_boxes,
                                            const FrameContext& context) const;

            /// Stage C: IoU against the last observation. No motion gate, by design (OCR).
            std::vector<float> stage_c_cost(const std::vector<int>& rows,
                                            const std::vector<int>& columns,
                                            const std::vector<float>& detection_boxes) const;

            /// Stage D: IoU, and nothing else. A tentative track has no history worth gating on.
            std::vector<float> stage_d_cost(const std::vector<int>& rows,
                                            const std::vector<int>& columns,
                                            const std::vector<float>& detection_boxes) const;

            /// The subset of `rows` stage C is allowed to try.
            ///
            /// Confirmed and lost tracks only, and — when the frame size is known — only those
            /// whose last observation was not against the frame edge. A frame size of zero
            /// means the caller did not supply one and the border rule is skipped rather than
            /// guessed: deriving the frame size from the boxes would make the rule depend on
            /// where the objects happen to be, which works until the first frame where everyone
            /// stands on one side.
            std::vector<int> recoverable(const std::vector<int>& rows,
                                         const FrameContext& context) const;

            Options options_;
            TrackPool pool_;
    };

}  // namespace shipvision::tracking
