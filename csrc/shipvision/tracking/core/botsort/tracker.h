// BoT-SORT: ByteTrack that knows the camera can move, and fuses appearance by minimum.
//
// Aharon, Orfaig and Bobrovsky, "BoT-SORT: Robust Associations Multi-Pedestrian Tracking",
// 2022. The C++ twin of `shipvision/tracking/core/botsort/tracker.py`, and — like the Python —
// a subclass of ByteTrack rather than a copy, because the paper *is* a two-line diff against
// it. Expressing it as two overridden methods is what keeps that claim true of this code.
//
// **Camera-motion compensation.** ByteTrack's Kalman prediction is in the previous frame's
// coordinate system. On a fixed camera those are the same system; on a panning one they differ
// by the pan, so *every* track's prediction is wrong by the same amount on the same frame, the
// whole association fails at once, and the tracker re-births the entire scene. Warping the
// predictions by the estimated frame-to-frame affine before association fixes it. How the
// affine is obtained is a separate, pluggable question and it stays in Python, in
// `shipvision.tracking.motion.cmc` — a PTZ head's own encoder beats any estimate made from
// pixels, and an estimator that needs OpenCV has no business inside a kernel library's
// translation unit.
//
// **Minimum fusion instead of a weighted sum.** DeepSORT adds an appearance distance to a
// motion distance, so a pair that is unambiguous on one signal can be dragged over the
// threshold by the other. BoT-SORT takes the element-wise minimum of two independently gated
// costs, so either signal on its own suffices. See `min_fuse` in `association.h`.
//
// Deliberately not implemented, matching the Python: the paper's third and smaller change —
// inflating the Kalman noise model's width/height terms — because this library's state is
// `(cx, cy, aspect, height)` rather than `(cx, cy, w, h)` and the corresponding term is
// already height-scaled here.

#pragma once

#include <vector>

#include "shipvision/tracking/core/bytetrack/tracker.h"

namespace shipvision::tracking {

    class BotSortTracker : public ByteTrackTracker {
        public:
            /// ByteTrack's knobs plus the two appearance ones, with the Python's defaults.
            ///
            /// The camera-motion estimator is absent because it lives in Python: this class is
            /// handed the resulting `(2, 3)` affine in `FrameContext`, which is what lets one
            /// implementation serve optical flow, a ground-plane homography and PTZ telemetry
            /// without any of them reaching a C++ translation unit.
            struct Options {
                    ByteTrackTracker::Options byte;
                    /// Cosine distance above which appearance contributes nothing.
                    float appearance_gate = 0.25f;
                    /// The paper halves the cosine distance before the minimum, because
                    /// `1 - IoU` and a cosine distance are not on the same scale.
                    float appearance_weight = 0.5f;
            };

            explicit BotSortTracker(const Options& options);

        protected:
            void compensate(const FrameContext& context) override;

            /// The minimum of the gated IoU cost and the gated, halved appearance cost.
            ///
            /// Note what is *not* here: `fuse_score`, which ByteTrack uses to scale similarity
            /// by detector confidence. Folding confidence into a cost that is already a minimum
            /// of two gated terms double-counts it — the high-score stage has by definition
            /// already filtered on confidence — and it pushes the fused cost above the
            /// appearance gate for exactly the medium-confidence detections appearance is
            /// supposed to rescue.
            ///
            /// With no appearance evidence this degrades to gated geometry rather than
            /// treating a missing distance as zero: a zero would mean "identical appearance",
            /// which is the strongest possible claim, made on no evidence.
            std::vector<float> first_cost(const std::vector<int>& rows,
                                          const std::vector<int>& columns,
                                          const std::vector<Detection>& detections,
                                          const FrameContext& context) const override;

        private:
            float appearance_gate_;
            float appearance_weight_;
    };

}  // namespace shipvision::tracking
