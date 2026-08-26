// ByteTrack: associate the confident detections, then give the rest a second chance.
//
// Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", ECCV
// 2022. The C++ twin of `shipvision/mot/trackers/bytetrack/tracker.py`, written from the same
// reading of the paper.
//
// Every tracker throws away low-confidence detections, because most of them are noise.
// ByteTrack's observation is that *some* of them are not: when a tracked person walks behind a
// pillar the detector does not stop seeing them, it sees them at 0.3 instead of 0.9. Matching
// that box against an EXISTING track keeps the identity through the occlusion.
//
// The asymmetry is what makes it safe. High-score detections may start new tracks; low-score
// ones may only continue existing ones. So a low-confidence false positive can never create an
// identity, and the cost of being wrong is one frame of a slightly misplaced box rather than a
// spurious object.

#pragma once

#include <vector>

#include "shipvision/mot/association.h"
#include "shipvision/mot/pool.h"

namespace shipvision::mot {

    class ByteTrackTracker {
        public:
            /// The same knobs as the Python constructor, with the same defaults.
            ///
            /// `embedding_momentum` is absent on purpose: ByteTrack's own association is purely
            /// geometric, and the appearance vector a track carries for the cross-camera tier
            /// is averaged by the Python wrapper. Marshalling a 512-float vector per track per
            /// frame into C++ to blend it and marshal it straight back would cost more than the
            /// blend.
            struct Options {
                    float track_threshold = 0.5f;  ///< at or above this a detection may start a
                                                   ///< track
                    float low_threshold = 0.1f;    ///< below this a detection is discarded
                    float match_threshold = 0.2f;  ///< minimum IoU for stage one
                    /// Minimum IoU for stage two. Deliberately stricter than stage one: the
                    /// evidence is weaker, so the geometry has to be better.
                    float second_match_threshold = 0.5f;
                    int max_age = 30;
                    int min_hits = 3;
                    bool gate = true;
            };

            explicit ByteTrackTracker(const Options& options);

            /// Virtual because BoT-SORT derives from this class, exactly as it does in Python.
            virtual ~ByteTrackTracker() = default;

            /// Advance one frame. Returns the tracks that are confirmed and were seen.
            std::vector<Track> update(const std::vector<Detection>& detections,
                                      const FrameContext& context = FrameContext{});

            void reset() { pool_.reset(); }

            const std::vector<Track>& tracks() const { return pool_.tracks(); }

            size_t size() const { return pool_.size(); }

        protected:
            /// Warp the predictions into this frame's coordinates. ByteTrack does nothing here.
            ///
            /// The extension point exists because BoT-SORT's first contribution is *exactly*
            /// this step and nothing else, so a subclass that fills it in is a faithful reading
            /// of the paper rather than a fork of the association logic. ByteTrack assumes a
            /// bolted-down camera, which is the right assumption for most of a fifty-camera
            /// installation and the wrong one for a PTZ head.
            virtual void compensate(const FrameContext& context);

            /// Stage one's cost: `1 - IoU` scaled by detector confidence, then gated.
            ///
            /// Confidence is folded in here and NOT in stage two: it is not trustworthy on a
            /// 0.3 detection, and folding an unreliable signal into the cost is how the second
            /// stage starts inventing matches instead of rescuing them.
            ///
            /// BoT-SORT replaces this method and nothing else, which is what keeps the paper's
            /// "ByteTrack plus two things" claim true of this code.
            ///
            /// @param rows    track rows, in the pool's own indices
            /// @param columns detections, in the indices of the frame's own list
            virtual std::vector<float> first_cost(const std::vector<int>& rows,
                                                  const std::vector<int>& columns,
                                                  const std::vector<Detection>& detections,
                                                  const FrameContext& context) const;

            /// `1 - IoU` for the named rows and columns, with the motion model given a veto.
            ///
            /// Stage two's whole cost, and the geometric half of every other stage in this
            /// class hierarchy. Shared rather than written twice because the SORT baseline and
            /// ByteTrack's second pass are the same expression, and two copies is how the two
            /// stop being comparable while both still pass their tests.
            std::vector<float> gated_iou(const std::vector<int>& rows,
                                         const std::vector<int>& columns,
                                         const std::vector<Detection>& detections) const;

            Options options_;
            float max_cost_;
            float second_max_cost_;
            TrackPool pool_;
    };

}  // namespace shipvision::mot
