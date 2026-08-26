// The lifecycle every tracker here shares: predict, promote, age, kill.
//
// A port of `shipvision/mot/pool.py`, and the reason it is shared there is the reason it
// is shared here: the trackers differ only in *how they associate*, and duplicating the
// lifecycle is how two trackers in one codebase drift until only one of them is correct. That
// drift is invisible — each tracker keeps working, they just quietly stop agreeing about when
// a track dies, which is precisely what a comparison between them depends on.
//
// Two capabilities are OFF by default and switched on by the trackers that need them, exactly
// as in the numpy pool: a bounded ring of past observations, and OC-SORT's re-update along a
// virtual trajectory. Both live here rather than in `ocsort/tracker.cpp` because the state
// they need is the filter state, and a tracker reaching into another object's covariance to
// rewind it is how a shared component stops being shared.
//
// WHAT THIS POOL DELIBERATELY DOES NOT HOLD
//
// * **Tags.** `(camera_id, frame_id)` stays in Python, where `BaseTracker.begin` already
//   refuses a camera swap and a replayed frame. Copying a string per track per frame across
//   the binding would cost more than the association, and a second implementation of the tag
//   discipline is a second place for it to be wrong.
// * **Embeddings.** A track's appearance vector is averaged on the Python side, and the
//   trackers that *associate* on appearance are handed the finished `(tracks, detections)`
//   cosine-distance matrix instead — see `FrameContext`. Marshalling a 512-float vector per
//   track per frame to blend it and marshal it straight back would cost far more than the
//   blend, while an `(n, m)` matrix at this sizing is a few hundred floats.
// * **Process-wide track ids.** Ids here are pool-local and start at 1. The library's contract
//   is that an id is unique across every tracker in the process — camera 3's track 7 and
//   camera 9's track 7 must not collide when their output meets downstream — and that counter
//   lives in `shipvision.mot.base.next_track_id`. The wrapper maps local to global once
//   per birth.

#pragma once

#include <array>
#include <cstddef>
#include <deque>
#include <utility>
#include <vector>

#include "shipvision/mot/kalman.h"

namespace shipvision::mot {

    /// One detection, as the trackers consume it. `xyxy` float32 absolute pixels, always.
    struct Detection {
            float box[4];
            float score = 1.f;
            int class_id = 0;
    };

    /// A track's lifecycle stage. The names and the transitions match `TrackState` in the
    /// Python, because the two are compared directly in the parity tests.
    enum class TrackState { Tentative = 0, Confirmed = 1, Lost = 2, Removed = 3 };

    /// One identity within one camera, at one frame.
    struct Track {
            int track_id = 0;  ///< pool-local; see the note in this file's header comment
            float box[4] = {0.f, 0.f, 0.f, 0.f};
            TrackState state = TrackState::Tentative;
            float score = 1.f;
            int class_id = 0;
            int age = 1;
            int hits = 1;
            int time_since_update = 0;
            /// Which detection of THIS frame corrected this track, or -1 for none.
            ///
            /// Exists for one caller: the Python wrapper keeps the appearance vector (see this
            /// file's header comment) and has to know which detection's vector to fold in. It
            /// is an index into the list `update` was handed, not into whatever tier the
            /// tracker split it into — a tracker that reported a position within its own
            /// high-score subset would have the wrapper average the wrong crop's appearance,
            /// which is invisible until two people in similar clothing swap identities in the
            /// cross-camera tier a minute later.
            int last_match = -1;

            /// Confirmed and updated on this frame.
            ///
            /// Both halves matter. A LOST track's box is a prediction no detector saw, and
            /// emitting it as an observation is how a phantom object drifts across a scene.
            bool is_publishable() const {
                return state == TrackState::Confirmed && time_since_update == 0;
            }
    };

    /// Everything about one frame that is not a detection box.
    ///
    /// One struct rather than three arguments so a tracker that needs none of it (SORT) and one
    /// that needs all of it (DeepSORTv2) have the same `update` signature. Every field has a
    /// meaningful "the caller did not supply this" state, because each of them genuinely can be
    /// absent: a camera with no re-ID pass has no appearance, a fixed camera has no motion, and
    /// an evaluation over an MOT ground-truth file has no frame size.
    struct FrameContext {
            /// `(pool size, detections)` cosine distances, or EMPTY when this frame carries no
            /// appearance evidence at all.
            ///
            /// Empty is a different answer from a matrix of zeros, and the difference is the
            /// whole of the appearance policy: a zero asserts that every pair looks identical,
            /// which is the strongest claim available and made on no evidence. A tracker handed
            /// an empty matrix falls back to geometry.
            ///
            /// Rows are pool rows as they stand when `update` is entered — every association
            /// runs before any track is born, so the row order cannot shift underneath it.
            /// Columns index the frame's own detection list.
            std::vector<float> appearance;

            /// `(2, 3)` row-major image-to-image affine mapping a point in the PREVIOUS frame
            /// to where it appears in this one. Identity means "the camera did not move".
            float affine[6] = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f};

            /// Frame height and width in pixels, or 0 when the caller did not supply them.
            ///
            /// Zero rather than a guess. Deriving the frame size from the boxes would make
            /// DeepSORTv2's border rule depend on where the objects happen to be, which works
            /// until the first frame where everyone stands on one side.
            int height = 0;
            int width = 0;
    };

    /// `(detections.size(), 4)` xyxy, packed once per frame so each stage can gather from it.
    std::vector<float> pack_boxes(const std::vector<Detection>& detections);

    /// Refuse an appearance matrix that is not `(rows, detections)`.
    ///
    /// Checked rather than trusted: the matrix is built on the Python side from a map keyed on
    /// pool-local ids, and a wrapper that fell one frame behind the C++ pool would hand over a
    /// matrix whose rows name different tracks. That reads as a plausible cost matrix and
    /// associates the wrong objects, which is the failure with no symptom.
    void check_appearance(const FrameContext& context, size_t rows, size_t detections);

    /// Holds every live track's filter state and lifecycle counters.
    ///
    /// The invariant the whole class exists to keep is that `tracks()[i]` and `states_[i]`
    /// describe the same track. `sweep()` rebuilds both from one pass for exactly that reason.
    class TrackPool {
        public:
            /// @param max_age  frames a confirmed track survives unmatched before it is dropped
            /// @param min_hits matches before a track is published
            /// @param observation_history how many past measurements to remember per track. `0`
            ///        keeps only the most recent, which is all an IoU-against-last-observation
            ///        recovery needs; a momentum term needs `delta_t + 1`. Bounded because a
            ///        process here runs for weeks.
            /// @param re_update rebuild the filter along a virtual trajectory when a gapped
            ///        track is re-found, instead of feeding one distant measurement to a filter
            ///        whose covariance has been inflating for the whole gap.
            TrackPool(int max_age, int min_hits, int observation_history = 0,
                      bool re_update = false);

            /// Open a frame: age every track and advance its filter.
            void predict();

            /// Warp every predicted state by a `(2, 3)` row-major image-to-image affine.
            ///
            /// BoT-SORT's camera-motion compensation. Without it a camera that pans makes every
            /// track's prediction wrong by the pan, the association fails for all of them on
            /// the same frame, and the tracker re-births the entire scene.
            ///
            /// **Everything the pool remembers about image positions is warped, not only the
            /// prediction.** The last-observation array and the observation ring are image
            /// coordinates too, and leaving them in the previous frame's system would give the
            /// observation-centric stages a stale frame of reference. That combination is
            /// unreachable today — only BoT-SORT compensates and it has no recovery stage —
            /// which is exactly why it is done here rather than left as a trap for whoever
            /// combines the two.
            void apply_camera_motion(const float affine[6]);

            /// Correct the matched filters and promote anything that has earned it.
            ///
            /// The indices are the caller's own: `detections` is the list `update` was given
            /// and the columns index into it, not into whatever subset the association stage
            /// happened to run over. Translating at the association stage rather than here is
            /// what lets `Track::last_match` mean something a caller can use.
            ///
            /// @param matches (track row, detection index) pairs; a row may appear at most once
            void apply_matches(const std::vector<std::pair<int, int>>& matches,
                               const std::vector<Detection>& detections);

            /// Age the tracks that found nothing this frame.
            void mark_missed(const std::vector<int>& rows);

            /// Start a track for each of the given unmatched detections.
            void spawn(const std::vector<Detection>& detections, const std::vector<int>& columns);

            /// Drop removed tracks, keeping the filter states aligned with the track list.
            void sweep();

            /// Forget every track. Called when a camera reconnects and continuity is broken.
            void reset();

            /// `(n, 4)` predicted xyxy, one row per live track, row-major.
            std::vector<float> boxes() const;

            /// `(n, 4)` xyxy of each track's **last real observation**. Not the prediction.
            ///
            /// Association against this is what recovers a track whose object stopped moving
            /// while it was hidden: the filter kept extrapolating the old velocity and its
            /// prediction has walked away, while the object is still sitting where it was last
            /// seen.
            std::vector<float> observed_boxes() const;

            /// `(n,)` `time_since_update` per track, for a cascade to band on.
            std::vector<int> ages() const;

            /// `(n, 2)` unit heading per track, measured between two real observations.
            ///
            /// Taken from the observation roughly `delta_t` frames back to the most recent one.
            /// Measured over a span rather than between consecutive frames because a single
            /// frame's displacement is mostly detector jitter — at 20 fps a person moves a few
            /// pixels and the box wobbles by a few pixels, so a one-frame heading is noise.
            ///
            /// A track with too little history gets `(0, 0)`, which every consumer must treat
            /// as "no information" rather than "not moving".
            std::vector<float> directions(int delta_t) const;

            /// `(rows.size(), m)` squared Mahalanobis distances for the named rows.
            std::vector<float> gating_distance(const std::vector<int>& rows, const float* boxes,
                                               int m) const;

            const std::vector<Track>& tracks() const { return tracks_; }

            /// Confirmed tracks seen on this frame — what a consumer should be shown.
            ///
            /// Tentative tracks are withheld because publishing one means handing downstream an
            /// identity for what may be a false positive, and downstream cannot tell.
            std::vector<Track> output() const;

            size_t size() const { return tracks_.size(); }

        private:
            void promote_if_earned(Track& track) const;

            /// OC-SORT's observation-centric re-update, for one gapped row.
            ///
            /// A track that has coasted for a gap is holding two things: a position that is a
            /// pure extrapolation and a covariance that has been inflated once per frame.
            /// Handing that filter a single distant measurement produces a *velocity*
            /// correction proportional to the whole accumulated position error, so the next
            /// prediction overshoots, misses, and the track is lost again — this time for good,
            /// and a new identity is born.
            ///
            /// The fix is to stop pretending the gap was observed: rewind to the last real
            /// observation, invent the measurements the detector would have produced had it not
            /// blinked (a straight line between the two observations — the only interpolation
            /// two points justify), and run the filter through them.
            void re_update_gap(size_t row, const float measurement[4]);

            KalmanFilter filter_;
            int max_age_;
            int min_hits_;
            int history_;
            bool re_update_;
            std::vector<Track> tracks_;
            std::vector<KalmanState> states_;
            /// The filter state at the last real observation, kept so a gapped track can be
            /// re-derived from a measurement rather than from its own extrapolation.
            std::vector<KalmanState> observed_states_;
            /// `(cx, cy, a, h)` of each track's last real observation.
            std::vector<std::array<float, 4>> observed_;
            /// `(track age, measurement)` per track, newest last, capped at `history_` entries.
            std::vector<std::deque<std::pair<int, std::array<float, 4>>>> observations_;
            /// Monotonic within one pool, and never rewound by `reset()`: handing the next
            /// track the id a dead one had is how a consumer stitches two different objects
            /// into one history.
            int last_track_id_ = 0;
    };

}  // namespace shipvision::mot
