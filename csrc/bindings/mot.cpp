#include "bindings/mot.h"

#include <pybind11/numpy.h>

#include <cstring>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#include "shipvision/mot/trackers/botsort/tracker.h"
#include "shipvision/mot/trackers/bytetrack/tracker.h"
#include "shipvision/mot/trackers/deepsortv2/tracker.h"
#include "shipvision/mot/trackers/ocsort/tracker.h"
#include "shipvision/mot/trackers/sort/tracker.h"

namespace py = pybind11;

namespace shipvision::bindings {

    namespace {

        using F32Array = py::array_t<float, py::array::c_style | py::array::forcecast>;
        using I32Array = py::array_t<int, py::array::c_style | py::array::forcecast>;

        /// One frame's detections, resolved out of numpy into plain structs.
        ///
        /// A copy rather than a view, so the tracker below reads plain PODs and never a
        /// `py::` object: the boundary is one hop wide, and an association loop that took
        /// numpy strides would have the interpreter in its inner loop.
        std::vector<mot::Detection> read_detections(const F32Array& boxes, const F32Array& scores,
                                                    const I32Array& class_ids) {
            const auto box_info = boxes.request();
            const auto score_info = scores.request();
            const auto class_info = class_ids.request();
            if (box_info.ndim != 2 || box_info.shape[1] != 4) {
                throw std::invalid_argument(
                    "boxes must be (n, 4) xyxy float32; an empty frame is (0, 4), not (0,)");
            }
            if (score_info.ndim != 1 || score_info.shape[0] != box_info.shape[0]) {
                throw std::invalid_argument("scores must be (n,) float32 matching boxes");
            }
            if (class_info.ndim != 1 || class_info.shape[0] != box_info.shape[0]) {
                throw std::invalid_argument("class_ids must be (n,) int32 matching boxes");
            }

            const auto* box_data = static_cast<const float*>(box_info.ptr);
            const auto* score_data = static_cast<const float*>(score_info.ptr);
            const auto* class_data = static_cast<const int*>(class_info.ptr);

            std::vector<mot::Detection> detections(static_cast<size_t>(box_info.shape[0]));
            for (size_t index = 0; index < detections.size(); ++index) {
                for (int i = 0; i < 4; ++i)
                    detections[index].box[i] = box_data[index * 4 + i];
                detections[index].score = score_data[index];
                detections[index].class_id = class_data[index];
            }
            return detections;
        }

        /// Tracks as two arrays: the float fields and the integer ones.
        ///
        /// `update` wraps the **live** set rather than the publishable subset the C++ tracker
        /// returns, and the Python wrapper filters it with `Track.is_publishable` — the same
        /// property every other tracker's output is filtered by, so the rule stays in one
        /// place. Sending only the published subset would cost a second crossing per frame,
        /// because the wrapper needs the live set anyway: its local-id-to-global-id map and
        /// its appearance vectors are evicted by following the C++ pool's own lifecycle, and
        /// a map that could not see a LOST track would either forget an identity that is
        /// about to come back or grow by one entry per object ever seen.
        ///
        /// Two arrays rather than eight, and rather than a list of objects. A list of pybind
        /// classes would allocate one Python object per track per frame — 15 000 a second at
        /// the fleet's sizing, which is the per-frame overhead a native tracker exists to
        /// remove — and one mixed float array would carry `track_id` as a float32, which stops
        /// being exact past 2^24 and would renumber identities on a long run without failing.
        ///
        /// Layout, matching `_decode` in `shipvision/mot/backends/native.py`:
        ///   geometry: (k, 5) float32 [x1, y1, x2, y2, score]
        ///   meta:     (k, 7) int32   [track_id, class_id, state, age, hits,
        ///                             time_since_update, last_match]
        py::tuple wrap_tracks(const std::vector<mot::Track>& tracks) {
            const auto count = static_cast<py::ssize_t>(tracks.size());
            auto geometry = py::array_t<float>({count, static_cast<py::ssize_t>(5)});
            auto meta = py::array_t<int>({count, static_cast<py::ssize_t>(7)});
            auto* geometry_data = geometry.mutable_data();
            auto* meta_data = meta.mutable_data();
            for (size_t index = 0; index < tracks.size(); ++index) {
                const mot::Track& track = tracks[index];
                for (int i = 0; i < 4; ++i)
                    geometry_data[index * 5 + i] = track.box[i];
                geometry_data[index * 5 + 4] = track.score;
                meta_data[index * 7 + 0] = track.track_id;
                meta_data[index * 7 + 1] = track.class_id;
                meta_data[index * 7 + 2] = static_cast<int>(track.state);
                meta_data[index * 7 + 3] = track.age;
                meta_data[index * 7 + 4] = track.hits;
                meta_data[index * 7 + 5] = track.time_since_update;
                meta_data[index * 7 + 6] = track.last_match;
            }
            return py::make_tuple(geometry, meta);
        }

        /// One frame's non-geometric context, resolved out of numpy.
        ///
        /// `appearance` is `(live tracks, detections)` cosine distances, and a `(0, 0)` array
        /// means "this frame carries no appearance evidence" — which is a different answer from
        /// a matrix of zeros. A zero asserts that every pair looks identical, which is the
        /// strongest claim available and made on no evidence at all; the tracker that receives
        /// nothing falls back to geometry instead.
        ///
        /// The vectors themselves never cross: the appearance EMA lives in Python, where the
        /// numpy pool's own `blend_embedding` is, so the two backends cannot produce different
        /// track vectors. What crosses is the finished `(n, m)` matrix, which at this sizing is
        /// a few hundred floats against the 512 per track per frame a vector would cost.
        mot::FrameContext read_context(const F32Array& appearance, size_t rows, size_t detections) {
            mot::FrameContext context;
            const auto info = appearance.request();
            if (info.ndim != 2) {
                throw std::invalid_argument(
                    "appearance must be a (tracks, detections) float32 matrix; pass a (0, 0) "
                    "array when this frame has no appearance evidence");
            }
            if (info.shape[0] == 0 || info.shape[1] == 0)
                return context;
            if (static_cast<size_t>(info.shape[0]) != rows ||
                static_cast<size_t>(info.shape[1]) != detections) {
                throw std::invalid_argument(
                    "the appearance matrix must be (live tracks, detections); one that is not "
                    "names different tracks than the pool holds and would associate the wrong "
                    "objects while looking entirely plausible");
            }
            const auto* data = static_cast<const float*>(info.ptr);
            context.appearance.assign(data, data + rows * detections);
            return context;
        }

        /// The `(2, 3)` image-to-image affine, copied into the context.
        void read_affine(const F32Array& affine, mot::FrameContext& context) {
            const auto info = affine.request();
            if (info.ndim != 2 || info.shape[0] != 2 || info.shape[1] != 3) {
                throw std::invalid_argument(
                    "a camera-motion affine must be (2, 3): it maps a point in the previous "
                    "frame to where it appears in this one");
            }
            const auto* data = static_cast<const float*>(info.ptr);
            for (int i = 0; i < 6; ++i)
                context.affine[i] = data[i];
        }

        // -- trackers ------------------------------------------------------------------------

        /// The lifecycle every tracker session shares, and the lock that makes it safe.
        ///
        /// The wrapper owns the tracker rather than exposing it directly because a tracker is
        /// stateful and single-camera by construction: one instance serves one camera for its
        /// whole life, and the Python side is what enforces that (`BaseTracker.begin` refuses a
        /// frame whose camera disagrees with the one this instance has been serving).
        ///
        /// One mutex per session, held for the whole of a call. In the deployment this is
        /// written for it is never contended — one camera, one thread — and it is here because
        /// this library takes no GIL policy (see `bindings/mot.h`). The embedding server decides
        /// whether its workers overlap; if it decides they may, two `update` calls interleaved
        /// into one pool would produce plausible tracks that no log could explain afterwards. A
        /// lock that costs an uncontended atomic per frame is the cheap side of that trade.
        ///
        /// The subclass is what differs: which extra evidence its `update` takes, and how it
        /// reaches the tracker. It calls `tracker_` directly rather than this class's `size()`,
        /// which is not a style choice — the lock is not recursive.
        template <typename Tracker> class TrackerSession {
            public:
                explicit TrackerSession(const typename Tracker::Options& options)
                    : tracker_(options) {}

                py::tuple tracks() const {
                    const std::lock_guard<std::mutex> guard(mutex_);
                    return wrap_tracks(tracker_.tracks());
                }

                void reset() {
                    const std::lock_guard<std::mutex> guard(mutex_);
                    tracker_.reset();
                }

                size_t size() const {
                    const std::lock_guard<std::mutex> guard(mutex_);
                    return tracker_.size();
                }

            protected:
                Tracker tracker_;
                mutable std::mutex mutex_;
        };

        /// `SortTracker` with its numpy edge.
        class SortSession : public TrackerSession<mot::SortTracker> {
            public:
                SortSession(float det_threshold, float iou_threshold, int max_age, int min_hits,
                            bool gate)
                    : TrackerSession(mot::SortTracker::Options{det_threshold, iou_threshold,
                                                               max_age, min_hits, gate}) {}

                py::tuple update(const F32Array& boxes, const F32Array& scores,
                                 const I32Array& class_ids) {
                    const std::lock_guard<std::mutex> guard(mutex_);
                    auto detections = read_detections(boxes, scores, class_ids);
                    tracker_.update(detections);
                    return wrap_tracks(tracker_.tracks());
                }
        };

        /// `ByteTrackTracker` with its numpy edge.
        class ByteTrackSession : public TrackerSession<mot::ByteTrackTracker> {
            public:
                ByteTrackSession(float track_threshold, float low_threshold, float match_threshold,
                                 float second_match_threshold, int max_age, int min_hits, bool gate)
                    : TrackerSession(mot::ByteTrackTracker::Options{
                          track_threshold, low_threshold, match_threshold, second_match_threshold,
                          max_age, min_hits, gate}) {}

                py::tuple update(const F32Array& boxes, const F32Array& scores,
                                 const I32Array& class_ids) {
                    const std::lock_guard<std::mutex> guard(mutex_);
                    auto detections = read_detections(boxes, scores, class_ids);
                    tracker_.update(detections);
                    return wrap_tracks(tracker_.tracks());
                }
        };

        /// `OcSortTracker` with its numpy edge.
        class OcSortSession : public TrackerSession<mot::OcSortTracker> {
            public:
                OcSortSession(float det_threshold, float iou_threshold,
                              float recovery_iou_threshold, int delta_t, float momentum_weight,
                              int max_age, int min_hits, bool gate, bool re_update, bool recover)
                    : TrackerSession(mot::OcSortTracker::Options{
                          det_threshold, iou_threshold, recovery_iou_threshold, delta_t,
                          momentum_weight, max_age, min_hits, gate, re_update, recover}) {}

                py::tuple update(const F32Array& boxes, const F32Array& scores,
                                 const I32Array& class_ids) {
                    const std::lock_guard<std::mutex> guard(mutex_);
                    auto detections = read_detections(boxes, scores, class_ids);
                    tracker_.update(detections);
                    return wrap_tracks(tracker_.tracks());
                }
        };

        /// `BotSortTracker` with its numpy edge: appearance and camera motion cross too.
        class BotSortSession : public TrackerSession<mot::BotSortTracker> {
            public:
                BotSortSession(float track_threshold, float low_threshold, float match_threshold,
                               float second_match_threshold, int max_age, int min_hits, bool gate,
                               float appearance_gate, float appearance_weight)
                    : TrackerSession(mot::BotSortTracker::Options{
                          mot::ByteTrackTracker::Options{track_threshold, low_threshold,
                                                         match_threshold, second_match_threshold,
                                                         max_age, min_hits, gate},
                          appearance_gate, appearance_weight}) {}

                py::tuple update(const F32Array& boxes, const F32Array& scores,
                                 const I32Array& class_ids, const F32Array& appearance,
                                 const F32Array& affine) {
                    const std::lock_guard<std::mutex> guard(mutex_);
                    auto detections = read_detections(boxes, scores, class_ids);
                    auto context = read_context(appearance, tracker_.size(), detections.size());
                    read_affine(affine, context);
                    tracker_.update(detections, context);
                    return wrap_tracks(tracker_.tracks());
                }
        };

        /// `DeepSortV2Tracker` with its numpy edge: appearance and the frame size cross too.
        class DeepSortV2Session : public TrackerSession<mot::DeepSortV2Tracker> {
            public:
                DeepSortV2Session(float det_threshold, float appearance_weight,
                                  float appearance_gate, float giou_gate, float stage_a_max_cost,
                                  int cascade_stride, float stage_b_max_cost, int stage_b_max_age,
                                  float stage_c_max_cost, bool recover, float stage_d_max_cost,
                                  float border_fraction, bool skip_border_recovery, int max_age,
                                  int min_hits, bool re_update)
                    : TrackerSession(mot::DeepSortV2Tracker::Options{
                          det_threshold, appearance_weight, appearance_gate, giou_gate,
                          stage_a_max_cost, cascade_stride, stage_b_max_cost, stage_b_max_age,
                          stage_c_max_cost, recover, stage_d_max_cost, border_fraction,
                          skip_border_recovery, max_age, min_hits, re_update}) {}

                py::tuple update(const F32Array& boxes, const F32Array& scores,
                                 const I32Array& class_ids, const F32Array& appearance, int height,
                                 int width) {
                    const std::lock_guard<std::mutex> guard(mutex_);
                    auto detections = read_detections(boxes, scores, class_ids);
                    auto context = read_context(appearance, tracker_.size(), detections.size());
                    context.height = height;
                    context.width = width;
                    tracker_.update(detections, context);
                    return wrap_tracks(tracker_.tracks());
                }
        };

    }  // namespace

    void bind_mot(py::module_& module) {
        py::class_<SortSession>(module, "SortTracker",
                                "SORT: Kalman prediction, IoU, one assignment per frame.")
            .def(py::init<float, float, int, int, bool>(), py::arg("det_threshold") = 0.5f,
                 py::arg("iou_threshold") = 0.3f, py::arg("max_age") = 30, py::arg("min_hits") = 3,
                 py::arg("gate") = true)
            .def("update", &SortSession::update, py::arg("boxes"), py::arg("scores"),
                 py::arg("class_ids"),
                 "Advance one frame. Yields (geometry (k, 5) float32, meta (k, 7) int32) for every "
                 "LIVE track — the caller filters for the publishable ones.")
            .def("tracks", &SortSession::tracks,
                 "Every live track, including tentative and lost ones, in the same two-array "
                 "layout as update().")
            .def("reset", &SortSession::reset,
                 "Forget every track. Track ids are not rewound: reusing a dead track's id is how "
                 "a consumer stitches two different objects into one history.")
            .def_property_readonly("size", &SortSession::size);

        py::class_<ByteTrackSession>(
            module, "ByteTrackTracker",
            "ByteTrack: associate the confident detections, then give the rest a second chance.")
            .def(py::init<float, float, float, float, int, int, bool>(),
                 py::arg("track_threshold") = 0.5f, py::arg("low_threshold") = 0.1f,
                 py::arg("match_threshold") = 0.2f, py::arg("second_match_threshold") = 0.5f,
                 py::arg("max_age") = 30, py::arg("min_hits") = 3, py::arg("gate") = true)
            .def("update", &ByteTrackSession::update, py::arg("boxes"), py::arg("scores"),
                 py::arg("class_ids"),
                 "Advance one frame. Yields (geometry (k, 5) float32, meta (k, 7) int32) for every "
                 "LIVE track.")
            .def("tracks", &ByteTrackSession::tracks, "Every live track, in the same layout.")
            .def("reset", &ByteTrackSession::reset, "Forget every track.")
            .def_property_readonly("size", &ByteTrackSession::size);

        py::class_<OcSortSession>(
            module, "OcSortTracker",
            "OC-SORT: observation-centric momentum, recovery and re-update over SORT.")
            .def(py::init<float, float, float, int, float, int, int, bool, bool, bool>(),
                 py::arg("det_threshold") = 0.5f, py::arg("iou_threshold") = 0.3f,
                 py::arg("recovery_iou_threshold") = 0.5f, py::arg("delta_t") = 3,
                 py::arg("momentum_weight") = 0.2f, py::arg("max_age") = 30,
                 py::arg("min_hits") = 3, py::arg("gate") = true, py::arg("re_update") = true,
                 py::arg("recover") = true)
            .def("update", &OcSortSession::update, py::arg("boxes"), py::arg("scores"),
                 py::arg("class_ids"),
                 "Advance one frame. Yields (geometry (k, 5) float32, meta (k, 7) int32) for "
                 "every LIVE track.")
            .def("tracks", &OcSortSession::tracks, "Every live track, in the same layout.")
            .def("reset", &OcSortSession::reset, "Forget every track.")
            .def_property_readonly("size", &OcSortSession::size);

        py::class_<BotSortSession>(
            module, "BotSortTracker",
            "BoT-SORT: ByteTrack plus camera-motion compensation and min-fused appearance.")
            .def(py::init<float, float, float, float, int, int, bool, float, float>(),
                 py::arg("track_threshold") = 0.5f, py::arg("low_threshold") = 0.1f,
                 py::arg("match_threshold") = 0.2f, py::arg("second_match_threshold") = 0.5f,
                 py::arg("max_age") = 30, py::arg("min_hits") = 3, py::arg("gate") = true,
                 py::arg("appearance_gate") = 0.25f, py::arg("appearance_weight") = 0.5f)
            .def("update", &BotSortSession::update, py::arg("boxes"), py::arg("scores"),
                 py::arg("class_ids"), py::arg("appearance"), py::arg("affine"),
                 "Advance one frame. `appearance` is (live tracks, detections) cosine "
                 "distances, or (0, 0) when the frame has none; `affine` is the (2, 3) "
                 "previous-to-current image transform the estimator produced.")
            .def("tracks", &BotSortSession::tracks, "Every live track, in the same layout.")
            .def("reset", &BotSortSession::reset, "Forget every track.")
            .def_property_readonly("size", &BotSortSession::size);

        py::class_<DeepSortV2Session>(
            module, "DeepSortV2Tracker",
            "DeepSORTv2: a four-stage cascade with observation-centric re-update and recovery.")
            .def(py::init<float, float, float, float, float, int, float, int, float, bool, float,
                          float, bool, int, int, bool>(),
                 py::arg("det_threshold") = 0.5f, py::arg("appearance_weight") = 0.9f,
                 py::arg("appearance_gate") = 0.15f, py::arg("giou_gate") = 1.2f,
                 py::arg("stage_a_max_cost") = 0.45f, py::arg("cascade_stride") = 5,
                 py::arg("stage_b_max_cost") = 0.55f, py::arg("stage_b_max_age") = 6,
                 py::arg("stage_c_max_cost") = 0.65f, py::arg("recover") = true,
                 py::arg("stage_d_max_cost") = 0.8f, py::arg("border_fraction") = 0.05f,
                 py::arg("skip_border_recovery") = true, py::arg("max_age") = 30,
                 py::arg("min_hits") = 3, py::arg("re_update") = true)
            .def("update", &DeepSortV2Session::update, py::arg("boxes"), py::arg("scores"),
                 py::arg("class_ids"), py::arg("appearance"), py::arg("height"), py::arg("width"),
                 "Advance one frame. `height` and `width` are the frame size in pixels, or 0 "
                 "when the caller does not know it — in which case the border rule that keeps "
                 "stage C off half-visible objects is skipped rather than guessed.")
            .def("tracks", &DeepSortV2Session::tracks, "Every live track, in the same layout.")
            .def("reset", &DeepSortV2Session::reset, "Forget every track.")
            .def_property_readonly("size", &DeepSortV2Session::size);
    }

}  // namespace shipvision::bindings
