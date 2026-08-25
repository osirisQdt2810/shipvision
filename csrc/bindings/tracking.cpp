#include "bindings/tracking.h"

#include <pybind11/numpy.h>

#include <cstring>
#include <stdexcept>
#include <vector>

#include "shipvision/mtmc/matcher.h"
#include "shipvision/tracking/core/botsort/tracker.h"
#include "shipvision/tracking/core/bytetrack/tracker.h"
#include "shipvision/tracking/core/deepsortv2/tracker.h"
#include "shipvision/tracking/core/ocsort/tracker.h"
#include "shipvision/tracking/core/sort/tracker.h"

namespace py = pybind11;

namespace shipvision::bindings {

    namespace {

        using F32Array = py::array_t<float, py::array::c_style | py::array::forcecast>;
        using F64Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
        using I32Array = py::array_t<int, py::array::c_style | py::array::forcecast>;
        using U8Array = py::array_t<unsigned char, py::array::c_style | py::array::forcecast>;

        /// One frame's detections, resolved out of numpy into plain structs.
        ///
        /// Built with the GIL held and consumed after it is released, which is the only reason
        /// it is a copy rather than a view: the arrays belong to the caller and nothing may
        /// touch a py:: object once the lock is gone.
        std::vector<tracking::Detection> read_detections(const F32Array& boxes,
                                                         const F32Array& scores,
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

            std::vector<tracking::Detection> detections(static_cast<size_t>(box_info.shape[0]));
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
        /// Layout, matching `_decode` in `shipvision/tracking/backends/native.py`:
        ///   geometry: (k, 5) float32 [x1, y1, x2, y2, score]
        ///   meta:     (k, 7) int32   [track_id, class_id, state, age, hits,
        ///                             time_since_update, last_match]
        py::tuple wrap_tracks(const std::vector<tracking::Track>& tracks) {
            const auto count = static_cast<py::ssize_t>(tracks.size());
            auto geometry = py::array_t<float>({count, static_cast<py::ssize_t>(5)});
            auto meta = py::array_t<int>({count, static_cast<py::ssize_t>(7)});
            auto* geometry_data = geometry.mutable_data();
            auto* meta_data = meta.mutable_data();
            for (size_t index = 0; index < tracks.size(); ++index) {
                const tracking::Track& track = tracks[index];
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

        /// The `(n, n)` side length of a square matrix argument, or a readable refusal.
        int square_side(const py::buffer_info& info, const char* what) {
            if (info.ndim != 2 || info.shape[0] != info.shape[1]) {
                throw std::invalid_argument(std::string(what) +
                                            " must be a square (n, n) matrix: every entry is one "
                                            "ordered pair of the same synchronised group");
            }
            return static_cast<int>(info.shape[0]);
        }

        /// A `(n, n)` float32 result built from a vector, without a second copy.
        py::array_t<float> wrap_square(const std::vector<float>& values, int n) {
            auto result =
                py::array_t<float>({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(n)});
            if (!values.empty())
                std::memcpy(result.mutable_data(), values.data(), values.size() * sizeof(float));
            return result;
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
        tracking::FrameContext read_context(const F32Array& appearance, size_t rows,
                                            size_t detections) {
            tracking::FrameContext context;
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
        void read_affine(const F32Array& affine, tracking::FrameContext& context) {
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

        /// `SortTracker` with its numpy edge.
        ///
        /// The wrapper owns the tracker rather than exposing it directly because a tracker is
        /// stateful and single-camera by construction: one instance serves one camera for its
        /// whole life, and the Python side is what enforces that (`BaseTracker.begin` refuses a
        /// frame whose camera disagrees with the one this instance has been serving).
        class SortSession {
            public:
                SortSession(float det_threshold, float iou_threshold, int max_age, int min_hits,
                            bool gate)
                    : tracker_(tracking::SortTracker::Options{det_threshold, iou_threshold, max_age,
                                                              min_hits, gate}) {}

                py::tuple update(const F32Array& boxes, const F32Array& scores,
                                 const I32Array& class_ids) {
                    auto detections = read_detections(boxes, scores, class_ids);
                    {
                        py::gil_scoped_release release;
                        tracker_.update(detections);
                    }
                    return wrap_tracks(tracker_.tracks());
                }

                py::tuple tracks() const { return wrap_tracks(tracker_.tracks()); }

                void reset() { tracker_.reset(); }

                size_t size() const { return tracker_.size(); }

            private:
                tracking::SortTracker tracker_;
        };

        class ByteTrackSession {
            public:
                ByteTrackSession(float track_threshold, float low_threshold, float match_threshold,
                                 float second_match_threshold, int max_age, int min_hits, bool gate)
                    : tracker_(tracking::ByteTrackTracker::Options{
                          track_threshold, low_threshold, match_threshold, second_match_threshold,
                          max_age, min_hits, gate}) {}

                py::tuple update(const F32Array& boxes, const F32Array& scores,
                                 const I32Array& class_ids) {
                    auto detections = read_detections(boxes, scores, class_ids);
                    {
                        py::gil_scoped_release release;
                        tracker_.update(detections);
                    }
                    return wrap_tracks(tracker_.tracks());
                }

                py::tuple tracks() const { return wrap_tracks(tracker_.tracks()); }

                void reset() { tracker_.reset(); }

                size_t size() const { return tracker_.size(); }

            private:
                tracking::ByteTrackTracker tracker_;
        };

        class OcSortSession {
            public:
                OcSortSession(float det_threshold, float iou_threshold,
                              float recovery_iou_threshold, int delta_t, float momentum_weight,
                              int max_age, int min_hits, bool gate, bool re_update, bool recover)
                    : tracker_(tracking::OcSortTracker::Options{
                          det_threshold, iou_threshold, recovery_iou_threshold, delta_t,
                          momentum_weight, max_age, min_hits, gate, re_update, recover}) {}

                py::tuple update(const F32Array& boxes, const F32Array& scores,
                                 const I32Array& class_ids) {
                    auto detections = read_detections(boxes, scores, class_ids);
                    {
                        py::gil_scoped_release release;
                        tracker_.update(detections);
                    }
                    return wrap_tracks(tracker_.tracks());
                }

                py::tuple tracks() const { return wrap_tracks(tracker_.tracks()); }

                void reset() { tracker_.reset(); }

                size_t size() const { return tracker_.size(); }

            private:
                tracking::OcSortTracker tracker_;
        };

        class BotSortSession {
            public:
                BotSortSession(float track_threshold, float low_threshold, float match_threshold,
                               float second_match_threshold, int max_age, int min_hits, bool gate,
                               float appearance_gate, float appearance_weight)
                    : tracker_(tracking::BotSortTracker::Options{
                          tracking::ByteTrackTracker::Options{
                              track_threshold, low_threshold, match_threshold,
                              second_match_threshold, max_age, min_hits, gate},
                          appearance_gate, appearance_weight}) {}

                py::tuple update(const F32Array& boxes, const F32Array& scores,
                                 const I32Array& class_ids, const F32Array& appearance,
                                 const F32Array& affine) {
                    auto detections = read_detections(boxes, scores, class_ids);
                    auto context = read_context(appearance, tracker_.size(), detections.size());
                    read_affine(affine, context);
                    {
                        py::gil_scoped_release release;
                        tracker_.update(detections, context);
                    }
                    return wrap_tracks(tracker_.tracks());
                }

                py::tuple tracks() const { return wrap_tracks(tracker_.tracks()); }

                void reset() { tracker_.reset(); }

                size_t size() const { return tracker_.size(); }

            private:
                tracking::BotSortTracker tracker_;
        };

        class DeepSortV2Session {
            public:
                DeepSortV2Session(float det_threshold, float appearance_weight,
                                  float appearance_gate, float giou_gate, float stage_a_max_cost,
                                  int cascade_stride, float stage_b_max_cost, int stage_b_max_age,
                                  float stage_c_max_cost, bool recover, float stage_d_max_cost,
                                  float border_fraction, bool skip_border_recovery, int max_age,
                                  int min_hits, bool re_update)
                    : tracker_(tracking::DeepSortV2Tracker::Options{
                          det_threshold, appearance_weight, appearance_gate, giou_gate,
                          stage_a_max_cost, cascade_stride, stage_b_max_cost, stage_b_max_age,
                          stage_c_max_cost, recover, stage_d_max_cost, border_fraction,
                          skip_border_recovery, max_age, min_hits, re_update}) {}

                py::tuple update(const F32Array& boxes, const F32Array& scores,
                                 const I32Array& class_ids, const F32Array& appearance, int height,
                                 int width) {
                    auto detections = read_detections(boxes, scores, class_ids);
                    auto context = read_context(appearance, tracker_.size(), detections.size());
                    context.height = height;
                    context.width = width;
                    {
                        py::gil_scoped_release release;
                        tracker_.update(detections, context);
                    }
                    return wrap_tracks(tracker_.tracks());
                }

                py::tuple tracks() const { return wrap_tracks(tracker_.tracks()); }

                void reset() { tracker_.reset(); }

                size_t size() const { return tracker_.size(); }

            private:
                tracking::DeepSortV2Tracker tracker_;
        };

        // -- mtmc ----------------------------------------------------------------------------

        py::array_t<float> mtmc_threshold_similarity(const F32Array& similarity, float threshold) {
            const auto info = similarity.request();
            const int n = square_side(info, "a similarity matrix");
            std::vector<float> values(static_cast<size_t>(n) * n);
            if (!values.empty())
                std::memcpy(values.data(), info.ptr, values.size() * sizeof(float));
            {
                py::gil_scoped_release release;
                mtmc::threshold_similarity(values, n, threshold);
            }
            return wrap_square(values, n);
        }

        py::array_t<double> mtmc_ground_distances(const F32Array& points, const U8Array& known) {
            const auto point_info = points.request();
            const auto known_info = known.request();
            if (point_info.ndim != 2 || point_info.shape[1] != 2) {
                throw std::invalid_argument("ground points must be (n, 2)");
            }
            if (known_info.ndim != 1 || known_info.shape[0] != point_info.shape[0]) {
                throw std::invalid_argument(
                    "one 'is this camera calibrated' flag per point is required; an "
                    "uncalibrated camera's point is not a place on the map");
            }
            const int n = static_cast<int>(point_info.shape[0]);
            const auto* point_data = static_cast<const float*>(point_info.ptr);
            const auto* known_data = static_cast<const unsigned char*>(known_info.ptr);

            std::vector<double> distances;
            {
                py::gil_scoped_release release;
                distances = mtmc::ground_distances(point_data, known_data, n);
            }
            auto result =
                py::array_t<double>({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(n)});
            if (!distances.empty()) {
                std::memcpy(result.mutable_data(), distances.data(),
                            distances.size() * sizeof(double));
            }
            return result;
        }

        py::array_t<float> mtmc_spatial_similarity(const F64Array& distances, float threshold) {
            const auto info = distances.request();
            const int n = square_side(info, "a ground-distance matrix");
            const auto* data = static_cast<const double*>(info.ptr);
            std::vector<float> similarity;
            {
                py::gil_scoped_release release;
                similarity = mtmc::spatial_similarity(data, n, threshold);
            }
            return wrap_square(similarity, n);
        }

        py::array_t<bool> mtmc_spatial_gate(const F64Array& distances, float threshold) {
            const auto info = distances.request();
            const int n = square_side(info, "a ground-distance matrix");
            const auto* data = static_cast<const double*>(info.ptr);
            std::vector<unsigned char> allowed;
            {
                py::gil_scoped_release release;
                allowed = mtmc::spatial_gate(data, n, threshold);
            }
            auto result =
                py::array_t<bool>({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(n)});
            auto* out = result.mutable_data();
            for (size_t index = 0; index < allowed.size(); ++index)
                out[index] = allowed[index] != 0;
            return result;
        }

        py::array_t<float> mtmc_veto(const F32Array& similarity, const py::array& allowed) {
            const auto info = similarity.request();
            const int n = square_side(info, "a similarity matrix");
            const auto gate =
                py::array_t<unsigned char, py::array::c_style | py::array::forcecast>(allowed);
            const auto gate_info = gate.request();
            if (square_side(gate_info, "a gate") != n) {
                throw std::invalid_argument(
                    "the gate and the similarity it gates describe the same pairs of the same "
                    "synchronised group, so they must be the same shape");
            }
            std::vector<float> values(static_cast<size_t>(n) * n);
            if (!values.empty())
                std::memcpy(values.data(), info.ptr, values.size() * sizeof(float));
            const auto* gate_data = static_cast<const unsigned char*>(gate_info.ptr);
            {
                py::gil_scoped_release release;
                mtmc::veto(values, gate_data, n);
            }
            return wrap_square(values, n);
        }

        py::array_t<float> mtmc_to_distance(const F32Array& similarity,
                                            const I32Array& camera_codes) {
            const auto info = similarity.request();
            const int n = square_side(info, "a similarity matrix");
            const auto code_info = camera_codes.request();
            if (code_info.ndim != 1 || code_info.shape[0] != n) {
                throw std::invalid_argument(
                    "one camera code per track is required; the same-camera exclusion is the "
                    "one rule this matrix must never lose");
            }
            const auto* values = static_cast<const float*>(info.ptr);
            const auto* codes = static_cast<const int*>(code_info.ptr);
            std::vector<float> distance;
            {
                py::gil_scoped_release release;
                distance = mtmc::to_distance(values, codes, n);
            }
            return wrap_square(distance, n);
        }

    }  // namespace

    void bind_tracking(py::module_& module) {
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

        module.def("mtmc_threshold_similarity", &mtmc_threshold_similarity, py::arg("similarity"),
                   py::arg("threshold"),
                   "Zero every similarity at or below the threshold. Weak evidence must not be "
                   "expressible, or average linkage chains three strangers into one identity.");
        module.def("mtmc_ground_distances", &mtmc_ground_distances, py::arg("points"),
                   py::arg("known"),
                   "(n, n) float64 euclidean distance on the ground plane; inf where at least "
                   "one of the pair's cameras is uncalibrated.");
        module.def("mtmc_spatial_similarity", &mtmc_spatial_similarity, py::arg("distances"),
                   py::arg("threshold"),
                   "(n, n) float32 in [0, 1] from ground distances: 1 at zero separation, 0 at "
                   "the threshold and beyond, and 1 on the diagonal.");
        module.def("mtmc_spatial_gate", &mtmc_spatial_gate, py::arg("distances"),
                   py::arg("threshold"),
                   "(n, n) bool: true where geometry does not object, INCLUDING where it cannot "
                   "judge — an uncalibrated camera takes part on appearance alone.");
        module.def("mtmc_veto", &mtmc_veto, py::arg("similarity"), py::arg("allowed"),
                   "(n, n) float32 similarity with every gated pair set to exactly zero.");
        module.def("mtmc_to_distance", &mtmc_to_distance, py::arg("similarity"),
                   py::arg("camera_codes"),
                   "(n, n) float32 clusterable distance: zero similarity and same-camera pairs "
                   "both become NEVER_MERGE, symmetric, zero on the diagonal.");
        module.attr("MTMC_NEVER_MERGE") = mtmc::kNeverMerge;
    }

}  // namespace shipvision::bindings
