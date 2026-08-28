#include "bindings/mtmc.h"

#include <pybind11/numpy.h>

#include <cstring>
#include <stdexcept>
#include <vector>

#include "shipvision/mtmc/clustering/agglomerative.h"
#include "shipvision/mtmc/frames.h"
#include "shipvision/mtmc/matcher.h"
#include "shipvision/mtmc/matchers/appearance/matcher.h"
#include "shipvision/mtmc/matchers/gated/matcher.h"
#include "shipvision/mtmc/matchers/spatial/matcher.h"
#include "shipvision/mtmc/matchers/spatial/utils.h"
#include "shipvision/mtmc/topology/homography.h"

namespace py = pybind11;

namespace shipvision::bindings {

    namespace {

        using F32Array = py::array_t<float, py::array::c_style | py::array::forcecast>;
        using F64Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
        using I32Array = py::array_t<int, py::array::c_style | py::array::forcecast>;
        using U8Array = py::array_t<unsigned char, py::array::c_style | py::array::forcecast>;

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

        /// One synchronised instant, resolved out of numpy into plain structs.
        ///
        /// A copy rather than a view, so the matcher below reads plain PODs and never a
        /// `py::` object: the boundary is one hop wide, and a kernel that took numpy strides
        /// would have the interpreter in its inner loop.
        std::vector<mtmc::Observation> read_observations(const F32Array& boxes,
                                                         const I32Array& frame_sizes,
                                                         const I32Array& camera_codes) {
            const auto box_info = boxes.request();
            const auto size_info = frame_sizes.request();
            const auto code_info = camera_codes.request();
            if (box_info.ndim != 2 || box_info.shape[1] != 4) {
                throw std::invalid_argument(
                    "boxes must be (n, 4) xyxy float32; an instant with no tracks is (0, 4), "
                    "not (0,)");
            }
            const auto count = static_cast<size_t>(box_info.shape[0]);
            if (size_info.ndim != 2 || size_info.shape[1] != 2 ||
                static_cast<size_t>(size_info.shape[0]) != count) {
                throw std::invalid_argument(
                    "frame_sizes must be (n, 2) int32 [width, height], one per box. MTMC needs "
                    "them for the bottom-truncated-box test and to scale into the homography's "
                    "domain; a zero default would make both silently wrong");
            }
            if (code_info.ndim != 1 || static_cast<size_t>(code_info.shape[0]) != count) {
                throw std::invalid_argument(
                    "one camera code per box is required; the same-camera exclusion is the one "
                    "rule this matrix must never lose");
            }

            const auto* box_data = static_cast<const float*>(box_info.ptr);
            const auto* size_data = static_cast<const int*>(size_info.ptr);
            const auto* code_data = static_cast<const int*>(code_info.ptr);

            std::vector<mtmc::Observation> observations(count);
            for (size_t index = 0; index < count; ++index) {
                for (int axis = 0; axis < 4; ++axis)
                    observations[index].box[axis] = box_data[index * 4 + axis];
                observations[index].frame_width = size_data[index * 2];
                observations[index].frame_height = size_data[index * 2 + 1];
                observations[index].camera_code = code_data[index];
            }
            return observations;
        }

        /// The ground plane as three arrays indexed by camera code.
        ///
        /// Indexed rather than keyed by name because the codes are already the boundary's
        /// currency, and because a map lookup per track per instant is the cost this whole file
        /// exists to avoid. `calibrated` is separate from the matrices so that "this camera has
        /// no homography" has a representation that is not a matrix — an identity matrix would be
        /// a real mapping, and every track on that camera would land at a plausible place.
        mtmc::GroundPlane read_ground_plane(const F64Array& homographies,
                                            const I32Array& calibration_sizes,
                                            const U8Array& calibrated) {
            const auto matrix_info = homographies.request();
            const auto size_info = calibration_sizes.request();
            const auto flag_info = calibrated.request();
            if (matrix_info.ndim != 3 || matrix_info.shape[1] != 3 || matrix_info.shape[2] != 3) {
                throw std::invalid_argument(
                    "homographies must be (cameras, 3, 3) float64, indexed by camera code");
            }
            const auto cameras = static_cast<size_t>(matrix_info.shape[0]);
            if (size_info.ndim != 2 || size_info.shape[1] != 2 ||
                static_cast<size_t>(size_info.shape[0]) != cameras) {
                throw std::invalid_argument(
                    "calibration_sizes must be (cameras, 2) int32 [width, height]: a homography "
                    "fitted on 1080p stills does not apply to the 720p stream the same camera "
                    "serves at night, and the size is what lets the projection rescale");
            }
            if (flag_info.ndim != 1 || static_cast<size_t>(flag_info.shape[0]) != cameras) {
                throw std::invalid_argument(
                    "one 'is this camera calibrated' flag per homography is required; an "
                    "uncalibrated camera is the normal case, not an error");
            }

            const auto* matrix_data = static_cast<const double*>(matrix_info.ptr);
            const auto* size_data = static_cast<const int*>(size_info.ptr);
            const auto* flag_data = static_cast<const unsigned char*>(flag_info.ptr);

            std::vector<mtmc::Homography> plane(cameras);
            std::vector<unsigned char> flags(cameras);
            for (size_t index = 0; index < cameras; ++index) {
                for (int cell = 0; cell < 9; ++cell)
                    plane[index].matrix[cell] = matrix_data[index * 9 + cell];
                plane[index].camera_width = size_data[index * 2];
                plane[index].camera_height = size_data[index * 2 + 1];
                flags[index] = flag_data[index] != 0 ? 1 : 0;
            }
            return mtmc::GroundPlane(std::move(plane), std::move(flags));
        }

        // -- the matchers --------------------------------------------------------------------

        /// `AppearanceMatcher` with its numpy edge.
        class AppearanceSession {
            public:
                explicit AppearanceSession(float appearance_threshold)
                    : matcher_(appearance_threshold) {}

                py::array_t<float> similarities(const F32Array& gram) const {
                    const auto info = gram.request();
                    const int n = square_side(info, "a cosine similarity matrix");
                    const auto* data = static_cast<const float*>(info.ptr);
                    std::vector<float> similarity;
                    similarity = matcher_.similarities(data, n);
                    return wrap_square(similarity, n);
                }

                py::array_t<float> build(const F32Array& gram, const I32Array& camera_codes) const {
                    const auto info = gram.request();
                    const int n = square_side(info, "a cosine similarity matrix");
                    const auto code_info = camera_codes.request();
                    if (code_info.ndim != 1 || code_info.shape[0] != n) {
                        throw std::invalid_argument(
                            "one camera code per track is required; the same-camera exclusion is "
                            "the one rule this matrix must never lose");
                    }
                    // The appearance half reads nothing but the camera code, so nothing else
                    // crosses: asking a caller with no geometry for boxes and frame sizes would
                    // be asking it to invent them.
                    const auto* codes = static_cast<const int*>(code_info.ptr);
                    const auto* data = static_cast<const float*>(info.ptr);

                    std::vector<float> distance;
                    distance = matcher_.build(data, codes, n);
                    return wrap_square(distance, n);
                }

                float appearance_threshold() const { return matcher_.appearance_threshold(); }

            private:
                mtmc::AppearanceMatcher matcher_;
        };

        /// `SpatialMatcher` with its numpy edge.
        class SpatialSession {
            public:
                SpatialSession(float spatial_threshold, double foot_ratio, double aspect_ratio,
                               const F64Array& homographies, const I32Array& calibration_sizes,
                               const U8Array& calibrated)
                    : matcher_(mtmc::SpatialMatcher::Options{spatial_threshold, foot_ratio,
                                                             aspect_ratio},
                               read_ground_plane(homographies, calibration_sizes, calibrated)) {}

                /// `((n, 2) float32 ground points, (n,) bool "this one is calibrated")`.
                py::tuple ground_positions(const F32Array& boxes, const I32Array& frame_sizes,
                                           const I32Array& camera_codes) const {
                    auto observations = read_observations(boxes, frame_sizes, camera_codes);
                    const auto n = static_cast<int>(observations.size());
                    std::vector<float> points(static_cast<size_t>(n) * 2);
                    std::vector<unsigned char> known(static_cast<size_t>(n));
                    matcher_.ground_positions(observations.data(), n, points.data(), known.data());
                    auto point_array = py::array_t<float>(
                        {static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(2)});
                    if (!points.empty()) {
                        std::memcpy(point_array.mutable_data(), points.data(),
                                    points.size() * sizeof(float));
                    }
                    auto known_array = py::array_t<bool>({static_cast<py::ssize_t>(n)});
                    auto* known_out = known_array.mutable_data();
                    for (size_t index = 0; index < known.size(); ++index)
                        known_out[index] = known[index] != 0;
                    return py::make_tuple(point_array, known_array);
                }

                py::array_t<double> ground_distances(const F32Array& boxes,
                                                     const I32Array& frame_sizes,
                                                     const I32Array& camera_codes) const {
                    auto observations = read_observations(boxes, frame_sizes, camera_codes);
                    const auto n = static_cast<int>(observations.size());
                    std::vector<double> distances;
                    distances = matcher_.ground_distances(observations.data(), n);
                    auto result = py::array_t<double>(
                        {static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(n)});
                    if (!distances.empty()) {
                        std::memcpy(result.mutable_data(), distances.data(),
                                    distances.size() * sizeof(double));
                    }
                    return result;
                }

                py::array_t<float> similarities(const F32Array& boxes, const I32Array& frame_sizes,
                                                const I32Array& camera_codes) const {
                    auto observations = read_observations(boxes, frame_sizes, camera_codes);
                    const auto n = static_cast<int>(observations.size());
                    std::vector<float> similarity;
                    similarity = matcher_.similarities(observations.data(), n);
                    return wrap_square(similarity, n);
                }

                py::array_t<bool> gate(const F32Array& boxes, const I32Array& frame_sizes,
                                       const I32Array& camera_codes) const {
                    auto observations = read_observations(boxes, frame_sizes, camera_codes);
                    const auto n = static_cast<int>(observations.size());
                    std::vector<unsigned char> allowed;
                    allowed = matcher_.gate(observations.data(), n);
                    auto result = py::array_t<bool>(
                        {static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(n)});
                    auto* out = result.mutable_data();
                    for (size_t index = 0; index < allowed.size(); ++index)
                        out[index] = allowed[index] != 0;
                    return result;
                }

                py::array_t<float> build(const F32Array& boxes, const I32Array& frame_sizes,
                                         const I32Array& camera_codes) const {
                    auto observations = read_observations(boxes, frame_sizes, camera_codes);
                    const auto n = static_cast<int>(observations.size());
                    std::vector<float> distance;
                    distance = matcher_.build(observations.data(), n);
                    return wrap_square(distance, n);
                }

                float spatial_threshold() const { return matcher_.options().spatial_threshold; }

                size_t cameras() const { return matcher_.ground_plane().size(); }

                const mtmc::SpatialMatcher& matcher() const { return matcher_; }

            private:
                mtmc::SpatialMatcher matcher_;
        };

        /// `GatedMatcher` with its numpy edge: appearance and geometry in ONE crossing.
        ///
        /// That is the whole point of binding the composed matcher rather than its halves. Called
        /// pass by pass from numpy, one instant costs five crossings and five full (n, n)
        /// temporaries — threshold, ground distance, gate, veto, distance conversion — and at
        /// fifty cameras with fifteen tracks each that is 560 000 entries walked five times, a
        /// thousand times a second.
        class GatedSession {
            public:
                GatedSession(float appearance_threshold, float spatial_threshold, double foot_ratio,
                             double aspect_ratio, const F64Array& homographies,
                             const I32Array& calibration_sizes, const U8Array& calibrated)
                    : matcher_(
                          mtmc::AppearanceMatcher(appearance_threshold),
                          mtmc::SpatialMatcher(
                              mtmc::SpatialMatcher::Options{spatial_threshold, foot_ratio,
                                                            aspect_ratio},
                              read_ground_plane(homographies, calibration_sizes, calibrated))) {}

                py::array_t<float> similarities(const F32Array& gram, const F32Array& boxes,
                                                const I32Array& frame_sizes,
                                                const I32Array& camera_codes) const {
                    const auto info = gram.request();
                    const int n = square_side(info, "a cosine similarity matrix");
                    auto observations = read_observations(boxes, frame_sizes, camera_codes);
                    if (static_cast<int>(observations.size()) != n) {
                        throw std::invalid_argument(
                            "the appearance matrix and the tracks it describes disagree on how "
                            "many tracks this instant has");
                    }
                    const auto* data = static_cast<const float*>(info.ptr);
                    std::vector<float> similarity;
                    similarity = matcher_.similarities(data, observations.data(), n);
                    return wrap_square(similarity, n);
                }

                py::array_t<float> build(const F32Array& gram, const F32Array& boxes,
                                         const I32Array& frame_sizes,
                                         const I32Array& camera_codes) const {
                    const auto info = gram.request();
                    const int n = square_side(info, "a cosine similarity matrix");
                    auto observations = read_observations(boxes, frame_sizes, camera_codes);
                    if (static_cast<int>(observations.size()) != n) {
                        throw std::invalid_argument(
                            "the appearance matrix and the tracks it describes disagree on how "
                            "many tracks this instant has");
                    }
                    const auto* data = static_cast<const float*>(info.ptr);
                    std::vector<float> distance;
                    distance = matcher_.build(data, observations.data(), n);
                    return wrap_square(distance, n);
                }

                float appearance_threshold() const {
                    return matcher_.appearance().appearance_threshold();
                }

                float spatial_threshold() const {
                    return matcher_.spatial().options().spatial_threshold;
                }

                size_t cameras() const { return matcher_.spatial().ground_plane().size(); }

            private:
                mtmc::GatedMatcher matcher_;
        };

        /// `AgglomerativeClusterer` with its numpy edge.
        class AgglomerativeSession {
            public:
                explicit AgglomerativeSession(double distance_threshold)
                    : clusterer_(distance_threshold) {}

                py::array_t<int> fit_predict(const F64Array& distances) const {
                    const auto info = distances.request();
                    const int n = square_side(info, "a pairwise distance matrix");
                    const auto* data = static_cast<const double*>(info.ptr);
                    std::vector<int> labels;
                    labels = clusterer_.fit_predict(data, n);
                    auto result = py::array_t<int>({static_cast<py::ssize_t>(n)});
                    if (!labels.empty()) {
                        std::memcpy(result.mutable_data(), labels.data(),
                                    labels.size() * sizeof(int));
                    }
                    return result;
                }

                double distance_threshold() const { return clusterer_.distance_threshold(); }

            private:
                mtmc::AgglomerativeClusterer clusterer_;
        };

        // -- the individual fused passes, for a caller that composes them itself -------------
        //
        // The whole matchers above are what production runs. These are the passes underneath
        // them, bound one by one because `shipvision/mtmc/backends/native.py` subclasses the
        // numpy matchers and replaces exactly the passes worth replacing — it keeps the
        // projection and the gemm in Python, so it needs the tail of the pipeline and not the
        // whole of it.

        py::array_t<float> mtmc_threshold_similarity(const F32Array& similarity, float threshold) {
            const auto info = similarity.request();
            const int n = square_side(info, "a similarity matrix");
            std::vector<float> values(static_cast<size_t>(n) * n);
            if (!values.empty())
                std::memcpy(values.data(), info.ptr, values.size() * sizeof(float));
            mtmc::threshold_similarity(values, n, threshold);
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
            distances = mtmc::ground_distances(point_data, known_data, n);
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
            similarity = mtmc::spatial_similarity(data, n, threshold);
            return wrap_square(similarity, n);
        }

        py::array_t<bool> mtmc_spatial_gate(const F64Array& distances, float threshold) {
            const auto info = distances.request();
            const int n = square_side(info, "a ground-distance matrix");
            const auto* data = static_cast<const double*>(info.ptr);
            std::vector<unsigned char> allowed;
            allowed = mtmc::spatial_gate(data, n, threshold);
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
            mtmc::veto(values, gate_data, n);
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
            distance = mtmc::to_distance(values, codes, n);
            return wrap_square(distance, n);
        }

        // -- the one primitive worth its own entry point -------------------------------------

        py::array_t<double> mtmc_foot_points(const F32Array& boxes, const F64Array& frame_heights,
                                             double foot_ratio, double aspect_ratio) {
            const auto box_info = boxes.request();
            const auto height_info = frame_heights.request();
            if (box_info.ndim != 2 || box_info.shape[1] != 4) {
                throw std::invalid_argument("boxes must be (n, 4) xyxy float32");
            }
            if (height_info.ndim != 1 || height_info.shape[0] != box_info.shape[0]) {
                throw std::invalid_argument(
                    "one frame height per box is required: whether a box was cut off by the "
                    "bottom of the frame is what decides where its feet are");
            }
            const auto n = static_cast<int>(box_info.shape[0]);
            const auto* box_data = static_cast<const float*>(box_info.ptr);
            const auto* height_data = static_cast<const double*>(height_info.ptr);
            std::vector<double> points(static_cast<size_t>(n) * 2);
            mtmc::foot_points(box_data, height_data, n, foot_ratio, aspect_ratio, points.data());
            auto result =
                py::array_t<double>({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(2)});
            if (!points.empty())
                std::memcpy(result.mutable_data(), points.data(), points.size() * sizeof(double));
            return result;
        }

    }  // namespace

    void bind_mtmc(py::module_& module) {
        py::class_<AppearanceSession>(
            module, "MtmcAppearanceMatcher",
            "Cosine appearance similarity, hard-thresholded, with same-camera pairs excluded. "
            "The gemm stays in numpy — this takes the gram matrix and owns everything around it.")
            .def(py::init<float>(), py::arg("appearance_threshold") = 0.86f)
            .def("similarities", &AppearanceSession::similarities, py::arg("similarity"),
                 "(n, n) float32 thresholded cosine similarity. Zero means 'no appearance "
                 "evidence', which to_distance turns into NEVER_MERGE.")
            .def("build", &AppearanceSession::build, py::arg("similarity"), py::arg("camera_codes"),
                 "(n, n) float32 clusterable distance: threshold, same-camera exclusion, "
                 "symmetrise, zero diagonal.")
            .def_property_readonly("appearance_threshold",
                                   &AppearanceSession::appearance_threshold);

        py::class_<SpatialSession>(
            module, "MtmcSpatialMatcher",
            "Euclidean distance between track foot points projected onto a shared ground plane.")
            .def(
                py::init<float, double, double, const F64Array&, const I32Array&, const U8Array&>(),
                py::arg("spatial_threshold") = 280.0f, py::arg("foot_ratio") = 1.0,
                py::arg("aspect_ratio") = 0.25, py::arg("homographies"),
                py::arg("calibration_sizes"), py::arg("calibrated"))
            .def("ground_positions", &SpatialSession::ground_positions, py::arg("boxes"),
                 py::arg("frame_sizes"), py::arg("camera_codes"),
                 "((n, 2) float32 ground points, (n,) bool calibrated). An uncalibrated camera's "
                 "point is NaN, not the origin — the origin is a real place on the map.")
            .def("ground_distances", &SpatialSession::ground_distances, py::arg("boxes"),
                 py::arg("frame_sizes"), py::arg("camera_codes"),
                 "(n, n) float64 separation on the ground plane; inf where at least one of the "
                 "pair's cameras is uncalibrated.")
            .def("similarities", &SpatialSession::similarities, py::arg("boxes"),
                 py::arg("frame_sizes"), py::arg("camera_codes"),
                 "(n, n) float32 in [0, 1]: 1 at zero separation, 0 at the threshold and beyond.")
            .def("gate", &SpatialSession::gate, py::arg("boxes"), py::arg("frame_sizes"),
                 py::arg("camera_codes"),
                 "(n, n) bool: true where geometry does not object, INCLUDING where it cannot "
                 "judge — an uncalibrated camera takes part on appearance alone.")
            .def("build", &SpatialSession::build, py::arg("boxes"), py::arg("frame_sizes"),
                 py::arg("camera_codes"),
                 "(n, n) float32 clusterable distance from position alone. Here an unknowable "
                 "pair becomes NEVER_MERGE — the opposite of what gate() does with it.")
            .def_property_readonly("spatial_threshold", &SpatialSession::spatial_threshold)
            .def_property_readonly("cameras", &SpatialSession::cameras);

        py::class_<GatedSession>(
            module, "MtmcGatedMatcher",
            "Appearance vetoed by geometry: the production matcher, in one crossing.")
            .def(py::init<float, float, double, double, const F64Array&, const I32Array&,
                          const U8Array&>(),
                 py::arg("appearance_threshold") = 0.86f, py::arg("spatial_threshold") = 280.0f,
                 py::arg("foot_ratio") = 1.0, py::arg("aspect_ratio") = 0.25,
                 py::arg("homographies"), py::arg("calibration_sizes"), py::arg("calibrated"))
            .def("similarities", &GatedSession::similarities, py::arg("similarity"),
                 py::arg("boxes"), py::arg("frame_sizes"), py::arg("camera_codes"),
                 "(n, n) float32 appearance similarity with geometrically impossible pairs set "
                 "to EXACTLY zero — a veto, never a penalty.")
            .def("build", &GatedSession::build, py::arg("similarity"), py::arg("boxes"),
                 py::arg("frame_sizes"), py::arg("camera_codes"),
                 "(n, n) float32 clusterable distance: appearance, the geometric veto and the "
                 "shared conversion, in one pass.")
            .def_property_readonly("appearance_threshold", &GatedSession::appearance_threshold)
            .def_property_readonly("spatial_threshold", &GatedSession::spatial_threshold)
            .def_property_readonly("cameras", &GatedSession::cameras);

        py::class_<AgglomerativeSession>(
            module, "MtmcAgglomerativeClusterer",
            "Average-linkage agglomerative clustering cut at a distance. No cluster count: the "
            "number of identities in front of a camera group is what is being asked.")
            .def(py::init<double>(), py::arg("distance_threshold") = 0.14)
            .def("fit_predict", &AgglomerativeSession::fit_predict, py::arg("distances"),
                 "(n,) int32 labels, numbered by first appearance. Only equality between labels "
                 "carries meaning.")
            .def_property_readonly("distance_threshold", &AgglomerativeSession::distance_threshold);

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
        module.def("mtmc_foot_points", &mtmc_foot_points, py::arg("boxes"),
                   py::arg("frame_heights"), py::arg("foot_ratio") = 1.0,
                   py::arg("aspect_ratio") = 0.25,
                   "(n, 2) float64 image points where each object meets the ground. A box the "
                   "bottom of the frame cut off has its feet extrapolated from its width.");
    }

}  // namespace shipvision::bindings
