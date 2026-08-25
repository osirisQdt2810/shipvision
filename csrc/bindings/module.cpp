// The pybind11 surface: `shipvision._C`.
//
// THE BOUNDARY THIS FILE EXISTS TO HOLD
//
// `include/` and `src/` are a plain C++/CUDA library: transformations and kernels that run
// on a CPU or a GPU and have never heard of Python. This file is the only one that knows
// an interpreter exists, and therefore the only one that may touch the GIL.
//
// Within it the rule is one transition per public entry point, in one shape:
//
//     1. GIL HELD    — read numpy buffers, validate, compute geometry, allocate results.
//                      Everything that needs Python happens here and produces plain PODs.
//     2. GIL RELEASED — one scoped release around the whole compute. The code inside must
//                      not touch a py:: object; it works from the PODs prepared above.
//     3. GIL HELD    — wrap the result.
//
// The previous version released the GIL in five places across nested helpers, so one call
// crossed the boundary three times and a reader could not tell from a helper's body whether
// the lock was held. Scattered releases are how a `py::` access ends up on the wrong side of
// one, and that failure is a hard interpreter crash rather than an exception.
//
// Two entry-point forms per operation:
//
//     letterbox_into(ptr)  -> writes into a caller-owned device buffer. The production path.
//     letterbox_batch(...) -> returns numpy, paying a device-to-host copy. Convenience and
//                             parity testing.
//
// Preprocessing feeds an engine on the same device, so round-tripping its output through
// host memory would undo most of what the fused kernel saved (ADR-007).

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstring>
#include <vector>

#include "bindings/mtmc.h"
#include "bindings/tracking.h"
#include "shipvision/core/buffers.h"
#include "shipvision/imgproc/image_ops.h"

namespace py = pybind11;
using shipvision::ImageView;
using shipvision::NormalizeParams;
using shipvision::Nv12View;
using shipvision::StagingRing;
using shipvision::StagingSlot;

namespace {

    using U8Array = py::array_t<unsigned char, py::array::c_style | py::array::forcecast>;
    using F32Array = py::array_t<float, py::array::c_style | py::array::forcecast>;

    NormalizeParams make_params(const std::vector<float>& mean, const std::vector<float>& std,
                                bool swap_rb) {
        if (mean.size() != 3 || std.size() != 3) {
            throw std::invalid_argument("mean and std must each have three entries");
        }
        NormalizeParams params;
        for (int i = 0; i < 3; ++i) {
            // Finite and positive, not merely non-zero. A negative divisor inverts that channel and
            // a NaN makes the whole tensor NaN, and neither raises anywhere downstream — the model
            // simply gets worse. Checked here as well as in Python because these entry points are
            // reachable without it.
            if (!std::isfinite(std[i]) || std[i] <= 0.f)
                throw std::invalid_argument("normalisation std must be finite and positive");
            if (!std::isfinite(mean[i]))
                throw std::invalid_argument("normalisation mean must be finite");
            params.mean[i] = mean[i];
            params.std[i] = std[i];
        }
        params.swap_rb = swap_rb;
        return params;
    }

    /// A whole letterbox fill level in [0, 255].
    ///
    /// `static_cast<unsigned char>` is silent: 256 fills with 0 and -1 fills with 255, so a
    /// config typo produced white bars on this path and the requested value on the numpy one,
    /// with nothing to say the two disagreed.
    unsigned char checked_pad_value(int pad_value) {
        if (pad_value < 0 || pad_value > 255) {
            throw std::invalid_argument("pad_value must be in [0, 255]");
        }
        return static_cast<unsigned char>(pad_value);
    }

    /// One source frame, resolved to a raw pointer and its letterbox geometry.
    ///
    /// Produced while the GIL is held and consumed after it is released — which is the only
    /// reason it exists. It deliberately holds no py:: object: the numpy arrays are kept
    /// alive by the caller's argument vector for the duration of the call.
    struct FramePlan {
            const void* data;
            int height;
            int width;
            size_t bytes;
            float scale;
            int pad_x;
            int pad_y;
            int out_h;
            int out_w;
    };

    /// Where one source extent lands inside the destination canvas.
    ///
    /// Extracted so that the BGR and NV12 entry points cannot drift: the rounding rule
    /// (`lroundf`, floored at 1) and the pad rule (integer `/ 2`, so the odd pixel goes to the
    /// bottom and right) are checked against `LetterboxGeometry.plan` in Python on every call,
    /// and two copies of a rule that is verified once is how the check starts passing for one
    /// path while the other quietly moves.
    struct LetterboxPlan {
            float scale;
            int pad_x;
            int pad_y;
            int out_h;
            int out_w;
    };

    LetterboxPlan compute_letterbox(int h, int w, int dst_h, int dst_w) {
        if (h <= 0 || w <= 0) {
            throw std::invalid_argument("each image extent must be positive");
        }
        const float scale = std::min(static_cast<float>(dst_h) / h, static_cast<float>(dst_w) / w);
        const int out_h = std::max(1, static_cast<int>(lroundf(h * scale)));
        const int out_w = std::max(1, static_cast<int>(lroundf(w * scale)));
        return LetterboxPlan{scale, (dst_w - out_w) / 2, (dst_h - out_h) / 2, out_h, out_w};
    }

    /// One NV12 source frame, resolved to plane pointers, strides and letterbox geometry.
    ///
    /// Carries `y_bytes` separately from `bytes` because the two planes are staged as one
    /// contiguous upload but the kernel needs to know where the chroma plane starts — and for a
    /// pitched device buffer that offset is `y_stride * height`, not `width * height`.
    struct Nv12Plan {
            const void* data;  ///< the whole frame: luma rows then chroma rows
            size_t bytes;
            size_t y_bytes;  ///< offset of the chroma plane inside `data`
            int height;
            int width;
            int y_stride;
            int uv_stride;
            LetterboxPlan box;
    };

    /// Rows in a packed NV12 buffer for `height` luma rows: `height * 3 / 2`.
    ///
    /// Inverted rather than passed in, so the Python side has one fewer parallel list to keep
    /// in step. `rows` must be a multiple of 3 and the recovered height even, which together
    /// rule out every off-by-one that a `(rows, stride)` array could otherwise hide.
    int nv12_height_from_rows(int rows) {
        if (rows <= 0 || rows % 3 != 0) {
            throw std::invalid_argument(
                "an NV12 frame must be (height * 3 / 2, stride) uint8, so its row count must be a "
                "positive multiple of 3");
        }
        const int height = rows / 3 * 2;
        if (height % 2 != 0) {
            throw std::invalid_argument(
                "NV12 requires an even height; 4:2:0 has no half chroma row");
        }
        return height;
    }

    /// Fused pre/post-processing bound to one device.
    /// Binds the calling thread to a device — as a *member*, so that it runs first.
    ///
    /// This exists because of an ordering rule that reads as pedantry until it costs a day.
    /// Members are constructed in declaration order, before the constructor body runs. The
    /// staging ring below creates three CUDA events in *its* constructor, and an event belongs
    /// to whichever device is current when it is created. With `gpuSetDevice` in the body,
    /// the events were created on the thread's default device — 0 — and then recorded on this
    /// instance's stream on device 5, which CUDA reports as `invalid resource handle`. Every
    /// letterbox on a non-zero device failed, from the first frame, while `crop_batch` and
    /// `nms` — which never record the slot event — worked, so the failure looked like a
    /// letterbox bug and was a construction-order bug. Declaring this before `ring_` is what
    /// makes the fix a guarantee of the language rather than a comment asking to be kept.
    struct BoundDevice {
        explicit BoundDevice(int device_index) {
            shipvision::check(gpuSetDevice(device_index), "gpuSetDevice");
        }
    };

    class ImageOps {
        public:
            explicit ImageOps(int device_index)
                : device_index_(device_index), bound_(device_index) {}

            int device_index() const { return device_index_; }

            py::dict scratch_bytes() const {
                py::dict out;
                out["staging_ring"] = ring_.bytes();
                out["pinned_download"] = pinned_download_.capacity();
                out["output"] = output_.capacity();
                out["nms"] = nms_boxes_.capacity() + nms_mask_.capacity();
                return out;
            }

            // -- letterbox -------------------------------------------------------------------

            /// Preprocess into a caller-owned device buffer. The production path.
            py::tuple letterbox_into(const std::vector<U8Array>& images, uintptr_t out_ptr,
                                     size_t out_bytes, int dst_h, int dst_w,
                                     const std::vector<float>& mean, const std::vector<float>& std,
                                     bool swap_rb, int pad_value, uintptr_t stream_handle) {
                auto scales = py::array_t<float>(static_cast<py::ssize_t>(images.size()));
                auto pads = py::array_t<float>(
                    {static_cast<py::ssize_t>(images.size()), static_cast<py::ssize_t>(2)});
                auto extents = py::array_t<int>(
                    {static_cast<py::ssize_t>(images.size()), static_cast<py::ssize_t>(2)});
                const auto params = make_params(mean, std, swap_rb);
                const auto fill = checked_pad_value(pad_value);
                const auto plans =
                    plan_frames(images, dst_h, dst_w, out_bytes, scales.mutable_data(),
                                pads.mutable_data(), extents.mutable_data());
                {
                    py::gil_scoped_release release;
                    run_letterbox(plans, reinterpret_cast<float*>(out_ptr), dst_h, dst_w, params,
                                  fill, reinterpret_cast<gpuStream_t>(stream_handle));
                }
                return py::make_tuple(scales, pads, extents);
            }

            /// Preprocess and bring the result back to the host. Convenience and parity testing.
            py::tuple letterbox_batch(const std::vector<U8Array>& images, int dst_h, int dst_w,
                                      const std::vector<float>& mean, const std::vector<float>& std,
                                      bool swap_rb, int pad_value, uintptr_t stream_handle) {
                const size_t elems = images.size() * 3 * static_cast<size_t>(dst_h) * dst_w;
                const size_t bytes = elems * sizeof(float);

                auto result = py::array_t<float>(
                    {static_cast<py::ssize_t>(images.size()), static_cast<py::ssize_t>(3),
                     static_cast<py::ssize_t>(dst_h), static_cast<py::ssize_t>(dst_w)});
                auto scales = py::array_t<float>(static_cast<py::ssize_t>(images.size()));
                auto pads = py::array_t<float>(
                    {static_cast<py::ssize_t>(images.size()), static_cast<py::ssize_t>(2)});
                auto extents = py::array_t<int>(
                    {static_cast<py::ssize_t>(images.size()), static_cast<py::ssize_t>(2)});
                const auto params = make_params(mean, std, swap_rb);
                const auto fill = checked_pad_value(pad_value);
                const auto plans = plan_frames(images, dst_h, dst_w, bytes, scales.mutable_data(),
                                               pads.mutable_data(), extents.mutable_data());
                auto* host_out = result.mutable_data();
                {
                    py::gil_scoped_release release;
                    const auto stream = reinterpret_cast<gpuStream_t>(stream_handle);
                    auto* device_out = static_cast<float*>(output_.reserve(bytes));
                    run_letterbox(plans, device_out, dst_h, dst_w, params, fill, stream);
                    download(device_out, host_out, bytes, stream);
                }
                return py::make_tuple(result, scales, pads, extents);
            }

            // -- letterbox, straight from a decoder's NV12 -----------------------------------

            /// Host NV12 in, caller-owned device buffer out. The path for a decoder whose output
            /// reaches system memory (upstream GStreamer without DeepStream, or software decode).
            py::tuple nv12_letterbox_into(const std::vector<U8Array>& frames,
                                          const std::vector<int>& widths, uintptr_t out_ptr,
                                          size_t out_bytes, int dst_h, int dst_w,
                                          const std::vector<float>& mean,
                                          const std::vector<float>& std, bool swap_rb,
                                          int pad_value, uintptr_t stream_handle) {
                auto scales = py::array_t<float>(static_cast<py::ssize_t>(frames.size()));
                auto pads = py::array_t<float>(
                    {static_cast<py::ssize_t>(frames.size()), static_cast<py::ssize_t>(2)});
                auto extents = py::array_t<int>(
                    {static_cast<py::ssize_t>(frames.size()), static_cast<py::ssize_t>(2)});
                const auto params = make_params(mean, std, swap_rb);
                const auto fill = checked_pad_value(pad_value);
                const auto plans =
                    plan_nv12(frames, widths, dst_h, dst_w, out_bytes, scales.mutable_data(),
                              pads.mutable_data(), extents.mutable_data());
                {
                    py::gil_scoped_release release;
                    run_nv12_letterbox(plans, reinterpret_cast<float*>(out_ptr), dst_h, dst_w,
                                       params, fill, reinterpret_cast<gpuStream_t>(stream_handle));
                }
                return py::make_tuple(scales, pads, extents);
            }

            /// Host NV12 in, numpy out. Convenience and the parity oracle's counterpart.
            py::tuple nv12_letterbox_batch(const std::vector<U8Array>& frames,
                                           const std::vector<int>& widths, int dst_h, int dst_w,
                                           const std::vector<float>& mean,
                                           const std::vector<float>& std, bool swap_rb,
                                           int pad_value, uintptr_t stream_handle) {
                const size_t elems = frames.size() * 3 * static_cast<size_t>(dst_h) * dst_w;
                const size_t bytes = elems * sizeof(float);

                auto result = py::array_t<float>(
                    {static_cast<py::ssize_t>(frames.size()), static_cast<py::ssize_t>(3),
                     static_cast<py::ssize_t>(dst_h), static_cast<py::ssize_t>(dst_w)});
                auto scales = py::array_t<float>(static_cast<py::ssize_t>(frames.size()));
                auto pads = py::array_t<float>(
                    {static_cast<py::ssize_t>(frames.size()), static_cast<py::ssize_t>(2)});
                auto extents = py::array_t<int>(
                    {static_cast<py::ssize_t>(frames.size()), static_cast<py::ssize_t>(2)});
                const auto params = make_params(mean, std, swap_rb);
                const auto fill = checked_pad_value(pad_value);
                const auto plans =
                    plan_nv12(frames, widths, dst_h, dst_w, bytes, scales.mutable_data(),
                              pads.mutable_data(), extents.mutable_data());
                auto* host_out = result.mutable_data();
                {
                    py::gil_scoped_release release;
                    const auto stream = reinterpret_cast<gpuStream_t>(stream_handle);
                    auto* device_out = static_cast<float*>(output_.reserve(bytes));
                    run_nv12_letterbox(plans, device_out, dst_h, dst_w, params, fill, stream);
                    download(device_out, host_out, bytes, stream);
                }
                return py::make_tuple(result, scales, pads, extents);
            }

            /// **Device NV12 in, device out — nothing crosses PCIe.**
            ///
            /// The reason the whole NV12 path exists. A GPU decoder has already put the frame in
            /// device memory; this takes the pointers it published and letterboxes straight into
            /// the engine's input, so a 1080p frame costs zero host traffic instead of a 6.2 MB
            /// download followed by a 6.2 MB upload.
            ///
            /// `descriptors` is an `(n, 6)` int64 array of
            /// `[y_ptr, uv_ptr, height, width, y_stride, uv_stride]`. One array rather than a list
            /// of tuples because this is the dispatch path: a caller reuses the array and the call
            /// allocates nothing per frame.
            ///
            /// The pointers are trusted. They cannot be validated from here — a device pointer from
            /// another context looks exactly like one from this context — so the *caller* owns the
            /// invariant that they belong to this instance's device and stay mapped for the
            /// duration. Getting it wrong is `cudaErrorIllegalAddress`, which is sticky.
            py::tuple nv12_letterbox_device_into(const py::array_t<int64_t>& descriptors,
                                                 uintptr_t out_ptr, size_t out_bytes, int dst_h,
                                                 int dst_w, const std::vector<float>& mean,
                                                 const std::vector<float>& std, bool swap_rb,
                                                 int pad_value, uintptr_t stream_handle) {
                const auto info = descriptors.request();
                if (info.ndim != 2 || info.shape[1] != 6) {
                    throw std::invalid_argument(
                        "descriptors must be (n, 6) int64: [y_ptr, uv_ptr, height, width, "
                        "y_stride, "
                        "uv_stride]");
                }
                const int batch = static_cast<int>(info.shape[0]);
                if (batch == 0) {
                    throw std::invalid_argument("nv12 device letterbox needs at least one frame");
                }
                const size_t required =
                    static_cast<size_t>(batch) * 3 * dst_h * dst_w * sizeof(float);
                if (out_bytes < required) {
                    throw std::invalid_argument("output buffer is too small for this batch");
                }
                const auto params = make_params(mean, std, swap_rb);
                const auto fill = checked_pad_value(pad_value);

                auto scales = py::array_t<float>(static_cast<py::ssize_t>(batch));
                auto pads = py::array_t<float>(
                    {static_cast<py::ssize_t>(batch), static_cast<py::ssize_t>(2)});
                auto extents = py::array_t<int>(
                    {static_cast<py::ssize_t>(batch), static_cast<py::ssize_t>(2)});
                auto* scales_out = scales.mutable_data();
                auto* pads_out = pads.mutable_data();
                auto* extents_out = extents.mutable_data();

                const auto* rows = static_cast<const int64_t*>(info.ptr);
                nv12_views_.clear();
                nv12_views_.reserve(static_cast<size_t>(batch));
                for (int i = 0; i < batch; ++i) {
                    const int64_t* row = rows + static_cast<size_t>(i) * 6;
                    const int height = static_cast<int>(row[2]);
                    const int width = static_cast<int>(row[3]);
                    const int y_stride = static_cast<int>(row[4]);
                    const int uv_stride = static_cast<int>(row[5]);
                    validate_nv12_extents(height, width, y_stride, uv_stride);
                    if (row[0] == 0 || row[1] == 0) {
                        throw std::invalid_argument("a null plane pointer is not a frame");
                    }
                    const LetterboxPlan box = compute_letterbox(height, width, dst_h, dst_w);
                    nv12_views_.push_back(Nv12View{reinterpret_cast<const unsigned char*>(row[0]),
                                                   reinterpret_cast<const unsigned char*>(row[1]),
                                                   height, width, y_stride, uv_stride, box.scale,
                                                   box.pad_x, box.pad_y, box.out_h, box.out_w});
                    scales_out[i] = box.scale;
                    pads_out[i * 2 + 0] = static_cast<float>(box.pad_x);
                    pads_out[i * 2 + 1] = static_cast<float>(box.pad_y);
                    extents_out[i * 2 + 0] = box.out_h;
                    extents_out[i * 2 + 1] = box.out_w;
                }

                {
                    py::gil_scoped_release release;
                    shipvision::check(gpuSetDevice(device_index_), "gpuSetDevice");
                    const auto stream = reinterpret_cast<gpuStream_t>(stream_handle);
                    // Only the descriptor table is uploaded — a few hundred bytes for a batch of
                    // eight, against 50 MB of frames on the host path.
                    auto& slot = ring_.acquire();
                    auto* device_views = static_cast<Nv12View*>(
                        slot.views().reserve(nv12_views_.size() * sizeof(Nv12View)));
                    shipvision::check(gpuMemcpyAsync(device_views, nv12_views_.data(),
                                                     nv12_views_.size() * sizeof(Nv12View),
                                                     gpuMemcpyHostToDevice, stream),
                                      "upload nv12 views");
                    shipvision::nv12_letterbox_batch(device_views, batch,
                                                     reinterpret_cast<float*>(out_ptr), dst_h,
                                                     dst_w, params, fill, stream);
                    slot.record(stream);
                }
                return py::make_tuple(scales, pads, extents);
            }

            // -- crops -----------------------------------------------------------------------

            void crop_into(const U8Array& image, const F32Array& boxes, uintptr_t out_ptr,
                           size_t out_bytes, int dst_h, int dst_w, const std::vector<float>& mean,
                           const std::vector<float>& std, bool swap_rb, uintptr_t stream_handle) {
                const auto params = make_params(mean, std, swap_rb);
                const auto plan = plan_crop(image, boxes, dst_h, dst_w, out_bytes);
                if (plan.num_boxes == 0)
                    return;
                {
                    py::gil_scoped_release release;
                    run_crop(plan, reinterpret_cast<float*>(out_ptr), dst_h, dst_w, params,
                             reinterpret_cast<gpuStream_t>(stream_handle));
                }
            }

            F32Array crop_batch(const U8Array& image, const F32Array& boxes, int dst_h, int dst_w,
                                const std::vector<float>& mean, const std::vector<float>& std,
                                bool swap_rb, uintptr_t stream_handle) {
                const auto params = make_params(mean, std, swap_rb);
                const auto plan = plan_crop(image, boxes, dst_h, dst_w, /*out_bytes=*/SIZE_MAX);

                auto result = py::array_t<float>(
                    {static_cast<py::ssize_t>(plan.num_boxes), static_cast<py::ssize_t>(3),
                     static_cast<py::ssize_t>(dst_h), static_cast<py::ssize_t>(dst_w)});
                if (plan.num_boxes == 0)
                    return result;

                const size_t bytes =
                    static_cast<size_t>(plan.num_boxes) * 3 * dst_h * dst_w * sizeof(float);
                auto* host_out = result.mutable_data();
                {
                    py::gil_scoped_release release;
                    const auto stream = reinterpret_cast<gpuStream_t>(stream_handle);
                    auto* device_out = static_cast<float*>(output_.reserve(bytes));
                    run_crop(plan, device_out, dst_h, dst_w, params, stream);
                    download(device_out, host_out, bytes, stream);
                }
                return result;
            }

            // -- nms -------------------------------------------------------------------------

            py::array_t<int64_t> nms(const F32Array& boxes, const F32Array& scores,
                                     float iou_threshold, float score_threshold, int max_output,
                                     uintptr_t stream_handle) {
                const auto box_info = boxes.request();
                const auto score_info = scores.request();
                if (box_info.ndim != 2 || box_info.shape[1] != 4) {
                    throw std::invalid_argument("boxes must be (N, 4) float32");
                }
                if (score_info.ndim != 1 || score_info.shape[0] != box_info.shape[0]) {
                    throw std::invalid_argument("scores must be (N,) float32 matching boxes");
                }
                const int n = static_cast<int>(box_info.shape[0]);
                const auto* box_data = static_cast<const float*>(box_info.ptr);
                const auto* score_data = static_cast<const float*>(score_info.ptr);

                std::vector<int64_t> kept;
                {
                    py::gil_scoped_release release;
                    kept = run_nms(box_data, score_data, n, iou_threshold, score_threshold,
                                   max_output, reinterpret_cast<gpuStream_t>(stream_handle));
                }

                auto result = py::array_t<int64_t>(static_cast<py::ssize_t>(kept.size()));
                if (!kept.empty()) {
                    std::memcpy(result.mutable_data(), kept.data(), kept.size() * sizeof(int64_t));
                }
                return result;
            }

        private:
            // ================================================================================
            // GIL HELD. These read Python objects and produce plain PODs.
            // ================================================================================

            /// Resolve a ragged batch to raw pointers plus its letterbox geometry.
            ///
            /// The geometry is three multiplies per image and is done here rather than in the
            /// kernel: it keeps the kernel free of divergence, and the scales and pads have to
            /// reach numpy anyway so that post-processing can invert the letterbox with exactly
            /// the numbers that applied it.
            ///
            /// `extents_out` carries `out_h` and `out_w` back for the same reason the scales go
            /// back, and it is not redundant with them: `out_h` is what the kernel divides by, so
            /// it is the number that decides the sampling ratio, and Python re-derives it from the
            /// scale rather than being told. Those two derivations can disagree by a pixel while
            /// the scale and the pad both still match — `pad = (T - r) / 2` is the same for `r` and
            /// `r + 1` whenever `T - r` is even — and then every row of the image is sampled from
            /// the wrong ratio with the drift guard none the wiser.
            std::vector<FramePlan> plan_frames(const std::vector<U8Array>& images, int dst_h,
                                               int dst_w, size_t out_bytes, float* scales_out,
                                               float* pads_out, int* extents_out) const {
                if (images.empty())
                    throw std::invalid_argument("letterbox needs at least one image");
                const size_t required =
                    images.size() * 3 * static_cast<size_t>(dst_h) * dst_w * sizeof(float);
                if (out_bytes < required) {
                    throw std::invalid_argument("output buffer is too small for this batch");
                }

                std::vector<FramePlan> plans;
                plans.reserve(images.size());
                for (size_t i = 0; i < images.size(); ++i) {
                    const auto info = images[i].request();
                    if (info.ndim != 3 || info.shape[2] != 3) {
                        throw std::invalid_argument("each image must be (H, W, 3) uint8");
                    }
                    const int h = static_cast<int>(info.shape[0]);
                    const int w = static_cast<int>(info.shape[1]);
                    // Both extents must be positive, and this is not a redundant check on top of
                    // the Python one. `out_h` below floors at 1, so a zero-row frame still gets a
                    // row of output, the kernel samples `src_y = -0.5`, and `sample_bilinear`
                    // clamps its high tap to `min(y0 + 1, h - 1) = -1` — a read before the staging
                    // allocation at batch index 0, and a read of the *previous frame's* pixels at
                    // any higher index. The first raises cudaErrorIllegalAddress, which is sticky
                    // and kills the worker for the life of the process; the second is silent. This
                    // entry point is reachable from any caller, so the guard cannot live only in
                    // Python.
                    if (h <= 0 || w <= 0) {
                        throw std::invalid_argument("each image extent must be positive");
                    }
                    const LetterboxPlan box = compute_letterbox(h, w, dst_h, dst_w);

                    plans.push_back(FramePlan{info.ptr, h, w, static_cast<size_t>(h) * w * 3,
                                              box.scale, box.pad_x, box.pad_y, box.out_h,
                                              box.out_w});
                    scales_out[i] = box.scale;
                    pads_out[i * 2 + 0] = static_cast<float>(box.pad_x);
                    pads_out[i * 2 + 1] = static_cast<float>(box.pad_y);
                    extents_out[i * 2 + 0] = box.out_h;
                    extents_out[i * 2 + 1] = box.out_w;
                }
                return plans;
            }

            /// The three things about an NV12 frame that cannot be recovered later.
            ///
            /// Checked here rather than in Python alone because these entry points are reachable
            /// from any caller and each failure is silent. An odd extent has no chroma sample to
            /// share, so the `>> 1` indexing runs off the end of the last row; a stride below the
            /// width makes every row start inside the previous one, which produces a sheared image
            /// and no error at all.
            static void validate_nv12_extents(int height, int width, int y_stride, int uv_stride) {
                if (height <= 0 || width <= 0) {
                    throw std::invalid_argument("each frame extent must be positive");
                }
                if (height % 2 != 0 || width % 2 != 0) {
                    throw std::invalid_argument(
                        "NV12 is 4:2:0, so both extents must be even; one chroma sample serves a "
                        "2x2 "
                        "luma block and there is no half block");
                }
                if (y_stride < width || uv_stride < width) {
                    throw std::invalid_argument(
                        "an NV12 stride below the width would make every row start inside the "
                        "previous "
                        "one — a sheared image, with nothing to report it");
                }
            }

            /// Resolve a ragged batch of host NV12 frames to pointers, strides and geometry.
            ///
            /// Each frame is a `(height * 3 / 2, stride)` uint8 array: luma rows followed by the
            /// interleaved chroma rows, which is one allocation and exactly what a decoder hands
            /// over. The height comes from the row count and the stride from the array's own second
            /// extent, so the only thing the caller has to say is the *visible* width — the one
            /// number no buffer layout records.
            std::vector<Nv12Plan> plan_nv12(const std::vector<U8Array>& frames,
                                            const std::vector<int>& widths, int dst_h, int dst_w,
                                            size_t out_bytes, float* scales_out, float* pads_out,
                                            int* extents_out) const {
                if (frames.empty())
                    throw std::invalid_argument("nv12 letterbox needs at least one frame");
                if (widths.size() != frames.size()) {
                    throw std::invalid_argument("one visible width per frame is required");
                }
                const size_t required =
                    frames.size() * 3 * static_cast<size_t>(dst_h) * dst_w * sizeof(float);
                if (out_bytes < required) {
                    throw std::invalid_argument("output buffer is too small for this batch");
                }

                std::vector<Nv12Plan> plans;
                plans.reserve(frames.size());
                for (size_t i = 0; i < frames.size(); ++i) {
                    const auto info = frames[i].request();
                    if (info.ndim != 2) {
                        throw std::invalid_argument(
                            "each NV12 frame must be a 2-D (height * 3 / 2, stride) uint8 array");
                    }
                    const int rows = static_cast<int>(info.shape[0]);
                    const int stride = static_cast<int>(info.shape[1]);
                    const int height = nv12_height_from_rows(rows);
                    const int width = widths[i];
                    validate_nv12_extents(height, width, stride, stride);
                    if (stride < width) {
                        throw std::invalid_argument("stride is narrower than the visible width");
                    }
                    const LetterboxPlan box = compute_letterbox(height, width, dst_h, dst_w);
                    plans.push_back(Nv12Plan{info.ptr, static_cast<size_t>(rows) * stride,
                                             static_cast<size_t>(height) * stride, height, width,
                                             stride, stride, box});
                    scales_out[i] = box.scale;
                    pads_out[i * 2 + 0] = static_cast<float>(box.pad_x);
                    pads_out[i * 2 + 1] = static_cast<float>(box.pad_y);
                    extents_out[i * 2 + 0] = box.out_h;
                    extents_out[i * 2 + 1] = box.out_w;
                }
                return plans;
            }

            struct CropPlan {
                    const void* frame;
                    const void* boxes;
                    int height;
                    int width;
                    int num_boxes;
            };

            CropPlan plan_crop(const U8Array& image, const F32Array& boxes, int dst_h, int dst_w,
                               size_t out_bytes) const {
                const auto image_info = image.request();
                const auto box_info = boxes.request();
                if (image_info.ndim != 3 || image_info.shape[2] != 3) {
                    throw std::invalid_argument("image must be (H, W, 3) uint8");
                }
                if (box_info.ndim != 2 || box_info.shape[1] != 4) {
                    throw std::invalid_argument("boxes must be (N, 4) float32");
                }
                if (image_info.shape[0] <= 0 || image_info.shape[1] <= 0) {
                    throw std::invalid_argument("each image extent must be positive");
                }
                const int num_boxes = static_cast<int>(box_info.shape[0]);
                const size_t required =
                    static_cast<size_t>(num_boxes) * 3 * dst_h * dst_w * sizeof(float);
                if (out_bytes != SIZE_MAX && out_bytes < required) {
                    throw std::invalid_argument("output buffer is too small");
                }
                return CropPlan{image_info.ptr, box_info.ptr, static_cast<int>(image_info.shape[0]),
                                static_cast<int>(image_info.shape[1]), num_boxes};
            }

            // ================================================================================
            // GIL RELEASED. Nothing below may touch a py:: object.
            // ================================================================================

            void run_letterbox(const std::vector<FramePlan>& plans, float* out, int dst_h,
                               int dst_w, const NormalizeParams& params, unsigned char pad_value,
                               gpuStream_t stream) {
                shipvision::check(gpuSetDevice(device_index_), "gpuSetDevice");

                // A slot the previous few batches have finished with. Rotating is what makes the
                // asynchronous path safe: with one set of buffers, this call's host memcpy would
                // overwrite pinned bytes a previous DMA is still reading, and its upload would
                // overwrite device frames a previous kernel is still sampling. Both produce
                // plausible output, and neither is visible to a benchmark that submits the same
                // image twice.
                auto& slot = ring_.acquire();

                size_t total = 0;
                for (const auto& plan : plans)
                    total += plan.bytes;

                auto* pinned = slot.pinned().reserve(total);
                auto* device = static_cast<unsigned char*>(slot.frames().reserve(total));

                views_.clear();
                views_.reserve(plans.size());
                size_t offset = 0;
                for (const auto& plan : plans) {
                    std::memcpy(pinned + offset, plan.data, plan.bytes);
                    views_.push_back(ImageView{device + offset, plan.height, plan.width, plan.scale,
                                               plan.pad_x, plan.pad_y, plan.out_h, plan.out_w});
                    offset += plan.bytes;
                }

                // One transfer, not one per frame: eight 6 MB copies cost eight driver round
                // trips and eight chances to serialise the stream.
                shipvision::check(
                    gpuMemcpyAsync(device, pinned, total, gpuMemcpyHostToDevice, stream),
                    "upload frames");

                auto* device_views = static_cast<ImageView*>(
                    slot.views().reserve(views_.size() * sizeof(ImageView)));
                shipvision::check(
                    gpuMemcpyAsync(device_views, views_.data(), views_.size() * sizeof(ImageView),
                                   gpuMemcpyHostToDevice, stream),
                    "upload image views");

                shipvision::letterbox_batch(device_views, static_cast<int>(plans.size()), out,
                                            dst_h, dst_w, params, pad_value, stream);
                // Not a synchronise: it marks when this slot's buffers stop being read, so the
                // rotation can reuse them without anybody waiting here.
                slot.record(stream);
            }

            /// Stage a batch of host NV12 frames and launch. Mirrors `run_letterbox`.
            ///
            /// One pinned copy and one upload for the whole batch, as on the BGR path — but the
            /// batch is half the size, because NV12 is 12 bits per pixel where BGR is 24. That
            /// halving is the entire point of this path when the frames are in host memory, and it
            /// is measurable: the upload is what the letterbox has to wait for.
            void run_nv12_letterbox(const std::vector<Nv12Plan>& plans, float* out, int dst_h,
                                    int dst_w, const NormalizeParams& params,
                                    unsigned char pad_value, gpuStream_t stream) {
                shipvision::check(gpuSetDevice(device_index_), "gpuSetDevice");
                auto& slot = ring_.acquire();

                size_t total = 0;
                for (const auto& plan : plans)
                    total += plan.bytes;

                auto* pinned = slot.pinned().reserve(total);
                auto* device = static_cast<unsigned char*>(slot.frames().reserve(total));

                nv12_views_.clear();
                nv12_views_.reserve(plans.size());
                size_t offset = 0;
                for (const auto& plan : plans) {
                    std::memcpy(pinned + offset, plan.data, plan.bytes);
                    nv12_views_.push_back(Nv12View{device + offset, device + offset + plan.y_bytes,
                                                   plan.height, plan.width, plan.y_stride,
                                                   plan.uv_stride, plan.box.scale, plan.box.pad_x,
                                                   plan.box.pad_y, plan.box.out_h, plan.box.out_w});
                    offset += plan.bytes;
                }

                shipvision::check(
                    gpuMemcpyAsync(device, pinned, total, gpuMemcpyHostToDevice, stream),
                    "upload nv12 frames");

                auto* device_views = static_cast<Nv12View*>(
                    slot.views().reserve(nv12_views_.size() * sizeof(Nv12View)));
                shipvision::check(gpuMemcpyAsync(device_views, nv12_views_.data(),
                                                 nv12_views_.size() * sizeof(Nv12View),
                                                 gpuMemcpyHostToDevice, stream),
                                  "upload nv12 views");

                shipvision::nv12_letterbox_batch(device_views, static_cast<int>(plans.size()), out,
                                                 dst_h, dst_w, params, pad_value, stream);
                slot.record(stream);
            }

            void run_crop(const CropPlan& plan, float* out, int dst_h, int dst_w,
                          const NormalizeParams& params, gpuStream_t stream) {
                shipvision::check(gpuSetDevice(device_index_), "gpuSetDevice");

                const size_t frame_bytes = static_cast<size_t>(plan.height) * plan.width * 3;
                const size_t box_bytes = static_cast<size_t>(plan.num_boxes) * 4 * sizeof(float);
                // The boxes share one allocation with the frame, and h*w*3 is a multiple of 4
                // only by luck — a 1079x1919 frame is 6,211,803 bytes, so a float* placed
                // straight after it is misaligned. CUDA then raises cudaErrorMisalignedAddress,
                // which is STICKY: it poisons the context for the life of the process, so one
                // odd-sized camera would kill that worker permanently.
                const size_t box_offset = shipvision::align_up(frame_bytes);
                const size_t packed = box_offset + box_bytes;

                auto& slot = ring_.acquire();
                auto* pinned = slot.pinned().reserve(packed);
                std::memcpy(pinned, plan.frame, frame_bytes);
                std::memcpy(pinned + box_offset, plan.boxes, box_bytes);

                auto* device = static_cast<unsigned char*>(slot.frames().reserve(packed));
                shipvision::check(
                    gpuMemcpyAsync(device, pinned, packed, gpuMemcpyHostToDevice, stream),
                    "upload frame and boxes");

                const ImageView view{device, plan.height, plan.width,  1.f,
                                     0,      0,           plan.height, plan.width};
                shipvision::crop_batch(view, reinterpret_cast<const float*>(device + box_offset),
                                       plan.num_boxes, out, dst_h, dst_w, params, stream);
                slot.record(stream);
            }

            std::vector<int64_t> run_nms(const float* boxes, const float* scores, int n,
                                         float iou_threshold, float score_threshold, int max_output,
                                         gpuStream_t stream) {
                shipvision::check(gpuSetDevice(device_index_), "gpuSetDevice");
                const size_t mask_words = shipvision::nms_mask_words(n);

                shipvision::NmsScratch scratch;
                scratch.box_floats = static_cast<size_t>(n) * 4;
                scratch.mask_words = mask_words;
                scratch.boxes =
                    static_cast<float*>(nms_boxes_.reserve(scratch.box_floats * sizeof(float)));
                scratch.mask = static_cast<unsigned long long*>(
                    nms_mask_.reserve(mask_words * sizeof(unsigned long long)));
                // The pinned download buffer the other entry points already stage through.
                // Sized to the mask, not shared with an in-flight download: `run_nms` is
                // serialised with every other method by the instance lock.
                scratch.host_mask = reinterpret_cast<unsigned long long*>(
                    pinned_download_.reserve(mask_words * sizeof(unsigned long long)));

                return shipvision::nms(boxes, scores, n, iou_threshold, score_threshold, max_output,
                                       scratch, stream);
            }

            /// Device to host, through pinned memory.
            ///
            /// A direct copy into a freshly allocated numpy array is startlingly slow: the
            /// destination is pageable *and* untouched, so the driver stages it through a small
            /// bounce buffer while the kernel faults in tens of megabytes of new pages. Going
            /// through pinned memory and then doing a plain memcpy is an order of magnitude
            /// faster, and it is why the convenience entry points are merely slower than the
            /// `_into` ones rather than unusable.
            void download(const void* device_src, void* host_dst, size_t bytes,
                          gpuStream_t stream) {
                auto* staging = pinned_download_.reserve(bytes);
                shipvision::check(
                    gpuMemcpyAsync(staging, device_src, bytes, gpuMemcpyDeviceToHost, stream),
                    "download");
                shipvision::check(gpuStreamSynchronize(stream), "download synchronize");
                std::memcpy(host_dst, staging, bytes);
            }

            int device_index_;
            BoundDevice bound_;  ///< MUST precede ring_: its events are created on the current device
            StagingRing ring_;  ///< rotated per call, so reuse never races a live copy
            shipvision::DeviceScratch output_;     ///< host-returning entry points only
            shipvision::DeviceScratch nms_boxes_;  ///< score-sorted boxes for one NMS call
            shipvision::DeviceScratch nms_mask_;   ///< the (box, box) overlap bitmask
            shipvision::PinnedScratch pinned_download_;
            std::vector<ImageView> views_;      ///< host-side descriptor table, rebuilt per call
            std::vector<Nv12View> nv12_views_;  ///< the same, for the NV12 paths
    };

}  // namespace

PYBIND11_MODULE(_C, m) {
    m.doc() = "shipvision native data plane: fused preprocessing and device-side NMS";
    m.attr("__version__") = "0.1.0";

#if defined(SHIPVISION_WITH_HIP)
    m.attr("platform") = "hip";
#else
    m.attr("platform") = "cuda";
#endif

    py::register_exception<shipvision::GpuError>(m, "GpuError", PyExc_RuntimeError);

    m.def("cuda_available", &shipvision::gpu_available,
          "True when this build has GPU kernels and a visible device.");
    m.def("device_count", &shipvision::device_count, "Number of visible devices.");

    // The trackers and the cross-camera matrices. Host work, no stream, no device pointer, so
    // they live in their own translation unit — see bindings/tracking.h. They are reachable
    // even where `cuda_available()` is false, and that is the point: a build without a visible
    // device still runs the association loops the fleet's per-frame budget is spent in.
    shipvision::bindings::bind_tracking(m);

    // The whole cross-camera matchers and their clusterer — see bindings/mtmc.h. Host work too,
    // and a separate translation unit for the same reason: the single-camera trackers and the
    // cross-camera matchers are two families with two lifecycles.
    shipvision::bindings::bind_mtmc(m);

    py::class_<ImageOps>(m, "ImageOps", "Fused pre/post-processing kernels bound to one device.")
        .def(py::init<int>(), py::arg("device_index") = 0)
        .def_property_readonly("device_index", &ImageOps::device_index)
        .def("scratch_bytes", &ImageOps::scratch_bytes,
             "Persistent scratch held by this instance, in bytes.")
        .def("letterbox_batch", &ImageOps::letterbox_batch, py::arg("images"), py::arg("dst_h"),
             py::arg("dst_w"), py::arg("mean"), py::arg("std"), py::arg("swap_rb"),
             py::arg("pad_value") = 114, py::arg("stream") = 0,
             "Fused resize+pad+convert+normalise+NCHW, returned as numpy. Yields "
             "(tensor, scales, pads, resized_extents).")
        .def("letterbox_into", &ImageOps::letterbox_into, py::arg("images"), py::arg("out_ptr"),
             py::arg("out_bytes"), py::arg("dst_h"), py::arg("dst_w"), py::arg("mean"),
             py::arg("std"), py::arg("swap_rb"), py::arg("pad_value") = 114, py::arg("stream") = 0,
             "Same, written straight into a caller-owned device buffer. The fast path. "
             "Yields (scales, pads, resized_extents).")
        .def("nv12_letterbox_batch", &ImageOps::nv12_letterbox_batch, py::arg("frames"),
             py::arg("widths"), py::arg("dst_h"), py::arg("dst_w"), py::arg("mean"), py::arg("std"),
             py::arg("swap_rb"), py::arg("pad_value") = 114, py::arg("stream") = 0,
             "Fused NV12 convert+resize+pad+normalise+NCHW from host NV12, returned as numpy. "
             "Yields (tensor, scales, pads, resized_extents).")
        .def("nv12_letterbox_into", &ImageOps::nv12_letterbox_into, py::arg("frames"),
             py::arg("widths"), py::arg("out_ptr"), py::arg("out_bytes"), py::arg("dst_h"),
             py::arg("dst_w"), py::arg("mean"), py::arg("std"), py::arg("swap_rb"),
             py::arg("pad_value") = 114, py::arg("stream") = 0,
             "Same, written straight into a caller-owned device buffer. "
             "Yields (scales, pads, resized_extents).")
        .def("nv12_letterbox_device_into", &ImageOps::nv12_letterbox_device_into,
             py::arg("descriptors"), py::arg("out_ptr"), py::arg("out_bytes"), py::arg("dst_h"),
             py::arg("dst_w"), py::arg("mean"), py::arg("std"), py::arg("swap_rb"),
             py::arg("pad_value") = 114, py::arg("stream") = 0,
             "Fused NV12 letterbox from frames ALREADY on the device: (n, 6) int64 "
             "[y_ptr, uv_ptr, height, width, y_stride, uv_stride]. Nothing crosses PCIe. "
             "Yields (scales, pads, resized_extents).")
        .def("crop_batch", &ImageOps::crop_batch, py::arg("image"), py::arg("boxes"),
             py::arg("dst_h"), py::arg("dst_w"), py::arg("mean"), py::arg("std"),
             py::arg("swap_rb"), py::arg("stream") = 0,
             "Extract and resize N boxes, returned as numpy.")
        .def("crop_into", &ImageOps::crop_into, py::arg("image"), py::arg("boxes"),
             py::arg("out_ptr"), py::arg("out_bytes"), py::arg("dst_h"), py::arg("dst_w"),
             py::arg("mean"), py::arg("std"), py::arg("swap_rb"), py::arg("stream") = 0,
             "Same, written straight into a caller-owned device buffer.")
        .def("nms", &ImageOps::nms, py::arg("boxes"), py::arg("scores"), py::arg("iou_threshold"),
             py::arg("score_threshold"), py::arg("max_output"), py::arg("stream") = 0,
             "Class-agnostic NMS on the device; returns kept indices.");
}
