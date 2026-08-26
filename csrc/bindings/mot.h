// The single-camera tracking half of `shipvision._C`.
//
// A second translation unit rather than more of `module.cpp`, for one reason: the entry points
// there are all *device* work, and every one of them is a variation on "stage through pinned
// memory, launch, record an event". These are host work with no stream and no device pointer
// in sight, and interleaving the two would leave one file with two unrelated shapes in it.
//
// This file obeys `module.cpp`'s two standing rules, restated here because they are what make
// a binding readable:
//
//     1. MARSHALLING ONLY — read numpy buffers, validate, build plain PODs; call the library
//        in `csrc/shipvision/mot/`; wrap the result. No algorithm lives in this directory.
//     2. NO GIL POLICY — nothing here releases or acquires the interpreter lock. A tracker
//        runs once per camera per frame on the thread that owns that camera, and whether
//        those fifty threads may overlap is a decision the embedding server makes with the
//        rest of its pipeline in view. What this file owes it instead is that overlapping is
//        *safe*: each session serialises its own tracker with a mutex, so a caller that does
//        release the lock cannot interleave two `update` calls into one pool.
//
// WHAT CROSSES, AND WHAT DOES NOT. One frame is three arrays — boxes, scores, class ids — and
// the answer is two, geometry and metadata. Never a list of objects: one pybind class per track
// per frame is 15 000 Python allocations a second at the fleet's sizing, which is the per-frame
// overhead the native backend exists to remove. Appearance crosses as the finished
// `(tracks, detections)` cosine matrix rather than as embeddings, because the EMA that produces
// those vectors lives in Python next to the numpy pool's — one definition of a track's
// appearance, not two.

#pragma once

#include <pybind11/pybind11.h>

namespace shipvision::bindings {

    /// Add the tracker classes to the module.
    void bind_mot(pybind11::module_& module);

}  // namespace shipvision::bindings
