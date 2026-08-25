// The tracking and cross-camera half of `shipvision._C`.
//
// A second translation unit rather than more of `module.cpp`, for one reason: the entry points
// there are all *device* work, and every one of them is a variation on "stage through pinned
// memory, launch, record an event". These are host work with no stream and no device pointer
// in sight, and interleaving the two would make the file's own rule — the one GIL transition
// per entry point — harder to check by reading.
//
// The rule itself does not change and is restated here because this file must obey it:
//
//     1. GIL HELD     — read numpy buffers, validate, build plain PODs.
//     2. GIL RELEASED — one scoped release around the whole compute, touching no py:: object.
//     3. GIL HELD     — wrap the result.
//
// Releasing the GIL matters more here than it looks. A tracker runs once per camera per frame
// on the thread that owns that camera, and at fifty cameras those threads are the pipeline; a
// tracker that held the lock for its association would serialise all fifty of them behind one
// interpreter, which is the exact cost the native backend exists to remove.

#pragma once

#include <pybind11/pybind11.h>

namespace shipvision::bindings {

    /// Add the tracker classes and the MTMC matrix helpers to the module.
    void bind_tracking(pybind11::module_& module);

}  // namespace shipvision::bindings
