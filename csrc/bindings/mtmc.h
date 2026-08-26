// The cross-camera half of `shipvision._C`: the whole matchers, not just their inner passes.
//
// A third translation unit, beside `bindings/mot.h`, for two reasons. The first is the one
// that file gives: these are host passes with no stream and no device pointer, so interleaving
// them with `module.cpp`'s device work would leave one file with two unrelated shapes in it.
// The second is ownership — the single-camera trackers and the cross-camera matchers are two
// families with two lifecycles, and keeping them apart is what lets one be worked on without
// rebuilding the other's opinions.
//
// The two standing rules do not change and are restated here because this file must obey them:
// marshalling only, with every algorithm in `csrc/shipvision/mtmc/`, and no GIL policy — see
// `bindings/mot.h`. The matchers here are immutable once constructed, so unlike a tracker they
// need no mutex to make a concurrent call safe.
//
// WHAT CROSSES, AND WHAT DOES NOT. An instant is three arrays — boxes, frame sizes, camera
// codes — and never a list of objects: a list of pybind classes would allocate one Python object
// per track per instant, which at the fleet's sizing is the per-frame overhead the native
// backend exists to remove. Camera identity crosses as an integer code, for the reason
// `shipvision/mtmc/frames.h` gives. The embeddings do not cross at all; their gram matrix does,
// because `features @ features.T` is BLAS's job and numpy already has it — see
// `core/appearance/matcher.h`.
//
// One contract the codes carry that is worth stating: a matcher's ground plane is INDEXED BY
// THE SAME CODES the instant uses, so the two are chosen together. Any consistent numbering
// works — nothing here compares a code against another call's — but a caller that renumbers per
// instant has to hand over a plane in that instant's order, and a caller that wants to build a
// matcher once should number its camera group once.

#pragma once

#include <pybind11/pybind11.h>

namespace shipvision::bindings {

    /// Add the cross-camera matchers and the clusterer to the module.
    void bind_mtmc(pybind11::module_& module);

}  // namespace shipvision::bindings
