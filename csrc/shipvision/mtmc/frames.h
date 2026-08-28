// The input unit, as plain data: one camera's track at one synchronised instant.
//
// The C++ twin of `shipvision/mtmc/frames.py`, reduced to the fields the (n, n) passes read.
// `TrackKey`, the `Track` itself and the embedding do not cross. An identity map keyed on
// (camera, track) is Python's to own — it is the stateful half — and the embeddings reach here
// already multiplied into a gram matrix, for the reason `matchers/appearance/matcher.h` gives.
//
// Camera identity crosses as an INTEGER CODE, never as a name. The reference implementation
// this library replaces compared camera strings pairwise to build the same-camera exclusion,
// which at fifty cameras and fifteen tracks each is 560 000 string compares per synchronised
// instant — on its own more expensive than the clustering it feeds. Codes are assigned once on
// the Python side, by first appearance; nothing here compares them across calls, so any
// consistent numbering describes the same mask.
//
// Frame dimensions are fields rather than optionals because three separate decisions need
// them — the test for a box the bottom of the frame cut off, the rescale into the domain a
// homography was calibrated in, and the height gate upstream. A zero default would make all
// three silently wrong rather than loudly absent.

#pragma once

#include <cstddef>
#include <vector>

namespace shipvision::mtmc {

    /// One single-camera track at one instant, with the frame context the geometry needs.
    struct Observation {
            int camera_code = 0;  ///< equal for two tracks exactly when they share a camera
            float box[4] = {0.f, 0.f, 0.f, 0.f};  ///< xyxy, absolute pixels of this frame
            int frame_width = 0;
            int frame_height = 0;
    };

    /// The camera code of every observation, contiguous, for the passes that take a raw pointer.
    ///
    /// `to_distance` wants the codes as one array because that is the shape the same-camera
    /// exclusion is applied in: a strided read through a struct once per *pair* is n^2 cache
    /// misses over 560 000 entries, and gathering them once is n.
    inline std::vector<int> camera_codes(const Observation* observations, int n) {
        std::vector<int> codes(static_cast<size_t>(n < 0 ? 0 : n));
        for (int index = 0; index < n; ++index)
            codes[static_cast<size_t>(index)] = observations[index].camera_code;
        return codes;
    }

}  // namespace shipvision::mtmc
