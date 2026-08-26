#include "shipvision/mtmc/core/spatial/utils.h"

#include <algorithm>

namespace shipvision::mtmc {

    void foot_points(const float* boxes, const double* frame_heights, int n, double foot_ratio,
                     double aspect_ratio, double* out) {
        // Clamped exactly as the numpy version clamps it. A zero aspect ratio is a configuration
        // error the constructor already refuses, but the division happens once per track per
        // instant and an inf here would travel all the way to a map coordinate.
        const double safe_aspect = std::max(aspect_ratio, 1e-6);
        for (int index = 0; index < n; ++index) {
            const double x1 = static_cast<double>(boxes[index * 4]);
            const double y1 = static_cast<double>(boxes[index * 4 + 1]);
            const double x2 = static_cast<double>(boxes[index * 4 + 2]);
            const double y2 = static_cast<double>(boxes[index * 4 + 3]);
            const double width = x2 - x1;
            const double height = y2 - y1;
            // `>= h - 1` rather than `>= h`: a detector clipping to the frame writes the last
            // valid pixel, which is h - 1, so testing for h exactly would miss every truncated
            // box there is.
            const bool truncated = y2 >= frame_heights[index] - 1.0;
            const double drop =
                truncated ? std::max(height, width / safe_aspect) : height * foot_ratio;
            out[index * 2] = (x1 + x2) * 0.5;
            out[index * 2 + 1] = y1 + drop;
        }
    }

}  // namespace shipvision::mtmc
