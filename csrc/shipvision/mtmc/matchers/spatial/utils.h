// Where an object meets the ground, in image pixels.
//
// Split from the matcher for the reason `shipvision/mtmc/matchers/spatial/utils.py` gives: it is
// pure image geometry — boxes and frame heights in, points out — with no homography, no camera
// group and no distance in it, which is what lets the one interesting case be checked against
// hand-computed numbers. Nobody would notice it going wrong otherwise: a foot point a hundred
// pixels too high still projects to a plausible place on the map.

#pragma once

namespace shipvision::mtmc {

    /// `(n, 4)` xyxy boxes to `(n, 2)` image points where each object meets the ground.
    ///
    /// A person's ground position is under their feet, so the foot point is the bottom-centre of
    /// the box — unless the bottom edge of the frame cut the box off, in which case the feet are
    /// *outside* the image and the bottom-centre is somewhere around the waist. That is detected
    /// with an aspect test: a person is roughly four times taller than they are wide, so a box
    /// touching the bottom edge has its foot estimated at `width / aspect_ratio` below its top.
    /// Skip it and every track in the near field of every camera projects metres short of where
    /// it is, consistently — which reads as a systematic map offset rather than as a bug.
    ///
    /// Double throughout, matching the numpy version's dtype: the boxes arrive as float32 but
    /// the projection that consumes these points is a 3x3 product whose third component is a
    /// difference of large numbers, and float32 there loses the digits the threshold is made of.
    ///
    /// @param boxes `(n, 4)` xyxy float32, absolute pixels
    /// @param frame_heights `(n,)` the frame height each box was measured in
    /// @param foot_ratio where the ground is within an un-clipped box, as a fraction of its
    ///        height from the top; 1.0 is its bottom edge, right for a person standing
    /// @param aspect_ratio width over height for a whole, un-clipped object
    /// @param out `(n, 2)` doubles, written by the caller's allocation
    void foot_points(const float* boxes, const double* frame_heights, int n, double foot_ratio,
                     double aspect_ratio, double* out);

}  // namespace shipvision::mtmc
