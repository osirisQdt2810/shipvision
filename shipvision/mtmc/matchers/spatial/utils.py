"""Where an object meets the ground, in image pixels.

Split from the matcher because it is pure image geometry — boxes and frame heights in, points
out — with no homography, no camera group and no distance in it. That is what lets the one
interesting case below be tested against hand-computed numbers, which is the only way anybody
would notice it going wrong: a foot point that is a hundred pixels too high still projects to
a plausible place on the map.
"""

from __future__ import annotations

import numpy as np

from shipvision.errors import ConfigurationError

__all__ = ["foot_points"]


def foot_points(
    boxes: np.ndarray,
    frame_heights: np.ndarray,
    *,
    foot_ratio: float = 1.0,
    aspect_ratio: float = 0.25,
) -> np.ndarray:
    """``(n, 4)`` xyxy boxes to ``(n, 2)`` image points where each object meets the ground.

    Args:
        boxes: ``(n, 4)`` xyxy, absolute pixels.
        frame_heights: ``(n,)`` the frame height each box was measured in.
        foot_ratio: where the ground is within an un-clipped box, as a fraction of its height
            from the top. 1.0 is its bottom edge, which is right for a person standing.
        aspect_ratio: width-to-height ratio of a whole, un-clipped object — 0.25 meaning a
            person is four times taller than they are wide. Used only to extrapolate a box
            that the bottom of the frame cut off.

    A person's ground position is under their feet, so the foot point is the bottom-centre of
    the box — unless the box is clipped by the bottom edge of the frame, in which case the
    feet are *outside* the image and the bottom-centre is somewhere around the waist. The
    reference detects that with an aspect test: a person is roughly four times taller than
    they are wide, so a box touching the bottom edge has its foot estimated at
    ``width / aspect_ratio`` below its top rather than at its own bottom. Skip that and every
    track in the near field of every camera projects metres short of where it is,
    consistently, which reads as a systematic map offset rather than as a bug.

    Vectorised over the whole group: this runs once per synchronised instant over every track
    in flight, and the arithmetic is four ufuncs against a thousand Python-level branches.
    """
    box = np.atleast_2d(np.asarray(boxes, dtype=np.float64))
    heights = np.asarray(frame_heights, dtype=np.float64).reshape(-1)
    if box.shape[-1] != 4:
        raise ConfigurationError(f"boxes must be (n, 4) xyxy, got shape {box.shape}")
    if heights.shape[0] != box.shape[0]:
        raise ConfigurationError(
            f"{box.shape[0]} boxes against {heights.shape[0]} frame heights"
        )
    width = box[:, 2] - box[:, 0]
    height = box[:, 3] - box[:, 1]
    truncated = box[:, 3] >= heights - 1.0
    drop = np.where(
        truncated,
        np.maximum(height, width / max(aspect_ratio, 1e-6)),
        height * foot_ratio,
    )
    return np.stack([(box[:, 0] + box[:, 2]) * 0.5, box[:, 1] + drop], axis=1)
