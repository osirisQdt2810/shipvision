"""Shared geometry for the suppression tests.

Every case in this directory is hand-built from these boxes, because "it returned some
indices" is not a result. The decisions that matter — whether the overlap test is strict, what
a decayed score means, how a tie breaks — only become visible on an input small enough to
reason about, so the whole directory works from four boxes whose pairwise IoUs are exact
fractions.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipvision.imgproc import IMGPROC
from shipvision.registry import PYTHON

ANCHOR = [0.0, 0.0, 10.0, 10.0]
"""10x10 at the origin. Area 100."""

HALF_SHIFTED = [5.0, 0.0, 15.0, 10.0]
"""The anchor slid five pixels right: intersection 50, union 150, so IoU is exactly 1/3."""

ONE_THIRD = float(np.float32(1.0 / 3.0))
"""That IoU, rounded exactly as float32 will compute it, so an at-the-threshold test is
deterministic rather than nearly deterministic."""

FAR_AWAY = [500.0, 500.0, 510.0, 510.0]
"""Overlaps nothing."""

INSIDE_SMALL = [3.0, 3.0, 5.0, 5.0]
"""Wholly inside the anchor, IoU 4/100 = 0.04 — containment is not overlap."""

INSIDE_LARGE = [0.0, 0.0, 9.0, 9.0]
"""Also wholly inside the anchor, but IoU 81/100 = 0.81."""

DEGENERATE = [7.0, 7.0, 7.0, 7.0]
"""No area, so an IoU against itself is 0/0."""


@pytest.fixture()
def ops():
    """The numpy backend: the definition of the answer for every method."""
    return IMGPROC.build("default", backend=PYTHON)


def run(ops, boxes: list[list[float]], scores: list[float], **kwargs) -> list[int]:
    """``ops.nms`` on hand-written lists, with the return contract asserted every time."""
    keep = ops.nms(
        np.array(boxes, dtype=np.float32), np.array(scores, dtype=np.float32), **kwargs
    )
    assert keep.dtype == np.int64, "indices must be int64 so they can index without a cast"
    return keep.tolist()
