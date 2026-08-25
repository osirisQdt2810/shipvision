"""Scenario builders. Every tracker test here is a situation, not a call.

The point of testing a tracker this way is that its internals are negotiable — a different
cost, a different gate — while "two people walking past each other keep their own identities"
is not. So the vocabulary in this file is positions over time, and the assertions downstream
are about identities.

``drive`` exists so no test has to build a :class:`~shipvision.types.Detections` by hand. That
matters more than convenience: the tag is the thing a tracker must never lose, and a helper
that constructs it once, monotonically, from one camera is a helper that cannot accidentally
test a mis-tagged sequence and call the result a tracking bug.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from shipvision.types import Detection, Detections, FrameTag, Track

CAMERA = "quay-3"
FRAME_HEIGHT = 1080
FRAME_WIDTH = 1920


def box(cx: float, cy: float, w: float = 40.0, h: float = 100.0) -> np.ndarray:
    """An xyxy box centred on ``(cx, cy)``. Person-shaped by default: taller than wide."""
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)


def det(
    cx: float,
    cy: float,
    score: float = 0.9,
    *,
    w: float = 40.0,
    h: float = 100.0,
    embedding: np.ndarray | None = None,
    class_id: int = 0,
) -> Detection:
    return Detection(box=box(cx, cy, w, h), score=score, class_id=class_id, embedding=embedding)


def frame(
    items: Sequence[Detection],
    frame_id: int,
    *,
    camera: str = CAMERA,
    height: int = FRAME_HEIGHT,
    width: int = FRAME_WIDTH,
) -> Detections:
    return Detections(
        tag=FrameTag(camera_id=camera, frame_id=frame_id),
        items=list(items),
        height=height,
        width=width,
    )


def drive(
    tracker: object,
    frames: Sequence[Sequence[Detection]],
    *,
    camera: str = CAMERA,
    images: Sequence[np.ndarray] | None = None,
    start: int = 0,
) -> list[list[Track]]:
    """Feed a whole sequence and collect what was published on each frame.

    ``frames`` may contain empty lists, and they are not skipped: an empty frame is
    information — the tracks still age — and a tracker that treats it as a no-op keeps dead
    objects alive forever. Several tests exist only to check that.
    """
    published: list[list[Track]] = []
    for offset, items in enumerate(frames):
        image = None if images is None else images[offset]
        published.append(
            tracker.update(frame(items, start + offset, camera=camera), image=image)
        )
    return published


def ids(published: Sequence[Sequence[Track]]) -> list[set[int]]:
    """Per-frame sets of published track ids."""
    return [{track.track_id for track in step} for step in published]


def all_ids(published: Sequence[Sequence[Track]]) -> set[int]:
    return {t.track_id for step in published for t in step}


def centres(published: Sequence[Track]) -> list[float]:
    return sorted(float(t.box[0] + t.box[2]) / 2.0 for t in published)


@pytest.fixture
def tracker_names() -> list[str]:
    from shipvision.mot import TRACKERS

    return TRACKERS.names()
