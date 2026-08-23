"""Cases small enough that a study over them finishes in milliseconds.

A tuning test must not need a dataset: the whole point of the offline tier is that a change to
a search space or an objective is checkable in under a second. So the sequences here are five
frames of two people walking, built from the same vocabulary as ``tests/eval``, and the tests
assert *properties* of the machinery rather than quality of the result — which is the only
thing five frames can support.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from shipvision.eval.sequence import EvaluationCase, ObjectFrame, TrackSequence
from shipvision.types import Detection, Detections, FrameTag

FRAMES = 5


def box(x: float, y: float = 0.0, w: float = 30.0, h: float = 60.0) -> np.ndarray:
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def positions(frame_id: int) -> tuple[float, ...]:
    """Two people: one still, one walking right at ten pixels a frame."""
    return (0.0, 200.0 + 10.0 * frame_id)


def make_case(name: str, *, camera: str | None = None, frames: int = FRAMES) -> EvaluationCase:
    """A case whose public detections are exactly its ground truth.

    Exact input means a tracker's only possible errors are lifecycle ones, so a difference
    between two configurations is attributable to the parameter that changed rather than to the
    detector — which is what a tuning test needs in order to assert anything at all.
    """
    camera = camera or name
    ground_truth = TrackSequence(
        name=name,
        frames=tuple(
            ObjectFrame(
                frame_id=t,
                ids=np.array([1, 2], dtype=np.int64),
                boxes=np.stack([box(x) for x in positions(t)]),
            )
            for t in range(1, frames + 1)
        ),
        length=frames,
    )
    detections: Sequence[Detections] = tuple(
        Detections(
            tag=FrameTag(camera_id=camera, frame_id=t),
            items=[Detection(box=box(x), score=0.9) for x in positions(t)],
            height=1080,
            width=1920,
        )
        for t in range(1, frames + 1)
    )
    return EvaluationCase(
        name=name,
        detections=tuple(detections),
        ground_truth=ground_truth,
        height=1080,
        width=1920,
    )


@pytest.fixture
def case() -> EvaluationCase:
    return make_case("synthetic-a")


@pytest.fixture
def two_cases() -> tuple[EvaluationCase, EvaluationCase]:
    """Two cases on *different* cameras, so a tracker reused across them would raise."""
    return make_case("synthetic-a", camera="cam-a"), make_case("synthetic-b", camera="cam-b")
