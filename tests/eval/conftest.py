"""Scenarios small enough to check by hand.

Every fixture here is a situation with a *stated* correct answer, and the answer is written
out arithmetically in the test that uses it. That is the only kind of metric test worth
having: "the number did not change" catches a regression but cannot catch a metric that has
been wrong since the day it was written, which is the failure mode that matters — a wrong
metric does not crash, it flatters.

Boxes are person-shaped (taller than wide) and 30x60, which makes the IoU of two boxes offset
by ``d`` on one axis exactly ``(30 - d) / (30 + d)``. Every threshold in these tests is chosen
from that identity, so a reader can verify the admissibility claims without running anything.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from shipvision.eval.sequence import EvaluationCase, ObjectFrame, TrackSequence
from shipvision.types import Detection, Detections, FrameTag

CAMERA = "quay-3"
WIDTH = 30.0
HEIGHT = 60.0

#: Where the real MOT17 train split lives on the machines this was developed on. The
#: environment variable wins, so a different checkout does not need a code change; the tests
#: that use it skip when it is absent, which is what keeps the offline tier dataset-free.
MOT17_ENV = "SHIPVISION_MOT17_TRAIN"
MOT17_DEFAULT = Path("/home/dungha15/workspaces/phucnp/shipinfer/data/mot17/train")


def box(x: float, y: float, w: float = WIDTH, h: float = HEIGHT) -> np.ndarray:
    """An xyxy box with its top-left corner at ``(x, y)``."""
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def frame(frame_id: int, entries: Sequence[tuple[int, float]]) -> ObjectFrame:
    """``(id, x)`` pairs at a fixed y. One axis of freedom keeps the IoU arithmetic readable."""
    return ObjectFrame(
        frame_id=frame_id,
        ids=np.array([i for i, _ in entries], dtype=np.int64),
        boxes=np.stack([box(x, 0.0) for _, x in entries]) if entries else np.zeros((0, 4)),
    )


def sequence(name: str, frames: Sequence[ObjectFrame], *, length: int = 0) -> TrackSequence:
    return TrackSequence(name=name, frames=tuple(frames), length=length or len(frames))


def detections(frame_id: int, xs: Sequence[float], *, score: float = 0.9) -> Detections:
    return Detections(
        tag=FrameTag(camera_id=CAMERA, frame_id=frame_id),
        items=[Detection(box=box(x, 0.0), score=score) for x in xs],
        height=1080,
        width=1920,
    )


# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def two_objects() -> TrackSequence:
    """Ground truth: two people, three frames, walking apart. GT = 6 detections.

    Object 1 sits still at x=0; object 2 walks from x=200 outwards. Far enough apart that no
    prediction can be ambiguous, so any error a metric reports is the metric's.
    """
    return sequence(
        "two-objects",
        [
            frame(1, [(1, 0.0), (2, 200.0)]),
            frame(2, [(1, 0.0), (2, 210.0)]),
            frame(3, [(1, 0.0), (2, 220.0)]),
        ],
    )


@pytest.fixture
def perfect(two_objects: TrackSequence) -> TrackSequence:
    """The same boxes under different ids. A metric must not reward id *equality*."""
    return sequence(
        "perfect",
        [
            frame(1, [(71, 0.0), (82, 200.0)]),
            frame(2, [(71, 0.0), (82, 210.0)]),
            frame(3, [(71, 0.0), (82, 220.0)]),
        ],
    )


@pytest.fixture
def four_frames() -> TrackSequence:
    """Two people over four frames. GT = 8 detections, for the ID-switch arithmetic."""
    return sequence(
        "four-frames",
        [frame(t, [(1, 0.0), (2, 200.0 + 10.0 * t)]) for t in range(1, 5)],
    )


@pytest.fixture
def four_frames_perfect() -> TrackSequence:
    return sequence(
        "perfect",
        [frame(t, [(71, 0.0), (82, 200.0 + 10.0 * t)]) for t in range(1, 5)],
    )


@pytest.fixture
def four_frames_one_switch() -> TrackSequence:
    """Object 2 is id 82 for two frames and id 83 for the next two. Exactly one switch."""
    return sequence(
        "one-switch",
        [
            frame(1, [(71, 0.0), (82, 210.0)]),
            frame(2, [(71, 0.0), (82, 220.0)]),
            frame(3, [(71, 0.0), (83, 230.0)]),
            frame(4, [(71, 0.0), (83, 240.0)]),
        ],
    )


@pytest.fixture
def crossing() -> tuple[TrackSequence, TrackSequence]:
    """Two people crossing, built so that a free re-solve prefers the swap.

    Frame 1 has them 40 px apart, which for 30-wide boxes means the cross pairs do not
    overlap at all and the assignment is forced. Frame 2 has ground truth at x=0 and x=10 with
    predictions at x=8 and x=2, so:

    * carrying frame 1's mapping forward pairs (0, 8) and (10, 2), each at IoU
      ``(30-8)/(30+8) = 22/38 = 0.579`` — admissible at 0.5, total 1.158;
    * swapping pairs (0, 2) and (10, 8), each at IoU ``28/32 = 0.875`` — also admissible,
      total 1.750.

    A matcher that maximises IoU alone therefore takes the swap and reports two identity
    switches the tracker never made. This is the case that proves the CLEAR matcher.
    """
    ground_truth = sequence(
        "crossing",
        [frame(1, [(1, 0.0), (2, 40.0)]), frame(2, [(1, 0.0), (2, 10.0)])],
    )
    predictions = sequence(
        "crossing-pred",
        [frame(1, [(71, 0.0), (82, 40.0)]), frame(2, [(71, 8.0), (82, 2.0)])],
    )
    return ground_truth, predictions


@pytest.fixture
def split_track() -> tuple[TrackSequence, TrackSequence]:
    """One person over four frames, tracked as two half-length identities.

    Ground truth is one trajectory of length 4. The prediction is id 71 on frames 1-2 and id
    82 on frames 3-4, boxes exact. A per-frame count of agreements says every frame was
    right; the global trajectory matching can only credit one of the two halves.
    """
    ground_truth = sequence("split", [frame(t, [(1, 0.0)]) for t in range(1, 5)])
    predictions = sequence(
        "split-pred",
        [
            frame(1, [(71, 0.0)]),
            frame(2, [(71, 0.0)]),
            frame(3, [(82, 0.0)]),
            frame(4, [(82, 0.0)]),
        ],
    )
    return ground_truth, predictions


@pytest.fixture
def simple_case(two_objects: TrackSequence) -> EvaluationCase:
    """A three-frame case whose public detections are exactly the ground-truth boxes.

    Scores at 0.9 so that every tracker's default ``det_threshold`` accepts them, and the
    positions are the ground truth's, so a tracker's *only* possible errors are lifecycle
    ones — which is what a runner test wants to see.
    """
    return EvaluationCase(
        name="two-objects",
        detections=tuple(
            detections(f.frame_id, [float(b[0]) for b in f.boxes]) for f in two_objects
        ),
        ground_truth=two_objects,
        height=1080,
        width=1920,
    )


@pytest.fixture(scope="session")
def mot17_root() -> Path:
    """The real MOT17 train split, or a skip.

    A skip and not a failure: the offline tier must stay runnable with no dataset on disk,
    and a test that silently passed by evaluating nothing would be worse than one that says
    it did not run.
    """
    root = Path(os.environ.get(MOT17_ENV, MOT17_DEFAULT))
    if not root.is_dir():
        pytest.skip(f"no MOT17 train split at {root}; set {MOT17_ENV} to point at one")
    return root
