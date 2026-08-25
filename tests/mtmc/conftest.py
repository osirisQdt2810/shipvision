"""Scenarios with a known-correct answer, built from embeddings that behave like real ones.

"It produced clusters" proves nothing about cross-camera tracking: a builder that returns
zeros produces clusters, and so does one that merges everything. Every test here therefore
starts from a scene somebody could describe out loud — *one person, two cameras* — and asserts
the number of identities that scene has.

Random vectors will not do. In 128 dimensions two random unit vectors are nearly orthogonal,
so every pair looks equally unlike every other and a broken similarity function passes. These
fixtures instead give each identity a direction and place each view as a small perturbation
around it, which is the structure real re-ID embeddings have and the only structure that can
tell a working association apart from a coincidence.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipvision.mtmc import CameraTracks, FrameTrackCluster, GroundPlane, Homography
from shipvision.types import FrameTag, Track

DIM = 64
FRAME_HEIGHT = 1080
FRAME_WIDTH = 1920

#: A box big enough to clear the default height gate (1/9 of 1080 = 120px) and not touching
#: the bottom edge, so the foot point is its own bottom rather than an extrapolation.
DEFAULT_BOX = (100.0, 300.0, 200.0, 700.0)


def identity_direction(identity: int, dim: int = DIM) -> np.ndarray:
    """A stable unit direction for one physical object."""
    vector = np.random.default_rng(7000 + identity).normal(size=dim)
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def view_of(
    identity: int, *, view: int = 0, jitter: float = 0.02, dim: int = DIM
) -> np.ndarray:
    """One camera's embedding of ``identity``, ``jitter`` away from its true direction.

    ``jitter`` is the knob the tests turn. At 0.02 every view of an object is far closer to
    its siblings than to any other object, which is the regime where association must be
    perfect; raise it and association must degrade rather than break.
    """
    base = identity_direction(identity, dim)
    noise = np.random.default_rng(900_000 + identity * 131 + view).normal(size=dim)
    vector = base + jitter * noise
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def make_track(
    *,
    camera: str,
    track_id: int,
    identity: int,
    frame_id: int = 0,
    view: int = 0,
    jitter: float = 0.02,
    box: tuple[float, float, float, float] = DEFAULT_BOX,
    embedding: np.ndarray | None = None,
) -> Track:
    """One single-camera track carrying the embedding of ``identity``."""
    return Track(
        track_id=track_id,
        box=np.array(box, dtype=np.float32),
        tag=FrameTag(camera_id=camera, frame_id=frame_id, timestamp=float(frame_id) / 20.0),
        embedding=(
            view_of(identity, view=view, jitter=jitter) if embedding is None else embedding
        ),
    )


def make_cluster(
    tracks_by_camera: dict[str, list[Track]],
    *,
    frame_id: int = 0,
    height: int = FRAME_HEIGHT,
    width: int = FRAME_WIDTH,
) -> FrameTrackCluster:
    """A synchronised instant from ``{camera: [tracks]}``."""
    return FrameTrackCluster.from_views(
        CameraTracks(
            tag=FrameTag(camera_id=camera, frame_id=frame_id, timestamp=frame_id / 20.0),
            tracks=tuple(tracks),
            height=height,
            width=width,
        )
        for camera, tracks in tracks_by_camera.items()
    )


def one_person_two_cameras(frame_id: int = 0, identity: int = 0) -> FrameTrackCluster:
    """The simplest scene with a right answer: one object, seen twice. Expect ONE id."""
    return make_cluster(
        {
            "cam-a": [
                make_track(
                    camera="cam-a", track_id=1, identity=identity, view=0, frame_id=frame_id
                )
            ],
            "cam-b": [
                make_track(
                    camera="cam-b", track_id=1, identity=identity, view=1, frame_id=frame_id
                )
            ],
        },
        frame_id=frame_id,
    )


def two_people_two_cameras(frame_id: int = 0) -> FrameTrackCluster:
    """Two objects, each seen by both cameras. Expect TWO ids, four assigned tracks."""
    return make_cluster(
        {
            "cam-a": [
                make_track(camera="cam-a", track_id=1, identity=0, view=0, frame_id=frame_id),
                make_track(
                    camera="cam-a",
                    track_id=2,
                    identity=1,
                    view=0,
                    frame_id=frame_id,
                    box=(600.0, 300.0, 700.0, 700.0),
                ),
            ],
            "cam-b": [
                make_track(camera="cam-b", track_id=1, identity=0, view=1, frame_id=frame_id),
                make_track(
                    camera="cam-b",
                    track_id=2,
                    identity=1,
                    view=1,
                    frame_id=frame_id,
                    box=(600.0, 300.0, 700.0, 700.0),
                ),
            ],
        },
        frame_id=frame_id,
    )


def run_for(tracker, cluster_at, steps: int) -> list:
    """Feed ``steps`` instants, returning the last result. Clears the tentative-track gate."""
    results: list = []
    for frame_id in range(steps):
        results = tracker.track(cluster_at(frame_id))
    return results


def identity_homography(*, scale: float = 1.0) -> Homography:
    """A homography that maps image pixels straight onto map units, optionally scaled.

    Not a shortcut around the geometry: a real ground plane is a projective warp, and one is
    fitted from points in ``test_topology.py``. It is used where the *decision* under test is
    the gate rather than the projection, because a test whose expected answer requires
    mentally inverting a 3x3 matrix is a test nobody can tell is wrong.
    """
    matrix = np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return Homography(matrix=matrix)


@pytest.fixture()
def flat_plane() -> GroundPlane:
    """Both test cameras mapping identically onto the ground plane."""
    return GroundPlane({"cam-a": identity_homography(), "cam-b": identity_homography()})
