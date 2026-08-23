"""Cross-camera tracking: one identity per object, across every camera that can see it.

Single-camera tracking answers "is this the same person as the last frame". This package
answers "and is that the same person camera 7 is looking at, and the one we were calling 41 a
minute ago". It runs **once per time-synchronised group of frames** — not offline over
tracklets — because the answer is needed while the object is still there.

Four components, one call:

    from shipvision.mtmc import MTMC, CameraTracks, FrameTrackCluster

    tracker = MTMC.build("cluster", ground_plane=plane, spatial_threshold=280.0)

    instant = FrameTrackCluster.from_views([
        CameraTracks(tag=tag_a, tracks=tracks_a, height=1080, width=1920),
        CameraTracks(tag=tag_b, tracks=tracks_b, height=1080, width=1920),
    ])
    for result in tracker.track(instant):
        if result.is_assigned:
            publish(result.global_id, result.track)

The pieces, each replaceable by name from config:

* :data:`MATRIX_BUILDERS` — tracks to an ``(n, n)`` distance matrix. ``appearance`` (cosine,
  thresholded), ``spatial`` (ground-plane separation), ``gated`` (appearance vetoed by
  geometry — the production one).
* :data:`CLUSTERERS` — the matrix to a label per track. ``agglomerative``: average linkage,
  cut at a distance, on a precomputed matrix.
* :class:`~shipvision.mtmc.identity.GlobalIdAssigner` — labels to *stable* ids, carrying state
  between calls. The stateful part, and where the reference implementation's two real bugs
  were: unbounded growth and a process-global id counter.
* :class:`~shipvision.mtmc.topology.GroundPlane` — the homographies the spatial half needs,
  and :func:`~shipvision.mtmc.topology.calculate_homography` to fit one *and say how wrong it
  is*.

Three properties this package is built around, all of them lessons from the system it
replaces:

**Two tracks in one camera never merge.** Enforced in one place,
:meth:`~shipvision.mtmc.matrix.base.BaseMatrixBuilder.mergeable_mask`. Without it MTMC
silently becomes a within-camera deduplicator: every count falls, every metric improves, and
the system is worse.

**Nothing grows without bound.** The identity maps have a maximum age *and* a capacity, and
eviction runs on every instant, including empty ones.

**Unassigned means `None`.** A track the gate held back comes back as a
:class:`~shipvision.types.GlobalTrack` with ``global_id=None``, not ``-1`` and not omitted.
"""

from __future__ import annotations

from shipvision.mtmc.base import MTMC, BaseMTMCTracker
from shipvision.mtmc.clustering import CLUSTERERS, AgglomerativeClusterer, BaseClusterer
from shipvision.mtmc.frames import (
    CameraTracks,
    FrameTrackCluster,
    TrackKey,
    TrackObservation,
)
from shipvision.mtmc.gating import ObservationGate
from shipvision.mtmc.identity import GlobalIdAssigner
from shipvision.mtmc.matrix import (
    MATRIX_BUILDERS,
    NEVER_MERGE,
    AppearanceMatrixBuilder,
    BaseMatrixBuilder,
    GatedMatrixBuilder,
    SpatialMatrixBuilder,
    foot_points,
)
from shipvision.mtmc.topology import GroundPlane, Homography, calculate_homography, project
from shipvision.mtmc.tracker import ClusterMTMCTracker

__all__ = [
    "CLUSTERERS",
    "MATRIX_BUILDERS",
    "MTMC",
    "NEVER_MERGE",
    "AgglomerativeClusterer",
    "AppearanceMatrixBuilder",
    "BaseClusterer",
    "BaseMTMCTracker",
    "BaseMatrixBuilder",
    "CameraTracks",
    "ClusterMTMCTracker",
    "FrameTrackCluster",
    "GatedMatrixBuilder",
    "GlobalIdAssigner",
    "GroundPlane",
    "Homography",
    "ObservationGate",
    "SpatialMatrixBuilder",
    "TrackKey",
    "TrackObservation",
    "calculate_homography",
    "foot_points",
    "project",
]
