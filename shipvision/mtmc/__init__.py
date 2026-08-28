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

* :data:`~shipvision.mtmc.registry.MTMC_MATCHERS` — tracks to an ``(n, n)`` distance matrix.
  ``appearance`` (cosine, thresholded), ``spatial`` (ground-plane separation), ``gated``
  (appearance vetoed by geometry — the production one). One package each under
  :mod:`shipvision.mtmc.matchers`, so a fourth strategy is a fourth directory and a decorator.
  All three also exist compiled, in :mod:`shipvision.mtmc.backends.native`, registered under
  the same names — ``MTMC_MATCHERS.build("gated")`` takes the fastest one this machine can
  build and ``backend="python"`` pins the reference the compiled one is checked against.
* :data:`~shipvision.mtmc.registry.MTMC_CLUSTERERS` — the matrix to a label per track.
  ``agglomerative``: average linkage, cut at a distance, on a precomputed matrix.
* :class:`~shipvision.mtmc.identity.GlobalIdAssigner` — labels to *stable* ids, carrying state
  between calls. The stateful part, and where the reference implementation's two real bugs
  were: unbounded growth and a process-global id counter.
* :class:`~shipvision.mtmc.topology.GroundPlane` — the homographies the spatial half needs,
  and :func:`~shipvision.mtmc.topology.calculate_homography` to fit one *and say how wrong it
  is*.

All three registries are declared in :mod:`shipvision.mtmc.registry`, which imports nothing
from this package — that is what lets a matcher import both its base class and the decorator
it registers with. Importing this package imports the implementations, which is what runs
those decorators.

Three properties this package is built around, all of them lessons from the system it
replaces:

**Two tracks in one camera never merge.** Enforced in one place,
:meth:`~shipvision.mtmc.base.BaseMatcher.mergeable_mask`. Without it MTMC silently becomes a
within-camera deduplicator: every count falls, every metric improves, and the system is worse.

**Nothing grows without bound.** The identity maps have a maximum age *and* a capacity, and
eviction runs on every instant, including empty ones.

**Unassigned means `None`.** A track the gate held back comes back as a
:class:`~shipvision.types.GlobalTrack` with ``global_id=None``, not ``-1`` and not omitted.

The names that end in ``MatrixBuilder``, plus ``MATRIX_BUILDERS`` and ``CLUSTERERS``, are
compatibility shims for the spelling this package used before the matchers moved into
:mod:`shipvision.mtmc.matchers`; see :mod:`shipvision.mtmc.matrix`.
"""

from __future__ import annotations

from shipvision.mtmc.backends.native import (
    NativeAppearanceMatcher,
    NativeGatedMatcher,
    NativeSpatialMatcher,
    native_available,
)
from shipvision.mtmc.base import NEVER_MERGE, BaseMatcher, BaseMTMCTracker
from shipvision.mtmc.clustering import AgglomerativeClusterer, BaseClusterer
from shipvision.mtmc.frames import CameraTracks, FrameTrackCluster, TrackKey, TrackObservation
from shipvision.mtmc.gating import ObservationGate
from shipvision.mtmc.identity import GlobalIdAssigner
from shipvision.mtmc.matchers import (
    AppearanceMatcher,
    GatedMatcher,
    SpatialMatcher,
    foot_points,
)
from shipvision.mtmc.matrix import (
    AppearanceMatrixBuilder,
    BaseMatrixBuilder,
    GatedMatrixBuilder,
    SpatialMatrixBuilder,
)
from shipvision.mtmc.registry import MTMC, MTMC_CLUSTERERS, MTMC_MATCHERS
from shipvision.mtmc.topology import GroundPlane, Homography, calculate_homography, project
from shipvision.mtmc.tracker import ClusterMTMCTracker

#: The registries under the names they had before the repackaging. The same objects — see
#: :mod:`shipvision.mtmc.matrix` and :mod:`shipvision.mtmc.clustering`.
CLUSTERERS = MTMC_CLUSTERERS
MATRIX_BUILDERS = MTMC_MATCHERS

__all__ = [
    "CLUSTERERS",
    "MATRIX_BUILDERS",
    "MTMC",
    "MTMC_CLUSTERERS",
    "MTMC_MATCHERS",
    "NEVER_MERGE",
    "AgglomerativeClusterer",
    "AppearanceMatcher",
    "AppearanceMatrixBuilder",
    "BaseClusterer",
    "BaseMTMCTracker",
    "BaseMatcher",
    "BaseMatrixBuilder",
    "CameraTracks",
    "ClusterMTMCTracker",
    "FrameTrackCluster",
    "GatedMatcher",
    "GatedMatrixBuilder",
    "GlobalIdAssigner",
    "GroundPlane",
    "Homography",
    "NativeAppearanceMatcher",
    "NativeGatedMatcher",
    "NativeSpatialMatcher",
    "ObservationGate",
    "SpatialMatcher",
    "SpatialMatrixBuilder",
    "TrackKey",
    "TrackObservation",
    "calculate_homography",
    "foot_points",
    "native_available",
    "project",
]
