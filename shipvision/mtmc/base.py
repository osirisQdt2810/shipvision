"""The two cross-camera contracts: the tracker, and the matcher it runs.

``base.py`` holds the abstract classes and ``registry.py`` holds the families that select
between their implementations — the split every package here uses, and the reason a matcher
can import both its base class and its decorator without the two importing each other. The
clusterer's ABC is the exception, and it earns it: :mod:`shipvision.mtmc.clustering` is a
package of implementations already, so its contract lives with them.

The tracker has one method, called once per synchronised group of frames: tracks in, global
identities out. Everything a caller needs to know is in that signature — no start-up
handshake, no per-camera registration, no "call this before that". A camera that appears for
the first time in the middle of a call is handled by the call.

**No state updater seam.** The reference implementation has a fourth component,
``StateUpdater``, whose every implementation is an empty ``run()``: a base class, a registry
entry, a config section and a constructor argument that between them do nothing. It is not
ported. An extension point with no implementations is not a design, it is an invitation to
hang per-frame work off the one component nobody has thought about, and the state it was
presumably meant to own already has a home — ageing, eviction and the invariant check belong
to :class:`~shipvision.mtmc.identity.GlobalIdAssigner`, which is where the state actually
lives. If something later needs a genuine post-assignment pass, adding a seam for it will be
a smaller change than removing a wrong one.

**Same-camera pairs can never merge, and exactly one place says so.** Two tracks in one
camera view are, by definition of single-camera tracking, two different objects: if they were
the same object the tracker upstream had one job and failed at it. Merge them anyway and MTMC
quietly becomes a within-camera deduplicator — every count drops, every metric improves, and
the system is worse. :meth:`BaseMatcher.mergeable_mask` lives in the shared base rather than
in each matcher because a matcher that forgets it produces plausible output.

**"Never merge" is a large finite number, not infinity.** :data:`NEVER_MERGE` is ``1e5``,
inherited from the reference implementation, and the reason is mechanical rather than
stylistic: hierarchical clustering on a precomputed matrix cannot take non-finite input.
``scipy.spatial.distance.squareform`` rejects it outright ("must contain only finite
values"), and any average-linkage update that got past that would compute ``inf - inf`` and
produce NaN, which silently poisons the rest of the dendrogram instead of failing. A finite
sentinel that is simply enormous next to a threshold of ~0.15 gives the arithmetic something
to work with while keeping the semantics.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence

import numpy as np

from shipvision.mtmc.frames import FrameTrackCluster, TrackObservation
from shipvision.mtmc.registry import MTMC
from shipvision.registry import PYTHON
from shipvision.types import GlobalTrack

__all__ = ["MTMC", "NEVER_MERGE", "BaseMTMCTracker", "BaseMatcher"]

NEVER_MERGE = 1e5
"""The distance between two tracks that must not be grouped. Finite on purpose."""


class BaseMTMCTracker(abc.ABC):
    """Assigns stable cross-camera identities to the tracks of one synchronised instant."""

    name: str = "mtmc"
    backend: str = PYTHON

    @abc.abstractmethod
    def track(self, cluster: FrameTrackCluster) -> list[GlobalTrack]:
        """One :class:`GlobalTrack` per input track, in input order.

        Every input track comes back, including the ones no identity was assigned to — those
        carry ``global_id=None``. Returning fewer results than were passed in would make the
        caller diff two lists to discover which of its tracks were judged too new or too
        small to identify, and a caller that skips that diff silently loses tracks. `None`
        says "not identified" in a way that fails at first use, which ``-1`` does not.
        """

    @abc.abstractmethod
    def reset(self) -> None:
        """Forget every identity. Must not make a fresh global id collide with a used one."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} backend={self.backend}>"


class BaseMatcher(abc.ABC):
    """Turns a synchronised group of tracks into an ``(n, n)`` distance matrix.

    This is where every piece of evidence about "are these two tracks the same object" is
    turned into a single number. Everything downstream — the clusterer, the id assigner —
    reads only the matrix, which is what lets an appearance-only, a geometry-only and a gated
    matcher be swapped from config without either of them knowing.
    """

    name: str = "matcher"
    backend: str = PYTHON

    @abc.abstractmethod
    def build(self, observations: Sequence[TrackObservation]) -> np.ndarray:
        """``(n, n)`` float32 distances, smaller meaning more likely the same object.

        Guaranteed properties, relied on by every clusterer: symmetric, zero on the diagonal,
        finite everywhere, and exactly :data:`NEVER_MERGE` for any pair that must not be
        grouped. An empty group returns ``(0, 0)`` rather than ``(0,)`` — an instant with no
        tracks is ordinary input, and the wrong shape turns it into an IndexError three
        frames later.
        """

    # -- shared machinery -----------------------------------------------------------------

    @staticmethod
    def mergeable_mask(observations: Sequence[TrackObservation]) -> np.ndarray:
        """``(n, n)`` bool: true where a pair is *allowed* to be the same object.

        False on the diagonal and false for every same-camera pair. Vectorised through
        integer camera codes rather than the nested string comparison the reference used —
        at 50 cameras and 15 tracks each that loop is 560 000 string compares per
        synchronised group, which on its own is more expensive than the clustering.
        """
        count = len(observations)
        if count == 0:
            return np.zeros((0, 0), dtype=bool)
        codes: dict[str, int] = {}
        camera = np.empty(count, dtype=np.int32)
        for index, observation in enumerate(observations):
            camera[index] = codes.setdefault(observation.camera_id, len(codes))
        return camera[:, None] != camera[None, :]

    @staticmethod
    def to_distance(similarity: np.ndarray, mergeable: np.ndarray) -> np.ndarray:
        """Similarities (higher is closer, 0 meaning "no evidence") to clusterable distances.

        Zero similarity becomes :data:`NEVER_MERGE` rather than a distance of 1. That
        distinction is the whole point of thresholding earlier: "these two scored 0.2, which
        is below the bar" and "these two are in the same camera" both mean *do not group*,
        and expressing both as 1.0 would let average linkage merge them anyway once a
        threshold moved.
        """
        distance = np.where(similarity > 0.0, 1.0 - similarity, NEVER_MERGE)
        distance = np.where(mergeable, distance, NEVER_MERGE)
        # Symmetrise explicitly. Both inputs are symmetric by construction, but BLAS does not
        # promise bitwise symmetry for A @ A.T, and squareform's tolerance for asymmetry is
        # zero — it silently reads the upper triangle.
        distance = 0.5 * (distance + distance.T)
        np.fill_diagonal(distance, 0.0)
        return distance.astype(np.float32)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} backend={self.backend}>"


# -- compatibility shim -------------------------------------------------------------------
#
# `MTMC` was declared here before the three registries moved to `mtmc/registry.py`, and
# `from shipvision.mtmc.base import MTMC` is a documented path. Re-exported rather than
# re-declared: a second Registry instance would mean a tracker registered through one of them
# being invisible through the other. It stays in `__all__`, which is what keeps it a
# re-export rather than an unused import.
