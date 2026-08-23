"""The matrix-builder contract: tracks in, one pairwise distance matrix out.

This is where every piece of evidence about "are these two tracks the same object" is turned
into a single number, and where the one rule that makes cross-camera tracking *cross-camera*
is enforced. Everything downstream — the clusterer, the id assigner — reads only the matrix,
which is what lets an appearance-only, a geometry-only and a gated builder be swapped from
config without either of them knowing.

**Same-camera pairs can never merge, and exactly one place says so.** Two tracks in one
camera view are, by definition of single-camera tracking, two different objects: if they were
the same object the tracker upstream had one job and failed at it. Merge them anyway and MTMC
quietly becomes a within-camera deduplicator — every count drops, every metric improves, and
the system is worse. The mask lives here, in the shared base, rather than in each builder,
because a builder that forgets it produces plausible output.

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

from shipvision.mtmc.frames import TrackObservation
from shipvision.registry import PYTHON, Registry

__all__ = ["MATRIX_BUILDERS", "NEVER_MERGE", "BaseMatrixBuilder"]

NEVER_MERGE = 1e5
"""The distance between two tracks that must not be grouped. Finite on purpose."""


class BaseMatrixBuilder(abc.ABC):
    """Turns a synchronised group of tracks into an ``(n, n)`` distance matrix."""

    name: str = "matrix builder"
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


#: The matrix-builder family. Appearance alone, geometry alone, and the gated combination are
#: the same question answered with different evidence, and which one a site wants depends on
#: whether its cameras are calibrated — so it is chosen by name from config, not by an import.
MATRIX_BUILDERS: Registry[BaseMatrixBuilder] = Registry("mtmc matrix builder")
