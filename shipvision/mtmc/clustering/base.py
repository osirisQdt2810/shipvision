"""The clusterer contract: a distance matrix in, one label per track out.

Deliberately the narrowest interface in the package. A clusterer sees no cameras, no
embeddings, no history and no time — only an ``(n, n)`` matrix — which means every scenario
that has ever gone wrong in cross-camera tracking can be reproduced here as a small matrix of
literals, and a candidate algorithm can be swapped in without knowing what the numbers mean.

**The number of clusters is never an input.** It is not known: at one instant a group of
cameras might be watching one person or eleven, and that is the answer being asked for.
Anything with a ``k`` in its signature has to be handed a guess, and the reference threaded
an unused ``n_clusters`` argument through four layers on the way to a function that ignored
it — a parameter that looks like a control and is not.
"""

from __future__ import annotations

import abc

import numpy as np

from shipvision.errors import ConfigurationError, TrackingError
from shipvision.registry import PYTHON, Registry

__all__ = ["CLUSTERERS", "BaseClusterer"]


class BaseClusterer(abc.ABC):
    """Groups tracks from a precomputed pairwise distance matrix."""

    name: str = "clusterer"
    backend: str = PYTHON

    @abc.abstractmethod
    def fit_predict(self, distances: np.ndarray) -> np.ndarray:
        """``(n, n)`` distances to ``(n,)`` int32 labels.

        Labels are arbitrary integers: only equality between them carries meaning, never
        their order or their value. ``n == 0`` returns ``(0,)`` and ``n == 1`` returns
        ``[0]`` — an instant with one visible track is the common case on a quiet site, not
        an edge case.
        """

    # -- shared machinery -----------------------------------------------------------------

    @staticmethod
    def check_matrix(distances: np.ndarray) -> np.ndarray:
        """Validate and coerce to a square float64 matrix, or raise a typed error.

        The non-finite check is the one worth having. ``inf`` is the obvious way to say "never
        merge these" and it is wrong: ``scipy``'s condensed form rejects it outright, and any
        linkage that averages distances would compute ``inf - inf`` and get NaN, which does
        not fail — it produces a dendrogram whose merges are arbitrary. That is why
        :data:`shipvision.mtmc.matrix.NEVER_MERGE` is a large finite number, and this is where
        a builder that ignored that finds out.
        """
        matrix = np.asarray(distances, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ConfigurationError(
                f"a pairwise distance matrix must be square, got shape {matrix.shape}"
            )
        if matrix.size and not np.all(np.isfinite(matrix)):
            raise TrackingError(
                "the distance matrix contains inf or NaN. Use the finite NEVER_MERGE "
                "sentinel for pairs that must not be grouped: hierarchical clustering "
                "cannot consume non-finite distances, and average linkage turns them into "
                "NaN rather than into a refusal"
            )
        return matrix

    def __repr__(self) -> str:
        return f"<{type(self).__name__} backend={self.backend}>"


#: The clusterer family. One implementation today; the seam exists because the choice of
#: linkage and cut is the single most consequential tuning decision in cross-camera tracking,
#: and comparing two of them on one recorded stream must not require a code change.
CLUSTERERS: Registry[BaseClusterer] = Registry("mtmc clusterer")
