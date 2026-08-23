"""Average-linkage agglomerative clustering with a distance cut.

Four words carry the whole algorithm, and each of them is a decision:

**Agglomerative**, because the number of identities in front of a camera group at one instant
is exactly what is being asked and cannot be supplied.

**Average linkage**, because single linkage chains — A near B, B near C, so A, B and C are one
person even though A and C are strangers — and complete linkage refuses to admit a third view
of an identity whose worst pairing is mediocre, which is what a partially-occluded crop always
is. Average is also what makes the ``NEVER_MERGE`` sentinel work: one forbidden pair drags a
candidate merge's mean distance to ~50 000, so the group is never formed.

**A distance cut**, not a cluster count: ``fcluster(..., criterion="distance")``.

**On a precomputed matrix**, because the evidence combined in that matrix — cosine appearance,
a ground-plane veto, a same-camera exclusion — is not a metric embedding of anything and there
are no coordinates to hand a clusterer instead.

This is a ~15-line call into ``scipy``. The reference vendors 2 400 lines of hand-written C++
reciprocal-agglomerative clustering to do it, which is the exact shape of code the ponytail
principle exists to prevent: a well-optimised library primitive, reimplemented, now needing
its own tests and its own maintainer.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from shipvision.errors import BackendUnavailableError, ConfigurationError
from shipvision.mtmc.clustering.base import CLUSTERERS, BaseClusterer
from shipvision.registry import PYTHON

__all__ = ["AgglomerativeClusterer"]


def _load_scipy() -> tuple[Any, Any, Any]:
    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
    except ImportError as exc:  # pragma: no cover - depends on the host
        raise BackendUnavailableError(
            "agglomerative clustering needs scipy; install the 'solvers' extra "
            "(pip install 'shipvision[solvers]')"
        ) from exc
    return linkage, fcluster, squareform


@CLUSTERERS.register("agglomerative", backend=PYTHON, aliases=("average_linkage", "aic"))
class AgglomerativeClusterer(BaseClusterer):
    """Hierarchical clustering cut at a distance threshold."""

    def __init__(self, *, distance_threshold: float = 0.14) -> None:
        """
        Args:
            distance_threshold: the cut. In the same units as the matrix, so with the
                appearance builder it is ``1 - cosine_similarity`` — 0.14 meaning "group
                things that are at least 0.86 alike", which is the reference's production
                value and deliberately the same number as its appearance threshold. The two
                doing the same work is not redundant: the appearance threshold removes weak
                *pairs* before linkage, and this one bounds the *average* over a group, so
                without the first a group of mediocre pairs still averages under the cut.
        """
        if distance_threshold <= 0.0:
            raise ConfigurationError(
                f"distance_threshold must be positive, got {distance_threshold}"
            )
        self.distance_threshold = float(distance_threshold)

    def fit_predict(self, distances: np.ndarray) -> np.ndarray:
        matrix = self.check_matrix(distances)
        count = matrix.shape[0]
        if count == 0:
            return np.zeros(0, dtype=np.int32)
        if count == 1:
            return np.zeros(1, dtype=np.int32)

        linkage, fcluster, squareform = _load_scipy()
        # Symmetrise and zero the diagonal before condensing. squareform's own checks are
        # exact — a 1e-16 asymmetry out of BLAS fails them — and with checks off it silently
        # keeps the upper triangle, so doing it explicitly is what makes the result the
        # matrix the builder meant rather than half of it.
        symmetric = 0.5 * (matrix + matrix.T)
        np.fill_diagonal(symmetric, 0.0)
        condensed = squareform(symmetric, checks=False)
        tree = linkage(condensed, method="average")
        labels = fcluster(tree, t=self.distance_threshold, criterion="distance")
        # scipy labels from 1; zero-based keeps "label 0" from meaning two things depending
        # on which library produced it.
        return (np.asarray(labels, dtype=np.int32) - 1).astype(np.int32)

    def __repr__(self) -> str:
        return (
            f"<AgglomerativeClusterer distance_threshold={self.distance_threshold} "
            f"backend={self.backend}>"
        )
