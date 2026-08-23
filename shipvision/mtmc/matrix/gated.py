"""Appearance, vetoed by geometry. The production builder.

Ten lines of logic, and they are the ten lines that make cross-camera tracking work on a real
site: take the appearance similarity, and zero it wherever the two tracks project to ground
positions further apart than they could possibly be for one object. Appearance decides
*which* of several candidates; geometry decides *whether* any of them is possible.

The composition matters more than the arithmetic. Both halves already exist as builders with
their own tests, so this class owns no distance function, no mask and no threshold logic — it
owns a decision about how two independent pieces of evidence combine. The reference
implemented the same idea by multiple-inheriting from both builders and calling protected
methods across the hierarchy; composing instances instead means the gate can be tested with a
hand-built appearance matrix, and either half can be replaced without touching this file.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from shipvision.mtmc.frames import TrackObservation
from shipvision.mtmc.matrix.appearance import AppearanceMatrixBuilder
from shipvision.mtmc.matrix.base import MATRIX_BUILDERS, BaseMatrixBuilder
from shipvision.mtmc.matrix.spatial import SpatialMatrixBuilder
from shipvision.mtmc.topology import GroundPlane
from shipvision.registry import PYTHON

__all__ = ["GatedMatrixBuilder"]


@MATRIX_BUILDERS.register("gated", backend=PYTHON, aliases=("spatial_gating",))
class GatedMatrixBuilder(BaseMatrixBuilder):
    """Cosine appearance similarity, zeroed where the ground-plane separation is too large."""

    def __init__(
        self,
        *,
        ground_plane: GroundPlane | None = None,
        appearance_threshold: float = 0.86,
        spatial_threshold: float = 280.0,
        foot_ratio: float = 1.0,
        aspect_ratio: float = 0.25,
        appearance: AppearanceMatrixBuilder | None = None,
        spatial: SpatialMatrixBuilder | None = None,
    ) -> None:
        """
        Args:
            ground_plane: the homographies. An empty one degrades this to a pure appearance
                builder, which is the correct behaviour on an uncalibrated site and is why
                ``gated`` is a safe default.
            appearance_threshold: see :class:`AppearanceMatrixBuilder`.
            spatial_threshold: see :class:`SpatialMatrixBuilder`.
            foot_ratio: see :func:`~shipvision.mtmc.matrix.spatial.foot_points`.
            aspect_ratio: see :func:`~shipvision.mtmc.matrix.spatial.foot_points`.
            appearance: a pre-built appearance builder, overriding the threshold arguments.
                For A/B-ing one half of the gate without rebuilding the other.
            spatial: a pre-built spatial builder, likewise.
        """
        self.appearance = appearance or AppearanceMatrixBuilder(
            appearance_threshold=appearance_threshold
        )
        self.spatial = spatial or SpatialMatrixBuilder(
            ground_plane=ground_plane,
            spatial_threshold=spatial_threshold,
            foot_ratio=foot_ratio,
            aspect_ratio=aspect_ratio,
        )

    @property
    def ground_plane(self) -> GroundPlane:
        return self.spatial.ground_plane

    def similarities(self, observations: Sequence[TrackObservation]) -> np.ndarray:
        """``(n, n)`` appearance similarity with geometrically impossible pairs zeroed."""
        similarity = self.appearance.similarities(observations)
        if similarity.size == 0:
            return similarity
        return np.where(self.spatial.gate(observations), similarity, 0.0).astype(np.float32)

    def build(self, observations: Sequence[TrackObservation]) -> np.ndarray:
        return self.to_distance(
            self.similarities(observations), self.mergeable_mask(observations)
        )

    def __repr__(self) -> str:
        return (
            f"<GatedMatrixBuilder appearance_threshold="
            f"{self.appearance.appearance_threshold} spatial_threshold="
            f"{self.spatial.spatial_threshold} cameras={len(self.ground_plane)} "
            f"backend={self.backend}>"
        )
