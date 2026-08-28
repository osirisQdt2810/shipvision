"""Appearance, vetoed by geometry. The production matcher.

Ten lines of logic, and they are the ten lines that make cross-camera tracking work on a real
site: take the appearance similarity, and zero it wherever the two tracks project to ground
positions further apart than they could possibly be for one object. Appearance decides
*which* of several candidates; geometry decides *whether* any of them is possible.

The composition matters more than the arithmetic. Both halves already exist as matchers with
their own tests, so this class owns no distance function, no mask and no threshold logic — it
owns a decision about how two independent pieces of evidence combine. The reference
implemented the same idea by multiple-inheriting from both builders and calling protected
methods across the hierarchy; composing instances instead means the gate can be tested with a
hand-built appearance matrix, and either half can be replaced without touching this file.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from shipvision.mtmc.base import BaseMatcher
from shipvision.mtmc.frames import TrackObservation
from shipvision.mtmc.matchers.appearance import AppearanceMatcher
from shipvision.mtmc.matchers.gated.utils import veto
from shipvision.mtmc.matchers.spatial import SpatialMatcher
from shipvision.mtmc.registry import MTMC_MATCHERS
from shipvision.mtmc.topology import GroundPlane
from shipvision.registry import PYTHON

__all__ = ["GatedMatcher"]


@MTMC_MATCHERS.register("gated", backend=PYTHON, aliases=("spatial_gating",))
class GatedMatcher(BaseMatcher):
    """Cosine appearance similarity, zeroed where the ground-plane separation is too large."""

    def __init__(
        self,
        *,
        ground_plane: GroundPlane | None = None,
        appearance_threshold: float = 0.86,
        spatial_threshold: float = 280.0,
        foot_ratio: float = 1.0,
        aspect_ratio: float = 0.25,
        appearance: AppearanceMatcher | None = None,
        spatial: SpatialMatcher | None = None,
    ) -> None:
        """
        Args:
            ground_plane: the homographies. An empty one degrades this to a pure appearance
                matcher, which is the correct behaviour on an uncalibrated site and is why
                ``gated`` is a safe default.
            appearance_threshold: see :class:`AppearanceMatcher`.
            spatial_threshold: see :class:`SpatialMatcher`.
            foot_ratio: see :func:`~shipvision.mtmc.matchers.spatial.utils.foot_points`.
            aspect_ratio: see :func:`~shipvision.mtmc.matchers.spatial.utils.foot_points`.
            appearance: a pre-built appearance matcher, overriding the threshold arguments.
                For A/B-ing one half of the gate without rebuilding the other.
            spatial: a pre-built spatial matcher, likewise.
        """
        self.appearance = appearance or AppearanceMatcher(
            appearance_threshold=appearance_threshold
        )
        self.spatial = spatial or SpatialMatcher(
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
        return veto(similarity, self.spatial.gate(observations))

    def build(self, observations: Sequence[TrackObservation]) -> np.ndarray:
        return self.to_distance(
            self.similarities(observations), self.mergeable_mask(observations)
        )

    def __repr__(self) -> str:
        return (
            f"<GatedMatcher appearance_threshold="
            f"{self.appearance.appearance_threshold} spatial_threshold="
            f"{self.spatial.spatial_threshold} cameras={len(self.ground_plane)} "
            f"backend={self.backend}>"
        )
