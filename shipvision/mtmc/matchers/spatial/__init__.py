"""Spatial matching: two views of one object project to the same place on the ground.

Importing this package registers the matcher, which is what makes
``MTMC_MATCHERS.build("spatial")`` work from a config string.
"""

from __future__ import annotations

from shipvision.mtmc.matchers.spatial.matcher import SpatialMatcher
from shipvision.mtmc.matchers.spatial.utils import foot_points

__all__ = ["SpatialMatcher", "foot_points"]
