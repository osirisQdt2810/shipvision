"""The cross-camera matchers: tracks in, one pairwise distance matrix out.

One package per strategy, because a strategy is more than a class — appearance owns a rule
about what input it will accept, spatial owns a piece of image geometry, and gated owns a rule
about how two answers combine. Each of those has a ``utils.py`` next to its ``matcher.py`` so
it can be tested without building a matcher at all, and adding a fourth strategy is a fourth
directory plus a ``@MTMC_MATCHERS.register`` decorator, not an edit to anything here beyond
one import line.

That import line is the registration: importing this package imports the three, and importing
those runs the decorators, which is what makes ``MTMC_MATCHERS.build("gated", ...)`` work from
a config string.
"""

from __future__ import annotations

from shipvision.mtmc.matchers.appearance import AppearanceMatcher, stack_embeddings
from shipvision.mtmc.matchers.gated import GatedMatcher, veto
from shipvision.mtmc.matchers.spatial import SpatialMatcher, foot_points

__all__ = [
    "AppearanceMatcher",
    "GatedMatcher",
    "SpatialMatcher",
    "foot_points",
    "stack_embeddings",
    "veto",
]
