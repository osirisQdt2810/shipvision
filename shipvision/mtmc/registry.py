"""The three cross-camera families, declared in one leaf module.

A registry object is needed in two places at once: by the base class it is typed on, and by
every implementation that decorates itself with it. Declaring it next to the ABC makes that a
cycle as soon as an implementation package wants both — ``core.gated.matcher`` imports
:class:`~shipvision.mtmc.base.BaseMatcher` for the subclass *and* the registry for the
decorator, and ``base`` would then have to be imported before itself. This module imports
nothing from :mod:`shipvision.mtmc`, so it is always safe to import from anywhere in it, and
the ABCs and the implementations both depend on it rather than on each other.

Three families, because there are three independent choices a site makes:

* :data:`MTMC` — the whole cross-camera algorithm.
* :data:`MTMC_MATCHERS` — *how alike are these two tracks*: appearance alone, geometry alone,
  or appearance vetoed by geometry. This is the one that changes when a site is calibrated.
* :data:`MTMC_CLUSTERERS` — *which of them are the same object right now*, from the matcher's
  matrix alone.

Adding a cross-camera association strategy is therefore a new package under
:mod:`shipvision.mtmc.matchers` and a ``@MTMC_MATCHERS.register`` decorator — never an edit to a
branch in the tracker.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from shipvision.registry import Registry

if TYPE_CHECKING:  # imported for typing only; see the module docstring on the cycle
    from shipvision.mtmc.base import BaseMatcher, BaseMTMCTracker
    from shipvision.mtmc.clustering.base import BaseClusterer

__all__ = ["MTMC", "MTMC_CLUSTERERS", "MTMC_MATCHERS"]

#: The cross-camera tracker family. Registered by name so a site can be moved from
#: appearance-only to spatially-gated association, or onto something not written yet, by
#: editing config rather than code.
MTMC: Registry[BaseMTMCTracker] = Registry("mtmc tracker")

#: The matcher family. Appearance alone, geometry alone, and the gated combination are the
#: same question answered with different evidence, and which one a site wants depends on
#: whether its cameras are calibrated — so it is chosen by name from config, not by an import.
MTMC_MATCHERS: Registry[BaseMatcher] = Registry("mtmc matcher")

#: The clusterer family. One implementation today; the seam exists because the choice of
#: linkage and cut is the single most consequential tuning decision in cross-camera tracking,
#: and comparing two of them on one recorded stream must not require a code change.
MTMC_CLUSTERERS: Registry[BaseClusterer] = Registry("mtmc clusterer")
