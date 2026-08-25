"""Grouping the tracks of one instant from a precomputed distance matrix.

Importing this package registers every clusterer, which is what makes
``MTMC_CLUSTERERS.build("agglomerative", ...)`` work from a config string.
"""

from __future__ import annotations

from shipvision.mtmc.clustering.agglomerative import AgglomerativeClusterer
from shipvision.mtmc.clustering.base import BaseClusterer
from shipvision.mtmc.registry import MTMC_CLUSTERERS

__all__ = ["MTMC_CLUSTERERS", "AgglomerativeClusterer", "BaseClusterer"]

# -- compatibility shim -------------------------------------------------------------------
#
# `CLUSTERERS` is what this registry was called before it moved to `mtmc/registry.py`, and
# `from shipvision.mtmc.clustering import CLUSTERERS` is a documented path. The same object
# under a second name, never a second registry.
CLUSTERERS = MTMC_CLUSTERERS
