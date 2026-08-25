"""Compatibility shim: the names ``shipvision.mtmc.matrix`` exported before the repackaging.

Nothing new lives here and nothing here is the definition of anything — every name below is
the object that :mod:`shipvision.mtmc.core` and :mod:`shipvision.mtmc.base` define, under the
spelling it had when the matchers were four modules in a ``matrix/`` package. Aliases rather
than subclasses, so ``isinstance`` and ``is`` comparisons cannot start disagreeing depending
on which name a caller reached for.

The move it covers, name by name:

==================================  ============================================
was                                 is
==================================  ============================================
``matrix.MATRIX_BUILDERS``          :data:`shipvision.mtmc.registry.MTMC_MATCHERS`
``matrix.BaseMatrixBuilder``        :class:`shipvision.mtmc.base.BaseMatcher`
``matrix.NEVER_MERGE``              :data:`shipvision.mtmc.base.NEVER_MERGE`
``matrix.AppearanceMatrixBuilder``  ``core.appearance.matcher.AppearanceMatcher``
``matrix.SpatialMatrixBuilder``     ``core.spatial.matcher.SpatialMatcher``
``matrix.GatedMatrixBuilder``       ``core.gated.matcher.GatedMatcher``
``matrix.stack_embeddings``         ``core.appearance.utils.stack_embeddings``
``matrix.foot_points``              ``core.spatial.utils.foot_points``
==================================  ============================================

What this shim does **not** restore are the leaf module paths — ``mtmc.matrix.base``,
``mtmc.matrix.gated`` and their siblings. Those were the implementation files of a package,
not its interface; nothing in this repository or in ShipInfer imported one, and keeping five
stub modules alive to preserve paths nobody used would be a larger lie than the move itself.
Import from :mod:`shipvision.mtmc` or :mod:`shipvision.mtmc.core` instead.
"""

from __future__ import annotations

from shipvision.mtmc.base import NEVER_MERGE, BaseMatcher
from shipvision.mtmc.core import (
    AppearanceMatcher,
    GatedMatcher,
    SpatialMatcher,
    foot_points,
    stack_embeddings,
)
from shipvision.mtmc.registry import MTMC_MATCHERS

__all__ = [
    "MATRIX_BUILDERS",
    "NEVER_MERGE",
    "AppearanceMatrixBuilder",
    "BaseMatrixBuilder",
    "GatedMatrixBuilder",
    "SpatialMatrixBuilder",
    "foot_points",
    "stack_embeddings",
]

MATRIX_BUILDERS = MTMC_MATCHERS
BaseMatrixBuilder = BaseMatcher
AppearanceMatrixBuilder = AppearanceMatcher
SpatialMatrixBuilder = SpatialMatcher
GatedMatrixBuilder = GatedMatcher
