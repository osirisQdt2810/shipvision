"""Compatibility shim: ``shipvision.mtmc.matrix.gated`` before the repackaging.

The definitions live in :mod:`shipvision.mtmc.matchers` and :mod:`shipvision.mtmc.base` now. This
module re-exports them under the module path they had, because a package-level re-export is
not the same promise as a module path — ``from shipvision.mtmc.matrix import
GatedMatrixBuilder`` kept working after the move while ``import
shipvision.mtmc.matrix.gated`` raised ``ModuleNotFoundError``, and both were public.

Aliases, never subclasses, so ``isinstance`` and ``is`` cannot start disagreeing depending on
which spelling a caller reached for.
"""

from shipvision.mtmc.matrix import GatedMatrixBuilder

__all__ = ["GatedMatrixBuilder"]
