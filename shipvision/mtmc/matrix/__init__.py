"""Turning a synchronised group of tracks into one pairwise distance matrix.

Importing the package registers every builder, which is what makes
``MATRIX_BUILDERS.build("gated", ...)`` work from a config string.
"""

from __future__ import annotations

from shipvision.mtmc.matrix.appearance import AppearanceMatrixBuilder, stack_embeddings
from shipvision.mtmc.matrix.base import MATRIX_BUILDERS, NEVER_MERGE, BaseMatrixBuilder
from shipvision.mtmc.matrix.gated import GatedMatrixBuilder
from shipvision.mtmc.matrix.spatial import SpatialMatrixBuilder, foot_points

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
