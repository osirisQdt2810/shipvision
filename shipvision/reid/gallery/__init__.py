"""The memory a query is compared against."""

from __future__ import annotations

from shipvision.reid.gallery.base import GALLERIES, BaseGallery
from shipvision.reid.gallery.centroid import CentroidGallery
from shipvision.reid.gallery.flat import FlatGallery

__all__ = [
    "GALLERIES",
    "BaseGallery",
    "CentroidGallery",
    "FlatGallery",
]
