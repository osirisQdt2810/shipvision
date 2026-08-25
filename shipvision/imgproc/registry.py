"""The image-ops registry.

Its own module rather than a name in ``__init__``, so an implementation can import the
registry to decorate itself without importing the package that imports it. That circular
import is otherwise unavoidable the moment a second backend appears.
"""

from __future__ import annotations

from shipvision.imgproc.base import ImageOps
from shipvision.registry import Registry

__all__ = ["IMGPROC"]

IMGPROC = Registry[ImageOps]("image ops")
"""Letterbox, crop and NMS, keyed on ``(name, backend)``.

There is exactly one algorithm name — ``"default"`` — and three backends behind it. That is
the shape the whole library uses: the numpy one is the oracle the compiled one is checked
against, and it is also what runs where there is no build.
"""
