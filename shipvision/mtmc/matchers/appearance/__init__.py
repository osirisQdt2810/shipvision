"""Appearance matching: two crops of the same object look alike, whichever camera saw them.

Importing this package registers the matcher, which is what makes
``MTMC_MATCHERS.build("appearance")`` work from a config string.
"""

from __future__ import annotations

from shipvision.mtmc.matchers.appearance.matcher import AppearanceMatcher
from shipvision.mtmc.matchers.appearance.utils import stack_embeddings

__all__ = ["AppearanceMatcher", "stack_embeddings"]
