"""Gated matching: appearance decides which candidate, geometry decides whether any.

Importing this package registers the matcher, which is what makes
``MTMC_MATCHERS.build("gated")`` work from a config string.
"""

from __future__ import annotations

from shipvision.mtmc.core.gated.matcher import GatedMatcher
from shipvision.mtmc.core.gated.utils import veto

__all__ = ["GatedMatcher", "veto"]
