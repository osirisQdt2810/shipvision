"""Compatibility shim. The trackers moved to :mod:`shipvision.tracking.core`.

Each algorithm is now a package of its own — ``tracker.py``, ``tracklet.py``, ``utils.py`` —
under ``core/``, in the shape the rest of the library uses. This module re-exports the five
classes so that ``from shipvision.tracking.trackers import SortTracker`` keeps resolving, and
it exists for no other reason.

It is a shim rather than the real home, so: **do not add a tracker here.** A new algorithm is
a new package under :mod:`shipvision.tracking.core` plus a ``@TRACKERS.register`` decorator,
and adding one here instead would make this file a second, silently divergent list of what
exists — which is the failure the registry was introduced to remove.

The import below is what registers all five, because ``core`` imports each package and each
package imports its ``tracker`` module. Keeping that side effect on this path matters: a
caller that has always imported this module to make ``TRACKERS.build("sort")`` work must not
find the registry empty after a rename it never made.
"""

from shipvision.tracking.core import (
    BotSortTracker,
    ByteTrackTracker,
    DeepSortV2Tracker,
    OcSortTracker,
    SortTracker,
)

__all__ = [
    "BotSortTracker",
    "ByteTrackTracker",
    "DeepSortV2Tracker",
    "OcSortTracker",
    "SortTracker",
]
