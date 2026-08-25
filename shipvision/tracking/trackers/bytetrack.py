"""Compatibility shim: ``shipvision.tracking.trackers.bytetrack`` before the repackaging.

The algorithm is a package now — :mod:`shipvision.tracking.core.bytetrack`, with ``tracker.py``,
``tracklet.py`` and ``utils.py`` — and this module re-exports its class so the module path a
caller may have imported still resolves.

A package-level re-export is **not** the same promise as a module path, which is what the
first attempt at this refactor got wrong: ``from shipvision.tracking.trackers import
ByteTrackTracker`` kept working while ``import shipvision.tracking.trackers.bytetrack`` raised
``ModuleNotFoundError``. Both were public, so both have to survive.

Do not add anything here. This file exists to be a redirect and nothing else.
"""

from shipvision.tracking.core.bytetrack.tracker import ByteTrackTracker

__all__ = ["ByteTrackTracker"]
