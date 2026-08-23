"""One tracker per file. Importing this package is what registers them.

They are deliberately ordered as a chain of one-idea-at-a-time differences, so that a claim
about any of them can be tested against the one below rather than asserted:

``sort`` -> ``bytetrack`` -> ``ocsort``, and ``bytetrack`` -> ``botsort``, with ``deepsortv2``
combining the cascade from the internal C++ tracker with OC-SORT's two observation-centric
corrections.
"""

from shipvision.tracking.trackers.botsort import BotSortTracker
from shipvision.tracking.trackers.bytetrack import ByteTrackTracker
from shipvision.tracking.trackers.deepsortv2 import DeepSortV2Tracker
from shipvision.tracking.trackers.ocsort import OcSortTracker
from shipvision.tracking.trackers.sort import SortTracker

__all__ = [
    "BotSortTracker",
    "ByteTrackTracker",
    "DeepSortV2Tracker",
    "OcSortTracker",
    "SortTracker",
]
