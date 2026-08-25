"""One package per tracking algorithm. Importing this package is what registers all of them.

Each subpackage is the same three files, and the split is the point rather than the tidiness:

``tracker.py``
    The class, and only the class: the per-frame sequence of association stages. Reading it
    top to bottom should read as the paper's algorithm, which it cannot do while the cost
    arithmetic is inlined between the stages.
``tracklet.py``
    What this algorithm asks of the shared track state. There is no per-algorithm tracklet
    *class* here and that is deliberate — see :mod:`shipvision.mot.trackers.sort.tracklet`
    for the reason — so what the file states is which of
    :class:`~shipvision.mot.pool.TrackPool`'s optional capabilities the algorithm turns
    on. Five files that answer that question are readable as a diff; five ``update`` methods
    are not.
``utils.py``
    Helpers used by **this algorithm alone**. Anything a second algorithm reaches for moves to
    :mod:`shipvision.mot.association` or :mod:`shipvision.mot.motion` instead of
    being copied, because a copied cost function is one that gets fixed in one place: the two
    trackers keep passing their tests and quietly stop being comparable, which defeats the
    only reason five of them exist.

The five are deliberately a chain of one-idea-at-a-time differences, so a claim about any of
them can be tested against the one below it rather than asserted:
``sort`` -> ``bytetrack`` -> ``ocsort``, and ``bytetrack`` -> ``botsort``, with ``deepsortv2``
combining the internal C++ tracker's cascade with OC-SORT's two observation-centric
corrections.
"""

from shipvision.mot.trackers.botsort import BotSortTracker
from shipvision.mot.trackers.bytetrack import ByteTrackTracker
from shipvision.mot.trackers.deepsortv2 import DeepSortV2Tracker
from shipvision.mot.trackers.ocsort import OcSortTracker
from shipvision.mot.trackers.sort import SortTracker

__all__ = [
    "BotSortTracker",
    "ByteTrackTracker",
    "DeepSortV2Tracker",
    "OcSortTracker",
    "SortTracker",
]
