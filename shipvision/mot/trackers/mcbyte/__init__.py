"""McByte — BoT-SORT that refuses to trade away a pair nothing else was bidding for."""

from shipvision.mot.trackers.mcbyte.tracker import McByteTracker
from shipvision.mot.trackers.mcbyte.tracklet import new_pool
from shipvision.mot.trackers.mcbyte.utils import (
    ambiguous_candidates,
    clear_matches,
    isolated_candidates,
    reduce_problem,
)

__all__ = [
    "McByteTracker",
    "ambiguous_candidates",
    "clear_matches",
    "isolated_candidates",
    "new_pool",
    "reduce_problem",
]
