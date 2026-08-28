"""McByte's track state: BoT-SORT's, which is ByteTrack's, which is the shared pool.

McByte changes *how a stage is solved*, not what a track remembers, so it asks the pool for
nothing BoT-SORT does not already ask for. Re-exporting BoT-SORT's factory rather than
declaring a second one keeps the paper's "BoT-SORT plus locking" claim true of this code — see
:mod:`shipvision.mot.trackers.botsort.tracklet` for the same argument one level down.

The propagated segmentation masks the paper conditions on are the piece that will change this:
they are per-track evidence with a lifetime, and when they arrive they arrive here.
"""

from __future__ import annotations

from shipvision.mot.trackers.botsort.tracklet import new_pool

__all__ = ["new_pool"]
