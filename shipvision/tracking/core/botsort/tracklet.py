"""BoT-SORT's track state: ByteTrack's, unchanged, and that is the finding worth recording.

Track state is one :class:`~shipvision.tracking.pool.TrackPool` for every algorithm here, not
a per-algorithm tracklet class — see :mod:`shipvision.tracking.core.sort.tracklet` for why.

BoT-SORT asks the pool for nothing ByteTrack does not already ask for. Its two contributions
use capabilities the pool always has rather than ones it has to be built with:
:meth:`~shipvision.tracking.pool.TrackPool.apply_camera_motion` warps the predictions once per
frame, and :meth:`~shipvision.tracking.pool.TrackPool.embeddings` reads the appearance vectors
ByteTrack was already maintaining. So this module re-exports ByteTrack's factory instead of
declaring a second one: a copy would be four identical lines whose only future is to be
changed on one side, at which point the paper's "ByteTrack plus two things" claim quietly
stops being true of this code.
"""

from __future__ import annotations

from shipvision.tracking.core.bytetrack.tracklet import new_pool

__all__ = ["new_pool"]
