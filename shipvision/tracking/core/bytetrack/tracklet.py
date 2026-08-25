"""ByteTrack's track state: the shared pool, plus an appearance EMA it never reads itself.

Track state is one :class:`~shipvision.tracking.pool.TrackPool` for every algorithm here, not
a per-algorithm tracklet class — see :mod:`shipvision.tracking.core.sort.tracklet` for why.
This file states the one capability ByteTrack switches on.
"""

from __future__ import annotations

from shipvision.tracking.pool import TrackPool

__all__ = ["new_pool"]


def new_pool(*, max_age: int, min_hits: int, embedding_momentum: float) -> TrackPool:
    """The pool with an appearance EMA, which ByteTrack maintains but does not associate on.

    ByteTrack's association is purely geometric, so carrying an appearance vector looks like
    dead weight until you follow the track downstream: the cross-camera tier matches on
    appearance, and it has no access to the crops. Averaging the vector here — once, as the
    detections arrive — is strictly cheaper than re-deriving it from stored crops later, and
    it is the reason a ByteTrack tracklet is usable by MTMC at all.

    Args:
        max_age: frames a confirmed track survives unmatched before the pool drops it.
        min_hits: matches before the pool starts publishing a track.
        embedding_momentum: EMA retention for the appearance vector, high meaning "barely
            update".
    """
    return TrackPool(max_age=max_age, min_hits=min_hits, embedding_momentum=embedding_momentum)
