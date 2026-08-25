"""DeepSORTv2's track state: an appearance EMA and OC-SORT's re-update, together.

Track state is one :class:`~shipvision.mot.pool.TrackPool` for every algorithm here, not
a per-algorithm tracklet class — see :mod:`shipvision.mot.trackers.sort.tracklet` for why.
This file states the two capabilities DeepSORTv2 switches on, which is the whole of what makes
its state differ from ByteTrack's and from OC-SORT's.
"""

from __future__ import annotations

from shipvision.mot.pool import TrackPool

__all__ = ["new_pool"]


def new_pool(
    *, max_age: int, min_hits: int, embedding_momentum: float, re_update: bool
) -> TrackPool:
    """The pool with an appearance EMA *and* observation-centric re-update.

    It is the only one of the five that asks for both, and that combination is what
    "DeepSORTv2" names: DeepSORT's appearance memory with OC-SORT's correction to the filter.

    The EMA rate given here is the **lower** bound of the dynamic range, not a fixed rate. It
    is the value used for a detection the frame gives no reason to trust more than usual;
    :func:`~shipvision.mot.trackers.deepsortv2.utils.dynamic_momentum` computes a per-detection
    rate that rises towards the upper bound as a detection gets less confident or more crowded,
    and hands it to :meth:`~shipvision.mot.pool.TrackPool.apply_matches` per frame. Passing
    the lower bound here is what makes the constructor's value the floor rather than the
    default — so a frame with no dynamic rate degrades to "update normally" rather than to
    "barely update", which would freeze every gallery vector the first time a frame was empty.

    Args:
        max_age: frames a confirmed track survives unmatched before the pool drops it.
        min_hits: matches before the pool starts publishing a track.
        embedding_momentum: the lower bound of the dynamic EMA range; see above.
        re_update: enable ORU. See :mod:`shipvision.mot.trackers.ocsort.tracklet`, which
            turns on the same pool capability for the same reason.
    """
    return TrackPool(
        max_age=max_age,
        min_hits=min_hits,
        embedding_momentum=embedding_momentum,
        re_update=re_update,
    )
