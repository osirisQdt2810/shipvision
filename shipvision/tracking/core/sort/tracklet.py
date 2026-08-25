"""SORT's track state: the shared pool, with none of its optional capabilities enabled.

There is no ``SortTracklet`` class here, and that is the design rather than an omission.
Track state in this library is one :class:`~shipvision.tracking.pool.TrackPool` holding
struct-of-arrays state for *every* track — means, covariances, ages, observation history —
because five per-algorithm tracklet classes are five places for "when is a track confirmed,
when does it die" to drift apart, and that drift is invisible: each tracker keeps working,
they just quietly stop agreeing about a lifecycle their comparison depends on.

What genuinely differs between algorithms is which of the pool's **optional** capabilities
each one needs, and that is what this file states. Reading it answers "what does SORT ask of
the shared state" without reading the tracker.
"""

from __future__ import annotations

from shipvision.tracking.pool import TrackPool

__all__ = ["new_pool"]


def new_pool(*, max_age: int, min_hits: int) -> TrackPool:
    """The plain pool: no observation history, no re-update, no appearance EMA.

    SORT is the baseline precisely because it asks for none of the extras. Every optional
    capability left off here is one that a later tracker in the chain switched on, and a
    reader comparing this file with ``ocsort/tracklet.py`` sees the difference as a diff
    rather than having to infer it from two ``update`` methods.

    Args:
        max_age: frames a confirmed track survives unmatched before the pool drops it.
        min_hits: matches before the pool starts publishing a track.
    """
    return TrackPool(max_age=max_age, min_hits=min_hits)
