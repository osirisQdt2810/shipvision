"""OC-SORT's track state: the shared pool with its two observation-centric features on.

Track state is one :class:`~shipvision.tracking.pool.TrackPool` for every algorithm here, not
a per-algorithm tracklet class — see :mod:`shipvision.tracking.core.sort.tracklet` for why.
This file states the capabilities OC-SORT switches on, which is the whole of what makes its
state differ from SORT's.
"""

from __future__ import annotations

from shipvision.tracking.pool import TrackPool

__all__ = ["new_pool"]


def new_pool(*, max_age: int, min_hits: int, delta_t: int, re_update: bool) -> TrackPool:
    """The pool with a bounded observation ring and observation-centric re-update.

    The two extras are the two halves of "observation-centric" that live in the *state* rather
    than in the cost:

    **The observation ring** is what lets the pool answer
    :meth:`~shipvision.tracking.pool.TrackPool.directions` at all. A heading measured between
    two real detections is a measurement; the filter's velocity after a gap is a guess
    conditioned on its own earlier guesses, which is precisely what OC-SORT stopped trusting.

    **Re-update (ORU)** rewinds a re-found track to its last real observation and runs the
    filter through the measurements the detector would have produced. Without it the single
    distant measurement that ends a gap drives an enormous velocity correction — the
    covariance has been inflating for the whole gap — the next prediction overshoots, and the
    track is lost permanently while a new identity is born.

    Args:
        max_age: frames a confirmed track survives unmatched before the pool drops it.
        min_hits: matches before the pool starts publishing a track.
        delta_t: how many frames back the momentum term measures heading over. The ring is
            sized ``delta_t + 1`` because that is the smallest history that can measure a
            heading over ``delta_t`` frames — and it *is* bounded, because this process runs
            for weeks and an unbounded per-track history is how the previous system's memory
            grew until it was restarted.
        re_update: enable ORU. Off is not a deployment configuration; it is how the feature's
            contribution gets measured, and a feature nobody can switch off is a feature
            nobody can show is earning its keep.
    """
    return TrackPool(
        max_age=max_age,
        min_hits=min_hits,
        observation_history=delta_t + 1,
        re_update=re_update,
    )
