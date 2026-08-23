"""How several views of one identity become one vector.

The registry lives here rather than in a module of its own because it is typed on the base
class — ``Registry[FeatureAggregator]`` — and a file whose entire content is one line that
cannot be read without this one is a file that only adds an import to follow.
"""

from __future__ import annotations

import abc

import numpy as np

from shipvision.registry import PYTHON, Registry

__all__ = ["AGGREGATORS", "FeatureAggregator"]


class FeatureAggregator(abc.ABC):
    """Reduce many embeddings of one identity to the single vector that represents it.

    This is the part of re-identification that decides whether a gallery entry is any good.
    A ship seen from the bow, from the stern and half-occluded by a crane produces three
    very different vectors, and how they are combined decides whether the fourth view
    matches. Combining them badly is worse than keeping one: a mean of two genuinely
    different appearances lands between them and is close to neither.

    Two shapes of use, and every implementation must serve both:

    * **Batch** — :meth:`aggregate` over all the vectors at once, for building a gallery
      from a labelled set.
    * **Online** — :meth:`update` folding one new observation into the running vector, for
      a live track whose appearance is refreshed every frame. It never sees the history
      again, so an implementation that needs all the vectors must say so by raising.

    Implementations receive **already-normalised** rows and must return a normalised
    vector. Normalising is the caller's contract (see :mod:`shipvision.reid.distance`);
    re-normalising the output is the aggregator's, because a mean of unit vectors is not
    itself a unit vector.

    **One instance backs exactly one running vector.** :meth:`update` is permitted to keep
    state — an exact running mean needs the accumulated weight, and recomputing it from the
    renormalised previous vector is impossible, not merely slower. So an instance shared
    across identities would fold every identity's observations into one accumulator and
    hand each of them the result. A holder of many running vectors must therefore build one
    aggregator per vector; :class:`~shipvision.reid.gallery.centroid.CentroidGallery` takes
    a factory for exactly this reason. :meth:`aggregate` has no such constraint: it sees
    all its input at once and must be safe to call repeatedly on one instance.
    """

    name: str = "aggregator"
    backend: str = PYTHON

    @abc.abstractmethod
    def aggregate(
        self, vectors: np.ndarray, *, weights: np.ndarray | None = None
    ) -> np.ndarray:
        """Reduce ``(n, d)`` normalised rows to one normalised ``(d,)`` vector."""

    @abc.abstractmethod
    def update(
        self, current: np.ndarray | None, observation: np.ndarray, *, weight: float = 1.0
    ) -> np.ndarray:
        """Fold one observation into a running vector.

        Args:
            current: the vector so far, or `None` for the first observation.
            observation: one normalised ``(d,)`` row.
            weight: how much to trust it — a quality score, typically.
        """

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"


#: Which aggregation a deployment uses is an empirical question answered by rank-1 accuracy
#: on its own footage, not by argument. Selecting one by name makes that comparison a config
#: change rather than a code change.
AGGREGATORS: Registry[FeatureAggregator] = Registry("aggregator")
