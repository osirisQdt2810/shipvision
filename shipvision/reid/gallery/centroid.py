"""One aggregated vector per identity."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence

import numpy as np

from shipvision.errors import ConfigurationError, DimensionMismatchError
from shipvision.registry import PYTHON
from shipvision.reid.aggregation.base import AGGREGATORS, FeatureAggregator
from shipvision.reid.distance import cosine_similarity, normalize
from shipvision.reid.gallery.base import GALLERIES, BaseGallery
from shipvision.reid.types import Match, QueryResult
from shipvision.types import Embedding

__all__ = ["CentroidGallery"]


@GALLERIES.register("centroid", backend=PYTHON, aliases=("prototype",))
class CentroidGallery(BaseGallery):
    """Each identity is one vector, folded in as observations arrive.

    Memory stops depending on how long a ship stays in view and starts depending only on
    how many ships there are — the difference between a gallery that grows all day and one
    that does not. A thousand identities of 512 floats is 2 MB, and the search is a gemm
    against a thousand rows rather than fifty thousand.

    The trade is real and worth stating plainly. A single centroid cannot represent an
    identity with genuinely multi-modal appearance: a ship's bow and stern views average to
    a vector resembling neither, and a query from either side scores worse against the
    centroid than it would against the nearest stored view. Prefer :class:`FlatGallery`
    where views vary that much, and this where they do not — or where the identity count is
    large enough that keeping sixteen vectors each is not affordable.

    The aggregator decides the folding: ``ema`` (the default) tracks current appearance and
    forgets at a fixed rate; ``mean`` weights every observation equally forever. That choice
    is this class's entire behaviour, which is why it is injected rather than hard-coded.

    It is injected as a **factory**, and that is load-bearing rather than ceremony: an
    aggregator's :meth:`~FeatureAggregator.update` may carry state, so one shared instance
    would accumulate every identity into a single running vector and return it to all of
    them. One aggregator is built per identity and lives as long as that identity does.

    Camera exclusion is best-effort here and cannot be otherwise: a centroid has no single
    camera. The camera of the *most recent* observation is recorded and used, so a query is
    still prevented from matching an identity it just saw in its own view — the case that
    actually inflates scores — but an identity built from several cameras is not excluded
    on account of one of them.
    """

    def __init__(
        self,
        *,
        aggregator: str | Callable[[], FeatureAggregator] = "ema",
        aggregator_options: dict[str, object] | None = None,
        capacity: int = 10_000,
        dim: int | None = None,
    ) -> None:
        if capacity <= 0:
            raise ConfigurationError(f"capacity must be positive, got {capacity}")
        self.capacity = capacity
        options = dict(aggregator_options or {})
        if isinstance(aggregator, str):
            self._make_aggregator: Callable[[], FeatureAggregator] = lambda: AGGREGATORS.build(
                aggregator, **options
            )
            # Build one now rather than on the first add: an unknown name or a bad option
            # must fail where the gallery is configured, not on whichever camera happens to
            # see the first ship.
            self._make_aggregator()
        else:
            if options:
                raise ConfigurationError(
                    "aggregator_options applies to a name; a factory should close over its "
                    "own arguments"
                )
            self._make_aggregator = aggregator
        self.aggregator_name = aggregator if isinstance(aggregator, str) else "factory"

        self._dim = dim
        self._aggregators: dict[str, FeatureAggregator] = {}
        self._vectors: dict[str, np.ndarray] = {}
        self._camera: dict[str, str | None] = {}
        self._frame: dict[str, int | None] = {}
        self._observations: dict[str, int] = {}
        self._sequence: dict[str, int] = {}
        self._next_sequence = 0
        self._lock = threading.RLock()

    @property
    def dim(self) -> int | None:
        return self._dim

    def __len__(self) -> int:
        return len(self._vectors)

    @property
    def identities(self) -> Sequence[str]:
        with self._lock:
            return tuple(self._vectors)

    def observations_for(self, identity: str) -> int:
        """How many embeddings have been folded into this identity's centroid."""
        with self._lock:
            return self._observations.get(identity, 0)

    def add(self, embedding: Embedding) -> int:
        if embedding.identity is None:
            raise ConfigurationError(
                "a gallery entry needs an identity; an unlabelled vector is a query"
            )
        with self._lock:
            if self._dim is None:
                self._dim = embedding.dim
            elif embedding.dim != self._dim:
                raise DimensionMismatchError(
                    f"gallery holds {self._dim}-d vectors, got a {embedding.dim}-d one"
                )

            identity = embedding.identity
            known = self._vectors.get(identity)
            if known is None and len(self._vectors) >= self.capacity:
                self._evict_oldest()

            folder = self._aggregators.get(identity)
            if folder is None:
                folder = self._make_aggregator()
                self._aggregators[identity] = folder
            self._vectors[identity] = folder.update(
                known, normalize(embedding.vector), weight=embedding.quality
            )
            self._camera[identity] = embedding.camera_id
            self._frame[identity] = embedding.frame_id
            self._observations[identity] = self._observations.get(identity, 0) + 1
            self._sequence[identity] = self._next_sequence
            self._next_sequence += 1
            return len(self._vectors) - 1

    def query(
        self,
        vector: np.ndarray,
        *,
        top_k: int = 1,
        threshold: float | None = None,
        exclude_camera: str | None = None,
    ) -> QueryResult:
        if top_k <= 0:
            raise ConfigurationError(f"top_k must be positive, got {top_k}")
        with self._lock:
            if not self._vectors:
                return QueryResult(matches=())

            names = [
                name
                for name in self._vectors
                if exclude_camera is None or self._camera.get(name) != exclude_camera
            ]
            if not names:
                return QueryResult(matches=())

            probe = normalize(np.asarray(vector, dtype=np.float32).reshape(1, -1))
            if probe.shape[1] != self._dim:
                raise DimensionMismatchError(
                    f"query is {probe.shape[1]}-d, gallery is {self._dim}-d"
                )
            centroids = np.stack([self._vectors[n] for n in names])
            scores = cosine_similarity(probe, centroids)[0]

            k = min(top_k, len(names))
            top = np.argpartition(-scores, k - 1)[:k]
            top = top[np.argsort(-scores[top], kind="stable")]
            matches = tuple(
                Match(
                    identity=names[i],
                    score=float(scores[i]),
                    entry_index=int(i),
                    camera_id=self._camera.get(names[i]),
                    frame_id=self._frame.get(names[i]),
                )
                for i in top
            )

        accepted = matches[0] if threshold is None or matches[0].score >= threshold else None
        return QueryResult(matches=matches, accepted=accepted)

    def remove_identity(self, identity: str) -> int:
        with self._lock:
            if identity not in self._vectors:
                return 0
            for store in (
                self._vectors,
                self._camera,
                self._frame,
                self._observations,
                self._sequence,
                self._aggregators,
            ):
                store.pop(identity, None)
            return 1

    def clear(self) -> None:
        with self._lock:
            for store in (
                self._vectors,
                self._camera,
                self._frame,
                self._observations,
                self._sequence,
                self._aggregators,
            ):
                store.clear()

    def _evict_oldest(self) -> None:
        """Drop the identity least recently observed, not the one added longest ago.

        Recency of *observation* is the right key: an identity seen thirty seconds ago is
        more likely to come back than one first enrolled recently and not seen since.
        """
        oldest = min(self._sequence, key=lambda name: self._sequence[name])
        self.remove_identity(oldest)
