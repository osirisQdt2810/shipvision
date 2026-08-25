"""One aggregated vector per identity."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence

import numpy as np

from shipvision.errors import ConfigurationError, DimensionMismatchError
from shipvision.registry import PYTHON
from shipvision.reid.aggregation.base import AGGREGATORS, FeatureAggregator
from shipvision.reid.distance import cosine_similarity, normalize
from shipvision.reid.gallery._cameras import NO_CAMERA, CameraCodec
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

    **Storage is a dense matrix, for the same reason as in :class:`FlatGallery`.** Centroids
    live in one preallocated ``(capacity, dim)`` float32 array with an identity-to-row map,
    and camera, frame and recency are parallel integer arrays. Keeping them in a dict of
    per-identity arrays instead is the tempting shape and it is a per-query O(identities)
    Python cost: at this class's own default capacity, restacking 10 000 rows of 512 floats
    took 25 ms and re-materialised 20 MB *per query*, three orders of magnitude more than
    the gemm it was preparing. Two galleries behind one registry have to be substitutable,
    and a query cost that differs by three orders of magnitude is not a substitution.
    Adding a new identity writes one row; evicting swaps the last live row into the hole,
    which keeps the matrix contiguous for the gemm at the cost of stable row indices.

    Camera exclusion is best-effort here and cannot be otherwise: a centroid has no single
    camera. The camera of the *most recent* observation is recorded and used, so a query is
    still prevented from matching an identity it just saw in its own view — the case that
    actually inflates scores — but an identity built from several cameras is not excluded
    on account of one of them.

    **Concurrency.** The lock covers the bookkeeping, not the matrix product: see
    :meth:`query` for exactly what a concurrent writer can and cannot do to a result.
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

        self._dim: int | None = None
        self._centroids: np.ndarray | None = None
        self._size = 0

        self._identity: list[str] = []
        self._row_of: dict[str, int] = {}
        self._aggregators: dict[str, FeatureAggregator] = {}

        self._camera_code = np.full(capacity, NO_CAMERA, dtype=np.int32)
        self._frame = np.zeros(capacity, dtype=np.int64)
        self._has_frame = np.zeros(capacity, dtype=bool)
        self._observations = np.zeros(capacity, dtype=np.int64)
        #: Recency of *observation*, the key eviction uses. Bumped on every add, including a
        #: refresh of an identity already present.
        self._sequence = np.zeros(capacity, dtype=np.int64)
        #: Which entry is sitting in a row. Bumped only when a row changes *occupant*, never
        #: on a refresh, so a reader can tell "this row moved under me" from "this row's
        #: centroid was updated" — see :meth:`query`.
        self._occupant = np.zeros(capacity, dtype=np.int64)

        self._cameras = CameraCodec()
        self._next_sequence = 0
        self._next_occupant = 1
        self._lock = threading.RLock()

        if dim is not None:
            self._ensure_dim(dim)

    # -- introspection ----------------------------------------------------------------

    @property
    def dim(self) -> int | None:
        return self._dim

    def __len__(self) -> int:
        return self._size

    @property
    def identities(self) -> Sequence[str]:
        with self._lock:
            return tuple(self._identity)

    def observations_for(self, identity: str) -> int:
        """How many embeddings have been folded into this identity's centroid."""
        with self._lock:
            row = self._row_of.get(identity)
            return 0 if row is None else int(self._observations[row])

    # -- writing ----------------------------------------------------------------------

    def add(self, embedding: Embedding) -> int:
        """Fold one observation into its identity's centroid. Returns that identity's row.

        The row is what :attr:`Match.entry_index` reports for the same identity, so the two
        numberings agree — but it is only stable until the next eviction, which may swap a
        different identity into it.
        """
        if embedding.identity is None:
            raise ConfigurationError(
                "a gallery entry needs an identity; an unlabelled vector is a query"
            )
        with self._lock:
            self._ensure_dim(embedding.dim)
            # Both of these can refuse the embedding, and both run before anything is
            # written or evicted: a rejected add must leave the gallery exactly as it was.
            # Normalising is also where a non-finite vector is stopped.
            vector = normalize(embedding.vector)
            code = self._cameras.code_for(embedding.camera_id)
            assert self._centroids is not None

            identity = embedding.identity
            row = self._row_of.get(identity)
            if row is None:
                if self._size >= self.capacity:
                    self._evict_oldest()
                row = self._claim_row(identity)
                known = None
            else:
                known = self._centroids[row]

            folder = self._aggregators.get(identity)
            if folder is None:
                folder = self._make_aggregator()
                self._aggregators[identity] = folder
            self._centroids[row] = folder.update(known, vector, weight=embedding.quality)
            self._camera_code[row] = code
            self._has_frame[row] = embedding.frame_id is not None
            self._frame[row] = embedding.frame_id or 0
            self._observations[row] += 1
            self._sequence[row] = self._next_sequence
            self._next_sequence += 1
            return row

    def remove_identity(self, identity: str) -> int:
        with self._lock:
            row = self._row_of.get(identity)
            if row is None:
                return 0
            self._drop(row)
            return 1

    def clear(self) -> None:
        with self._lock:
            self._size = 0
            self._identity.clear()
            self._row_of.clear()
            self._aggregators.clear()
            self._camera_code[:] = NO_CAMERA
            self._has_frame[:] = False
            self._cameras.clear()

    # -- reading ----------------------------------------------------------------------

    def query(
        self,
        vector: np.ndarray,
        *,
        top_k: int = 1,
        threshold: float | None = None,
        exclude_camera: str | None = None,
    ) -> QueryResult:
        """Rank the centroids against one vector. See :meth:`BaseGallery.query`.

        The lock is held to take a snapshot and again to read the winning rows back; the
        matrix product and the top-k run outside it, so several worker threads searching one
        gallery actually search it at the same time. What that snapshot does and does not
        promise, precisely:

        * a concurrent :meth:`add` of a *new* identity may be missing from the result — the
          answer is at most one entry stale, which is what a frame-rate query is anyway;
        * a concurrent refresh of a centroid may be scored from either the old or the new
          vector;
        * a concurrent eviction can **never** attach a score to the wrong identity. Every
          row is stamped with which entry occupies it, and a winning row whose stamp moved
          while the gemm ran is dropped from the result rather than reported.

        The snapshot copies the stamp column — 8 bytes a row, one memcpy, under a
        microsecond per thousand identities against a gemm two orders of magnitude larger.
        """
        if top_k <= 0:
            raise ConfigurationError(f"top_k must be positive, got {top_k}")
        probe = normalize(np.asarray(vector, dtype=np.float32).reshape(1, -1))

        with self._lock:
            if self._size == 0 or self._centroids is None:
                return QueryResult(matches=())
            if probe.shape[1] != self._dim:
                raise DimensionMismatchError(
                    f"query is {probe.shape[1]}-d, gallery is {self._dim}-d"
                )
            size = self._size
            centroids = self._centroids[:size]
            codes = self._camera_code[:size]
            stamps = self._occupant[:size].copy()
            exclude = self._cameras.lookup(exclude_camera)

        scores = cosine_similarity(probe, centroids)[0]
        eligible = size
        if exclude is not None:
            same = codes == exclude
            # -inf rather than deletion: row indices stay aligned with the metadata arrays,
            # and compacting per query would cost more than the search.
            scores = np.where(same, -np.inf, scores)
            eligible -= int(same.sum())
        if eligible <= 0:
            return QueryResult(matches=())

        k = min(top_k, eligible)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]

        with self._lock:
            matches = tuple(
                Match(
                    identity=self._identity[row],
                    score=float(scores[row]),
                    entry_index=row,
                    camera_id=self._cameras.name_for(int(self._camera_code[row])),
                    frame_id=int(self._frame[row]) if self._has_frame[row] else None,
                )
                for row in (int(i) for i in top)
                if row < self._size
                and self._occupant[row] == stamps[row]
                and np.isfinite(scores[row])
                and (exclude is None or self._camera_code[row] != exclude)
            )

        if not matches:
            return QueryResult(matches=())
        accepted = matches[0] if threshold is None or matches[0].score >= threshold else None
        return QueryResult(matches=matches, accepted=accepted)

    # -- internals --------------------------------------------------------------------

    def _ensure_dim(self, dim: int) -> None:
        if self._dim is None:
            self._dim = dim
            self._centroids = np.zeros((self.capacity, dim), dtype=np.float32)
        elif dim != self._dim:
            raise DimensionMismatchError(
                f"gallery holds {self._dim}-d vectors, got a {dim}-d one"
            )

    def _claim_row(self, identity: str) -> int:
        """Give ``identity`` the next free row. The caller has already made space."""
        row = self._size
        self._identity.append(identity)
        self._row_of[identity] = row
        self._observations[row] = 0
        self._occupant[row] = self._next_occupant
        self._next_occupant += 1
        self._size += 1
        return row

    def _evict_oldest(self) -> None:
        """Drop the identity least recently observed, not the one added longest ago.

        Recency of *observation* is the right key: an identity seen thirty seconds ago is
        more likely to come back than one first enrolled recently and not seen since.

        A C-speed scan of an int64 column rather than a Python ``min`` over a dict, and it
        runs only once the gallery is full.
        """
        self._drop(int(np.argmin(self._sequence[: self._size])))

    def _drop(self, row: int) -> None:
        """Forget the identity in ``row``, swapping the last live row into its place."""
        assert self._centroids is not None
        last = self._size - 1
        victim = self._identity[row]
        del self._row_of[victim]
        self._aggregators.pop(victim, None)

        if row != last:
            moved = self._identity[last]
            self._centroids[row] = self._centroids[last]
            self._identity[row] = moved
            self._row_of[moved] = row
            self._camera_code[row] = self._camera_code[last]
            self._frame[row] = self._frame[last]
            self._has_frame[row] = self._has_frame[last]
            self._observations[row] = self._observations[last]
            self._sequence[row] = self._sequence[last]
            # A new occupant, so any reader mid-gemm holding the old stamp discards this row
            # instead of reporting `moved`'s score under `victim`'s name.
            self._occupant[row] = self._next_occupant
            self._next_occupant += 1

        self._identity.pop()
        self._camera_code[last] = NO_CAMERA
        self._has_frame[last] = False
        self._size -= 1
