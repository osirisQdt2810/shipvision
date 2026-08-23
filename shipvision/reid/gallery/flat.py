"""Exact search over a bounded, densely-packed matrix."""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from shipvision.errors import ConfigurationError, DimensionMismatchError
from shipvision.registry import PYTHON
from shipvision.reid.distance import cosine_similarity, normalize
from shipvision.reid.gallery.base import GALLERIES, BaseGallery
from shipvision.reid.types import Match, QueryResult
from shipvision.types import Embedding

__all__ = ["FlatGallery"]

_NO_CAMERA = -1


@GALLERIES.register("flat", backend=PYTHON, aliases=("exact",))
class FlatGallery(BaseGallery):
    """Every embedding kept, searched exhaustively with one matrix product.

    Exhaustive is the right default at this scale, and the arithmetic says so: 50 000
    entries of 512 floats is a 100 MB gemm per query — tens of microseconds on any modern
    CPU, a rounding error next to the detector that produced the crop. An approximate index
    earns its complexity somewhere past a million vectors; below that it buys latency that
    was not the bottleneck and pays in recall that was.

    **Bounded two ways, because one is not enough.** `capacity` caps the gallery;
    `per_identity` caps how many vectors any one identity may hold. Without the second, one
    ship that sits in view for an hour fills the gallery and evicts all fifty others — the
    exact starvation this project exists to avoid, one layer up. Per-identity eviction runs
    first, so a busy identity trims its own oldest before it can cost a neighbour a slot.

    **Everything the query path touches is a numpy array, not a Python list.** Vectors are
    a preallocated ``(capacity, dim)`` float32 matrix; camera, frame and insertion order are
    parallel integer arrays. That is not tidiness. A camera filter written as a generator
    over a list is an O(n) Python loop *per query* — about 5 ms at 50 000 entries, two
    orders of magnitude more than the gemm it is filtering, and enough on its own to put
    re-identification off the frame budget. Only the identity strings stay a list, because
    only the k selected rows are ever read from it.

    Adding writes one row; evicting swaps the last live row into the hole. Swap-with-last
    keeps the matrix contiguous for the gemm with no reallocation, at the cost of stable
    entry indices — which is why :class:`Match` carries an index for immediate use rather
    than as a durable handle.
    """

    def __init__(
        self,
        *,
        capacity: int = 50_000,
        per_identity: int = 16,
        dim: int | None = None,
    ) -> None:
        if capacity <= 0:
            raise ConfigurationError(f"capacity must be positive, got {capacity}")
        if per_identity <= 0:
            raise ConfigurationError(f"per_identity must be positive, got {per_identity}")
        self.capacity = capacity
        self.per_identity = per_identity

        self._dim: int | None = None
        self._vectors: np.ndarray | None = None
        self._size = 0

        self._identity: list[str] = []
        self._camera_code = np.full(capacity, _NO_CAMERA, dtype=np.int32)
        self._frame = np.zeros(capacity, dtype=np.int64)
        self._has_frame = np.zeros(capacity, dtype=bool)
        self._sequence = np.zeros(capacity, dtype=np.int64)

        self._camera_codes: dict[str, int] = {}
        self._camera_names: list[str] = []
        self._by_identity: dict[str, list[int]] = defaultdict(list)
        self._next_sequence = 0
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
            return tuple(self._by_identity)

    def count_for(self, identity: str) -> int:
        """How many vectors this identity currently holds."""
        with self._lock:
            return len(self._by_identity.get(identity, ()))

    # -- writing ----------------------------------------------------------------------

    def add(self, embedding: Embedding) -> int:
        if embedding.identity is None:
            raise ConfigurationError(
                "a gallery entry needs an identity; an unlabelled vector is a query"
            )
        with self._lock:
            self._ensure_dim(embedding.dim)
            self._make_room(embedding.identity)

            index = self._size
            assert self._vectors is not None
            self._vectors[index] = normalize(embedding.vector)
            self._identity.append(embedding.identity)
            self._camera_code[index] = self._code_for(embedding.camera_id)
            self._has_frame[index] = embedding.frame_id is not None
            self._frame[index] = embedding.frame_id or 0
            self._sequence[index] = self._next_sequence
            self._by_identity[embedding.identity].append(index)

            self._next_sequence += 1
            self._size += 1
            return index

    def remove_identity(self, identity: str) -> int:
        with self._lock:
            rows = self._by_identity.get(identity)
            if not rows:
                return 0
            # Descending: a swap-with-last only ever moves a row to a *lower* index, so by
            # taking the highest first this loop can never be handed a row it already
            # visited under a new number.
            doomed = sorted(rows, reverse=True)
            for index in doomed:
                self._drop(index)
            return len(doomed)

    def clear(self) -> None:
        with self._lock:
            self._size = 0
            self._identity.clear()
            self._by_identity.clear()
            self._camera_code[:] = _NO_CAMERA
            self._has_frame[:] = False

    # -- reading ----------------------------------------------------------------------

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
            if self._size == 0 or self._vectors is None:
                return QueryResult(matches=())

            probe = normalize(np.asarray(vector, dtype=np.float32).reshape(1, -1))
            if probe.shape[1] != self._dim:
                raise DimensionMismatchError(
                    f"query is {probe.shape[1]}-d, gallery is {self._dim}-d"
                )

            scores = cosine_similarity(probe, self._vectors[: self._size])[0]

            eligible = self._size
            if exclude_camera is not None:
                code = self._camera_codes.get(exclude_camera)
                if code is not None:
                    same = self._camera_code[: self._size] == code
                    # -inf rather than deletion: row indices stay aligned with the metadata
                    # arrays, and compacting per query would cost more than the search.
                    scores = np.where(same, -np.inf, scores)
                    eligible -= int(same.sum())
            if eligible <= 0:
                return QueryResult(matches=())

            k = min(top_k, eligible)
            # argpartition is O(n) where argsort is O(n log n); only the k it selects get
            # sorted. At 50 000 entries queried at frame rate that is the difference
            # between the search being invisible and being in the profile.
            top = np.argpartition(-scores, k - 1)[:k]
            top = top[np.argsort(-scores[top], kind="stable")]

            matches = tuple(
                Match(
                    identity=self._identity[i],
                    score=float(scores[i]),
                    entry_index=int(i),
                    camera_id=self._name_for(int(self._camera_code[i])),
                    frame_id=int(self._frame[i]) if self._has_frame[i] else None,
                )
                for i in top
            )

        if not matches:
            return QueryResult(matches=())
        accepted = matches[0] if threshold is None or matches[0].score >= threshold else None
        return QueryResult(matches=matches, accepted=accepted)

    # -- internals --------------------------------------------------------------------

    def _ensure_dim(self, dim: int) -> None:
        if self._dim is None:
            self._dim = dim
            self._vectors = np.zeros((self.capacity, dim), dtype=np.float32)
        elif dim != self._dim:
            raise DimensionMismatchError(
                f"gallery holds {self._dim}-d vectors, got a {dim}-d one; two models are "
                f"feeding one gallery"
            )

    def _code_for(self, camera_id: str | None) -> int:
        if camera_id is None:
            return _NO_CAMERA
        code = self._camera_codes.get(camera_id)
        if code is None:
            code = len(self._camera_names)
            self._camera_codes[camera_id] = code
            self._camera_names.append(camera_id)
        return code

    def _name_for(self, code: int) -> str | None:
        return None if code == _NO_CAMERA else self._camera_names[code]

    def _make_room(self, identity: str) -> None:
        """Evict until this identity may take one more row."""
        while len(self._by_identity[identity]) >= self.per_identity:
            rows = self._by_identity[identity]
            self._drop(min(rows, key=lambda i: int(self._sequence[i])))
        while self._size >= self.capacity:
            # A C-speed scan of an int64 array, not a Python min over a list. It is O(n),
            # and it runs only once the gallery is full; gallery writes are per *track*
            # rather than per crop — hundreds a second, not the 15 000 crops/s the server
            # ingests — so tens of microseconds here stays well inside the budget.
            self._drop(int(np.argmin(self._sequence[: self._size])))

    def _drop(self, index: int) -> None:
        """Remove one row, swapping the last live row into its place."""
        assert self._vectors is not None
        last = self._size - 1
        victim = self._identity[index]
        self._by_identity[victim].remove(index)

        if index != last:
            self._vectors[index] = self._vectors[last]
            moved = self._identity[last]
            self._identity[index] = moved
            self._camera_code[index] = self._camera_code[last]
            self._frame[index] = self._frame[last]
            self._has_frame[index] = self._has_frame[last]
            self._sequence[index] = self._sequence[last]
            rows = self._by_identity[moved]
            rows[rows.index(last)] = index

        self._identity.pop()
        self._camera_code[last] = _NO_CAMERA
        self._has_frame[last] = False
        self._size -= 1
        if not self._by_identity[victim]:
            del self._by_identity[victim]
