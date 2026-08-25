"""Cost and concurrency properties of a gallery query.

These are the two claims the gallery docstrings make that arithmetic on a laptop can check
and a correctness test cannot: that a query costs about what its matrix product costs, and
that a gallery shared between worker threads actually runs those products at the same time.
Both were false once, in different galleries, and both failures are invisible to a test that
only asks who the top match was.
"""

from __future__ import annotations

import sys
import threading
import time

import numpy as np
import pytest

from shipvision.reid import GALLERIES
from shipvision.reid.distance import cosine_similarity
from shipvision.types import Embedding

GALLERY_NAMES = GALLERIES.names()

DIM = 256
IDENTITIES = 4_000


def _fill(name: str, *, identities: int, dim: int, views: int = 1):
    gallery = GALLERIES.build(name, capacity=identities * views + 8)
    rng = np.random.default_rng(4)
    vectors = rng.normal(size=(identities, dim)).astype(np.float32)
    for i in range(identities):
        for view in range(views):
            gallery.add(
                Embedding(
                    vector=vectors[i] + 0.01 * view,
                    identity=f"ship-{i}",
                    camera_id=f"cam-{i % 50}",
                )
            )
    return gallery, vectors


def _best_of(call, repeats: int = 7) -> float:
    """Fastest of several runs. The box these tests run on is shared, so the minimum is the
    only statistic that is about the code rather than about the neighbours."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        best = min(best, time.perf_counter() - start)
    return best


class TestGalleryQueryCostsAboutWhatItsGemmCosts:
    """A query is one matrix product plus a top-k. Anything materially above that is Python
    work per gallery row, which is what put re-identification off the frame budget in the
    system this library replaces — and it is invisible to every correctness test.
    """

    @pytest.mark.parametrize("name", GALLERY_NAMES)
    def test_a_query_is_within_a_small_multiple_of_the_matrix_product(self, name: str) -> None:
        gallery, vectors = _fill(name, identities=IDENTITIES, dim=DIM)
        probe = np.asarray(vectors[7], dtype=np.float32)
        dense = np.ascontiguousarray(vectors / np.linalg.norm(vectors, axis=1, keepdims=True))
        row = probe.reshape(1, -1)

        gallery.query(probe, top_k=5, exclude_camera="cam-3")
        reference = _best_of(lambda: cosine_similarity(row, dense))
        measured = _best_of(lambda: gallery.query(probe, top_k=5, exclude_camera="cam-3"))

        # A generous multiple: the query also normalises the probe, masks a camera and sorts
        # k entries. Ten times the gemm leaves room for all of that on a loaded machine and
        # still fails an implementation that touches one Python object per row — those run at
        # 10x to 30x, because a per-row __getitem__ costs more than the multiply.
        assert measured < 10.0 * reference, (
            f"{name}: query {measured * 1e6:.0f} us against a {reference * 1e6:.0f} us gemm "
            f"over {IDENTITIES} rows of {DIM}"
        )

    @pytest.mark.parametrize("name", GALLERY_NAMES)
    def test_the_query_path_never_restacks_the_gallery(
        self, name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The structural form of the same claim, which does not depend on a clock.

        ``np.stack`` over a list of rows is the specific shape that was wrong: it looks like
        one numpy call and is in fact one Python ``__getitem__`` and one bounds check per
        gallery entry, and it re-materialises the whole gallery on every query.
        """
        gallery, vectors = _fill(name, identities=64, dim=32)

        def refuse(*args: object, **kwargs: object) -> object:
            raise AssertionError("the query path re-materialised the gallery row by row")

        module = type(gallery).__module__
        monkeypatch.setattr(f"{module}.np.stack", refuse, raising=False)
        monkeypatch.setattr(f"{module}.np.vstack", refuse, raising=False)
        monkeypatch.setattr(f"{module}.np.concatenate", refuse, raising=False)

        result = gallery.query(np.asarray(vectors[5], dtype=np.float32), top_k=4)

        assert result.best is not None and result.best.identity == "ship-5"


def _elapsed_over_threads(call, *, threads: int, per_thread: int) -> float:
    """Wall time for ``threads`` workers each making ``per_thread`` calls."""
    barrier = threading.Barrier(threads)

    def work() -> None:
        barrier.wait()
        for _ in range(per_thread):
            call()

    pool = [threading.Thread(target=work) for _ in range(threads)]
    start = time.perf_counter()
    for thread in pool:
        thread.start()
    for thread in pool:
        thread.join()
    return time.perf_counter() - start


class TestASharedGalleryRunsItsSearchesConcurrently:
    """:class:`BaseGallery` promises the server may call one gallery from several worker
    threads. That promise is worth nothing if the lock is held across the matrix product:
    the gemm is read-only and releases the GIL, so it is the one part of a query that *can*
    overlap, and holding the lock over it turns eight threads into one.
    """

    DELAY = 0.02
    THREADS = 4
    PER_THREAD = 4

    def _gallery(self, name: str):
        gallery, vectors = _fill(name, identities=256, dim=64)
        return gallery, np.asarray(vectors[11], dtype=np.float32)

    @pytest.mark.parametrize("name", GALLERY_NAMES)
    def test_the_matrix_product_is_not_serialised_by_the_lock(
        self, name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fixed sleep stands in for the gemm, deliberately.

        Timing a real gemm here would measure the BLAS thread pool and whatever else is
        running on the box, and would say nothing about the lock; a sleep releases the GIL
        exactly as a gemm does and makes the claim — "these overlap" — decidable in
        milliseconds whatever the machine is doing.
        """
        gallery, probe = self._gallery(name)
        module = sys.modules[type(gallery).__module__]
        real = module.cosine_similarity

        def slow(*args: object, **kwargs: object) -> object:
            time.sleep(self.DELAY)
            return real(*args, **kwargs)

        monkeypatch.setattr(module, "cosine_similarity", slow)
        call = lambda: gallery.query(probe, top_k=3, exclude_camera="cam-2")  # noqa: E731

        alone = _elapsed_over_threads(call, threads=1, per_thread=self.PER_THREAD)
        together = _elapsed_over_threads(call, threads=self.THREADS, per_thread=self.PER_THREAD)

        # The harness has to be able to see the injected cost at all, or the test below
        # would pass on a gallery that never called the patched function.
        assert alone >= self.PER_THREAD * self.DELAY * 0.9, "the sleep is not on the path"
        serialised = self.THREADS * self.PER_THREAD * self.DELAY
        assert together < 0.5 * serialised, (
            f"{name}: {self.THREADS} threads took {together * 1e3:.0f} ms where one thread "
            f"takes {alone * 1e3:.0f} ms — the searches did not overlap"
        )

    @pytest.mark.parametrize("name", GALLERY_NAMES)
    def test_a_concurrent_eviction_never_attaches_a_score_to_the_wrong_identity(
        self, name: str
    ) -> None:
        """The price of letting the gemm run outside the lock, and the line that must hold.

        A stale result is acceptable — a query may miss an identity added a microsecond ago.
        A *mis-attributed* result is not: rows are compacted by swapping the last live row
        into the hole, so a reader holding a row index while a writer evicts would otherwise
        report the score of one identity under the name of another. Every identity here is a
        distinct basis vector, so a score near 1 belongs to exactly one name and any
        confusion is visible rather than plausible.
        """
        dim, capacity, pool = 64, 24, 64
        gallery = GALLERIES.build(name, capacity=capacity)
        basis = np.eye(pool, dim, dtype=np.float32)
        for i in range(capacity):
            gallery.add(Embedding(vector=basis[i], identity=f"ship-{i}", camera_id="cam-a"))

        stop = threading.Event()
        problems: list[str] = []

        def churn() -> None:
            rng = np.random.default_rng(3)
            while not stop.is_set():
                i = int(rng.integers(pool))
                gallery.add(Embedding(vector=basis[i], identity=f"ship-{i}", camera_id="cam-a"))
                if rng.random() < 0.25:
                    gallery.remove_identity(f"ship-{int(rng.integers(pool))}")

        def read() -> None:
            rng = np.random.default_rng(7)
            for _ in range(400):
                i = int(rng.integers(pool))
                for match in gallery.query(basis[i], top_k=3).matches:
                    if match.score > 0.5 and match.identity != f"ship-{i}":
                        problems.append(f"{match.identity} scored {match.score:.3f} for {i}")

        writers = [threading.Thread(target=churn) for _ in range(2)]
        readers = [threading.Thread(target=read) for _ in range(4)]
        for thread in writers:
            thread.start()
        for thread in readers:
            thread.start()
        for thread in readers:
            thread.join()
        stop.set()
        for thread in writers:
            thread.join()

        assert not problems, problems[:5]
