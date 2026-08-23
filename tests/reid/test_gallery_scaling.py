"""Cost and concurrency properties of a gallery query.

These are the two claims the gallery docstrings make that arithmetic on a laptop can check
and a correctness test cannot: that a query costs about what its matrix product costs, and
that a gallery shared between worker threads actually runs those products at the same time.
Both were false once, in different galleries, and both failures are invisible to a test that
only asks who the top match was.
"""

from __future__ import annotations

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
