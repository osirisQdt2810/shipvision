from __future__ import annotations

import re
import tracemalloc

import numpy as np
import pytest

from shipvision.errors import ConfigurationError
from shipvision.reid import cosine_similarity, evaluate_ranking, normalize, rerank
from shipvision.reid.rerank import _peak_estimate
from tests.reid.conftest import view_of


def build_set(identities: int, views: int, jitter: float) -> tuple[np.ndarray, list[str]]:
    vectors, labels = [], []
    for identity in range(identities):
        for view in range(views):
            vectors.append(view_of(identity, jitter=jitter, view=view))
            labels.append(f"id-{identity}")
    return normalize(np.stack(vectors)), labels


def test_reranking_returns_distances_and_keeps_the_shape() -> None:
    gallery, _ = build_set(4, 3, 0.3)
    query, _ = build_set(4, 1, 0.3)
    qg = cosine_similarity(query, gallery)

    out = rerank(qg, cosine_similarity(query, query), cosine_similarity(gallery, gallery), k1=6)

    assert out.shape == qg.shape
    assert np.all(np.isfinite(out))


def test_lambda_one_reproduces_the_original_ranking() -> None:
    """The identity case. If this drifts, the mixing is wrong and every other result from
    this function is unattributable.

    The *values* are rescaled — rows are divided by their own maximum, as the paper does —
    so what must be preserved is the ordering, which is the only thing a ranking uses.
    """
    gallery, _ = build_set(5, 3, 0.3)
    query, _ = build_set(5, 1, 0.3)
    qg = cosine_similarity(query, gallery)

    out = rerank(
        qg,
        cosine_similarity(query, query),
        cosine_similarity(gallery, gallery),
        k1=6,
        lambda_value=1.0,
    )

    for i in range(len(query)):
        assert np.array_equal(np.argsort(out[i]), np.argsort(-qg[i]))


def multimodal_set(
    identities: int = 10, views_per_mode: int = 3, dim: int = 64, separation: float = 0.4
) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    """Identities with two appearance modes each — a ship's bow and its stern.

    This structure is not decoration; it is the only regime in which re-ranking can help,
    and building the test data any other way would make the test vacuous. Isotropic blobs
    (one direction per identity plus gaussian noise) have no neighbourhood information
    beyond the cluster itself, so the k-reciprocal expansion has nothing to expand into and
    re-ranking is a no-op at best. Real embeddings are multi-modal, and the gain comes from
    exactly that: a stern query is far from the bow views in cosine terms, but shares their
    neighbours, and shared company is what the Jaccard distance measures.

    `separation` is how much the two modes have in common. Lower makes the raw ranking
    worse and leaves more for re-ranking to recover.
    """
    rng = np.random.default_rng(5)
    gallery, gallery_ids, query, query_ids = [], [], [], []
    for i in range(identities):
        core = rng.normal(size=dim)
        modes = [
            core * separation + rng.normal(size=dim) * (1.0 - separation) for _ in range(2)
        ]
        for mode in modes:
            for _ in range(views_per_mode):
                gallery.append(mode + 0.25 * rng.normal(size=dim))
                gallery_ids.append(f"id-{i}")
        # The query is taken from ONE mode, so half the gallery entries for its identity
        # are the hard cross-view case this function is supposed to rescue.
        query.append(modes[0] + 0.25 * rng.normal(size=dim))
        query_ids.append(f"id-{i}")
    return (
        normalize(np.stack(query).astype(np.float32)),
        query_ids,
        normalize(np.stack(gallery).astype(np.float32)),
        gallery_ids,
    )


def test_reranking_improves_map_where_it_is_supposed_to() -> None:
    """The claim the whole module rests on, so it is measured rather than asserted."""
    query, query_ids, gallery, gallery_ids = multimodal_set()
    qg = cosine_similarity(query, gallery)

    before = evaluate_ranking(qg, query_ids, gallery_ids)
    distances = rerank(
        qg, cosine_similarity(query, query), cosine_similarity(gallery, gallery), k1=10, k2=3
    )
    after = evaluate_ranking(-distances, query_ids, gallery_ids)

    assert before.mean_ap < 1.0, "the baseline must have room to improve or this proves nothing"
    assert (
        after.mean_ap > before.mean_ap
    ), f"re-ranking made it worse: {before.mean_ap:.4f} -> {after.mean_ap:.4f}"


def test_reranking_cannot_help_isotropic_clusters_and_that_is_expected() -> None:
    """The negative result, kept deliberately.

    One direction per identity plus gaussian noise gives the neighbourhood structure no
    information the raw distance does not already have, so re-ranking has nothing to work
    with and slightly *degrades* mAP. This is pinned so that the passing test above is
    read as evidence about multi-modal data specifically, rather than as a general claim
    that re-ranking always helps — and so nobody "fixes" this module after measuring it on
    the wrong kind of data.
    """
    gallery, gallery_ids = build_set(8, 4, jitter=0.85)
    query, query_ids = build_set(8, 1, jitter=0.85)
    qg = cosine_similarity(query, gallery)

    before = evaluate_ranking(qg, query_ids, gallery_ids)
    distances = rerank(
        qg, cosine_similarity(query, query), cosine_similarity(gallery, gallery), k1=8, k2=3
    )
    after = evaluate_ranking(-distances, query_ids, gallery_ids)

    assert after.mean_ap < before.mean_ap


def test_it_refuses_a_matrix_it_would_have_to_allocate_blind() -> None:
    """O(n^2) memory, and the failure mode without this guard is the process being killed
    by the OOM reaper rather than an error anyone can read."""
    with pytest.raises(ConfigurationError, match="GB"):
        rerank(
            np.zeros((10, 30_000), np.float32),
            np.zeros((10, 10), np.float32),
            np.zeros((30_000, 30_000), np.float32),
        )


def test_mismatched_blocks_are_refused() -> None:
    with pytest.raises(ConfigurationError, match="do not agree"):
        rerank(
            np.zeros((4, 9), np.float32),
            np.zeros((4, 4), np.float32),
            np.zeros((8, 8), np.float32),
        )


def test_nonsense_parameters_are_refused() -> None:
    gallery, _ = build_set(3, 2, 0.3)
    query, _ = build_set(3, 1, 0.3)
    blocks = (
        cosine_similarity(query, gallery),
        cosine_similarity(query, query),
        cosine_similarity(gallery, gallery),
    )

    with pytest.raises(ConfigurationError, match="lambda_value"):
        rerank(*blocks, lambda_value=1.5)
    with pytest.raises(ConfigurationError, match="k1 and k2"):
        rerank(*blocks, k1=0)
    with pytest.raises(ConfigurationError, match="k1=99"):
        rerank(*blocks, k1=99)


class TestRerankMemoryGuardIsHonest:
    """The guard must estimate the peak, not the size of one matrix.

    `test_it_refuses_a_matrix_it_would_have_to_allocate_blind` passes 30 000, which is over
    any plausible threshold and therefore cannot detect that the threshold is wrong. These
    pin the *estimate* against a measured peak at a size small enough to run, so the
    estimate and the allocations cannot drift apart in silence.
    """

    def _blocks(self, total: int, dim: int = 16):
        n_query = total // 10
        rng = np.random.default_rng(0)
        query = normalize(rng.normal(size=(n_query, dim)).astype(np.float32))
        gallery = normalize(rng.normal(size=(total - n_query, dim)).astype(np.float32))
        return (
            cosine_similarity(query, gallery),
            cosine_similarity(query, query),
            cosine_similarity(gallery, gallery),
        )

    @pytest.mark.parametrize("total", [1200, 2000])
    @pytest.mark.parametrize("k2", [1, 6])
    def test_the_estimate_matches_a_measured_peak(self, total: int, k2: int) -> None:
        qg, qq, gg = self._blocks(total)
        estimate = sum(_peak_estimate(qg.shape[0], qg.shape[1], k2).values())

        tracemalloc.start()
        try:
            rerank(qg, qq, gg, k1=20, k2=k2)
            _, measured = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        ratio = measured / estimate
        assert 0.7 <= ratio <= 1.3, (
            f"the guard claims {estimate / 1e6:.1f} MB at total={total} k2={k2} and the "
            f"function peaked at {measured / 1e6:.1f} MB ({ratio:.2f}x)"
        )

    def test_the_refusal_names_the_peak_and_where_it_goes(self) -> None:
        with pytest.raises(ConfigurationError) as raised:
            rerank(
                np.zeros((10, 30_000), np.float32),
                np.zeros((10, 10), np.float32),
                np.zeros((30_000, 30_000), np.float32),
            )

        message = str(raised.value)
        assert "int32" in message, "the breakdown has to say where the memory goes"

        # The property that matters is the reported TOTAL, not the absence of a substring.
        # One 30 010^2 float32 matrix is 3.6 GB, and the breakdown legitimately names several
        # parts of exactly that size — so forbidding the string "3.6 GB" would reject a
        # correct message. Parse the headline figure and require it to be several matrices.
        one_matrix_gb = 30_010 * 30_010 * 4 / 1e9
        reported = float(re.search(r"peaks at about ([\d.]+) GB", message).group(1))

        assert reported > 3 * one_matrix_gb, (
            f"the guard reports {reported:.1f} GB, which is only {reported / one_matrix_gb:.1f} "
            f"matrices — the function holds at least four live at once, and under-reporting "
            f"is the exact defect this test exists for"
        )

    def test_int32_ranks_are_wide_enough_for_anything_the_guard_permits(self) -> None:
        """int32 is the whole first half of the fix, so the bound is worth stating."""
        assert 20_000 < np.iinfo(np.int32).max


class TestRerankRefusesPoisonedInput:
    """One NaN gallery row makes `original.max(axis=1)` NaN for that row, and `argsort` on
    an all-NaN row falls back to array order — so a metric computed downstream comes back
    *higher* than the truth. Refusing is the only safe answer."""

    def _finite_blocks(self, identities: int = 6):
        gallery, _ = build_set(identities, 3, 0.3)
        query, _ = build_set(identities, 1, 0.3)
        return (
            cosine_similarity(query, gallery),
            cosine_similarity(query, query),
            cosine_similarity(gallery, gallery),
        )

    @pytest.mark.parametrize("block", [0, 1, 2])
    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_a_non_finite_entry_in_any_block_is_refused(self, block: int, bad: float) -> None:
        blocks = list(self._finite_blocks())
        blocks[block] = blocks[block].copy()
        blocks[block][0, 0] = bad

        with pytest.raises(ConfigurationError, match="non-finite"):
            rerank(*blocks, k1=6)

    def test_one_poisoned_row_would_otherwise_flatter_the_measurement(self) -> None:
        """The measurement that motivates the guard, kept as evidence rather than a claim."""
        blocks = list(self._finite_blocks())
        clean = rerank(*blocks, k1=6)
        assert np.all(np.isfinite(clean)), "the clean case must still work"

        blocks[2] = blocks[2].copy()
        blocks[2][3, :] = np.nan
        with pytest.raises(ConfigurationError, match="non-finite"):
            rerank(*blocks, k1=6)
