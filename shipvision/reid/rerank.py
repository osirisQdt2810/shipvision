"""k-reciprocal re-ranking (Zhong et al., CVPR 2017).

The single largest accuracy gain available without touching the model, and the reason is
worth stating because it decides when to use it. A raw similarity asks "does the query look
like this gallery entry?". Re-ranking also asks "do the query and that entry have the same
*neighbours*?" — and appearance under a different viewpoint is far less stable than the
company an identity keeps. A ship photographed from the stern matches its own bow view
poorly and matches the bow view's neighbourhood well.

**It is a batch operation, not a query-time one.** The cost is O((n_query + n_gallery)^2) in
both time and memory, and the constant is not 1: several square matrices are live at once, so
10 000 combined entries peaks at roughly **1.6 GB**, not the 400 MB one matrix would suggest.
:func:`_peak_estimate` is the breakdown, and it is checked against a measured peak by a test
rather than reasoned about. Use this to evaluate a change offline, or on a bounded candidate
set retrieved first by a plain cosine search — never on a whole gallery inside a frame
budget. :func:`rerank` refuses sizes it cannot do justice to rather than quietly allocating
them.
"""

from __future__ import annotations

import numpy as np

from shipvision.errors import ConfigurationError

__all__ = ["rerank"]


#: Rows per `argsort` call. Small enough that the int64 indices numpy insists on producing
#: stay a rounding error, large enough that the per-call overhead is invisible.
_ARGSORT_CHUNK = 1024


def _argsort_int32(matrix: np.ndarray) -> np.ndarray:
    """Row-wise argsort into an int32 array, without ever materialising an int64 one.

    `np.argsort(m, axis=1).astype(np.int32)` is the obvious spelling and it allocates an
    int64 result first — **twice the size of the largest matrix in this function**, and
    measurement showed that intermediate, not anything later, was the true peak. numpy has
    no way to ask argsort for a narrower dtype, so the sort is done in row chunks and cast
    as it goes: the int64 temporary is bounded by the chunk rather than by the matrix.

    At the 20 000-entry ceiling this is the difference between a 164 MB temporary and a
    3.2 GB one.
    """
    rows = matrix.shape[0]
    out = np.empty(matrix.shape, dtype=np.int32)
    for start in range(0, rows, _ARGSORT_CHUNK):
        stop = min(start + _ARGSORT_CHUNK, rows)
        out[start:stop] = np.argsort(matrix[start:stop], axis=1, kind="stable")
    return out


def _peak_estimate(n_query: int, n_gallery: int, k2: int) -> dict[str, int]:
    """Bytes this function actually peaks at, broken down by what holds them.

    A breakdown rather than a single number, because the guard's whole failure mode was
    reporting one matrix and implying that was the cost. Naming the parts means the next
    person to add an intermediate can see whether the estimate still covers it — and a test
    compares this against a measured `tracemalloc` peak, which is what stops the two
    drifting apart again.

    The function has **two** high-water marks and they do not coincide, so this returns the
    parts of whichever dominates rather than summing both:

    * **construction** — the raw distance matrix is live alongside its normalised copy and
      the freshly-built rank array, then it is dropped;
    * **expansion** — with ``k2 > 1`` the local query expansion needs an accumulator and a
      reusable gather buffer on top of the three matrices that survive construction.

    Adding them would over-report by a third at ``k2 = 1``, and an estimate that is wrong in
    the safe direction still teaches the next reader the wrong shape of the cost.
    """
    total = n_query + n_gallery
    square = total * total
    survivors = {
        "normalised float32": square * 4,
        "rank int32": square * 4,
    }
    construction = {
        **survivors,
        # Assembled, normalised, then dropped — live alongside `normalised` but never
        # alongside `V`. Still a real high-water mark.
        "distance matrix float32 (construction)": square * 4,
        "argsort int64 chunk": min(_ARGSORT_CHUNK, total) * total * 8,
    }
    expansion = {
        **survivors,
        "V float32": square * 4,
        "jaccard float32": n_query * total * 4,
    }
    if k2 > 1:
        expansion["expansion accumulator float32"] = square * 4
        expansion["gather buffer float32"] = square * 4

    return construction if sum(construction.values()) > sum(expansion.values()) else expansion


def _k_reciprocal_neighbours(rank: np.ndarray, index: int, k: int) -> np.ndarray:
    """The entries in ``index``'s top-k that also have ``index`` in theirs.

    Plain k-nearest-neighbour sets are polluted: a hard positive that ranks 20th is missed,
    and an easy negative that ranks 3rd is included. Requiring the relation to hold *both
    ways* removes most of the second kind, which is what makes the expansion below safe.
    """
    forward = rank[index, : k + 1]
    backward = rank[forward, : k + 1]
    reciprocal = forward[(backward == index).any(axis=1)]
    return reciprocal


def rerank(
    query_gallery: np.ndarray,
    query_query: np.ndarray,
    gallery_gallery: np.ndarray,
    *,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
    max_entries: int = 20_000,
) -> np.ndarray:
    """Re-ranked ``(n_query, n_gallery)`` *distances*. Smaller is more alike.

    Args:
        query_gallery: ``(q, g)`` cosine **similarities**.
        query_query: ``(q, q)`` cosine similarities among the queries.
        gallery_gallery: ``(g, g)`` cosine similarities among the gallery.
        k1: neighbourhood size for the reciprocal set.
        k2: neighbourhood size for the local query expansion that smooths it.
        lambda_value: how much original distance to keep. 0 uses the Jaccard distance
            alone; 1 returns the original ranking unchanged. 0.3 is the paper's value.

    Similarities in, distances out — the asymmetry is deliberate. Every other function in
    this package speaks similarity, so that is what this one accepts; but the Jaccard
    distance it computes has no meaningful similarity form, and returning something called
    a score that ranked backwards would be the worst of both.
    """
    if not 0.0 <= lambda_value <= 1.0:
        raise ConfigurationError(f"lambda_value must be in [0, 1], got {lambda_value}")
    if k1 < 1 or k2 < 1:
        raise ConfigurationError(f"k1 and k2 must be positive, got {k1} and {k2}")

    qg = np.asarray(query_gallery, dtype=np.float32)
    qq = np.asarray(query_query, dtype=np.float32)
    gg = np.asarray(gallery_gallery, dtype=np.float32)
    n_query, n_gallery = qg.shape
    total = n_query + n_gallery
    if qq.shape != (n_query, n_query) or gg.shape != (n_gallery, n_gallery):
        raise ConfigurationError(
            f"blocks do not agree: query_gallery is {qg.shape}, query_query is "
            f"{qq.shape}, gallery_gallery is {gg.shape}"
        )
    for name, block in (("query_gallery", qg), ("query_query", qq), ("gallery_gallery", gg)):
        if not np.all(np.isfinite(block)):
            count = int((~np.isfinite(block)).sum())
            raise ConfigurationError(
                f"{name} has {count} non-finite value(s). Re-ranking cannot contain them: "
                f"the row-maximum normalisation is a reduction, so one bad row makes the "
                f"whole matrix NaN, and a ranking metric computed over all-NaN scores comes "
                f"back HIGHER than the truth because argsort falls back to array order"
            )

    if k1 >= total:
        raise ConfigurationError(f"k1={k1} needs more than {total} entries to be meaningful")
    if total > max_entries:
        parts = _peak_estimate(n_query, n_gallery, k2)
        peak = sum(parts.values())
        breakdown = ", ".join(f"{name} {size / 1e9:.1f} GB" for name, size in parts.items())
        raise ConfigurationError(
            f"re-ranking {n_query} queries against {n_gallery} entries peaks at about "
            f"{peak / 1e9:.1f} GB ({breakdown}) — not the {total}x{total} matrix alone. "
            f"Raise max_entries if that is genuinely intended, or retrieve a candidate set "
            f"with a plain cosine search first and re-rank only that."
        )

    # One symmetric distance matrix over queries and gallery together: re-ranking treats
    # them as a single set, and the query-query block is what lets query expansion work.
    original = np.zeros((total, total), dtype=np.float32)
    original[:n_query, :n_query] = 1.0 - qq
    original[:n_query, n_query:] = 1.0 - qg
    original[n_query:, :n_query] = (1.0 - qg).T
    original[n_query:, n_query:] = 1.0 - gg

    # Each row divided by its own maximum, which is what the paper does and is not
    # interchangeable with dividing by the row sum. The weights below are exp(-d): over a
    # row scaled to [0, 1] they span a factor of e and actually discriminate, while over a
    # row scaled to sum 1 every entry is ~1/n, exp(-d) is ~1 for all of them, and the
    # weighting silently degenerates to a uniform average of the neighbourhood.
    scale = np.maximum(original.max(axis=1, keepdims=True), 1e-12)
    normalised = original / scale
    del original  # nothing below reads it, and it is one of the largest live blocks

    rank = _argsort_int32(normalised)

    # `V` is the paper's own symbol for the weighted neighbour-set matrix (Zhong et al.,
    # eq. 8), kept verbatim so this function can be read next to it. Renaming it to satisfy
    # a lint rule would make the correspondence with the published algorithm unverifiable,
    # which is the only way anyone checks an implementation of it.
    V = np.zeros((total, total), dtype=np.float32)  # noqa: N806
    for i in range(total):
        neighbours = _k_reciprocal_neighbours(rank, i, k1)

        # Expansion: a hard positive can sit outside i's reciprocal set while being deep
        # inside one of its neighbours'. Each neighbour contributes its own (smaller)
        # reciprocal set, but only when it overlaps enough to be about the same identity —
        # the 2/3 test — which is what keeps the expansion from dragging in a whole
        # unrelated cluster.
        expanded = neighbours
        for candidate in neighbours:
            smaller = _k_reciprocal_neighbours(rank, int(candidate), round(k1 / 2))
            if len(np.intersect1d(smaller, neighbours)) > 2.0 / 3.0 * len(smaller):
                expanded = np.union1d(expanded, smaller)

        weight = np.exp(-normalised[i, expanded])
        V[i, expanded] = weight / weight.sum()

    if k2 > 1:
        # Local query expansion: replace each row by the mean of its k2 nearest rows, so a
        # single bad embedding is outvoted by its neighbourhood instead of deciding it.
        #
        # Accumulated in a loop rather than as `V[rank[:, :k2]].mean(axis=1)`. That
        # expression is shorter and allocates a `(total, k2, total)` intermediate — six
        # copies of the largest matrix in the function at the default k2, which is where
        # most of the 12x gap between the old guard's claim and reality came from. The loop
        # gathers into one reusable buffer instead, so the cost is two extra matrices
        # regardless of k2.
        accumulator = np.zeros_like(V)
        buffer = np.empty_like(V)
        for neighbour in range(k2):
            np.take(V, rank[:, neighbour], axis=0, out=buffer)
            accumulator += buffer
        accumulator /= k2
        V = accumulator  # noqa: N806
        del accumulator, buffer

    # Jaccard distance between the weighted neighbour sets. min/max over the sparse rows is
    # the set intersection and union generalised to weights.
    inverted: list[np.ndarray] = [np.flatnonzero(V[:, j] != 0) for j in range(total)]
    jaccard = np.ones((n_query, total), dtype=np.float32)
    for i in range(n_query):
        intersection = np.zeros(total, dtype=np.float32)
        nonzero = np.flatnonzero(V[i])
        for j in nonzero:
            rows = inverted[j]
            intersection[rows] += np.minimum(V[i, j], V[rows, j])
        jaccard[i] = 1.0 - intersection / (2.0 - intersection)

    # Mix against the *normalised* original, not the raw one: the Jaccard distance lives in
    # [0, 1] and lambda is meant to trade the two off evenly. Mixing a raw cosine distance
    # in [0, 2] against it would make lambda mean something different for every input.
    combined = (
        jaccard[:, n_query:] * (1.0 - lambda_value)
        + normalised[:n_query, n_query:] * lambda_value
    )
    return combined.astype(np.float32)
