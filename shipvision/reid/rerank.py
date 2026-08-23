"""k-reciprocal re-ranking (Zhong et al., CVPR 2017).

The single largest accuracy gain available without touching the model, and the reason is
worth stating because it decides when to use it. A raw similarity asks "does the query look
like this gallery entry?". Re-ranking also asks "do the query and that entry have the same
*neighbours*?" — and appearance under a different viewpoint is far less stable than the
company an identity keeps. A ship photographed from the stern matches its own bow view
poorly and matches the bow view's neighbourhood well.

**It is a batch operation, not a query-time one.** The cost is O((n_query + n_gallery)^2) in
both time and memory: 10 000 combined entries is a 400 MB float32 matrix. Use it to
evaluate a change offline, or on a bounded candidate set retrieved first by plain cosine —
never on the whole gallery inside a frame budget. :func:`rerank` refuses sizes it cannot do
justice to rather than quietly allocating them.
"""

from __future__ import annotations

import numpy as np

from shipvision.errors import ConfigurationError

__all__ = ["rerank"]


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
    if total > max_entries:
        raise ConfigurationError(
            f"re-ranking {n_query} queries against {n_gallery} entries needs a "
            f"{total}x{total} matrix ({total * total * 4 / 1e9:.1f} GB); raise max_entries "
            f"if that is genuinely intended, or retrieve a candidate set first"
        )
    if k1 >= total:
        raise ConfigurationError(f"k1={k1} needs more than {total} entries to be meaningful")

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
    rank = np.argsort(normalised, axis=1, kind="stable")

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
        V = V[rank[:, :k2]].mean(axis=1)  # noqa: N806

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
