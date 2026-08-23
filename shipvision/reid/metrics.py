"""Ranking metrics: CMC and mAP, under the standard evaluation protocol.

These decide whether a change to the embedder, the aggregator or the gallery actually
helped. That makes the *protocol* the load-bearing part, more than the arithmetic:

**A gallery entry with the query's identity AND the query's camera is discarded, not
counted as wrong.** It is neither a hit nor a miss — it is removed from the ranking
entirely, and every rank below it moves up. Matching yourself in your own camera measures
tracking, not re-identification. Counting such a pair as correct is how an implementation
reports rank-1 in the high nineties and fails in the field; dropping it as an error is
equally wrong in the other direction, and would punish a model for a match nobody asked it
to make.

**A query with no valid ground truth left after that filter is skipped**, and skipped
queries are excluded from the denominator. Averaging in a zero for a query that was
unanswerable would report the composition of the test set as if it were the model's.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from shipvision.errors import ConfigurationError

__all__ = ["RankingResult", "cmc_curve", "evaluate_ranking", "mean_average_precision"]


@dataclass(slots=True, frozen=True)
class RankingResult:
    """What an evaluation run produced.

    `cmc` is indexed from 0, so rank-1 accuracy is ``cmc[0]``. `evaluated` and `skipped`
    are reported rather than hidden: a run where half the queries were skipped is a
    statement about the test set, and it should be visible next to the score it produced.
    """

    cmc: np.ndarray
    mean_ap: float
    evaluated: int
    skipped: int

    def rank(self, k: int) -> float:
        """Rank-k accuracy, 1-indexed as everyone writes it (``rank(1)`` is rank-1)."""
        if k < 1:
            raise ConfigurationError(f"rank is 1-indexed; got {k}")
        return float(self.cmc[min(k, len(self.cmc)) - 1])

    def __repr__(self) -> str:
        return (
            f"<RankingResult rank1={self.rank(1):.4f} rank5={self.rank(5):.4f} "
            f"mAP={self.mean_ap:.4f} n={self.evaluated} skipped={self.skipped}>"
        )


def _valid_mask(
    query_identity: str,
    query_camera: str | None,
    gallery_identities: np.ndarray,
    gallery_cameras: np.ndarray | None,
) -> np.ndarray:
    """Which gallery entries this query may be scored against."""
    if gallery_cameras is None or query_camera is None:
        return np.ones(len(gallery_identities), dtype=bool)
    return ~((gallery_identities == query_identity) & (gallery_cameras == query_camera))


def evaluate_ranking(
    similarity: np.ndarray,
    query_identities: Sequence[str],
    gallery_identities: Sequence[str],
    *,
    query_cameras: Sequence[str | None] | None = None,
    gallery_cameras: Sequence[str | None] | None = None,
    max_rank: int = 50,
) -> RankingResult:
    """CMC and mAP from a precomputed similarity matrix.

    Args:
        similarity: ``(n_query, n_gallery)``, higher is more alike. Cosine similarity, not
            a distance — passing a distance silently inverts every ranking, so the argument
            is named for what it must be.
        max_rank: how far the CMC curve extends. Clamped to the gallery size.

    Both metrics come out of one ranking pass because they read the same sorted labels: CMC
    asks where the *first* hit landed, mAP asks how the hits are distributed over the whole
    ranking. A model can win one and lose the other — an identity with one easy view and
    nine hard ones scores rank-1 = 1 and a poor AP — which is why reporting only rank-1 is
    how a regression in the hard cases goes unnoticed.
    """
    scores = np.asarray(similarity, dtype=np.float32)
    if scores.ndim != 2:
        raise ConfigurationError(f"similarity must be 2-D, got shape {scores.shape}")
    q_ids = np.asarray(query_identities, dtype=object)
    g_ids = np.asarray(gallery_identities, dtype=object)
    if scores.shape != (len(q_ids), len(g_ids)):
        raise ConfigurationError(
            f"similarity is {scores.shape} but there are {len(q_ids)} queries and "
            f"{len(g_ids)} gallery entries"
        )
    if max_rank < 1:
        raise ConfigurationError(f"max_rank must be positive, got {max_rank}")

    q_cams = np.asarray(query_cameras, dtype=object) if query_cameras is not None else None
    g_cams = np.asarray(gallery_cameras, dtype=object) if gallery_cameras is not None else None
    if q_cams is not None and len(q_cams) != len(q_ids):
        raise ConfigurationError("query_cameras must be one per query")
    if g_cams is not None and len(g_cams) != len(g_ids):
        raise ConfigurationError("gallery_cameras must be one per gallery entry")

    rank_width = min(max_rank, len(g_ids))
    hits = np.zeros(rank_width, dtype=np.float64)
    average_precisions: list[float] = []
    skipped = 0

    for i in range(len(q_ids)):
        valid = _valid_mask(q_ids[i], None if q_cams is None else q_cams[i], g_ids, g_cams)
        # Sort the whole row once, then drop the filtered entries. Filtering first and
        # sorting the remainder gives the same order, but this way the ranks that survive
        # are already the *post-filter* ranks — which is exactly the protocol's "everything
        # below a discarded entry moves up".
        order = np.argsort(-scores[i], kind="stable")
        order = order[valid[order]]
        if order.size == 0:
            skipped += 1
            continue

        correct = g_ids[order] == q_ids[i]
        if not correct.any():
            # No positive to find. Not a failure of the model — the protocol has nothing to
            # ask it — so it does not enter either average.
            skipped += 1
            continue

        first = int(np.argmax(correct))
        if first < rank_width:
            hits[first:] += 1.0

        # AP for one query: mean precision at each rank where a positive appears.
        positions = np.flatnonzero(correct)
        precision_at_hit = (np.arange(len(positions)) + 1.0) / (positions + 1.0)
        average_precisions.append(float(precision_at_hit.mean()))

    evaluated = len(q_ids) - skipped
    if evaluated == 0:
        return RankingResult(
            cmc=np.zeros(rank_width), mean_ap=0.0, evaluated=0, skipped=skipped
        )
    return RankingResult(
        cmc=hits / evaluated,
        mean_ap=float(np.mean(average_precisions)),
        evaluated=evaluated,
        skipped=skipped,
    )


def cmc_curve(*args: object, **kwargs: object) -> np.ndarray:
    """Just the CMC curve. See :func:`evaluate_ranking`."""
    return evaluate_ranking(*args, **kwargs).cmc  # type: ignore[arg-type]


def mean_average_precision(*args: object, **kwargs: object) -> float:
    """Just the mAP. See :func:`evaluate_ranking`."""
    return evaluate_ranking(*args, **kwargs).mean_ap  # type: ignore[arg-type]
