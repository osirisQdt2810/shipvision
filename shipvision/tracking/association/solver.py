"""Turning a cost matrix into matches. One optimal solve, one threshold, three shapes of it.

:func:`scipy.optimize.linear_sum_assignment` solves the assignment optimally in O(n^3). There
is no reason to hand-roll a Hungarian implementation and every reason not to; what *is* worth
owning is everything around it, because that is where multi-stage trackers go wrong:

* the threshold belongs **after** the solve, not before — see :func:`associate`;
* a sub-problem's solution is in submatrix positions, and using those as track indices
  silently associates the wrong objects — see :func:`associate_subset`;
* a track last seen twenty frames ago should not outbid one last seen on the previous frame —
  see :func:`cascade_associate`.

Each of those is done once, here, so it cannot drift between one tracker's stages and
another's.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = ["associate", "associate_subset", "cascade_associate"]


def associate(
    cost: np.ndarray, max_cost: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Optimal one-to-one assignment, then a threshold.

    Returns ``(matches, unmatched_rows, unmatched_columns)``.

    The threshold is applied *after* the solve, not before. The solver optimises the total,
    so it will accept an expensive pair to enable two cheap ones — which is correct globally
    and wrong for that pair. Dropping the over-threshold matches afterwards keeps the global
    optimum where it helps and refuses the individual assignments that are not actually
    evidence.
    """
    rows, cols = cost.shape
    if rows == 0 or cols == 0:
        return [], list(range(rows)), list(range(cols))

    row_indices, col_indices = linear_sum_assignment(cost)

    matches: list[tuple[int, int]] = []
    matched_rows: set[int] = set()
    matched_cols: set[int] = set()
    for row, col in zip(row_indices, col_indices, strict=True):
        if cost[row, col] > max_cost:
            continue
        matches.append((int(row), int(col)))
        matched_rows.add(int(row))
        matched_cols.add(int(col))

    return (
        matches,
        [r for r in range(rows) if r not in matched_rows],
        [c for c in range(cols) if c not in matched_cols],
    )


def cascade_associate(
    build_cost: Callable[[Sequence[int], Sequence[int]], np.ndarray],
    max_cost: float,
    rows: Sequence[int],
    columns: Sequence[int],
    ages: np.ndarray,
    *,
    stride: int,
    max_depth: int,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """DeepSORT's matching cascade: recently-seen tracks choose first.

    A single global assignment treats a track last seen one frame ago and one last seen
    twenty frames ago as equally credible bidders for the same detection. They are not. The
    older track has a covariance the filter has been inflating for twenty frames, so its
    Mahalanobis gate is wide open and it will happily claim a detection that belongs to its
    neighbour. Solving in bands of increasing ``time_since_update`` gives the well-supported
    tracks first refusal, which is the whole reason DeepSORT keeps identities through crowds
    that plain SORT scrambles.

    ``stride`` widens each band. A stride of one is the original formulation and costs one
    solve per band; the internal reference uses five, which is a deliberate trade of a
    little precedence for a fifth of the solves, and at fifty cameras that is a real saving.

    Args:
        build_cost: called with ``(rows_subset, columns_subset)`` — both are indices into
            the *caller's* frame of reference, exactly as passed in — and returns the cost
            matrix for that sub-problem.
        max_cost: per-pair threshold, applied inside each band.
        rows: track indices to match.
        columns: detection indices to match.
        ages: ``time_since_update`` for every track, indexed by the values in ``rows``.
        stride: band width in frames.
        max_depth: bands stop once this age is reached.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if not rows or not columns:
        return [], list(rows), list(columns)

    matches: list[tuple[int, int]] = []
    remaining = list(columns)
    matched_rows: set[int] = set()

    for start in range(0, max(max_depth, 1), stride):
        if not remaining or len(matched_rows) == len(rows):
            break
        band = [r for r in rows if start <= ages[r] < start + stride]
        if not band:
            continue
        band_matches, _, remaining = associate_subset(build_cost, max_cost, band, remaining)
        matches.extend(band_matches)
        matched_rows.update(r for r, _ in band_matches)

    return matches, [r for r in rows if r not in matched_rows], remaining


def associate_subset(
    build_cost: Callable[[Sequence[int], Sequence[int]], np.ndarray],
    max_cost: float,
    rows: Sequence[int],
    columns: Sequence[int],
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """:func:`associate` on a sub-problem, translating positions back to caller indices.

    Every multi-stage tracker needs this and every one of them gets the translation wrong at
    least once: the solver returns positions within the submatrix, and using them as track
    indices silently associates the wrong objects. Doing it once, here, is the only version
    that cannot drift between stages.
    """
    if not rows or not columns:
        return [], list(rows), list(columns)
    cost = build_cost(rows, columns)
    matches, unmatched_rows, unmatched_cols = associate(cost, max_cost)
    return (
        [(rows[r], columns[c]) for r, c in matches],
        [rows[r] for r in unmatched_rows],
        [columns[c] for c in unmatched_cols],
    )
