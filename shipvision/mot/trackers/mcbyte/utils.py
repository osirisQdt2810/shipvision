# Ported from roboflow/trackers (Apache-2.0), src/trackers/core/mcbyte/mask_association.py,
# commit ced34f04886da91dc6bec3dfe02f0a0427231ce8. Changed: similarity space -> this library's
# cost space, and the mask conditioning the reference fuses in is a separate concern here.
"""McByte's pre-assignment bookkeeping: what is already decided, and what is still in doubt.

The idea the paper is built on is that a Hungarian solve optimises a *total*, so it will trade
away an obvious pair to buy two cheap ones — and after the threshold both of those are thrown
out anyway, leaving the obvious pair unmatched as well. A pair that is the only affordable
candidate in both its row and its column has no such trade to make, so it is locked first and
its row and column leave the problem.

Everything here is in **cost** space, minimising, where the reference maximises a similarity:
``cost = 1 - similarity`` and ``max_cost = 1 - minimum_similarity``, so ``sim >= t`` becomes
``cost <= max_cost`` and "positive IoU" becomes ``iou_cost < 1``. The sign is the whole risk in
this file, which is why every test uses an asymmetric matrix.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = [
    "ambiguous_candidates",
    "clear_matches",
    "isolated_candidates",
    "reduce_problem",
]


def clear_matches(cost: np.ndarray, max_cost: float) -> list[tuple[int, int]]:
    """``(row, column)`` pairs that are the only affordable candidate on both sides.

    Eligibility is read off the cost matrix and nothing else, so a pair a gate forbade —
    :data:`~shipvision.mot.association.costs.INFEASIBLE` — can never lock. That is the point:
    the gate said impossible, and locking would make it certain.

    Args:
        cost: ``(n, m)`` association cost. Lower is better.
        max_cost: the stage's threshold. A pair above it is not evidence.
    """
    eligible = np.asarray(cost) <= max_cost
    rows, columns = np.where(
        eligible & (eligible.sum(axis=1)[:, None] == 1) & (eligible.sum(axis=0)[None, :] == 1)
    )
    return list(zip(rows.tolist(), columns.tolist(), strict=True))


def ambiguous_candidates(cost: np.ndarray, max_cost: float) -> np.ndarray:
    """``(n, m)`` bool: affordable pairs whose row or column has a rival.

    Computed from the untouched cost, before anything is locked or conditioned. Ambiguity is a
    property of the situation the frame presented, and recomputing it on a matrix that has
    already been boosted would let the boost justify itself.
    """
    eligible = np.asarray(cost) <= max_cost
    rivals_in_row = eligible.sum(axis=1) > 1
    rivals_in_column = eligible.sum(axis=0) > 1
    return eligible & (rivals_in_row[:, None] | rivals_in_column[None, :])


def isolated_candidates(iou_cost: np.ndarray, max_cost: float) -> np.ndarray:
    """``(n, m)`` bool: the one pair that overlaps at all, but not enough to match.

    Read off **raw** ``1 - IoU``, never off a fused cost. Fusing moves a pair for reasons that
    are not geometry, and the claim here is "these two boxes touch and neither touches
    anything else", which is a fact about pixels.

    Args:
        iou_cost: ``(n, m)`` raw ``1 - IoU``. ``< 1`` means the boxes overlap.
        max_cost: the stage's threshold; a pair above it did not match on geometry alone.
    """
    iou_cost = np.asarray(iou_cost)
    overlapping = iou_cost < 1.0
    return (
        overlapping
        & (iou_cost > max_cost)
        & (overlapping.sum(axis=1) == 1)[:, None]
        & (overlapping.sum(axis=0) == 1)[None, :]
    )


def reduce_problem(
    cost: np.ndarray, locked: Sequence[tuple[int, int]]
) -> tuple[np.ndarray, list[int], list[int]]:
    """Drop the locked rows and columns, and hand back the map needed to undo that.

    Returns ``(reduced, kept_rows, kept_columns)``. The two index lists are the whole point:
    the solver's answer is in submatrix positions, and using those as track indices associates
    the wrong objects while every shape still agrees — the same trap
    :func:`~shipvision.mot.association.solver.associate_subset` exists to close one level up.
    """
    taken_rows = {row for row, _ in locked}
    taken_columns = {column for _, column in locked}
    kept_rows = [row for row in range(cost.shape[0]) if row not in taken_rows]
    kept_columns = [column for column in range(cost.shape[1]) if column not in taken_columns]
    return np.asarray(cost)[np.ix_(kept_rows, kept_columns)].copy(), kept_rows, kept_columns
