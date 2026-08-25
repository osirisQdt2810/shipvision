"""IDF1: one global matching between whole trajectories, not a count of agreeing frames.

Ristani et al., "Performance Measures and a Data Set for Multi-Target, Multi-Camera
Tracking", ECCV 2016 workshops. Written from the paper.

.. math::

    \\mathrm{IDF1} = \\frac{2\\,\\mathrm{IDTP}}{2\\,\\mathrm{IDTP} + \\mathrm{IDFP}
    + \\mathrm{IDFN}}

**The whole content of the metric is in what counts as an IDTP.** A ground-truth trajectory
is assigned to at most one predicted trajectory *for the entire sequence*, by a single
bipartite matching that maximises total overlap; IDTP is then the number of frames on which
that one pairing agrees. Everything else is a consequence: a tracker that splits an object
into two half-length tracks is credited for one half and charged for the other, which is the
behaviour the metric exists to produce.

**Getting this wrong is the commonest error in re-implementations, and it always inflates the
score.** The wrong version counts per-frame agreements — for each frame, how many
ground-truth boxes had *some* prediction on them at IoU 0.5 — which credits both halves of a
split track and therefore reports a tracker that switches identity on every frame as
perfect. There is a test for exactly that disagreement in
``tests/eval/test_identity.py::TestGlobalVersusPerFrameMatching``.

**The matching maximises co-occurrence, and unmatched trajectories are free to leave
unmatched.** Both sides may be left out: a spurious predicted track pairs with nothing and
contributes its whole length to IDFP. TrackEval encodes that by padding the cost matrix with
a diagonal of "matched to nobody" columns; the identity below is the same problem stated
directly, and the docstring of :func:`identity_counts` writes out why the two agree.

**Aggregation across sequences sums IDTP, IDFP and IDFN and divides once.** IDF1 is a ratio
of counts, so that is the only aggregation that means anything. Note what it does *not* do:
it does not re-run the global matching over the concatenated sequences. Trajectory identity
does not survive a sequence boundary — MOT17-09's person 3 is not MOT17-11's person 3 — so
the matching is per sequence and only the counts are pooled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from shipvision.eval.association import AlignedSequence

__all__ = ["IdentityCounts", "identity_counts"]


@dataclass(frozen=True, slots=True)
class IdentityCounts:
    """The three ID tallies. Additive across sequences; the scores are properties.

    Attributes:
        true_positives: frames on which the globally-matched pairing agreed (IDTP).
        false_positives: predicted detections not covered by the pairing (IDFP).
        false_negatives: ground-truth detections not covered by the pairing (IDFN).
    """

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    def __add__(self, other: IdentityCounts) -> IdentityCounts:
        if not isinstance(other, IdentityCounts):  # pragma: no cover - defensive
            return NotImplemented
        return IdentityCounts(
            true_positives=self.true_positives + other.true_positives,
            false_positives=self.false_positives + other.false_positives,
            false_negatives=self.false_negatives + other.false_negatives,
        )

    @property
    def idr(self) -> float:
        """Identity recall: of the ground-truth detections, how many were identified."""
        return self.true_positives / max(1, self.true_positives + self.false_negatives)

    @property
    def idp(self) -> float:
        """Identity precision."""
        return self.true_positives / max(1, self.true_positives + self.false_positives)

    @property
    def idf1(self) -> float:
        """The harmonic mean of :attr:`idp` and :attr:`idr`, written as a ratio of counts.

        Written this way rather than as ``2 * idp * idr / (idp + idr)`` so that a sequence
        with no ground truth and no predictions gives 0.0 rather than a division by zero,
        and so that the aggregate over sequences is a ratio of summed counts by construction.
        """
        denominator = 2 * self.true_positives + self.false_positives + self.false_negatives
        if denominator == 0:
            return 0.0
        return 2 * self.true_positives / denominator

    def __repr__(self) -> str:
        return (
            f"<IdentityCounts IDF1={self.idf1:.4f} IDP={self.idp:.4f} IDR={self.idr:.4f} "
            f"IDTP={self.true_positives}>"
        )


def identity_counts(aligned: AlignedSequence, *, threshold: float = 0.5) -> IdentityCounts:
    """One bipartite matching over the whole sequence, then three counts.

    Args:
        aligned: the two sequences, paired by frame id.
        threshold: IoU at which a box counts as covering a ground-truth box. Unlike HOTA this
            is a single hard threshold, which is the metric's definition and also its main
            weakness.

    **Why maximising co-occurrence is the same problem as TrackEval's padded matrix.** For a
    matched pair ``(i, j)`` with ``m`` co-occurring frames, the cost is
    ``(len(i) - m) + (len(j) - m)``; an unmatched trajectory costs its own length. Summing
    over any complete assignment gives ``total_gt + total_pred - 2 * sum(m)``, in which the
    first two terms are constants of the data. So minimising the cost and maximising
    ``sum(m)`` are the same optimisation, and the padding exists only to make the matrix
    square — which :func:`scipy.optimize.linear_sum_assignment` does not need, since it
    already solves the rectangular problem and a zero-overlap pair costs nothing to include.
    Half the matrix and none of the ``1e10`` sentinels.

    Note that ``co-occurrence`` is counted over *all* pairs above the threshold, not over a
    one-to-one per-frame matching: in a crowd one ground-truth box may be covered by two
    predictions at IoU 0.5, and both pairings are candidates for the global matching that
    then has to choose between them.
    """
    n_gt, n_pred = aligned.num_gt_ids, aligned.num_pred_ids
    total_gt = aligned.num_gt_dets
    total_pred = aligned.num_pred_dets
    if n_gt == 0 or n_pred == 0:
        return IdentityCounts(false_positives=total_pred, false_negatives=total_gt)

    co_occurrence = np.zeros((n_gt, n_pred), dtype=np.float64)
    for gt_ids, pred_ids, similarity in zip(
        aligned.gt_ids, aligned.pred_ids, aligned.similarity, strict=True
    ):
        if gt_ids.size == 0 or pred_ids.size == 0:
            continue
        rows, cols = np.nonzero(similarity >= threshold - np.finfo(np.float64).eps)
        # A plain fancy-indexed `+=` rather than `np.add.at`: within one frame both sides'
        # ids are unique, so `np.nonzero` cannot produce the same (gt, pred) pair twice and
        # there is no duplicate index for the buffered add to drop.
        co_occurrence[gt_ids[rows], pred_ids[cols]] += 1.0

    rows, cols = linear_sum_assignment(-co_occurrence)
    true_positives = int(co_occurrence[rows, cols].sum())
    return IdentityCounts(
        true_positives=true_positives,
        false_positives=total_pred - true_positives,
        false_negatives=total_gt - true_positives,
    )
