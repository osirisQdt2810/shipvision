"""CLEAR: MOTA, MOTP, identity switches, and the mostly-tracked family.

Bernardin and Stiefelhagen, "Evaluating Multiple Object Tracking Performance: The CLEAR MOT
Metrics", 2008. Written from the paper.

.. math::

    \\mathrm{MOTA} = 1 - \\frac{\\mathrm{FN} + \\mathrm{FP} + \\mathrm{IDSW}}{\\mathrm{GT}}

Everything interesting about MOTA is in that fraction rather than in the arithmetic:

**The three error types are added, not weighted.** So MOTA is dominated by whichever is
largest, and on a public-detection benchmark that is almost always FN — which is a property
of the detector, not of the tracker. A tracker change moves IDSW by tens and FN by nothing,
so a *tracker* judged by MOTA is being judged mostly on somebody else's work. This is why the
tuning objective in :mod:`shipvision.tune` optimises HOTA by default.

**MOTA is unbounded below.** A tracker that emits three boxes per object scores -1: FP is
twice GT, so the fraction is 2 and MOTA is -1. Clamping it at zero would hide the difference
between "found nothing" and "flooded the scene", and those need different fixes. Nothing here
clamps.

**Identity switches depend entirely on the matcher.** A matcher that re-solves each frame
from scratch invents switches, and the invented ones outnumber the real ones in a crowd. See
:func:`shipvision.eval.association.match_preferring`.

**Aggregation across sequences sums FN, FP, IDSW and GT and divides once.** MOTA is a ratio
of counts, so that is the only aggregation that means anything; averaging per-sequence MOTA
weights a 525-frame sequence like a 1050-frame one. MOTP is the mean IoU of the matched pairs
and aggregates the same way, as a sum of overlaps over a sum of true positives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shipvision.eval.association import AlignedSequence, match_preferring

__all__ = ["ClearCounts", "clear_counts"]


@dataclass(frozen=True, slots=True)
class ClearCounts:
    """The CLEAR tallies. Everything reportable is a property over these.

    Attributes:
        true_positives: matched (ground truth, prediction) pairs, summed over frames.
        false_positives: predictions that matched nothing.
        false_negatives: ground-truth objects that matched nothing.
        id_switches: a ground-truth object matched to a different prediction id than the one
            it was last matched to — *ever*, not merely on the previous frame. Re-birthing a
            track after a twenty-frame gap is an identity switch; counting only the previous
            frame would make it free.
        overlap_sum: total IoU over the matched pairs, so MOTP is a ratio of sums.
        num_frames: frames in the sequence, used for the false-positive rate.
        num_gt_ids: distinct ground-truth trajectories, the denominator of the MT/ML rates.
        mostly_tracked: trajectories matched in more than 80% of the frames they appear in.
        partly_tracked: matched in at least 20% but not more than 80%.
        mostly_lost: the rest.
        fragmentations: how many times a trajectory's coverage resumed after a break.
    """

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    id_switches: int = 0
    overlap_sum: float = 0.0
    num_frames: int = 0
    num_gt_ids: int = 0
    mostly_tracked: int = 0
    partly_tracked: int = 0
    mostly_lost: int = 0
    fragmentations: int = 0

    def __add__(self, other: ClearCounts) -> ClearCounts:
        if not isinstance(other, ClearCounts):  # pragma: no cover - defensive
            return NotImplemented
        return ClearCounts(
            true_positives=self.true_positives + other.true_positives,
            false_positives=self.false_positives + other.false_positives,
            false_negatives=self.false_negatives + other.false_negatives,
            id_switches=self.id_switches + other.id_switches,
            overlap_sum=self.overlap_sum + other.overlap_sum,
            num_frames=self.num_frames + other.num_frames,
            num_gt_ids=self.num_gt_ids + other.num_gt_ids,
            mostly_tracked=self.mostly_tracked + other.mostly_tracked,
            partly_tracked=self.partly_tracked + other.partly_tracked,
            mostly_lost=self.mostly_lost + other.mostly_lost,
            fragmentations=self.fragmentations + other.fragmentations,
        )

    @property
    def num_gt_dets(self) -> int:
        """Ground-truth detections: every one is either a true positive or a miss."""
        return self.true_positives + self.false_negatives

    @property
    def num_pred_dets(self) -> int:
        return self.true_positives + self.false_positives

    @property
    def mota(self) -> float:
        """``1 - (FN + FP + IDSW) / GT``. Negative when the errors outnumber the objects.

        Zero when there is no ground truth *and* nothing was predicted, which is the only
        sensible answer to a question nobody asked — and it is a float, never a NaN, because
        a NaN propagates silently through an aggregate.
        """
        if self.num_gt_dets == 0:
            return 0.0 if self.false_positives == 0 else -float(self.false_positives)
        errors = self.false_negatives + self.false_positives + self.id_switches
        return 1.0 - errors / self.num_gt_dets

    @property
    def motp(self) -> float:
        """Mean IoU of the matched pairs — a *similarity*, as MOTChallenge reports it.

        The 2008 paper defines MOTP as a mean distance, where lower is better. Every leaderboard
        since reports the overlap, where higher is better. Reporting the overlap and saying so
        is less confusing than reporting a distance under a name everyone reads as a score.
        """
        if self.true_positives == 0:
            return 0.0
        return self.overlap_sum / self.true_positives

    @property
    def recall(self) -> float:
        return self.true_positives / max(1, self.num_gt_dets)

    @property
    def precision(self) -> float:
        return self.true_positives / max(1, self.num_pred_dets)

    @property
    def false_positives_per_frame(self) -> float:
        return self.false_positives / max(1, self.num_frames)

    def __repr__(self) -> str:
        return (
            f"<ClearCounts MOTA={self.mota:.4f} MOTP={self.motp:.4f} "
            f"TP={self.true_positives} FP={self.false_positives} "
            f"FN={self.false_negatives} IDSW={self.id_switches}>"
        )


def clear_counts(aligned: AlignedSequence, *, threshold: float = 0.5) -> ClearCounts:
    """Walk the sequence once, matching each frame with a bias towards the last one.

    Args:
        aligned: the two sequences, already paired by frame id.
        threshold: minimum IoU for a pair to count as a match. 0.5 is the benchmark's value
            and it is a cliff, not a slope: a box at 0.49 is a false positive *and* a miss.

    Two pieces of per-trajectory state, and the difference between them is the subtle part:

    ``last_matched`` remembers the prediction each ground-truth object was matched to the
    last time it was matched at all, however long ago. Identity switches are counted against
    it, so a track that dies and is re-born under a new id is charged for the switch.

    ``matched_previous_frame`` remembers only the frame just gone, and it is what biases the
    matching. A prediction whose box was good enough five frames ago says nothing about where
    the object is now, so it gets no preference — the preference exists to resolve a *current*
    ambiguity, not to reward history.

    One deliberate edge case: a frame in which the tracker published nothing does not clear
    the preference, so the next frame still prefers the mapping from before the gap. That is
    the conservative reading — it declines to invent a switch across a one-frame publication
    gap — and it is what TrackEval does, which is why the numbers here can be compared with a
    leaderboard's.
    """
    n_gt = aligned.num_gt_ids
    appearances = np.zeros(n_gt, dtype=np.int64)
    matches = np.zeros(n_gt, dtype=np.int64)
    segments = np.zeros(n_gt, dtype=np.int64)
    last_matched = np.full(n_gt, -1, dtype=np.int64)
    matched_previous_frame = np.full(n_gt, -1, dtype=np.int64)

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    id_switches = 0
    overlap_sum = 0.0

    for gt_ids, pred_ids, similarity in zip(
        aligned.gt_ids, aligned.pred_ids, aligned.similarity, strict=True
    ):
        if gt_ids.size == 0:
            false_positives += int(pred_ids.size)
            continue
        if pred_ids.size == 0:
            false_negatives += int(gt_ids.size)
            appearances[gt_ids] += 1
            continue

        # Translate "which id was this object matched to last frame" into "which column of
        # this frame's similarity matrix", which is what the matcher works in. An id that is
        # absent from this frame has no column and therefore no preference, which is correct:
        # a preference for a prediction that does not exist is not a preference.
        column_of = {int(pred_id): index for index, pred_id in enumerate(pred_ids)}
        preferred = np.array(
            [column_of.get(int(matched_previous_frame[gt_id]), -1) for gt_id in gt_ids],
            dtype=np.int64,
        )

        rows, cols = match_preferring(similarity, preferred, threshold=threshold)
        matched_gt = gt_ids[rows]
        matched_pred = pred_ids[cols]

        previous = last_matched[matched_gt]
        id_switches += int(np.count_nonzero((previous >= 0) & (previous != matched_pred)))

        was_tracked = matched_previous_frame >= 0
        appearances[gt_ids] += 1
        matches[matched_gt] += 1
        last_matched[matched_gt] = matched_pred
        matched_previous_frame[:] = -1
        matched_previous_frame[matched_gt] = matched_pred
        segments += (~was_tracked) & (matched_previous_frame >= 0)

        true_positives += int(rows.size)
        false_negatives += int(gt_ids.size - rows.size)
        false_positives += int(pred_ids.size - rows.size)
        if rows.size:
            overlap_sum += float(similarity[rows, cols].sum())

    seen = appearances > 0
    ratio = np.zeros(n_gt, dtype=np.float64)
    ratio[seen] = matches[seen] / appearances[seen]
    mostly_tracked = int(np.count_nonzero(ratio > 0.8))
    at_least_partly = int(np.count_nonzero(ratio >= 0.2))
    return ClearCounts(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        id_switches=id_switches,
        overlap_sum=overlap_sum,
        num_frames=aligned.num_frames,
        num_gt_ids=n_gt,
        mostly_tracked=mostly_tracked,
        partly_tracked=at_least_partly - mostly_tracked,
        mostly_lost=n_gt - at_least_partly,
        fragmentations=int(np.sum(np.clip(segments - 1, 0, None))),
    )
