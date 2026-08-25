"""HOTA: Higher Order Tracking Accuracy, and the sub-scores it hides.

Luiten et al., "HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking",
IJCV 2021. Written from the paper, cross-checked against TrackEval's arithmetic.

.. math::

    \\mathrm{HOTA}_\\alpha = \\sqrt{\\mathrm{DetA}_\\alpha \\cdot \\mathrm{AssA}_\\alpha},
    \\qquad
    \\mathrm{HOTA} = \\frac{1}{|A|}\\sum_{\\alpha \\in A} \\mathrm{HOTA}_\\alpha

Four properties of that definition decide how this module is written:

**The geometric mean is the point.** A metric that added detection and association accuracy
could be maximised by being excellent at one and hopeless at the other; a product cannot. So
a tracker cannot buy association quality with detections it never found, which is exactly the
trade MOTA permits and the reason HOTA is the default objective in :mod:`shipvision.tune`.

**It is averaged over nineteen localisation thresholds**, :math:`\\alpha \\in \\{0.05, 0.10,
\\ldots, 0.95\\}`, so a single IoU cliff cannot decide a comparison. CLEAR at 0.5 turns a box
at IoU 0.49 into *two* errors and a box at 0.51 into a success; HOTA charges that box at the
strict thresholds and credits it at the loose ones.

**The averaging is of the geometric means, not a geometric mean of averages.** Those differ,
and reporting the second under the first's name is the commonest way a re-implementation comes
out flattering: :math:`\\overline{\\sqrt{xy}} \\le \\sqrt{\\bar{x}\\bar{y}}` by Cauchy-Schwarz,
so the wrong order is never lower and usually higher.

**The per-frame matching is scored by a sequence-wide alignment.** Every candidate pair is
weighted by how much the *whole* sequence agrees that this ground-truth trajectory and this
predicted trajectory are the same object, so a locally ambiguous frame resolves the way the
rest of the sequence says it should rather than by a hair of IoU. This is what makes HOTA
"higher order": no purely per-frame matcher can express it.

**Report DetA, AssA and LocA next to HOTA, always.** A change that trades detection for
association moves the two sub-scores in opposite directions and leaves the geometric mean
almost unmoved — which is a real and common outcome of tuning an association threshold, and
invisible in the combined number.

Aggregation across sequences sums TP, FN and FP per :math:`\\alpha` and takes a
TP-**weighted** mean of the association terms, then recomputes DetA and HOTA from the sums.
It never averages per-sequence HOTA: that would weight MOT17-09's 525 frames like
MOT17-04's 1050. See :class:`HotaCounts`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from shipvision.errors import ConfigurationError
from shipvision.eval.association import AlignedSequence

__all__ = ["ALPHAS", "HotaCounts", "hota_counts"]

#: The localisation thresholds HOTA is averaged over: 0.05 to 0.95 in steps of 0.05.
#:
#: Written as ``arange(0.05, 0.99, 0.05)`` rather than ``arange(0.05, 1.0, 0.05)`` because
#: floating-point accumulation in ``arange`` can put the last element a hair under 1.0 and
#: produce a twentieth threshold on some numpy builds. A silently different number of
#: thresholds changes every HOTA this package reports by a fraction of a point, which is
#: exactly the size of difference a tuning study is asked to resolve.
ALPHAS = np.arange(0.05, 0.99, 0.05)

#: Denominator floor. TrackEval's value, kept identical so the two agree bit-for-bit on the
#: degenerate case: with no true positives at all, LocA comes out as 1.0 rather than 0/0.
_TINY = 1e-10


def _zeros() -> np.ndarray:
    return np.zeros(len(ALPHAS), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class HotaCounts:
    """Per-threshold tallies. Every reported score is a property over these.

    All seven arrays have one entry per :data:`ALPHAS` threshold, and all seven are
    **additive across sequences** — which is the whole reason the association terms are
    stored pre-multiplied by the true-positive count they are an average over.

    The association scores cannot be written as a ratio of summed detection counts: AssA is
    an average over *pairs of trajectories*, weighted by how many detections each pair
    accounts for. Storing ``AssA * TP`` and dividing by the summed TP after addition
    reproduces the detection-weighted average exactly, and makes averaging-of-scores
    unrepresentable rather than merely discouraged.

    Attributes:
        true_positives: matched pairs at each threshold.
        false_negatives: unmatched ground-truth detections at each threshold.
        false_positives: unmatched predictions at each threshold.
        association: :math:`\\sum_c |\\mathrm{TPA}(c)| \\cdot A(c)`, the numerator of AssA.
        association_recall: the same sum for AssRe.
        association_precision: the same sum for AssPr.
        overlap: total IoU over the matched pairs, the numerator of LocA.
    """

    true_positives: np.ndarray = field(default_factory=_zeros)
    false_negatives: np.ndarray = field(default_factory=_zeros)
    false_positives: np.ndarray = field(default_factory=_zeros)
    association: np.ndarray = field(default_factory=_zeros)
    association_recall: np.ndarray = field(default_factory=_zeros)
    association_precision: np.ndarray = field(default_factory=_zeros)
    overlap: np.ndarray = field(default_factory=_zeros)

    def __post_init__(self) -> None:
        for name in (
            "true_positives",
            "false_negatives",
            "false_positives",
            "association",
            "association_recall",
            "association_precision",
            "overlap",
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64).reshape(-1)
            if value.shape[0] != len(ALPHAS):
                raise ConfigurationError(
                    f"{name} has {value.shape[0]} entries but HOTA is averaged over "
                    f"{len(ALPHAS)} thresholds; a mismatched array would silently change "
                    f"the score by however many thresholds it is missing"
                )
            object.__setattr__(self, name, value)

    def __add__(self, other: HotaCounts) -> HotaCounts:
        if not isinstance(other, HotaCounts):  # pragma: no cover - defensive
            return NotImplemented
        return HotaCounts(
            true_positives=self.true_positives + other.true_positives,
            false_negatives=self.false_negatives + other.false_negatives,
            false_positives=self.false_positives + other.false_positives,
            association=self.association + other.association,
            association_recall=self.association_recall + other.association_recall,
            association_precision=self.association_precision + other.association_precision,
            overlap=self.overlap + other.overlap,
        )

    # -- per-threshold curves ------------------------------------------------------------

    @property
    def det_a_curve(self) -> np.ndarray:
        """``TP / (TP + FN + FP)`` at each threshold."""
        total = self.true_positives + self.false_negatives + self.false_positives
        return self.true_positives / np.maximum(1.0, total)

    @property
    def ass_a_curve(self) -> np.ndarray:
        return self.association / np.maximum(1.0, self.true_positives)

    @property
    def loc_a_curve(self) -> np.ndarray:
        """Mean IoU of the matched pairs. 1.0 when there are no matches at all.

        The degenerate value is TrackEval's, not an accident: a tracker with no true
        positives has no localisation error to report, and 1.0 keeps the array free of NaN
        so that summing across sequences cannot poison the aggregate.
        """
        return np.maximum(_TINY, self.overlap) / np.maximum(_TINY, self.true_positives)

    @property
    def hota_curve(self) -> np.ndarray:
        return np.sqrt(self.det_a_curve * self.ass_a_curve)

    # -- reported scores -----------------------------------------------------------------

    @property
    def hota(self) -> float:
        """The mean over thresholds of the per-threshold geometric mean."""
        return float(self.hota_curve.mean())

    @property
    def det_a(self) -> float:
        return float(self.det_a_curve.mean())

    @property
    def ass_a(self) -> float:
        return float(self.ass_a_curve.mean())

    @property
    def det_re(self) -> float:
        recall = self.true_positives / np.maximum(
            1.0, self.true_positives + self.false_negatives
        )
        return float(recall.mean())

    @property
    def det_pr(self) -> float:
        precision = self.true_positives / np.maximum(
            1.0, self.true_positives + self.false_positives
        )
        return float(precision.mean())

    @property
    def ass_re(self) -> float:
        return float((self.association_recall / np.maximum(1.0, self.true_positives)).mean())

    @property
    def ass_pr(self) -> float:
        return float((self.association_precision / np.maximum(1.0, self.true_positives)).mean())

    @property
    def loc_a(self) -> float:
        return float(self.loc_a_curve.mean())

    def at(self, alpha: float) -> dict[str, float]:
        """The scores at one threshold, for a report that wants the 0.5 column too."""
        index = int(np.argmin(np.abs(ALPHAS - alpha)))
        return {
            "alpha": float(ALPHAS[index]),
            "HOTA": float(self.hota_curve[index]),
            "DetA": float(self.det_a_curve[index]),
            "AssA": float(self.ass_a_curve[index]),
            "LocA": float(self.loc_a_curve[index]),
            "TP": float(self.true_positives[index]),
            "FN": float(self.false_negatives[index]),
            "FP": float(self.false_positives[index]),
        }

    def __repr__(self) -> str:
        return (
            f"<HotaCounts HOTA={self.hota:.4f} DetA={self.det_a:.4f} "
            f"AssA={self.ass_a:.4f} LocA={self.loc_a:.4f}>"
        )


def _global_alignment(aligned: AlignedSequence) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """How much the whole sequence agrees that each ``(gt id, pred id)`` pair is one object.

    Returns ``(alignment, gt_counts, pred_counts)`` where ``alignment`` is
    ``(num_gt_ids, num_pred_ids)`` in [0, 1].

    The pair count is accumulated in *IoU-normalised* units rather than as a plain frame
    count: a frame contributes the pair's overlap divided by the total overlap that either
    side has in that frame, so one ground-truth object surrounded by five overlapping
    predictions cannot contribute five whole votes. Then the count is turned into a Jaccard
    score over the two trajectories' lengths, so a pair that co-occurs on ten of ten frames
    beats a pair that co-occurs on ten of a thousand.
    """
    n_gt, n_pred = aligned.num_gt_ids, aligned.num_pred_ids
    potential = np.zeros((n_gt, n_pred), dtype=np.float64)
    gt_counts = np.zeros((n_gt, 1), dtype=np.float64)
    pred_counts = np.zeros((1, n_pred), dtype=np.float64)

    for gt_ids, pred_ids, similarity in zip(
        aligned.gt_ids, aligned.pred_ids, aligned.similarity, strict=True
    ):
        gt_counts[gt_ids] += 1
        pred_counts[0, pred_ids] += 1
        if gt_ids.size == 0 or pred_ids.size == 0:
            continue
        sim = similarity.astype(np.float64)
        denominator = sim.sum(0)[None, :] + sim.sum(1)[:, None] - sim
        share = np.zeros_like(sim)
        usable = denominator > np.finfo(np.float64).eps
        share[usable] = sim[usable] / denominator[usable]
        # Ids are unique within a frame (ObjectFrame enforces it), so this fancy-indexed
        # `+=` cannot silently drop a duplicate index the way numpy would otherwise.
        potential[gt_ids[:, None], pred_ids[None, :]] += share

    alignment = potential / np.maximum(_TINY, gt_counts + pred_counts - potential)
    return alignment, gt_counts, pred_counts


def hota_counts(aligned: AlignedSequence) -> HotaCounts:
    """Score one aligned sequence at all nineteen thresholds in a single pass.

    Two passes over the frames, and they are not interchangeable. The first builds the
    sequence-wide alignment; the second matches each frame with the alignment as its
    tie-breaker. Doing it in one pass would mean matching frame 1 with an alignment derived
    from frame 1 alone, which is a per-frame matcher wearing HOTA's name.

    Within the second pass the *matching* is solved once — on ``alignment * IoU``, which is
    threshold-independent — and only the *acceptance* varies with :math:`\\alpha`. That is
    the paper's definition and not merely an optimisation: solving a separate assignment per
    threshold would let the matching itself jump between thresholds, making the HOTA curve
    non-monotonic in ways that say nothing about the tracker.
    """
    alignment, gt_counts, pred_counts = _global_alignment(aligned)
    n_alpha = len(ALPHAS)

    true_positives = _zeros()
    false_negatives = _zeros()
    false_positives = _zeros()
    overlap = _zeros()
    matched_pairs = [
        np.zeros((aligned.num_gt_ids, aligned.num_pred_ids), dtype=np.float64)
        for _ in range(n_alpha)
    ]

    for gt_ids, pred_ids, similarity in zip(
        aligned.gt_ids, aligned.pred_ids, aligned.similarity, strict=True
    ):
        if gt_ids.size == 0:
            false_positives += pred_ids.size
            continue
        if pred_ids.size == 0:
            false_negatives += gt_ids.size
            continue

        sim = similarity.astype(np.float64)
        score = alignment[gt_ids[:, None], pred_ids[None, :]] * sim
        rows, cols = linear_sum_assignment(-score)
        pair_similarity = sim[rows, cols]

        for index, alpha in enumerate(ALPHAS):
            accepted = pair_similarity >= alpha - np.finfo(np.float64).eps
            kept_rows, kept_cols = rows[accepted], cols[accepted]
            matches = int(kept_rows.size)
            true_positives[index] += matches
            false_negatives[index] += gt_ids.size - matches
            false_positives[index] += pred_ids.size - matches
            if matches:
                overlap[index] += float(pair_similarity[accepted].sum())
                matched_pairs[index][gt_ids[kept_rows], pred_ids[kept_cols]] += 1

    association = _zeros()
    association_recall = _zeros()
    association_precision = _zeros()
    for index in range(n_alpha):
        counts = matched_pairs[index]
        union = np.maximum(1.0, gt_counts + pred_counts - counts)
        association[index] = float(np.sum(counts * (counts / union)))
        association_recall[index] = float(
            np.sum(counts * (counts / np.maximum(1.0, gt_counts)))
        )
        association_precision[index] = float(
            np.sum(counts * (counts / np.maximum(1.0, pred_counts)))
        )

    return HotaCounts(
        true_positives=true_positives,
        false_negatives=false_negatives,
        false_positives=false_positives,
        association=association,
        association_recall=association_recall,
        association_precision=association_precision,
        overlap=overlap,
    )
