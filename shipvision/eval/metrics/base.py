"""What a metric returns, and the one rule about putting two of them together.

Every metric here is a small bag of **counts** plus properties that divide them. That is not
a style choice, it is the only shape that aggregates correctly.

**Aggregation across sequences sums the counts and divides once. It never averages the
scores.** MOT17-05 has 837 frames and 8.3 people in each; MOT17-04 has 1050 frames and 45.3.
Averaging their MOTA weights a fifth of the objects like four fifths of them, and the
resulting number describes a benchmark nobody ran. So :class:`SequenceResult` adds, and the
scores are properties computed after the addition — which makes the wrong thing hard to write
rather than merely discouraged.

The one metric that cannot be expressed as a ratio of summed counts is HOTA's association
term, because it is an average over *pairs of identities* rather than over detections. It is
stored pre-multiplied by the detection count it is an average over, so summing the stored
value and dividing by the summed count reproduces the detection-weighted average exactly.
See :class:`~shipvision.eval.metrics.hota.HotaCounts`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from shipvision.errors import ConfigurationError
from shipvision.eval.metrics.clear import ClearCounts
from shipvision.eval.metrics.hota import HotaCounts
from shipvision.eval.metrics.identity import IdentityCounts

__all__ = ["SequenceResult", "combine"]

COMBINED = "COMBINED"
"""The name an aggregate carries, so a table row cannot be mistaken for a sequence."""


@dataclass(frozen=True, slots=True)
class SequenceResult:
    """Every metric for one sequence, plus what it cost to produce.

    ``seconds`` is the wall-clock time inside the tracker, excluding decoding and file IO.
    It belongs next to the quality numbers because the target is 1000 frames per second
    across fifty cameras: a tracker that wins HOTA by a point and costs 4 ms a frame has not
    won anything, and separating the two measurements into two reports is how that trade goes
    unnoticed.
    """

    name: str
    num_frames: int
    num_gt_dets: int
    num_gt_ids: int
    num_pred_dets: int
    num_pred_ids: int
    clear: ClearCounts
    identity: IdentityCounts
    hota: HotaCounts
    seconds: float = 0.0

    def __add__(self, other: SequenceResult) -> SequenceResult:
        """Sum the counts. The result is named :data:`COMBINED`, never after a sequence."""
        if not isinstance(other, SequenceResult):  # pragma: no cover - defensive
            return NotImplemented
        return SequenceResult(
            name=COMBINED,
            num_frames=self.num_frames + other.num_frames,
            num_gt_dets=self.num_gt_dets + other.num_gt_dets,
            num_gt_ids=self.num_gt_ids + other.num_gt_ids,
            num_pred_dets=self.num_pred_dets + other.num_pred_dets,
            num_pred_ids=self.num_pred_ids + other.num_pred_ids,
            clear=self.clear + other.clear,
            identity=self.identity + other.identity,
            hota=self.hota + other.hota,
            seconds=self.seconds + other.seconds,
        )

    @property
    def ms_per_frame(self) -> float:
        """Milliseconds of tracker time per frame. ``0`` when nothing was timed."""
        if self.num_frames == 0:
            return 0.0
        return 1000.0 * self.seconds / self.num_frames

    def scores(self) -> dict[str, float]:
        """The flat view a table row and a tuning objective both want.

        One dictionary rather than three so that ``--metric`` in a study can be a string
        that means the same thing everywhere. A typo'd metric name then fails on lookup
        instead of quietly optimising something else.
        """
        return {
            "HOTA": self.hota.hota,
            "DetA": self.hota.det_a,
            "AssA": self.hota.ass_a,
            "AssRe": self.hota.ass_re,
            "AssPr": self.hota.ass_pr,
            "LocA": self.hota.loc_a,
            "IDF1": self.identity.idf1,
            "IDP": self.identity.idp,
            "IDR": self.identity.idr,
            "MOTA": self.clear.mota,
            "MOTP": self.clear.motp,
            "IDSW": float(self.clear.id_switches),
            "FP": float(self.clear.false_positives),
            "FN": float(self.clear.false_negatives),
            "MT": float(self.clear.mostly_tracked),
            "ML": float(self.clear.mostly_lost),
            "Frag": float(self.clear.fragmentations),
            "ms_per_frame": self.ms_per_frame,
        }

    def score(self, metric: str) -> float:
        """One named score. Raises rather than defaulting on an unknown name."""
        scores = self.scores()
        if metric not in scores:
            raise ConfigurationError(f"unknown metric {metric!r}; available: {sorted(scores)}")
        return scores[metric]

    def renamed(self, name: str) -> SequenceResult:
        return replace(self, name=name)

    def __repr__(self) -> str:
        return (
            f"<SequenceResult {self.name} HOTA={self.hota.hota:.4f} "
            f"IDF1={self.identity.idf1:.4f} MOTA={self.clear.mota:.4f} "
            f"IDSW={self.clear.id_switches}>"
        )


def combine(results: list[SequenceResult], *, name: str = COMBINED) -> SequenceResult:
    """Aggregate by summing counts, not by averaging scores. See the module docstring."""
    if not results:
        raise ConfigurationError("nothing to combine; an empty run is not a score of zero")
    total = results[0]
    for result in results[1:]:
        total = total + result
    return total.renamed(name)
