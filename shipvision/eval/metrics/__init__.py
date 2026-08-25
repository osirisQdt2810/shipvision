"""The three MOT metrics, and the container that carries all of them at once.

They are separate modules because they are separate *matchings*, not because they are
separate arithmetic — see :mod:`shipvision.eval.association`. They are re-exported together
because no report should ever quote one without the others: CLEAR alone is dominated by the
detector, IDF1 alone says nothing about localisation, and HOTA alone hides which of detection
and association moved.
"""

from shipvision.eval.metrics.base import COMBINED, SequenceResult, combine
from shipvision.eval.metrics.clear import ClearCounts, clear_counts
from shipvision.eval.metrics.hota import ALPHAS, HotaCounts, hota_counts
from shipvision.eval.metrics.identity import IdentityCounts, identity_counts

__all__ = [
    "ALPHAS",
    "COMBINED",
    "ClearCounts",
    "HotaCounts",
    "IdentityCounts",
    "SequenceResult",
    "clear_counts",
    "combine",
    "hota_counts",
    "identity_counts",
]
