"""How two independent pieces of evidence combine: a veto, not a penalty.

One function, and it is the whole of what "gated" means. It is here rather than inline in the
matcher because the property it has to hold is easy to state and easy to break: a vetoed pair
must come out as *exactly* zero, because :meth:`~shipvision.mtmc.base.BaseMatcher.to_distance`
is what turns zero into ``NEVER_MERGE``. Scale the similarity down instead, or subtract a
penalty from it, and a pair the geometry ruled impossible is merely expensive — which average
linkage will happily buy the moment somebody loosens a threshold, and no test of either half
would notice.
"""

from __future__ import annotations

import numpy as np

from shipvision.errors import DimensionMismatchError

__all__ = ["veto"]


def veto(similarity: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """``(n, n)`` similarity with every pair the gate refuses set to exactly zero.

    Args:
        similarity: ``(n, n)`` evidence for a merge, higher meaning more alike, zero meaning
            "no evidence".
        allowed: ``(n, n)`` bool, true where the gate does not object. A gate that cannot
            judge a pair must pass true — falling open is what lets an uncalibrated camera
            take part on appearance alone.

    Raises:
        DimensionMismatchError: the two disagree on shape. Worth a check rather than trusting
            the caller: numpy would broadcast an ``(n,)`` or ``(1, n)`` gate against an
            ``(n, n)`` similarity without complaint, and the result is a matrix that is
            plausible, asymmetric in the wrong places, and wrong for every row but one.
    """
    if similarity.shape != allowed.shape:
        raise DimensionMismatchError(
            f"the gate is {allowed.shape} and the similarity it gates is {similarity.shape}; "
            f"both describe the same pairs of the same synchronised group"
        )
    return np.where(allowed, similarity, 0.0).astype(np.float32)
