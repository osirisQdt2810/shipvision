"""Non-maximum suppression: five methods, one set of rules, one implementation each.

Suppression is the one operation in this package whose *answer* depends on decisions no
library makes for you — whether the overlap test is strict, what a decayed score means, how a
tie breaks — and there are five methods to keep consistent across three backends. Fifteen
copies of those decisions is fifteen chances to drift, so there is one copy, here, and the
backends call it. A backend accelerates the ``"classic"`` path with a device kernel or with
``torchvision.ops.nms`` because that path is O(n^2) overlap tests; the rest is a sequential
scalar loop over a few dozen survivors, which is not work a GPU wants.

The rules, stated once:

* **Admission** is ``score >= score_threshold``, inclusive — see
  :func:`~shipvision.imgproc.nms.candidates.prepare`, which also fixes the tie order.
* **Overlap** punishes when ``iou > iou_threshold``, strictly, for ``classic`` and
  ``linear``. A threshold of 1.0 is therefore a documented no-op for those two rather than a
  duplicate remover, and two boxes at exactly the threshold both survive. The CUDA kernel,
  ``torchvision.ops.nms`` and the C++ reference all agree on this one.
  **``gauss`` is not gated at all.** Soft-NMS's Eq. (4) — ``s_i <- s_i * exp(-iou^2 / sigma)``
  for every box not yet kept — has no threshold in it, and removing the discontinuity at the
  threshold is the paper's stated reason for preferring it over the linear rule. Under
  ``gauss`` the threshold is ignored and every live candidate is decayed by its overlap with
  the box just kept; a threshold of 1.0 still decays. This used to be gated like the others,
  which made it ``linear`` with a different curve — see the note in
  :func:`~shipvision.imgproc.nms.greedy.greedy`.
* **Departure**: a candidate leaves the pool when its decayed score drops below
  ``score_threshold`` or reaches zero — see :func:`~shipvision.imgproc.nms.greedy.greedy`.

Layout: :mod:`~shipvision.imgproc.nms.candidates` decides who competes and in what order,
:mod:`~shipvision.imgproc.nms.greedy` runs classic and soft NMS, and
:mod:`~shipvision.imgproc.nms.neighborhood` runs the corroboration variant. This module is
only the dispatcher.
"""

from __future__ import annotations

import numpy as np

from shipvision.imgproc.nms.candidates import (
    CLASSIC,
    GAUSS,
    LINEAR,
    METHODS,
    NEIGHBORHOOD,
    NONE,
    SOFT_METHODS,
    prepare,
)
from shipvision.imgproc.nms.greedy import decay_weights, greedy
from shipvision.imgproc.nms.neighborhood import neighborhood

__all__ = [
    "CLASSIC",
    "GAUSS",
    "LINEAR",
    "METHODS",
    "NEIGHBORHOOD",
    "NONE",
    "SOFT_METHODS",
    "decay_weights",
    "greedy",
    "neighborhood",
    "prepare",
    "suppress",
]


def suppress(
    boxes: np.ndarray,
    scores: np.ndarray,
    *,
    iou_threshold: float,
    method: str = CLASSIC,
    sigma: float = 0.5,
    score_threshold: float = 0.0,
    min_neighbors: int = 0,
    min_score_sum: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one of the five methods and return ``(kept_indices, kept_scores)``.

    ``kept_scores`` carries the decayed value for the soft methods and the original score for
    every other one, so a caller can threshold soft-NMS's real output instead of guessing at
    it. Both arrays are in descending score order.
    """
    box_array, score_array, order = prepare(
        boxes,
        scores,
        iou_threshold=iou_threshold,
        method=method,
        sigma=sigma,
        score_threshold=score_threshold,
    )
    if order.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
    if method == NONE:
        return order.astype(np.int64), score_array[order]
    if method == NEIGHBORHOOD:
        return neighborhood(
            box_array,
            score_array,
            order,
            iou_threshold=iou_threshold,
            min_neighbors=min_neighbors,
            min_score_sum=min_score_sum,
        )
    return greedy(
        box_array,
        score_array,
        order,
        iou_threshold=iou_threshold,
        method=method,
        sigma=sigma,
        score_threshold=score_threshold,
    )
