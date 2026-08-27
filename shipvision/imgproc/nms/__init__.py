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
* **The cap**, ``max_output``, is applied last and changes nothing about the suppression that
  produced the survivors. It keeps the best ``max_output`` of them by final score, which over
  an answer that is already in descending order is a truncation — so it is one line in
  :func:`suppress` rather than a rule the five methods each have to obey. ``None`` is no cap.

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
    validate_max_output,
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
    "validate_max_output",
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
    max_output: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one of the five methods and return ``(kept_indices, kept_scores)``.

    ``kept_scores`` carries the decayed value for the soft methods and the original score for
    every other one, so a caller can threshold soft-NMS's real output instead of guessing at
    it. Both arrays are in descending score order.

    ``max_output`` caps how many survivors come back, ``None`` meaning no cap. It is applied
    *here*, on the dispatcher's way out, and not inside the five methods: every one of them
    already returns descending final score — the greedy loops pick the running maximum and a
    decay only ever lowers a score, so the picks cannot rise, and ``none`` returns the
    admission order — so the cap is one truncation of an already-sorted pair rather than five
    copies of a top-k. Five copies is what it would have taken to put it inside the methods,
    and it is the shape this package exists to avoid: the departure rule and the tie order
    drifted between implementations in the reference this replaces, and they were far more
    visible than a cap would be.
    """
    box_array, score_array, order = prepare(
        boxes,
        scores,
        iou_threshold=iou_threshold,
        method=method,
        sigma=sigma,
        score_threshold=score_threshold,
        max_output=max_output,
    )
    if order.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)

    if method == NONE:
        kept, kept_scores = order.astype(np.int64), score_array[order]
    elif method == NEIGHBORHOOD:
        kept, kept_scores = neighborhood(
            box_array,
            score_array,
            order,
            iou_threshold=iou_threshold,
            min_neighbors=min_neighbors,
            min_score_sum=min_score_sum,
        )
    else:
        kept, kept_scores = greedy(
            box_array,
            score_array,
            order,
            iou_threshold=iou_threshold,
            method=method,
            sigma=sigma,
            score_threshold=score_threshold,
        )

    if max_output is None:
        return kept, kept_scores
    # A slice and not an argsort: the pair is already sorted by final score with ties broken
    # towards the lower input index, and re-sorting it would put that tie order back at the
    # mercy of the sort's stability. Both arrays are cut together — a capped index list beside
    # an uncapped score list is the kind of misalignment nothing downstream can detect.
    return kept[:max_output], kept_scores[:max_output]
