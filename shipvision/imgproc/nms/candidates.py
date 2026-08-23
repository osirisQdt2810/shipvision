"""Admission and ordering: who is allowed into the suppression pool, and in what order.

Its own module because both halves are shared by every method *and* by the backends'
accelerated paths — a device kernel must be handed exactly the candidate set the numpy
reference would have used, or the two are not comparable. The two decisions here are the ones
most often made differently by accident.
"""

from __future__ import annotations

import numpy as np

from shipvision.errors import ConfigurationError, DimensionMismatchError

__all__ = [
    "CLASSIC",
    "GAUSS",
    "LINEAR",
    "METHODS",
    "NEIGHBORHOOD",
    "NONE",
    "SOFT_METHODS",
    "prepare",
]

CLASSIC = "classic"
LINEAR = "linear"
GAUSS = "gauss"
NEIGHBORHOOD = "neighborhood"
NONE = "none"

METHODS: tuple[str, ...] = (CLASSIC, LINEAR, GAUSS, NEIGHBORHOOD, NONE)
SOFT_METHODS: frozenset[str] = frozenset({LINEAR, GAUSS})
"""The methods that change scores. Everything else only removes boxes."""


def prepare(
    boxes: np.ndarray,
    scores: np.ndarray,
    *,
    iou_threshold: float,
    method: str,
    sigma: float,
    score_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate the inputs, then return ``(boxes, scores, order)``.

    Admission is ``score >= score_threshold`` — inclusive, so the default of ``0.0`` admits
    everything. The C++ reference used a strict ``>`` here; the CUDA kernel in ``csrc/`` uses
    ``>=``, and matching the kernel is what keeps the backends comparable.

    Ties break towards the lower input index, via a **stable** descending sort. An unstable
    sort makes the same input give different output between runs, which turns a tracking
    regression into a heisenbug that nobody can reproduce. ``torchvision.ops.nms`` sorts
    stably too, so the backends agree even on a duplicated proposal.

    Returns:
        Contiguous float32 ``boxes`` and ``scores``, plus ``order``: the indices that clear
        ``score_threshold``, stably sorted by descending score.

    Raises:
        DimensionMismatchError: the box and score counts differ.
        ConfigurationError: an unknown method, an out-of-range threshold, or a non-positive
            sigma for the one method that reads it.
    """
    box_array = np.ascontiguousarray(np.asarray(boxes, dtype=np.float32).reshape(-1, 4))
    score_array = np.ascontiguousarray(np.asarray(scores, dtype=np.float32).reshape(-1))
    if box_array.shape[0] != score_array.shape[0]:
        raise DimensionMismatchError(
            f"nms got {box_array.shape[0]} boxes and {score_array.shape[0]} scores; a "
            f"broadcast here would silently score the wrong box"
        )
    if method not in METHODS:
        raise ConfigurationError(f"unknown nms method {method!r}; expected one of {METHODS}")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ConfigurationError(f"iou_threshold must be in [0, 1], got {iou_threshold}")
    if method == GAUSS and sigma <= 0.0:
        raise ConfigurationError(f"gauss nms needs a positive sigma, got {sigma}")

    admitted = np.flatnonzero(score_array >= np.float32(score_threshold))
    # Negating rather than reversing an ascending sort: reversing would flip the tie order
    # back again, and the tie order is the thing this is being careful about.
    order = admitted[np.argsort(-score_array[admitted], kind="stable")]
    return box_array, score_array, order
