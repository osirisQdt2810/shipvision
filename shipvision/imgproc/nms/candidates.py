"""Admission and ordering: who is allowed into the suppression pool, and in what order.

Its own module because both halves are shared by every method *and* by the backends'
accelerated paths — a device kernel must be handed exactly the candidate set the numpy
reference would have used, or the two are not comparable. The two decisions here are the ones
most often made differently by accident.
"""

from __future__ import annotations

import numpy as np

from shipvision.errors import ConfigurationError, DimensionMismatchError
from shipvision.imgproc.validation import reject_non_finite

__all__ = [
    "CLASSIC",
    "GAUSS",
    "LINEAR",
    "METHODS",
    "NEIGHBORHOOD",
    "NONE",
    "SOFT_METHODS",
    "prepare",
    "validate_max_output",
]

CLASSIC = "classic"
LINEAR = "linear"
GAUSS = "gauss"
NEIGHBORHOOD = "neighborhood"
NONE = "none"

METHODS: tuple[str, ...] = (CLASSIC, LINEAR, GAUSS, NEIGHBORHOOD, NONE)
SOFT_METHODS: frozenset[str] = frozenset({LINEAR, GAUSS})
"""The methods that change scores. Everything else only removes boxes."""


def validate_max_output(max_output: int | None) -> int | None:
    """A whole, non-negative survivor budget, or ``None`` for no cap.

    Named and exported rather than inlined into :func:`prepare`, whose validation block is
    already long: a cap usually arrives from a config file, and a consumer that wants to refuse
    a bad one at start-up — rather than on the first frame that reaches suppression — needs
    somewhere to ask. It returns the normalised value so that such a caller can keep it.

    Raises:
        ConfigurationError: negative, or not a whole number. ``0`` is allowed and means an
            empty answer — a budget of nothing is unusual but it is not ambiguous, and every
            backend agrees on it.
    """
    if max_output is None:
        return None
    if isinstance(max_output, bool) or int(max_output) != max_output:
        raise ConfigurationError(
            f"max_output must be a whole number of boxes or None, got {max_output!r}. A "
            f"fractional cap would be truncated by a slice and rounded by the binding's int "
            f"conversion, which are not the same number"
        )
    value = int(max_output)
    if value < 0:
        raise ConfigurationError(
            f"max_output must be non-negative, got {value}. Use None for no cap: a negative "
            f"one reaches numpy as a slice that drops the worst survivor and reaches the CUDA "
            f"sweep as a budget it can never meet, so the two backends would disagree silently"
        )
    return value


def prepare(
    boxes: np.ndarray,
    scores: np.ndarray,
    *,
    iou_threshold: float,
    method: str,
    sigma: float,
    score_threshold: float,
    max_output: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate the inputs, then return ``(boxes, scores, order)``.

    Admission is ``score >= score_threshold`` — inclusive, so the default of ``0.0`` admits
    everything. The C++ reference used a strict ``>`` here; the CUDA kernel in ``csrc/`` uses
    ``>=``, and matching the kernel is what keeps the backends comparable.

    Ties break towards the lower input index, via a **stable** descending sort. An unstable
    sort makes the same input give different output between runs, which turns a tracking
    regression into a heisenbug that nobody can reproduce. ``torchvision.ops.nms`` sorts
    stably too, so the backends agree even on a duplicated proposal.

    ``max_output`` is only *checked* here, not applied — the cap belongs after suppression,
    and each backend truncates in the place that costs it nothing. It is checked here because
    this is the one function every method and every backend passes through, so one check
    covers the numpy loop, ``torchvision.ops.nms`` and the CUDA sweep alike. A negative cap is
    the case worth refusing: Python's ``[:-1]`` drops the *worst* survivor and the C++ sweep's
    ``keep.size() < max_output`` is false from the start, so the two backends would return an
    almost-complete answer and an empty one for the same input, with no error on either side.

    Returns:
        Contiguous float32 ``boxes`` and ``scores``, plus ``order``: the indices that clear
        ``score_threshold``, stably sorted by descending score.

    Non-finite boxes and scores are refused here rather than downstream, because this is the
    one function every method and every backend passes through — including the device kernel,
    which is handed the candidate set this decides on. A NaN coordinate makes the kernel and
    numpy disagree about which boxes survive, and a NaN score makes the surviving *order*
    depend on the sort implementation; see
    :func:`~shipvision.imgproc.validation.reject_non_finite`.

    Raises:
        DimensionMismatchError: the box and score counts differ.
        ConfigurationError: an unknown method, an out-of-range threshold, a non-positive
            sigma for the one method that reads it, a negative or fractional ``max_output``,
            or a non-finite box or score.
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
    validate_max_output(max_output)
    reject_non_finite(box_array, "nms boxes")
    reject_non_finite(score_array, "nms scores")

    admitted = np.flatnonzero(score_array >= np.float32(score_threshold))
    # Negating rather than reversing an ascending sort: reversing would flip the tie order
    # back again, and the tie order is the thing this is being careful about.
    order = admitted[np.argsort(-score_array[admitted], kind="stable")]
    return box_array, score_array, order
