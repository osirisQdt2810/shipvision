"""The input rules more than one layer of this package has to enforce.

One rule lives here today, and it is here rather than in :mod:`shipvision.imgproc.base`
because of the import direction: :mod:`shipvision.imgproc.nms` is *below* ``base`` — ``base``
imports :func:`~shipvision.imgproc.nms.suppress` — so the suppression code cannot reach
``base``'s validators, and :func:`~shipvision.imgproc.nms.suppress` is called directly by the
detection heads without an :class:`~shipvision.imgproc.base.ImageOps` anywhere in the picture.
A rule that must hold on both paths therefore needs a module neither of them owns.

:mod:`shipvision.types` refuses non-finite values at :class:`~shipvision.types.Detection`
construction for the same reason. This is the second line of defence, for the arrays that
never went through it — a raw ``(n, 4)`` block straight out of a decode head.
"""

from __future__ import annotations

import numpy as np

from shipvision.errors import ConfigurationError

__all__ = ["reject_non_finite"]


def reject_non_finite(array: np.ndarray, what: str) -> None:
    """Refuse NaN and inf, naming how many there were and where the first one was.

    Refusing rather than clamping, because the three backends clamp differently and none of
    them can be called wrong. ``fmaxf(0.f, NaN)`` in the CUDA kernel returns 0, so a NaN box
    becomes a plausible crop of the top-left corner; numpy's ``int(nan)`` is
    ``-9223372036854775808`` and raises an untyped ``IndexError`` that loses the whole batch;
    torch's clamp produces a third answer again. In NMS the divergence changes the *result*:
    the kernel computes ``area = 0``, hence ``iou = 9.0``, and suppresses a box that numpy
    keeps because ``NaN > threshold`` is false.

    Counted and located because the usual cause is one bad row out of fifteen thousand, and
    "one crop" versus "the whole head" are different incidents with different fixes.

    Raises:
        ConfigurationError: any value is NaN or infinite.
    """
    bad = ~np.isfinite(array)
    if bad.any():
        first = int(np.flatnonzero(bad.reshape(-1))[0])
        raise ConfigurationError(
            f"{what} has {int(bad.sum())} non-finite value(s), the first at flat index "
            f"{first}. A NaN is not a large number: every backend clamps it differently, so "
            f"three implementations that must agree would return three different answers"
        )
