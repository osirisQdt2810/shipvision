"""Whether this machine has the compiled trackers, and how to compare two of them.

The comparison helper is the interesting half. Track ids come from a **process-wide** counter,
so two trackers driven over the same sequence in one test never produce the same integers — the
second one starts wherever the first left off. Comparing raw ids would therefore fail for a
reason that has nothing to do with either tracker, and "fix" itself if the counter were made
per-instance, which is the bug the process-wide counter exists to prevent.

What has to match is the *structure*: which detections were grouped into one identity, and in
what order those identities first appeared. :func:`relabel` renumbers by first appearance so
that is exactly what gets compared.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from shipvision.tracking.backends.native import native_available
from shipvision.types import Track

NATIVE_BUILT = native_available()

#: Reason string, written once: it is what a reader sees when the whole file skips, and it has
#: to say which of the two possible causes applies.
NO_BUILD = "shipvision._C is not built, or predates the native trackers"


#: The algorithms that have a compiled twin. Read from the registry rather than written out,
#: so adding a third native tracker adds it to the parity suite by construction — a
#: hand-maintained list here is how a compiled tracker ends up with no oracle.
def native_tracker_names() -> list[str]:
    from shipvision.registry import NATIVE
    from shipvision.tracking import TRACKERS

    return [name for name in TRACKERS.names() if NATIVE in TRACKERS.backends(name)]


def relabel(published: Sequence[Sequence[Track]]) -> list[list[tuple[int, Track]]]:
    """Per-frame ``(identity, track)`` pairs, with identities numbered by first appearance."""
    mapping: dict[int, int] = {}
    frames: list[list[tuple[int, Track]]] = []
    for step in published:
        frames.append([(mapping.setdefault(t.track_id, len(mapping)), t) for t in step])
    return frames


def assert_same_tracking(
    reference: Sequence[Sequence[Track]],
    candidate: Sequence[Sequence[Track]],
    *,
    tolerance: float = 1e-3,
) -> float:
    """Fail unless two runs grouped the detections into the same identities in the same frames.

    Boxes are compared with a tolerance and identities are not. That asymmetry is the whole
    claim: a Kalman update is a chain of float32 operations whose *order* differs between numpy
    and C++, so the boxes may disagree in the seventh digit — but an association is a
    comparison against a threshold, and a tracker whose associations depend on the seventh
    digit is one whose output is not reproducible at all.

    Returns:
        The largest box disagreement seen, so a test can report it rather than only assert it.
    """
    expected = relabel(reference)
    actual = relabel(candidate)

    assert len(expected) == len(actual), "the two runs saw a different number of frames"
    worst = 0.0
    for index, (want, got) in enumerate(zip(expected, actual, strict=True)):
        assert sorted(i for i, _ in want) == sorted(i for i, _ in got), (
            f"frame {index}: the compiled tracker published identities "
            f"{sorted(i for i, _ in got)} where the readable one published "
            f"{sorted(i for i, _ in want)}"
        )
        want_boxes = {i: track.box for i, track in want}
        got_boxes = {i: track.box for i, track in got}
        for identity, box in want_boxes.items():
            delta = float(np.abs(box - got_boxes[identity]).max())
            worst = max(worst, delta)
            assert delta < tolerance, (
                f"frame {index}, identity {identity}: box {got_boxes[identity].tolist()} "
                f"against {box.tolist()} — {delta} apart, which is a different association "
                f"rather than float32 rounding"
            )
    return worst


@pytest.fixture(params=native_tracker_names())
def algorithm(request) -> str:
    """One algorithm that exists in both backends."""
    return request.param
