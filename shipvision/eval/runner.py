"""Run a tracker over a case, and score two sequences against each other.

Two functions and one deliberate separation between them. :func:`score` compares a ground
truth with a prediction and knows nothing about trackers; :func:`run` drives a tracker and
knows nothing about metrics. Keeping them apart is what lets the same scoring path grade a
tracker built here, a submission file read off disk, and a hand-written three-frame scenario
in a unit test — and it is what makes it impossible for the scoring to depend on how the
prediction was produced.

**Timing is measured here and reported next to the quality numbers**, not in a separate
benchmark. The target is 1000 frames per second across fifty cameras, so a tracker that wins
HOTA by a point at 4 ms a frame has lost; splitting the two measurements into two reports is
exactly how that trade goes unnoticed. The clock is :func:`time.perf_counter` around the
``update`` call only — not around the loader, not around the metric — because the number that
matters is the one that will run inside the server.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence

import numpy as np

from shipvision.errors import ConfigurationError
from shipvision.eval.association import align
from shipvision.eval.metrics import SequenceResult, clear_counts, hota_counts, identity_counts
from shipvision.eval.sequence import EvaluationCase, ObjectFrame, TrackSequence
from shipvision.tracking.base import BaseTracker

__all__ = ["evaluate", "evaluate_all", "run", "score"]


def score(
    ground_truth: TrackSequence,
    predictions: TrackSequence,
    *,
    ignored: Sequence[ObjectFrame] = (),
    unscored: Sequence[ObjectFrame] = (),
    threshold: float = 0.5,
    seconds: float = 0.0,
    name: str | None = None,
) -> SequenceResult:
    """Align once, then compute all three metrics over the same alignment.

    Args:
        ground_truth: what should have come out.
        predictions: what did.
        ignored: per-frame boxes that absorb a prediction. See
            :func:`shipvision.eval.association.drop_predictions_matching`.
        unscored: per-frame boxes that compete for predictions but neither score nor absorb.
        threshold: the IoU cliff CLEAR and IDF1 use. HOTA ignores it — it sweeps nineteen
            thresholds of its own — which is why it is the better tuning objective.
        seconds: wall-clock tracker time, carried through to ``ms_per_frame``.
        name: what to call the row. Defaults to the ground truth's name.

    Aligning once rather than three times is not a saving of milliseconds: it is what makes
    it impossible for CLEAR and IDF1 to disagree about which predictions the ignore regions
    removed, or for HOTA to score a frame the other two never saw.
    """
    aligned = align(
        ground_truth,
        predictions,
        ignored=ignored,
        unscored=unscored,
        name=name or ground_truth.name,
    )
    return SequenceResult(
        name=aligned.name,
        num_frames=aligned.num_frames,
        num_gt_dets=aligned.num_gt_dets,
        num_gt_ids=aligned.num_gt_ids,
        num_pred_dets=aligned.num_pred_dets,
        num_pred_ids=aligned.num_pred_ids,
        clear=clear_counts(aligned, threshold=threshold),
        identity=identity_counts(aligned, threshold=threshold),
        hota=hota_counts(aligned),
        seconds=seconds,
    )


def run(
    tracker: BaseTracker,
    case: EvaluationCase,
    *,
    reset: bool = True,
) -> tuple[TrackSequence, float]:
    """Feed every frame of ``case`` to ``tracker``. Returns ``(predictions, seconds)``.

    Args:
        tracker: a live tracker. **Stateful**, so one instance evaluates one case: reusing it
            across two cases would carry camera 1's tracks into camera 2, which
            :meth:`~shipvision.tracking.base.BaseTracker.begin` refuses outright — the
            ``reset`` argument exists so a caller who *means* to reuse one can.
        case: the input frames and the answer.
        reset: forget the tracker's state first. Default on, because the common mistake is
            reusing an instance and the consequence is a run whose first hundred frames are
            polluted by the previous sequence's tracks.

    Every frame is passed, including the empty ones. A tracker ages its tracks on an empty
    frame and eventually forgets them; skipping those frames measures a tracker that never
    has to forget anything, which is a different and much easier problem.

    No image is passed. BoT-SORT's camera-motion estimator is the only consumer of pixels,
    and a public-detection evaluation deliberately has none — so what is measured here is
    BoT-SORT *without* CMC, which the report has to say out loud rather than letting the
    comparison look like a fair fight.

    **Each frame is snapshotted into an :class:`ObjectFrame` immediately, and that is not
    tidiness.** ``update`` returns the pool's *live* :class:`~shipvision.types.Track` objects,
    which the tracker mutates in place on the next frame. Buffering them and reading their ids
    afterwards gives every entry the last frame's state — the symptom is a sequence in which
    one identity appears twenty-six times in a single frame, which is how this was found.
    Copying the two fields a metric needs, per frame, costs nothing and is the only version
    that is correct.
    """
    if reset:
        tracker.reset()
    frames: list[ObjectFrame] = []
    elapsed = 0.0
    for detections in case.detections:
        started = time.perf_counter()
        tracks = tracker.update(detections)
        elapsed += time.perf_counter() - started
        if tracks:
            frames.append(
                ObjectFrame(
                    frame_id=detections.tag.frame_id,
                    ids=np.fromiter(
                        (t.track_id for t in tracks), dtype=np.int64, count=len(tracks)
                    ),
                    boxes=np.stack([t.box for t in tracks]),
                )
            )
    return (
        TrackSequence(name=case.name, frames=tuple(frames), length=case.num_frames),
        elapsed,
    )


def evaluate(
    tracker: BaseTracker,
    case: EvaluationCase,
    *,
    threshold: float = 0.5,
    reset: bool = True,
) -> SequenceResult:
    """:func:`run` then :func:`score`, which is what almost every caller wants."""
    predictions, seconds = run(tracker, case, reset=reset)
    return score(
        case.ground_truth,
        predictions,
        ignored=case.ignored,
        unscored=case.unscored,
        threshold=threshold,
        seconds=seconds,
        name=case.name,
    )


def evaluate_all(
    factory: Callable[[], BaseTracker],
    cases: Iterable[EvaluationCase],
    *,
    threshold: float = 0.5,
) -> list[SequenceResult]:
    """One result per case, with a **fresh tracker for each**.

    ``factory`` rather than a tracker, and this is the argument that has to be a callable: a
    tracker is stateful and single-camera by construction, so evaluating seven sequences with
    one instance either raises (the camera id changed) or, worse, silently carries process-
    global track ids and pool contents from one sequence into the next. Handing in a factory
    makes the correct thing the only thing that can be expressed.
    """
    cases = list(cases)
    if not cases:
        raise ConfigurationError("nothing to evaluate; an empty case list is not a score of 0")
    return [evaluate(factory(), case, threshold=threshold) for case in cases]
