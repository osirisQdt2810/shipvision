"""The matching every tracking metric is built out of, in one place.

Three metrics, three different matchings, and the differences *are* the metrics:

============  ==========================================================================
CLEAR/MOTA    One matching per frame, biased towards keeping the previous frame's
              assignment. Re-solving each frame from scratch invents identity switches
              that never happened; see :func:`match_preferring`.
IDF1          One matching for the whole sequence, between *trajectories*. Not a count of
              per-frame agreements; see :mod:`shipvision.eval.metrics.identity`.
HOTA          One matching per frame, but scored by a sequence-wide alignment so that a
              locally ambiguous frame resolves the way the rest of the sequence says it
              should; see :mod:`shipvision.eval.metrics.hota`.
============  ==========================================================================

What they share is underneath: line up two sequences frame by frame, compute IoU, and give
every identity a dense index so a metric can use an ``(n_gt, n_pred)`` array as a lookup
table. That is :func:`align`, and doing it once means the three metrics cannot disagree
about which ground-truth object frame 40 000 contained.

:func:`scipy.optimize.linear_sum_assignment` does the solving. Hand-rolling a Hungarian
implementation here would be slower, longer and less correct than the one already installed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from shipvision.errors import ConfigurationError
from shipvision.eval.sequence import ObjectFrame, TrackSequence
from shipvision.types import iou_matrix

__all__ = [
    "AlignedSequence",
    "align",
    "drop_predictions_matching",
    "iou_similarity",
    "match_preferring",
    "solve_maximum",
]

#: Tolerance when comparing a similarity against a threshold. A box read back from a text
#: file and one computed in float32 can differ in the last bit, and an IoU of exactly 0.5
#: must count as a match at threshold 0.5 or the same tracker scores differently depending
#: on whether its output went through a file.
EPS = float(np.finfo(np.float64).eps)


def iou_similarity(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    """``(n_gt, n_pred)`` IoU. Ground truth is always the row index.

    A one-line wrapper over :func:`shipvision.types.iou_matrix`, and it exists for the
    orientation alone. Every metric in this package indexes ``[gt, pred]``; a transposed
    similarity matrix produces numbers that are individually plausible, jointly wrong, and
    impossible to spot in a report. Naming the arguments makes the mistake unrepresentable.
    """
    return iou_matrix(gt_boxes, pred_boxes)


def solve_maximum(score: np.ndarray, *, minimum: float = EPS) -> tuple[np.ndarray, np.ndarray]:
    """Maximise total score under a one-to-one assignment, then drop the weak pairs.

    Returns ``(rows, cols)``, both int64.

    The threshold is applied *after* the solve, exactly as it is in
    :mod:`shipvision.tracking.association.solver`, and for the same reason: the solver
    optimises the total, so it will accept a poor pair to enable two good ones. That is right
    globally and wrong for that pair, so the pair is dropped afterwards rather than being
    made invisible beforehand.
    """
    if score.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    rows, cols = linear_sum_assignment(-score)
    keep = score[rows, cols] > minimum
    return rows[keep].astype(np.int64), cols[keep].astype(np.int64)


def match_preferring(
    similarity: np.ndarray,
    previous: np.ndarray,
    *,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """CLEAR's matcher: keep last frame's assignment where it is still admissible.

    Args:
        similarity: ``(n_gt, n_pred)`` IoU for this frame.
        previous: ``(n_gt,)`` int; for each ground-truth row, the *column* it was matched to
            on the previous frame, or ``-1`` for none. Columns, not identities, because the
            caller already has this frame's id-to-column map and rebuilding it here would be
            the same lookup done twice.
        threshold: minimum IoU for a pair to be admissible at all.

    Returns:
        ``(rows, cols)`` of the accepted matches.

    **Why the preference exists.** Two people cross. Both predicted boxes are admissible for
    both ground-truth objects, and the geometry happens to favour the swap by a hair. A
    matcher that maximises IoU alone takes the swap and records two identity switches; the
    tracker did not switch anything, and the two objects it will be blamed for are an
    artefact of the metric. So the CLEAR protocol (Bernardin and Stiefelhagen, 2008) carries
    the previous frame's mapping forward whenever it is still admissible, and only then
    optimises the remainder.

    **How the preference is made absolute.** Each surviving previous assignment is worth a
    bonus of ``1 + min(n_gt, n_pred)`` — strictly more than the largest total IoU any
    assignment can reach, since IoU is at most 1 per matched pair and there are at most
    ``min(n_gt, n_pred)`` pairs. Breaking one carried-over match therefore cannot be paid for
    by any amount of geometric improvement elsewhere, which is what "prefer" has to mean if
    it is to be worth anything. TrackEval hard-codes 1000 here, which is the same idea and is
    outbid by a frame containing more than a thousand objects.

    **What it costs.** Preferring a carried-over pair can leave a detection unmatched that a
    free re-solve would have matched, turning one true positive into a false positive and a
    false negative. That is deliberate: one frame of a slightly worse box is a smaller lie
    than an identity switch the tracker never made.
    """
    n_gt, n_pred = similarity.shape
    if previous.shape[0] != n_gt:
        raise ConfigurationError(
            f"previous has {previous.shape[0]} entries for {n_gt} ground-truth rows"
        )
    if n_gt == 0 or n_pred == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    admissible = similarity >= threshold - EPS
    bonus = 1.0 + float(min(n_gt, n_pred))
    score = np.where(admissible, similarity.astype(np.float64), 0.0)

    carried = np.zeros_like(score, dtype=bool)
    rows = np.flatnonzero(previous >= 0)
    carried[rows, previous[rows]] = True
    score = score + bonus * (carried & admissible)

    return solve_maximum(score)


def drop_predictions_matching(
    predictions: ObjectFrame,
    ignored: ObjectFrame,
    *,
    competing: np.ndarray | None = None,
    threshold: float = 0.5,
) -> ObjectFrame:
    """Remove the predictions that landed on a box the benchmark declines to ask about.

    Args:
        predictions: this frame's tracker output.
        ignored: the boxes that *absorb* a prediction — MOTChallenge's distractor classes:
            people on vehicles, static persons, distractors and reflections.
        competing: ``(m, 4)`` boxes that take part in the assignment but neither score nor
            absorb: the scored ground truth itself, plus the annotation classes that are
            neither pedestrians nor distractors (occluders, crowd regions, vehicles).
            Optional, and omitting it biases the result — see below.
        threshold: IoU at which a prediction counts as landing on a box.

    MOTChallenge's ground truth contains reflections, people sitting in vehicles and static
    mannequins. They are real objects, a good detector finds them, and the benchmark scores
    neither their presence nor their absence — so a prediction matched to one is *deleted*,
    not counted as a false positive. Skipping this step is worth several MOTA points and
    every one of them is a lie about the detector rather than about the tracker.

    **``competing`` is what makes the deletion conservative.** The assignment is solved over
    every annotated box in the frame at once and only the predictions assigned to an
    *ignored* row are dropped. Solve over the ignored boxes alone and a prediction sitting on
    a real pedestrian who happens to stand in front of a reflection gets deleted — removing a
    true positive from the tracker that found it while leaving the ground-truth box in the
    denominator. One frame of that is noise; over 1050 frames of MOT17-04 it is a systematic
    penalty on whichever tracker keeps the most tracks alive. MOTChallenge's own protocol does
    the joint solve, which is also what makes these numbers comparable with a leaderboard's.

    The matching is one-to-one, so a single ignore region cannot absorb an unlimited number
    of duplicate predictions — a tracker that emits ten boxes on one mannequin has nine false
    positives, which is the honest count.
    """
    if len(predictions) == 0 or len(ignored) == 0:
        return predictions
    if competing is None or len(competing) == 0:
        boxes, offset = ignored.boxes, 0
    else:
        others = np.asarray(competing, dtype=np.float32).reshape(-1, 4)
        boxes, offset = np.concatenate([others, ignored.boxes]), others.shape[0]
    similarity = iou_similarity(boxes, predictions.boxes)
    rows, cols = solve_maximum(np.where(similarity >= threshold - EPS, similarity, 0.0))
    drop = cols[rows >= offset]
    if drop.size == 0:
        return predictions
    keep = np.ones(len(predictions), dtype=bool)
    keep[drop] = False
    return ObjectFrame(
        frame_id=predictions.frame_id,
        ids=predictions.ids[keep],
        boxes=predictions.boxes[keep],
    )


@dataclass(frozen=True, slots=True)
class AlignedSequence:
    """Two sequences, lined up frame by frame with dense identity indices.

    Every metric reads this and nothing else, which is what stops them from disagreeing
    about the data. ``gt_ids[t]`` and ``pred_ids[t]`` hold *dense* indices into
    ``[0, num_gt_ids)`` and ``[0, num_pred_ids)``, so a metric can accumulate into an
    ``(num_gt_ids, num_pred_ids)`` array by fancy indexing. The original ids are kept in
    ``gt_labels``/``pred_labels`` so a report can name the identity it is complaining about.

    ``num_frames`` is the number of frames the *sequence* has, not the number of entries in
    these lists. Frames where neither side had anything are not stored — there is nothing to
    match — but they still belong in the denominator of a per-frame rate.
    """

    name: str
    frame_ids: tuple[int, ...]
    gt_ids: tuple[np.ndarray, ...]
    pred_ids: tuple[np.ndarray, ...]
    similarity: tuple[np.ndarray, ...]
    gt_labels: tuple[int, ...]
    pred_labels: tuple[int, ...]
    num_frames: int

    @property
    def num_gt_ids(self) -> int:
        return len(self.gt_labels)

    @property
    def num_pred_ids(self) -> int:
        return len(self.pred_labels)

    @property
    def num_gt_dets(self) -> int:
        return int(sum(a.shape[0] for a in self.gt_ids))

    @property
    def num_pred_dets(self) -> int:
        return int(sum(a.shape[0] for a in self.pred_ids))

    def __len__(self) -> int:
        return len(self.frame_ids)

    def __iter__(self) -> object:
        """``(gt_ids, pred_ids, similarity)`` per stored frame, in frame order."""
        return iter(zip(self.gt_ids, self.pred_ids, self.similarity, strict=True))


def _dense(labels: Sequence[np.ndarray]) -> tuple[dict[int, int], tuple[int, ...]]:
    """Map the ids that actually occur onto ``0..k-1``, in sorted order."""
    if not labels:
        return {}, ()
    unique = np.unique(np.concatenate(labels)) if labels else np.empty(0, dtype=np.int64)
    ordered = tuple(int(v) for v in unique)
    return {value: index for index, value in enumerate(ordered)}, ordered


def align(
    ground_truth: TrackSequence,
    predictions: TrackSequence,
    *,
    ignored: Sequence[ObjectFrame] = (),
    unscored: Sequence[ObjectFrame] = (),
    ignore_threshold: float = 0.5,
    name: str | None = None,
) -> AlignedSequence:
    """Pair two sequences by frame id, compute IoU, and densify the identities.

    Args:
        ground_truth: the answer.
        predictions: what the tracker said.
        ignored: per-frame boxes that absorb a prediction. A prediction matched to one is
            removed here, before any metric sees it — see :func:`drop_predictions_matching`.
        unscored: per-frame boxes that are annotated but are neither scored nor absorbing —
            occluders, crowd regions, vehicles. They exist only to compete for predictions in
            the ignore assignment, so that a distractor cannot absorb a prediction which
            plainly belongs to an occluder instead.
        ignore_threshold: IoU at which a prediction counts as landing on an ignored box.
        name: what to call the result. Defaults to the ground truth's name.

    Frames are paired by ``frame_id``, never by position. A tracker that publishes nothing
    for the first thirty frames produces a sequence whose first entry is frame 31, and
    zipping the two lists would shift every box in the run by thirty frames while leaving
    every number it produces plausible.
    """
    if not isinstance(ground_truth, TrackSequence) or not isinstance(
        predictions, TrackSequence
    ):
        raise ConfigurationError("align takes two TrackSequence objects")

    gt_by_frame = ground_truth.by_frame()
    pred_by_frame = predictions.by_frame()
    ignored_by_frame = {f.frame_id: f for f in ignored}
    unscored_by_frame = {f.frame_id: f for f in unscored}
    empty = ObjectFrame(frame_id=-1, ids=np.empty(0, dtype=np.int64), boxes=np.empty((0, 4)))

    paired: list[tuple[int, ObjectFrame, ObjectFrame]] = []
    for frame_id in sorted(set(gt_by_frame) | set(pred_by_frame)):
        gt_frame = gt_by_frame.get(frame_id, empty)
        pred_frame = pred_by_frame.get(frame_id, empty)
        ignore_frame = ignored_by_frame.get(frame_id)
        if ignore_frame is not None and len(pred_frame):
            other = unscored_by_frame.get(frame_id)
            competing = (
                gt_frame.boxes
                if other is None
                else np.concatenate([gt_frame.boxes, other.boxes])
            )
            pred_frame = drop_predictions_matching(
                pred_frame, ignore_frame, competing=competing, threshold=ignore_threshold
            )
        if len(gt_frame) == 0 and len(pred_frame) == 0:
            # Nothing to match either way. Not stored, but still counted in num_frames: a
            # false-positive rate per frame is wrong if the quiet frames are missing from
            # its denominator.
            continue
        paired.append((frame_id, gt_frame, pred_frame))

    gt_map, gt_labels = _dense([gt.ids for _, gt, _ in paired])
    pred_map, pred_labels = _dense([pred.ids for _, _, pred in paired])

    return AlignedSequence(
        name=name or ground_truth.name,
        frame_ids=tuple(frame_id for frame_id, _, _ in paired),
        gt_ids=tuple(
            np.array([gt_map[i] for i in gt.ids.tolist()], dtype=np.int64)
            for _, gt, _ in paired
        ),
        pred_ids=tuple(
            np.array([pred_map[i] for i in pred.ids.tolist()], dtype=np.int64)
            for _, _, pred in paired
        ),
        similarity=tuple(iou_similarity(gt.boxes, pred.boxes) for _, gt, pred in paired),
        gt_labels=gt_labels,
        pred_labels=pred_labels,
        num_frames=max(ground_truth.length, predictions.length, len(paired)),
    )
