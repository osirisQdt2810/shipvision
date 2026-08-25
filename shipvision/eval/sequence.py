"""What a metric is handed: identified boxes, per frame, for one sequence.

A tracker's output and a ground-truth file are the same shape — an id and a box per object
per frame — so they are the same type here. That is not a saving of a hundred lines; it is
what makes the symmetry of the metrics expressible. IDF1 matches whole ground-truth
trajectories against whole predicted ones, and a design where the two sides are different
types pushes the caller into writing the conversion twice, once per direction, which is
exactly where a transposed similarity matrix comes from.

Three deliberate constraints:

**Boxes arrive already in ``xyxy``.** MOTChallenge stores ``x, y, w, h`` top-left and the
conversion happens once, in the loader, at the file boundary. A metric that accepted either
format would need to be told which, and the day it is told wrong every number it produces is
plausible and wrong.

**Ids are whatever the producer used.** Ground truth numbers its people from one; this
library's trackers hand out process-global ids that run into the tens of thousands. Neither
is dense and neither has to be: :mod:`shipvision.eval.association` relabels both to a
contiguous range when it aligns them, once, so no metric has to think about it.

**The sequence knows how long it is.** ``length`` is the number of frames the *camera*
produced, not the number of frames something was detected in. False positives per frame and
the frame count in a report are both wrong if a sequence with detections on 40 of its 600
frames reports 40.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

import numpy as np

from shipvision.errors import ConfigurationError
from shipvision.types import Detections, Track

__all__ = ["EvaluationCase", "ObjectFrame", "TrackSequence"]


@dataclass(frozen=True, slots=True)
class ObjectFrame:
    """The identified objects present in one frame.

    Attributes:
        frame_id: the frame's number in its sequence. Comparable across the two sides of an
            evaluation, which is the only reason it is carried rather than implied by
            position: a prediction file that skips the frames it found nothing in must still
            line up with the ground truth, and lining them up by index silently shifts every
            box by however many frames were skipped.
        ids: ``(n,)`` int64. One per box. Duplicates are rejected — the same identity twice
            in one frame is not a degraded measurement, it is a file whose ids do not mean
            what the metric assumes.
        boxes: ``(n, 4)`` float32 ``xyxy`` in absolute pixels.
    """

    frame_id: int
    ids: np.ndarray
    boxes: np.ndarray

    def __post_init__(self) -> None:
        ids = np.asarray(self.ids, dtype=np.int64).reshape(-1)
        boxes = np.asarray(self.boxes, dtype=np.float32).reshape(-1, 4)
        if ids.shape[0] != boxes.shape[0]:
            raise ConfigurationError(
                f"frame {self.frame_id} has {ids.shape[0]} ids and {boxes.shape[0]} boxes"
            )
        if ids.shape[0] != len(np.unique(ids)):
            raise ConfigurationError(
                f"frame {self.frame_id} repeats an identity: {sorted(ids.tolist())}. Every "
                f"metric here assumes one box per identity per frame, so a repeat would be "
                f"counted as two objects that happen to be the same person"
            )
        object.__setattr__(self, "ids", ids)
        object.__setattr__(self, "boxes", boxes)

    def __len__(self) -> int:
        return int(self.ids.shape[0])


@dataclass(frozen=True, slots=True)
class TrackSequence:
    """Every identified box in one sequence, one :class:`ObjectFrame` per frame.

    Frames are held sorted by ``frame_id`` and a frame with nothing in it may be omitted:
    the metrics iterate the union of the two sides' frame ids, so an absent frame and an
    empty one mean the same thing. ``length`` still has to be right, because it is the
    denominator of every per-frame rate in a report.
    """

    name: str
    frames: tuple[ObjectFrame, ...] = ()
    length: int = 0

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.frames, key=lambda f: f.frame_id))
        seen = {f.frame_id for f in ordered}
        if len(seen) != len(ordered):
            raise ConfigurationError(
                f"sequence {self.name!r} has two entries for one frame id; a sequence is a "
                f"map from frame to objects, not a list of appends"
            )
        object.__setattr__(self, "frames", ordered)
        object.__setattr__(self, "length", max(int(self.length), len(ordered)))

    # -- construction --------------------------------------------------------------------

    @classmethod
    def from_tracks(
        cls, name: str, tracks: Iterable[Track], *, length: int = 0
    ) -> TrackSequence:
        """Group a flat stream of :class:`~shipvision.types.Track` by ``tag.frame_id``.

        The natural shape of a tracker's output — one list per call — is a list of lists, and
        flattening it here rather than asking the caller to keep the nesting means a caller
        that batches or reorders its publication still produces the right sequence. The frame
        number comes from the track's own tag, which is the whole reason the tag travels.

        **The tracks must be distinct objects.**
        :meth:`~shipvision.tracking.base.BaseTracker.update` returns the pool's live
        :class:`~shipvision.types.Track` instances and mutates them on the next frame, so
        accumulating them across a run and calling this afterwards reads the *last* frame's id
        and box for every entry. :func:`shipvision.eval.runner.run` snapshots per frame for
        that reason; this constructor is for a caller that already has one list per frame or
        is reading a file.
        """
        grouped: dict[int, tuple[list[int], list[np.ndarray]]] = {}
        for track in tracks:
            ids, boxes = grouped.setdefault(track.tag.frame_id, ([], []))
            ids.append(int(track.track_id))
            boxes.append(track.box)
        frames = tuple(
            ObjectFrame(frame_id=frame_id, ids=np.asarray(ids), boxes=np.asarray(boxes))
            for frame_id, (ids, boxes) in sorted(grouped.items())
        )
        return cls(name=name, frames=frames, length=length)

    @classmethod
    def empty(cls, name: str, *, length: int = 0) -> TrackSequence:
        """A tracker that published nothing. A real case, and it must score, not crash."""
        return cls(name=name, frames=(), length=length)

    # -- inspection ----------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterator[ObjectFrame]:
        return iter(self.frames)

    @property
    def num_detections(self) -> int:
        return sum(len(frame) for frame in self.frames)

    @property
    def num_ids(self) -> int:
        """How many distinct identities appear anywhere in the sequence."""
        if not self.frames:
            return 0
        return len(np.unique(np.concatenate([f.ids for f in self.frames])))

    @property
    def frame_ids(self) -> tuple[int, ...]:
        return tuple(f.frame_id for f in self.frames)

    def by_frame(self) -> dict[int, ObjectFrame]:
        return {f.frame_id: f for f in self.frames}


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """One tracker-evaluation problem: what to feed in, and what the answer should be.

    Bundling the input detections with the ground truth is what lets the same code drive a
    three-frame synthetic scenario in a unit test and a 1050-frame MOT17 sequence off disk.
    A comparison between two trackers is only a comparison if both saw exactly the same
    input, and the way that guarantee is usually lost is a benchmark that re-derives its
    input per tracker.

    Attributes:
        name: the sequence name, as it appears in a report.
        detections: the per-frame input, in frame order. Frames with no detections are
            included and must not be dropped — a tracker ages its tracks on an empty frame,
            and skipping them measures a tracker that never has to forget anything.
        ground_truth: what should have come out.
        ignored: boxes that are neither right nor wrong — MOTChallenge's distractor classes.
            A prediction that matches one is *removed* before scoring rather than counted as
            a false positive, because the object is really there and the benchmark simply
            declines to ask about it. Empty for a synthetic case.
        unscored: boxes that are annotated but neither scored nor absorbing — occluders,
            crowd regions, vehicles. They exist so that a distractor box cannot absorb a
            prediction that plainly belongs to one of them instead; see
            :func:`shipvision.eval.association.drop_predictions_matching`.
        height: frame height in pixels, passed through to the trackers that need it.
        width: frame width in pixels.
    """

    name: str
    detections: tuple[Detections, ...]
    ground_truth: TrackSequence
    ignored: tuple[ObjectFrame, ...] = ()
    unscored: tuple[ObjectFrame, ...] = ()
    height: int = 0
    width: int = 0
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "detections", tuple(self.detections))
        object.__setattr__(self, "ignored", tuple(self.ignored))
        object.__setattr__(self, "unscored", tuple(self.unscored))
        cameras = {d.tag.camera_id for d in self.detections}
        if len(cameras) > 1:
            raise ConfigurationError(
                f"case {self.name!r} mixes cameras {sorted(cameras)}. One tracker serves one "
                f"camera, so a case that spans two measures a configuration nobody deploys"
            )

    @property
    def num_frames(self) -> int:
        return len(self.detections)

    @property
    def num_input_detections(self) -> int:
        return sum(len(d) for d in self.detections)

    def truncated(self, frames: int) -> EvaluationCase:
        """The first ``frames`` frames, ground truth and ignore regions included.

        For a smoke run or a tuning study, where the point is to compare configurations
        rather than to publish a number. Truncating the input alone would leave the ground
        truth counting objects the tracker was never shown, which reads as a collapse in
        recall rather than as a shorter run.
        """
        if frames < 1:
            raise ConfigurationError(f"frames must be positive, got {frames}")
        kept = self.detections[:frames]
        if not kept:
            return self
        last = kept[-1].tag.frame_id
        return EvaluationCase(
            name=self.name,
            detections=kept,
            ground_truth=TrackSequence(
                name=self.ground_truth.name,
                frames=tuple(f for f in self.ground_truth if f.frame_id <= last),
                length=len(kept),
            ),
            ignored=tuple(f for f in self.ignored if f.frame_id <= last),
            unscored=tuple(f for f in self.unscored if f.frame_id <= last),
            height=self.height,
            width=self.width,
            metadata=dict(self.metadata),
        )
