"""The MOTChallenge loader: one directory in, one :class:`EvaluationCase` out.

The canonical layout, which MOT16/17/20 and every derivative share::

    MOT17-09-FRCNN/
    ├── seqinfo.ini      how many frames the camera produced
    ├── img1/            000001.jpg …            (not read here; boxes are enough)
    ├── det/det.txt      public detections — the tracker's *input*
    └── gt/gt.txt        ground truth — what should have come out

Four decisions in this file decide whether a number produced downstream means anything.

**Only ``class == 1`` with ``conf == 1`` is ground truth.** ``gt.txt`` also holds people on
vehicles, static persons, distractors, reflections, occluders and crowd regions. Counting
them inflates the crowd by a third on MOT17-09 (10 411 rows against 5 325 real pedestrians)
and every metric computed over the inflated set is wrong in a direction that looks like a
detector failure. The other classes are not discarded either — they are returned separately,
because the benchmark's protocol needs them to decide which predictions to *forgive*.

**``x, y, w, h`` becomes ``xyxy`` here and nowhere else.** The file format is top-left plus
extent; this library's format is corners. Converting at the loader boundary means no metric,
no tracker and no report has to know which one it is holding — and the day a converter is
applied twice, the boxes are visibly wrong instead of subtly small.

**Every frame from 1 to ``seqLength`` is yielded, including the empty ones.** A tracker ages
its tracks on an empty frame and eventually forgets them, so skipping the frames with no
public detections measures a tracker that never has to forget anything. It also fixes the
denominator of every per-frame rate in the report.

**The public detections are the input, by default.** A tracker has to be evaluable with no
detector present: that is what makes tracking changes measurable in the offline tier, and it
is the only way two trackers can be compared on identical input rather than on two runs of a
non-deterministic detector.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from shipvision.errors import ConfigurationError
from shipvision.eval.datasets.seqinfo import SeqInfo, read_seqinfo
from shipvision.eval.sequence import EvaluationCase, ObjectFrame, TrackSequence
from shipvision.types import Detection, Detections, FrameTag, Track

__all__ = [
    "DISTRACTOR_CLASSES",
    "PEDESTRIAN_CLASS",
    "MotSequenceFiles",
    "discover_sequences",
    "load_case",
    "load_cases",
    "load_detections",
    "load_ground_truth",
    "read_mot_file",
    "write_mot_file",
]

#: The only annotation class MOTChallenge scores.
PEDESTRIAN_CLASS = 1

#: Classes whose boxes *absorb* a prediction instead of counting it wrong. A person sitting
#: in a car, a static person, an explicit distractor and a reflection are all real things a
#: good detector finds, and the benchmark declines to ask about them. MOT20 adds class 6
#: (non-motorised vehicle); MOT17 does not, and adding it here would quietly change the score.
DISTRACTOR_CLASSES = (2, 7, 8, 12)

#: The narrowest row this loader will accept: ``frame, id, x, y, w, h``. Ground truth needs
#: two more (``conf, class``) and says so where it reads them.
_COLUMNS = 6


class MotSequenceFiles:
    """Where the four files of one sequence live, and whether they are there.

    A class rather than four ``Path`` arguments because "the ground truth is missing" and
    "the detections are missing" are different situations — a test-split sequence has no
    ``gt.txt`` and is still perfectly usable as tracker input — and the caller should be told
    which one it hit, at construction, rather than at the read.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise ConfigurationError(f"no such sequence directory: {self.root}")
        self.seqinfo = self.root / "seqinfo.ini"
        self.ground_truth = self.root / "gt" / "gt.txt"
        self.detections = self.root / "det" / "det.txt"

    @property
    def name(self) -> str:
        return self.root.name

    @property
    def has_ground_truth(self) -> bool:
        return self.ground_truth.is_file()

    @property
    def has_detections(self) -> bool:
        return self.detections.is_file()

    def info(self) -> SeqInfo:
        return read_seqinfo(self.seqinfo)

    def __repr__(self) -> str:
        return f"<MotSequenceFiles {self.name} gt={self.has_ground_truth} det={self.has_detections}>"


def discover_sequences(root: Path | str, *, pattern: str = "MOT*") -> list[MotSequenceFiles]:
    """Every sequence directory under ``root``, sorted by name.

    Sorted rather than in directory order so that a report's rows are in the same order on
    every machine — a table whose rows move between runs cannot be diffed, and diffing two
    runs is the main thing anyone does with one.
    """
    root = Path(root)
    if not root.is_dir():
        raise ConfigurationError(f"no such dataset directory: {root}")
    found = [
        MotSequenceFiles(path)
        for path in sorted(root.glob(pattern))
        if path.is_dir() and (path / "seqinfo.ini").is_file()
    ]
    if not found:
        raise ConfigurationError(
            f"no sequences under {root} matching {pattern!r}; a MOTChallenge sequence is a "
            f"directory containing seqinfo.ini"
        )
    return found


def read_mot_file(path: Path | str) -> np.ndarray:
    """``(n, k)`` float64 of a comma-separated MOTChallenge file. Empty files are legal.

    ``np.loadtxt`` rather than a hand-rolled split: it handles the trailing newline, the
    empty file and the single-row file, and all three occur in the wild. A file with one row
    comes back one-dimensional, which is the bug this wrapper exists to absorb.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigurationError(f"no such annotation file: {path}")
    if path.stat().st_size == 0:
        return np.zeros((0, _COLUMNS), dtype=np.float64)
    rows = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
    if rows.shape[1] < _COLUMNS:
        raise ConfigurationError(
            f"{path} has {rows.shape[1]} columns; a MOTChallenge row is at least "
            f"'frame, id, x, y, w, h'"
        )
    return rows


def _to_xyxy(rows: np.ndarray) -> np.ndarray:
    """Columns 2:6 of a MOT row — ``x, y, w, h`` top-left — as ``xyxy``. The one conversion."""
    boxes = np.empty((rows.shape[0], 4), dtype=np.float32)
    boxes[:, 0] = rows[:, 2]
    boxes[:, 1] = rows[:, 3]
    boxes[:, 2] = rows[:, 2] + rows[:, 4]
    boxes[:, 3] = rows[:, 3] + rows[:, 5]
    return boxes


def _group(rows: np.ndarray, boxes: np.ndarray) -> tuple[ObjectFrame, ...]:
    """One :class:`ObjectFrame` per frame present, in frame order."""
    if rows.shape[0] == 0:
        return ()
    order = np.argsort(rows[:, 0], kind="stable")
    rows, boxes = rows[order], boxes[order]
    frame_ids = rows[:, 0].astype(np.int64)
    boundaries = np.flatnonzero(np.diff(frame_ids)) + 1
    return tuple(
        ObjectFrame(
            frame_id=int(frame_ids[chunk[0]]),
            ids=rows[chunk, 1].astype(np.int64),
            boxes=boxes[chunk],
        )
        for chunk in np.split(np.arange(frame_ids.shape[0]), boundaries)
    )


def load_ground_truth(
    path: Path | str,
    *,
    name: str = "",
    length: int = 0,
    min_visibility: float = 0.0,
) -> tuple[TrackSequence, tuple[ObjectFrame, ...], tuple[ObjectFrame, ...]]:
    """Read ``gt.txt`` into ``(scored, ignored, unscored)``.

    Args:
        path: the ``gt/gt.txt`` file.
        name: what to call the resulting sequence. Defaults to the grandparent directory.
        length: the sequence length from ``seqinfo.ini``. Passing it matters — see
            :mod:`shipvision.eval.datasets.seqinfo`.
        min_visibility: drop scored pedestrians below this visibility. **Zero by default**,
            which is the MOTChallenge protocol: a fully-occluded person is still a person and
            a tracker that keeps its identity through the occlusion is doing the thing this
            library exists to do. Raising it makes every score go up and measures a different,
            easier benchmark — so it is an argument the caller has to type rather than a
            default that flatters.

    Returns:
        ``scored``: the pedestrians, as a :class:`TrackSequence`.
        ``ignored``: the distractor classes, which absorb a prediction.
        ``unscored``: everything else, which competes for predictions but neither scores nor
        absorbs.

    A pedestrian row with ``conf == 0`` is a *deliberately* unevaluated annotation and goes
    into ``unscored``, not ``ignored``: it competes for predictions but does not forgive one.
    That is MOTChallenge's own reading rather than the more generous one, and it is what makes
    these numbers comparable with the leaderboard's. On the MOT17 train split the question is
    moot — every ``class == 1`` row there also has ``conf == 1``, verified over all seven
    sequences — so the choice only bites on a dataset that actually uses the flag.
    """
    path = Path(path)
    rows = read_mot_file(path)
    if rows.shape[1] < 8:
        raise ConfigurationError(
            f"{path} has {rows.shape[1]} columns; ground truth needs at least "
            f"'frame, id, x, y, w, h, conf, class' or the class filter cannot be applied"
        )
    boxes = _to_xyxy(rows)
    classes = rows[:, 7].astype(np.int64)
    marked = rows[:, 6] != 0
    visibility = rows[:, 8] if rows.shape[1] > 8 else np.ones(rows.shape[0])

    pedestrian = classes == PEDESTRIAN_CLASS
    scored = pedestrian & marked & (visibility >= min_visibility)
    ignored = np.isin(classes, DISTRACTOR_CLASSES)
    unscored = ~(scored | ignored)

    sequence = TrackSequence(
        name=name or path.parent.parent.name,
        frames=_group(rows[scored], boxes[scored]),
        length=length,
    )
    return (
        sequence,
        _group(rows[ignored], boxes[ignored]),
        _group(rows[unscored], boxes[unscored]),
    )


def load_detections(
    path: Path | str,
    *,
    camera_id: str,
    length: int,
    height: int = 0,
    width: int = 0,
    min_score: float = 0.0,
    frame_rate: float = 0.0,
) -> tuple[Detections, ...]:
    """Read ``det.txt`` into one :class:`~shipvision.types.Detections` per frame.

    Args:
        path: the ``det/det.txt`` file.
        camera_id: the tag every frame carries. The sequence name, so that feeding one
            tracker two sequences fails loudly instead of merging their identities.
        length: how many frames to emit. Frames with no detections are emitted empty.
        height: frame height, from ``seqinfo.ini``. Trackers that decline to recover a track
            against the frame border need it, and a zero disables that policy silently.
        width: frame width.
        min_score: drop detections below this confidence. Zero by default: a tracker's own
            thresholds are its business, and pre-filtering the input would hide the fact that
            ByteTrack's whole contribution is what it does with the low-scoring boxes.
        frame_rate: written into each tag's timestamp as ``frame_id / frame_rate``, so a
            latency measured downstream is in seconds rather than in frames.

    Returns:
        ``length`` entries, in frame order, tagged 1..``length`` as MOTChallenge numbers them.
    """
    rows = read_mot_file(path)
    boxes = _to_xyxy(rows)
    scores = rows[:, 6] if rows.shape[1] > 6 else np.ones(rows.shape[0])
    if rows.shape[0] and (scores.min() < 0.0 or scores.max() > 1.0):
        raise ConfigurationError(
            f"{path} has detection confidences outside [0, 1] (min {scores.min():.4g}, max "
            f"{scores.max():.4g}). MOTChallenge public detections are normalised; an "
            f"unnormalised file would make every score threshold in every tracker meaningless"
        )
    frame_ids = rows[:, 0].astype(np.int64)

    by_frame: dict[int, list[Detection]] = {}
    for index in range(rows.shape[0]):
        score = float(scores[index])
        if score < min_score:
            continue
        by_frame.setdefault(int(frame_ids[index]), []).append(
            Detection(box=boxes[index], score=score, class_id=PEDESTRIAN_CLASS)
        )

    period = 1.0 / frame_rate if frame_rate > 0 else 0.0
    return tuple(
        Detections(
            tag=FrameTag(camera_id=camera_id, frame_id=frame, timestamp=frame * period),
            items=by_frame.get(frame, []),
            height=height,
            width=width,
        )
        for frame in range(1, length + 1)
    )


def load_case(
    root: Path | str,
    *,
    min_score: float = 0.0,
    min_visibility: float = 0.0,
    frames: int = 0,
) -> EvaluationCase:
    """One sequence directory into one :class:`EvaluationCase`.

    Args:
        root: the sequence directory, e.g. ``.../MOT17-09-FRCNN``.
        min_score: passed to :func:`load_detections`.
        min_visibility: passed to :func:`load_ground_truth`.
        frames: truncate to this many frames, for a smoke run. ``0`` means the whole sequence.

    Raises:
        ConfigurationError: the directory is not a MOTChallenge sequence, or has no ground
            truth. A missing ``gt.txt`` is the test split, which cannot be scored locally; a
            loader that returned an empty ground truth for it would report MOTA 0 and look
            like a broken tracker.
    """
    files = MotSequenceFiles(root)
    info = files.info()
    if not files.has_ground_truth:
        raise ConfigurationError(
            f"{files.root} has no gt/gt.txt. The MOTChallenge test split ships without one, "
            f"so it can be tracked but not scored — submit to the server for that"
        )
    if not files.has_detections:
        raise ConfigurationError(
            f"{files.root} has no det/det.txt. Public detections are what make a tracker "
            f"evaluable with no detector installed"
        )

    ground_truth, ignored, unscored = load_ground_truth(
        files.ground_truth,
        name=info.name,
        length=info.length,
        min_visibility=min_visibility,
    )
    detections = load_detections(
        files.detections,
        camera_id=info.name,
        length=info.length,
        height=info.height,
        width=info.width,
        min_score=min_score,
        frame_rate=info.frame_rate,
    )
    case = EvaluationCase(
        name=info.name,
        detections=detections,
        ground_truth=ground_truth,
        ignored=ignored,
        unscored=unscored,
        height=info.height,
        width=info.width,
        metadata={
            "root": str(files.root),
            "frame_rate": info.frame_rate,
            "people_per_frame": ground_truth.num_detections / max(1, info.length),
        },
    )
    return case.truncated(frames) if frames else case


def load_cases(
    root: Path | str,
    *,
    sequences: Sequence[str] | None = None,
    pattern: str = "MOT*",
    **options: object,
) -> list[EvaluationCase]:
    """Every sequence under ``root``, or the named subset, in name order.

    Args:
        root: the split directory, e.g. ``.../mot17/train``.
        sequences: names to keep. Matched on the directory name, and a name that matches
            nothing raises rather than being skipped — a typo in a sequence list that
            silently evaluates six sequences instead of seven produces a number nobody can
            reproduce.
        pattern: glob for the sequence directories.
        **options: forwarded to :func:`load_case`.
    """
    found = discover_sequences(root, pattern=pattern)
    if sequences is not None:
        wanted = list(sequences)
        by_name = {files.name: files for files in found}
        missing = [name for name in wanted if name not in by_name]
        if missing:
            raise ConfigurationError(
                f"no such sequence(s) under {root}: {missing}; available: {sorted(by_name)}"
            )
        found = [by_name[name] for name in wanted]
    return [load_case(files.root, **options) for files in found]  # type: ignore[arg-type]


def write_mot_file(path: Path | str, tracks: Iterable[Track]) -> int:
    """Write tracks in MOTChallenge submission format. Returns how many rows were written.

    ``frame, id, x, y, w, h, 1, -1, -1, -1`` — the boxes go back to ``x, y, w, h`` on the way
    out, in the same file that converts them on the way in, so the round trip is one function
    pair rather than a convention two modules have to share.

    This exists for the cross-check against TrackEval: the only way to be sure the metrics in
    this package agree with the reference implementation is to hand the reference the same
    tracker output in the format it reads.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for track in sorted(tracks, key=lambda t: (t.tag.frame_id, t.track_id)):
            x1, y1, x2, y2 = (float(v) for v in track.box)
            handle.write(
                f"{track.tag.frame_id},{track.track_id},{x1:.2f},{y1:.2f},"
                f"{x2 - x1:.2f},{y2 - y1:.2f},1,-1,-1,-1\n"
            )
            written += 1
    return written
