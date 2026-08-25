"""The shared vocabulary. Every stage speaks these and nothing else.

This module is the contract between the algorithm library and the server that calls it, and
between each algorithm and the next. It is deliberately small and deliberately boring: plain
dataclasses over numpy arrays, validation at construction, no behaviour that could differ
between two stages.

Three conventions are fixed here once so that nothing downstream has to ask, because every
reference implementation this library replaces got at least one of them wrong somewhere:

**Boxes are ``xyxy`` float32, absolute pixels.** Not ``xywh``, not normalised, not
``cxcywh``. Detectors emit whatever their head emits and convert at their own boundary; past
that boundary there is one format. The bug this prevents is real and was found in the
references: a tracker whose Kalman state is ``(cx, cy, aspect, height)`` fed by a converter
that wrote width where height belonged tracks square objects perfectly and falls apart on a
ship.

**Every frame carries ``(camera_id, frame_id)`` from ingest to output, including on error
paths.** A result that loses its tag is worse than a dropped frame: a dropped frame is
counted, while a mis-tagged one is attributed to the wrong camera and looks like a real
detection somewhere it never happened.

**Embeddings are stored L2-normalised.** Normalising once on the way in, rather than inside
every distance function, is the difference between a similarity search costing one gemm and
costing a gemm plus a full pass over the gallery.

**Nothing non-finite gets in.** A NaN or an inf is refused at construction rather than
carried, because it does not stay local: `np.maximum(norm, eps)` propagates NaN, any
reduction over a matrix containing one poisons every row, and `argsort` on an all-NaN row
falls back to array order — which means a ranking metric computed over poisoned scores comes
back *higher* than the true one. An fp16 engine emitting an inf is the ordinary way this
happens, and a failure that flatters the measurement is the worst kind to allow through.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from shipvision.errors import ConfigurationError

__all__ = [
    "Detection",
    "Detections",
    "Embedding",
    "Frame",
    "FrameTag",
    "GlobalTrack",
    "Track",
    "TrackState",
    "cxcyah_to_xyxy",
    "cxcywh_to_xyxy",
    "iou_matrix",
    "xyxy_to_cxcyah",
    "xyxy_to_cxcywh",
]


def _as_unit_vector(array: np.ndarray, what: str) -> np.ndarray:
    """One embedding, in the form this module promises everything stores it in.

    The module docstring says embeddings are stored L2-normalised, and until this existed that
    was a sentence rather than a fact: ``Embedding(vector=[3.0, 4.0]).vector`` came back with
    norm 5. Nothing raised, and nothing would — the consequence lands in whoever believes the
    contract. The whole point of normalising here is that a gallery can compute cosine
    similarity as a plain dot product, which is one gemm instead of a gemm plus a pass over the
    gallery; against un-normalised rows, two identical ``np.ones(512)`` vectors score 512, every
    ``sim > 0.5`` gate admits everything, and a quality-weighted aggregator ends up weighted by
    whichever crop had the largest activations rather than by ``quality``. The mAP simply comes
    out wrong.

    So the invariant is enforced in the one place all three carriers share — ``Embedding``,
    ``Detection`` and ``Track`` — rather than in three copies that can drift.

    A zero vector is refused rather than normalised. It has no direction: dividing gives NaN,
    and leaving it alone gives a row at cosine 0 from everything, which is a *plausible*
    answer to every query and therefore the worse of the two failures.

    Raises:
        ConfigurationError: empty, non-finite, or all-zero.
    """
    vector = np.asarray(array, dtype=np.float32).reshape(-1)
    if vector.size == 0:
        raise ConfigurationError(f"an {what} cannot be empty")
    _reject_non_finite(vector, what)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ConfigurationError(
            f"an all-zero {what} has no direction, so it cannot be normalised. Left alone it "
            f"sits at cosine 0 from every gallery entry, which is a plausible-looking answer "
            f"to every query rather than an obvious failure"
        )
    return vector / norm


def _reject_non_finite(array: np.ndarray, what: str) -> None:
    """Refuse NaN and inf at the boundary, naming how many and where.

    Reported rather than merely refused because the usual cause is one bad crop out of tens
    of thousands, and knowing it was one row of 512 rather than the whole batch is the
    difference between a bug hunt and a dropped frame.
    """
    bad = ~np.isfinite(array)
    if bad.any():
        first = int(np.flatnonzero(bad.reshape(-1))[0])
        raise ConfigurationError(
            f"{what} has {int(bad.sum())} non-finite value(s), the first at index {first}. "
            f"A NaN does not stay local: it propagates through every reduction, and a "
            f"ranking computed over poisoned scores comes back better than the truth"
        )


# --------------------------------------------------------------------------- identity


@dataclass(slots=True, frozen=True)
class FrameTag:
    """Where and when a frame came from. Immutable, and it travels with everything.

    ``frame_id`` is per-camera and monotonic, assigned by the ingest actor that owns that
    camera — not a global counter. A global counter would need synchronising across fifty
    threads to hand out numbers nobody compares across cameras anyway.

    ``timestamp`` is the capture time in seconds since the epoch, as reported by the
    decoder, *not* the time this object was built. The gap between them is queue latency,
    and measuring it is the whole point of carrying the field.
    """

    camera_id: str
    frame_id: int
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise ConfigurationError(f"frame_id must be non-negative, got {self.frame_id}")

    def __str__(self) -> str:
        return f"{self.camera_id}#{self.frame_id}"


@dataclass(slots=True)
class Frame:
    """One decoded image plus its tag.

    ``image`` is whatever the ingest backend produced — an ``np.ndarray`` in HWC BGR for a
    CPU decoder, or an opaque device handle for a GPU one. It is typed ``Any`` on purpose:
    the point of hardware decode is that the pixels never reach host memory, and a type
    that insisted on ``np.ndarray`` here would force a copy that costs more than the
    inference.
    """

    tag: FrameTag
    image: Any
    height: int = 0
    width: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def camera_id(self) -> str:
        return self.tag.camera_id

    @property
    def frame_id(self) -> int:
        return self.tag.frame_id


# ------------------------------------------------------------------------- detections


@dataclass(slots=True)
class Detection:
    """One detected object in one frame.

    Attributes:
        box: ``(4,)`` float32 ``xyxy`` in absolute pixels.
        score: detector confidence in [0, 1].
        class_id: integer class. Semantics are the model's; this library does not own a
            class list, because a hard-coded one is how a detector swap becomes a rewrite.
        embedding: appearance vector, once a re-ID model has run. `None` until then.
        mask: ``(h, w)`` instance mask in the box's frame of reference, if segmented.
        keypoints: ``(k, 2)`` or ``(k, 3)`` with confidence.
    """

    box: np.ndarray
    score: float = 1.0
    class_id: int = 0
    embedding: np.ndarray | None = None
    mask: np.ndarray | None = None
    keypoints: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.box = np.asarray(self.box, dtype=np.float32).reshape(-1)
        if self.box.shape[0] != 4:
            raise ConfigurationError(f"box must be xyxy with 4 values, got {self.box.shape}")
        if not np.all(np.isfinite(self.box)):
            raise ConfigurationError(
                f"box contains a non-finite value: {self.box.tolist()}. A NaN box compares "
                f"false against every threshold, so it is silently never matched rather than "
                f"reported"
            )
        if self.box[2] < self.box[0] or self.box[3] < self.box[1]:
            raise ConfigurationError(
                f"box is inside-out: {self.box.tolist()}. xyxy means (x1, y1, x2, y2) with "
                f"x2 >= x1 — a converter that wrote xywh here would produce exactly this"
            )
        if not 0.0 <= self.score <= 1.0:
            raise ConfigurationError(f"score must be in [0, 1], got {self.score}")
        if self.embedding is not None:
            self.embedding = _as_unit_vector(self.embedding, "embedding")

    @property
    def width(self) -> float:
        return float(self.box[2] - self.box[0])

    @property
    def height(self) -> float:
        return float(self.box[3] - self.box[1])

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def centre(self) -> tuple[float, float]:
        return (
            float((self.box[0] + self.box[2]) * 0.5),
            float((self.box[1] + self.box[3]) * 0.5),
        )


@dataclass(slots=True)
class Detections:
    """Every detection in one frame, with the frame's tag attached.

    A list of :class:`Detection` plus a :class:`FrameTag` — rather than a bare list — because
    the tag must not be reconstructible-by-convention at the next stage. Handing a tracker a
    list and telling it the camera separately is how a result ends up under the wrong
    camera's name.
    """

    tag: FrameTag
    items: list[Detection] = field(default_factory=list)
    height: int = 0
    width: int = 0

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[Detection]:
        return iter(self.items)

    def __getitem__(self, index: int) -> Detection:
        return self.items[index]

    @property
    def boxes(self) -> np.ndarray:
        """``(n, 4)`` float32. An empty frame gives ``(0, 4)``, not ``(0,)``.

        The shape matters: ``(0,)`` breaks every downstream ``[:, 2]`` with an IndexError
        instead of yielding an empty result, and an empty frame is normal input.
        """
        if not self.items:
            return np.zeros((0, 4), dtype=np.float32)
        return np.stack([d.box for d in self.items])

    @property
    def scores(self) -> np.ndarray:
        return np.array([d.score for d in self.items], dtype=np.float32)

    @property
    def class_ids(self) -> np.ndarray:
        return np.array([d.class_id for d in self.items], dtype=np.int32)

    @property
    def embeddings(self) -> np.ndarray | None:
        """``(n, d)`` if *every* detection has one, else `None`.

        All-or-nothing on purpose. A partially-embedded batch silently becomes a cost matrix
        where some rows are appearance-based and some are not, which is not a matrix anyone
        can reason about — the caller must decide what to do, so it is told.
        """
        if not self.items or any(d.embedding is None for d in self.items):
            return None
        return np.stack([d.embedding for d in self.items])  # type: ignore[misc]

    def filter(
        self, *, min_score: float = 0.0, class_ids: Sequence[int] | None = None
    ) -> Detections:
        """A new :class:`Detections` with the same tag and a subset of items."""
        keep = [
            d
            for d in self.items
            if d.score >= min_score and (class_ids is None or d.class_id in class_ids)
        ]
        return Detections(tag=self.tag, items=keep, height=self.height, width=self.width)


@dataclass(slots=True)
class Embedding:
    """One appearance vector with the context needed to judge it.

    ``camera_id`` is load-bearing rather than metadata: the standard re-identification
    protocol excludes gallery entries from the query's own camera, because matching an
    identity to itself in the same view measures tracking and inflates every score.
    """

    vector: np.ndarray
    identity: str | None = None
    camera_id: str | None = None
    frame_id: int | None = None
    quality: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vector = _as_unit_vector(self.vector, "embedding")
        if not 0.0 <= self.quality <= 1.0:
            raise ConfigurationError(f"quality must be in [0, 1], got {self.quality}")

    @property
    def dim(self) -> int:
        return int(self.vector.shape[0])


# ----------------------------------------------------------------------------- tracks


class TrackState:
    """A track's lifecycle stage.

    Plain class constants rather than an enum so the values survive JSON and Kafka
    round-trips unchanged — these cross a process boundary on the way to MTMC, and an enum
    that serialises as ``"TrackState.CONFIRMED"`` on one side and has to be parsed back on
    the other is a bug waiting for a version skew.
    """

    TENTATIVE = "tentative"
    """Seen, not yet trusted. Not published: publishing a track that dies after two frames
    hands downstream an identity that never existed."""
    CONFIRMED = "confirmed"
    LOST = "lost"
    """Not matched this frame, still within ``max_age``. Predicted but not published."""
    REMOVED = "removed"

    ALL = (TENTATIVE, CONFIRMED, LOST, REMOVED)


@dataclass(slots=True)
class Track:
    """One identity within one camera, at one frame."""

    track_id: int
    box: np.ndarray
    tag: FrameTag
    state: str = TrackState.CONFIRMED
    score: float = 1.0
    class_id: int = 0
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
    embedding: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.box = np.asarray(self.box, dtype=np.float32).reshape(-1)
        if self.box.shape[0] != 4:
            raise ConfigurationError(f"box must be xyxy with 4 values, got {self.box.shape}")
        # A Kalman filter that has diverged produces a NaN state, and a track carrying one
        # would be published to MTMC, where it lands in a distance matrix and poisons the
        # clustering for every camera at that instant — not just its own.
        _reject_non_finite(self.box, "track box")
        # A track's embedding is stored under the same contract as everyone else's. It was
        # checked nowhere before this: a track is what MTMC clusters on, so an un-normalised
        # one is the carrier that reaches the furthest.
        if self.embedding is not None:
            self.embedding = _as_unit_vector(self.embedding, "track embedding")
        if self.state not in TrackState.ALL:
            raise ConfigurationError(f"unknown state {self.state!r}; expected {TrackState.ALL}")

    @property
    def camera_id(self) -> str:
        return self.tag.camera_id

    @property
    def is_publishable(self) -> bool:
        """Confirmed and updated on this frame.

        Both halves matter. A LOST track has a predicted box that no detector saw, and
        emitting it as an observation is how a phantom object drifts across a scene.
        """
        return self.state == TrackState.CONFIRMED and self.time_since_update == 0


@dataclass(slots=True)
class GlobalTrack:
    """One identity across cameras — the MTMC output.

    ``global_id`` is `None` rather than ``-1`` when unassigned. ``-1`` is the reference
    implementations' convention and it leaks: it compares, sorts and serialises as a
    perfectly ordinary id, so an unassigned track flows downstream looking assigned. `None`
    fails loudly at the first use.
    """

    global_id: int | None
    track: Track
    cluster_id: str | None = None
    members: tuple[tuple[str, int], ...] = ()
    """Every ``(camera_id, track_id)`` currently believed to be this identity."""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_assigned(self) -> bool:
        return self.global_id is not None


# ------------------------------------------------------------------------ conversions


def xyxy_to_cxcyah(boxes: np.ndarray) -> np.ndarray:
    """``(n, 4)`` xyxy → ``(n, 4)`` ``(cx, cy, aspect, height)``.

    Aspect ratio and **height** — the DeepSORT/ByteTrack Kalman convention. Height rather
    than width because height is the more stable measurement under partial occlusion from
    below, which is the common case for a person walking behind something.
    """
    b = np.atleast_2d(np.asarray(boxes, dtype=np.float32))
    widths = b[:, 2] - b[:, 0]
    heights = b[:, 3] - b[:, 1]
    safe = np.maximum(heights, 1e-6)
    return np.stack(
        [
            (b[:, 0] + b[:, 2]) * 0.5,
            (b[:, 1] + b[:, 3]) * 0.5,
            widths / safe,
            heights,
        ],
        axis=1,
    )


def cxcyah_to_xyxy(states: np.ndarray) -> np.ndarray:
    """Inverse of :func:`xyxy_to_cxcyah`."""
    s = np.atleast_2d(np.asarray(states, dtype=np.float32))
    heights = s[:, 3]
    widths = s[:, 2] * heights
    return np.stack(
        [
            s[:, 0] - widths * 0.5,
            s[:, 1] - heights * 0.5,
            s[:, 0] + widths * 0.5,
            s[:, 1] + heights * 0.5,
        ],
        axis=1,
    )


def xyxy_to_cxcywh(boxes: np.ndarray) -> np.ndarray:
    """xyxy → ``(cx, cy, w, h)``, the format most detector heads emit."""
    b = np.atleast_2d(np.asarray(boxes, dtype=np.float32))
    return np.stack(
        [
            (b[:, 0] + b[:, 2]) * 0.5,
            (b[:, 1] + b[:, 3]) * 0.5,
            b[:, 2] - b[:, 0],
            b[:, 3] - b[:, 1],
        ],
        axis=1,
    )


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Inverse of :func:`xyxy_to_cxcywh`."""
    b = np.atleast_2d(np.asarray(boxes, dtype=np.float32))
    half_w = b[:, 2] * 0.5
    half_h = b[:, 3] * 0.5
    return np.stack(
        [b[:, 0] - half_w, b[:, 1] - half_h, b[:, 0] + half_w, b[:, 1] + half_h],
        axis=1,
    )


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``(n, m)`` pairwise IoU between two sets of xyxy boxes.

    Vectorised rather than looped because this runs once per camera per frame — 50 cameras
    at 20 fps is a thousand calls a second, and a Python loop over 15x15 boxes costs more
    than the whole rest of the association.
    """
    boxes_a = np.atleast_2d(np.asarray(a, dtype=np.float32))
    boxes_b = np.atleast_2d(np.asarray(b, dtype=np.float32))
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float32)

    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    overlap = np.prod(np.clip(bottom_right - top_left, 0.0, None), axis=2)

    area_a = np.prod(np.clip(boxes_a[:, 2:] - boxes_a[:, :2], 0.0, None), axis=1)
    area_b = np.prod(np.clip(boxes_b[:, 2:] - boxes_b[:, :2], 0.0, None), axis=1)
    union = area_a[:, None] + area_b[None, :] - overlap
    return (overlap / np.maximum(union, 1e-9)).astype(np.float32)
