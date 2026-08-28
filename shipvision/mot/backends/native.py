"""The machinery every native tracker shares: extension loading and marshalling.

Each algorithm's compiled wrapper lives in its own package beside its readable twin —
`mot/trackers/deepsortv2/native.py` beside `tracker.py` — because a tracker is one algorithm
with two implementations, and keeping them apart is what let three of the five go for a
release with no compiled version and nothing saying so.

What stays here is the part that is *not* per algorithm: finding `shipvision._C`, refusing
with a build command when it is absent, and converting between numpy and the binding. Five
copies of a marshalling layer would be five places for a subtle disagreement about, say,
whether an empty detection set is `(0, 4)` or `(0,)`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from shipvision.errors import BackendUnavailableError, ConfigurationError
from shipvision.mot.association import pairwise_appearance
from shipvision.mot.base import BaseTracker, next_track_id
from shipvision.mot.pool import blend_embedding
from shipvision.types import Detection, Detections, FrameTag, Track, TrackState

#: The shared machinery, not the trackers. Each algorithm's native class lives in its own
#: ``trackers/<name>/tracker.py`` beside the Python one (V48), because both are implementations
#: of the same algorithm and splitting them by *implementation* rather than by algorithm put the
#: two halves of DeepSORTv2 in different directories. What is left here is what none of them
#: owns: finding the extension, decoding its two arrays, and the lifecycle check.
__all__ = [
    "NativeTracker",
    "native_available",
    "require_extension",
    "validate_lifecycle",
]

try:  # pragma: no cover - depends on whether the extension was built, not on a branch
    from shipvision._native import load_extension

    _C, _IMPORT_ERROR = load_extension()
except ImportError as exc:  # pragma: no cover
    _C = None  # type: ignore[assignment]
    _IMPORT_ERROR = str(exc)

_BUILD_HINT = (
    "Build it with `cmake -S . -B build && cmake --build build -j`, or use backend='python'"
)

#: ``meta`` column order, mirroring ``wrap_tracks`` in ``csrc/bindings/mot.cpp``.
_TRACK_ID, _CLASS_ID, _STATE, _AGE, _HITS, _MISSES, _LAST_MATCH = range(7)
_META_COLUMNS = 7

#: ``TrackState`` in the same order as the C++ enum. A list rather than a dict because the
#: enum is an index by construction on both sides, and a mapping would invite the two to be
#: reordered independently.
_STATES = (TrackState.TENTATIVE, TrackState.CONFIRMED, TrackState.LOST, TrackState.REMOVED)

#: "This frame carries no appearance evidence", as the binding spells it. A distinct answer
#: from a matrix of zeros, which would assert that every pair looks identical — the strongest
#: claim available, made on no evidence at all.
_NO_APPEARANCE = np.zeros((0, 0), dtype=np.float32)


def native_available() -> bool:
    """True when ``shipvision._C`` imported and carries the compiled trackers.

    Deliberately **not** a device check, unlike
    :func:`shipvision.imgproc.backends.native_ops.native_available`. These trackers are host
    C++ — an association loop over fifteen boxes has nothing to gain from a GPU — so a build
    on a machine with no driver runs them perfectly well, and requiring a device here would
    skip the parity tests on exactly the machines where they are cheapest to run.
    """
    return _C is not None and hasattr(_C, "SortTracker")


def require_extension(what: str) -> Any:
    """Check for ``what``, then hand back the extension it is on.

    **Returns the module rather than None**, so a caller cannot hold the handle without having
    passed the check. That is not tidiness: when each algorithm's native class moved into its
    own ``tracker.py`` (V48), five of them called ``_C.XTracker(...)`` on a name only this
    module imports, and every one raised ``NameError: name '_C' is not defined`` — 207 tests,
    from a construction line that reads perfectly. Returning the module is what makes the
    working spelling the short one.

    Raises:
        BackendUnavailableError: there is no build here. Distinct from
            :class:`~shipvision.errors.ConfigurationError` on purpose: the configuration is
            fine, the machine is missing a runtime, and an operator fixes those in different
            places.
    """
    if _C is None:
        raise BackendUnavailableError(
            f"shipvision._C is not built: {_IMPORT_ERROR}. {_BUILD_HINT}"
        )
    if not hasattr(_C, what):
        raise BackendUnavailableError(
            f"shipvision._C is built but has no {what}: it predates the native trackers. "
            f"Rebuild it — {_BUILD_HINT}"
        )
    return _C


def validate_lifecycle(max_age: int, min_hits: int) -> None:
    """Refuse a lifecycle the pool cannot express, in Python, before the binding sees it.

    The C++ pool checks the same thing — it has to, since its constructor is reachable from
    the bindings without this — but a ``std::invalid_argument`` surfaces in Python as a bare
    ``ValueError``, and this library's contract is that a component built with arguments that
    cannot work raises :class:`~shipvision.errors.ConfigurationError`. A caller who wrote
    ``max_age=0`` must get the same typed failure whichever backend the registry resolved,
    because the ``except`` clause that handles it was written once.

    Raises:
        ConfigurationError: either value is below 1.
    """
    if max_age < 1 or min_hits < 1:
        raise ConfigurationError(
            f"max_age ({max_age}) and min_hits ({min_hits}) must both be >= 1"
        )


class _NativePool:
    """The small part of :class:`~shipvision.mot.pool.TrackPool` that
    :class:`~shipvision.mot.base.BaseTracker` actually uses, over the C++ pool.

    ``BaseTracker`` needs four things from whatever holds the track state: ``predict``,
    ``reset``, ``__len__`` and ``tracks``. Satisfying that shape instead of subclassing the
    numpy pool is what lets the shared contract — the tag discipline, ``pool_size``, ``reset``
    — apply unchanged to a tracker whose state lives on the other side of a binding.

    :meth:`predict` only records the tag. The C++ ``update`` advances its own filters, and
    splitting predict from update across the boundary would double the crossings per frame to
    reproduce a step whose result nothing on this side reads.
    """

    def __init__(self, session: Any) -> None:
        self._session = session
        self._tag: FrameTag | None = None
        #: Pool-local id to the process-wide one. Bounded by the live track count: an entry is
        #: dropped as soon as the C++ pool stops reporting the track, which is what keeps a
        #: process that runs for weeks from accumulating one entry per object ever seen.
        self._ids: dict[int, int] = {}
        self._embeddings: dict[int, np.ndarray] = {}

    def predict(self, tag: FrameTag) -> None:
        self._tag = tag

    def reset(self) -> None:
        self._session.reset()
        self._ids.clear()
        self._embeddings.clear()

    def __len__(self) -> int:
        return int(self._session.size)

    @property
    def tracks(self) -> list[Track]:
        """Every live track, including tentative and lost ones."""
        return self.decode(*self._session.tracks())

    def decode(self, geometry: np.ndarray, meta: np.ndarray) -> list[Track]:
        """The two arrays the binding returns, as :class:`~shipvision.types.Track` objects.

        Raises:
            ConfigurationError: the extension's array layout disagrees with this module's. A
                stale ``_C`` built before a column was added would otherwise be read with
                every field shifted by one, which produces plausible tracks with the wrong
                ages and states rather than an error.
        """
        if self._tag is None:
            raise ConfigurationError("predict() must open the frame before its tracks are read")
        if meta.ndim != 2 or meta.shape[1] != _META_COLUMNS:
            raise ConfigurationError(
                f"shipvision._C returned {meta.shape} track metadata where "
                f"(n, {_META_COLUMNS}) was expected; the extension and this module were built "
                f"from different revisions"
            )
        return [
            Track(
                track_id=self._global_id(int(row[_TRACK_ID])),
                box=geometry[index, :4],
                tag=self._tag,
                state=_STATES[int(row[_STATE])],
                score=float(geometry[index, 4]),
                class_id=int(row[_CLASS_ID]),
                age=int(row[_AGE]),
                hits=int(row[_HITS]),
                time_since_update=int(row[_MISSES]),
                embedding=self._embeddings.get(int(row[_TRACK_ID])),
            )
            for index, row in enumerate(meta)
        ]

    def track_embeddings(self) -> np.ndarray | None:
        """``(n, d)`` appearance vectors in the C++ pool's **row order**, or ``None``.

        All-or-nothing, matching :meth:`~shipvision.mot.pool.TrackPool.embeddings`: a cost
        matrix whose rows are half appearance-based and half not is not a matrix anyone can
        threshold.

        The row order is what makes this usable as a cost matrix's rows at all. It comes from
        the C++ pool itself rather than from this map's insertion order, because a dict ordered
        by when a track was born stops matching the pool the first time `sweep()` drops one
        from the middle — and the result is a plausible-looking matrix that scores every track
        against the wrong detections.
        """
        _, meta = self._session.tracks()
        vectors = [self._embeddings.get(int(row[_TRACK_ID])) for row in meta]
        if not vectors or any(vector is None for vector in vectors):
            return None
        return np.stack(vectors)

    def _global_id(self, local_id: int) -> int:
        """A process-wide id for a pool-local one, allocated once at the track's birth."""
        identity = self._ids.get(local_id)
        if identity is None:
            identity = next_track_id()
            self._ids[local_id] = identity
        return identity

    def absorb(
        self,
        geometry: np.ndarray,
        meta: np.ndarray,
        detections: Sequence[Detection],
        momentum: float | np.ndarray,
    ) -> list[Track]:
        """One frame's live set: fold in this frame's appearance, evict the dead, decode.

        The three happen together because they read the same array once. The order matters —
        the appearance is blended *before* the tracks are decoded, or a track would be
        published carrying the vector it had a frame ago.

        ``momentum`` is one rate for the whole frame, or one per detection indexed the way
        ``detections`` is. DeepSORTv2 derives the per-detection rate from confidence and
        crowding, so a clean isolated crop moves a track's appearance further than a
        half-occluded one does — the same signal the numpy pool takes through
        ``apply_matches(embedding_momentum=...)``.

        The eviction is what keeps this backend honest about "nothing grows without bound".
        Both maps are keyed on a pool-local id and follow the C++ pool's own lifecycle rather
        than keeping a second age rule, so there is one decision about when a track is gone
        instead of two that can disagree. It runs on **every** frame, including the ones with
        no embeddings and the ones with no detections at all: an earlier version evicted only
        while blending, so a purely geometric tracker — which is what ByteTrack normally is —
        accumulated one entry per object it had ever seen, and a camera running for a week
        would have shown it as a slow leak with no other symptom.
        """
        live: set[int] = set()
        for row in meta:
            local_id = int(row[_TRACK_ID])
            live.add(local_id)
            column = int(row[_LAST_MATCH])
            if column < 0:
                continue
            rate = momentum if isinstance(momentum, float) else float(momentum[column])
            blended = blend_embedding(
                self._embeddings.get(local_id), detections[column].embedding, rate
            )
            if blended is not None:
                self._embeddings[local_id] = blended
        self._ids = {key: value for key, value in self._ids.items() if key in live}
        self._embeddings = {
            key: value for key, value in self._embeddings.items() if key in live
        }
        return self.decode(geometry, meta)


class NativeTracker(BaseTracker):
    """Marshalling shared by the native trackers: detections in, tracks out.

    Five trackers, one translator. What the subclasses differ in is which C++ session they
    open and what else that session needs to see — which is the same relationship the five
    ``core/`` packages have to ``TrackPool``: the algorithm is the association, and everything
    around it is shared.
    """

    def __init__(self, session: Any, *, embedding_momentum: float = 0.9) -> None:
        if not 0.0 <= embedding_momentum < 1.0:
            raise ConfigurationError(
                f"embedding_momentum must be in [0, 1), got {embedding_momentum}"
            )
        pool = _NativePool(session)
        super().__init__(pool)  # type: ignore[arg-type]
        self._native_pool = pool
        self._session = session
        self._momentum = float(embedding_momentum)

    def update(self, detections: Detections, *, image: np.ndarray | None = None) -> list[Track]:
        """See :meth:`~shipvision.mot.base.BaseTracker.update`.

        ``image`` is accepted and ignored by every tracker here except BoT-SORT: a library that
        demanded pixels from every caller would be unusable by one that only has boxes, which
        is what an evaluation over an MOT ground-truth file is.
        """
        self.begin(detections)
        boxes, scores, class_ids = _as_arrays(detections)
        geometry, meta = self._advance(detections, boxes, scores, class_ids, image)
        live = self._native_pool.absorb(
            geometry, meta, detections.items, self._blend_rates(detections)
        )
        # The publishable rule is `Track.is_publishable` and lives there for both backends: a
        # LOST track's box is a prediction no detector saw, and emitting it is how a phantom
        # object drifts across a scene. Filtering here rather than in C++ is what stops the
        # rule from existing twice.
        return [track for track in live if track.is_publishable]

    def _advance(
        self,
        detections: Detections,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        image: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Hand one frame to the C++ session and take back the live set.

        The geometric trackers need nothing else. The two that associate on appearance, and the
        one that compensates for camera motion, override this to build the extra arguments —
        which is where the appearance matrix and the ``(2, 3)`` affine are assembled, on this
        side of the boundary, from the estimator and the vectors that live here.
        """
        return self._session.update(boxes, scores, class_ids)

    def _blend_rates(self, detections: Detections) -> float | np.ndarray:
        """The EMA retention to age each matched track's appearance by. One rate, by default."""
        return self._momentum

    def _appearance(self, detections: Detections, columns: Sequence[int]) -> np.ndarray:
        """``(live tracks, detections)`` cosine distance for ``columns``, or the empty matrix.

        ``columns`` is the tier the tracker's appearance-using stage may actually see — the
        high-score detections for BoT-SORT, the kept ones for DeepSORTv2 — because that is what
        the numpy tracker passes to
        :func:`~shipvision.mot.association.appearance.pairwise_appearance`, and "does this
        frame have appearance evidence" is an all-or-nothing question asked over exactly that
        set. Asking it over the whole frame instead would let one low-score detection with no
        crop turn appearance off for a stage that was never going to look at it.

        The result is widened back to the frame's full detection list because the C++ side
        indexes columns the way ``Track.last_match`` does — by position in the list ``update``
        was handed — so a matrix in tier coordinates would score the wrong crops. The columns no
        stage will read stay zero and are never consulted.
        """
        tracks = self._native_pool.track_embeddings()
        if tracks is None or not columns:
            return _NO_APPEARANCE
        distances = pairwise_appearance(
            tracks, range(len(tracks)), [detections.items[c].embedding for c in columns]
        )
        if distances is None:
            return _NO_APPEARANCE
        widened = np.zeros((len(tracks), len(detections.items)), dtype=np.float32)
        widened[:, list(columns)] = distances
        return widened


# -- shared between the wrappers -------------------------------------------------------------
#
# These live here rather than in one algorithm's `tracker.py` because three callers need them:
# `NativeTracker.update` below, BoT-SORT's `_advance`, and DeepSORTv2's. When each algorithm's
# native class moved into its own file (V48) they went with DeepSORTv2, and the two callers left
# behind raised `NameError` at their first frame — which the type checker could not see, because
# a name used only inside a method body is not resolved until it runs.


def _columns_above(detections: Detections, threshold: float) -> list[int]:
    """The positions of the detections at or above ``threshold``, in input order.

    Positions rather than a filtered :class:`~shipvision.types.Detections`, because everything
    that crosses the binding is indexed by position in the list ``update`` was handed — see
    ``Track.last_match``. The predicate is the same ``>=`` as
    :meth:`~shipvision.types.Detections.filter`, and it exists here as well because this side
    has to name the same tier the C++ tracker will build for itself: a wrapper that split at a
    different threshold would hand the cascade an appearance matrix over the wrong crops.
    """
    return [index for index, item in enumerate(detections.items) if item.score >= threshold]


def _as_arrays(detections: Detections) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One frame's detections as the three contiguous arrays the binding reads.

    Built here rather than in the binding because an empty frame is ordinary input — a quiet
    camera — and ``Detections.boxes`` already promises ``(0, 4)`` rather than ``(0,)`` for it.
    The binding refuses ``(0,)``, so getting the shape right on this side is what keeps a quiet
    camera from raising.
    """
    return (
        np.ascontiguousarray(detections.boxes, dtype=np.float32),
        np.ascontiguousarray(detections.scores, dtype=np.float32),
        np.ascontiguousarray(detections.class_ids, dtype=np.int32),
    )
