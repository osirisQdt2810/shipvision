"""The native backend: the C++ association loops in ``shipvision._C``.

This module is a translator, not an algorithm. The whole of SORT and ByteTrack exists in
``csrc/shipvision/tracking/`` — the Kalman filter, the IoU costs, the assignment solver and the
track lifecycle — and what is left in Python is marshalling plus the three things that must not
move:

**The tag.** ``(camera_id, frame_id)`` never crosses the boundary.
:meth:`~shipvision.tracking.base.BaseTracker.begin` already refuses a camera swap on a live
instance and a frame_id that does not advance, and both refusals are the difference between a
dropped frame and a real-looking detection on a camera where nothing happened. A second
implementation of that discipline in C++ would be a second place for it to be wrong, and it
would cost a string copy per track per frame to gain nothing.

**The track id.** The library's contract is that an id is unique across every tracker in the
process — camera 3's track 7 and camera 9's track 7 must not collide when their output meets
the cross-camera tier — so ids come from
:func:`~shipvision.tracking.base.next_track_id`. The C++ pool numbers its tracks locally and
this module maps a local id to a process-wide one, once, at birth.

**The appearance vector.** A track's embedding exists for the cross-camera tier downstream,
and two of the five algorithms also *associate* on it. Marshalling a 512-float vector per
track per frame into C++ to average it and marshal it straight back would cost far more than
the average, so the EMA stays here — through
:func:`~shipvision.tracking.pool.blend_embedding`, the same function the numpy pool calls, so
the two backends cannot produce different vectors. What crosses instead is the finished
``(tracks, detections)`` cosine-distance matrix, built by the same
:func:`~shipvision.tracking.association.appearance.pairwise_appearance` the numpy trackers
use: a few hundred floats a frame, and one decision about what "there is no appearance
evidence" means rather than two.

**The camera motion.** BoT-SORT's affine comes from
:mod:`shipvision.tracking.motion.cmc`, where a PTZ head's own encoder, an optical-flow
estimate and a ground-plane homography are three registered answers to one question. Keeping
the estimator in Python is what lets all three serve the compiled tracker; the binding is
handed the resulting ``(2, 3)`` matrix.

Importing this module never fails, even with no build. Only construction does, with
:class:`~shipvision.errors.BackendUnavailableError`, which is what lets
:mod:`shipvision.tracking` register the backend unconditionally and
``TRACKERS.build("sort")`` fall back to numpy without a try/import dance at the call site.

**What the native backend does not have is an algorithm of its own.** There are five classes
here and five under ``core/``, and every one of the five pairs is checked against the other in
``tests/tracking/backends/test_parity.py``: a compiled tracker nobody can compare against is a
compiled tracker nobody can trust, so the numpy twin is not a fallback, it is the oracle.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from shipvision.errors import BackendUnavailableError, ConfigurationError
from shipvision.registry import NATIVE
from shipvision.tracking.association import pairwise_appearance
from shipvision.tracking.base import BaseTracker, next_track_id
from shipvision.tracking.core.deepsortv2.utils import dynamic_momentum
from shipvision.tracking.motion.cmc import CAMERA_MOTION, CameraMotionEstimator
from shipvision.tracking.pool import blend_embedding
from shipvision.tracking.registry import TRACKERS
from shipvision.types import Detection, Detections, FrameTag, Track, TrackState

__all__ = [
    "NativeBotSortTracker",
    "NativeByteTrackTracker",
    "NativeDeepSortV2Tracker",
    "NativeOcSortTracker",
    "NativeSortTracker",
    "native_available",
]

try:  # pragma: no cover - depends on whether the extension was built, not on a branch
    from shipvision import _C

    _IMPORT_ERROR: str | None = None
except ImportError as exc:  # pragma: no cover
    _C = None  # type: ignore[assignment]
    _IMPORT_ERROR = str(exc)

_BUILD_HINT = (
    "Build it with `cmake -S . -B build && cmake --build build -j`, or use backend='python'"
)

#: ``meta`` column order, mirroring ``wrap_tracks`` in ``csrc/bindings/tracking.cpp``.
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


def _require_extension(what: str) -> None:
    """Raise the typed failure, naming the fix.

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


def _validate_lifecycle(max_age: int, min_hits: int) -> None:
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
    """The small part of :class:`~shipvision.tracking.pool.TrackPool` that
    :class:`~shipvision.tracking.base.BaseTracker` actually uses, over the C++ pool.

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

        All-or-nothing, matching :meth:`~shipvision.tracking.pool.TrackPool.embeddings`: a cost
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


class _NativeTracker(BaseTracker):
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
        """See :meth:`~shipvision.tracking.base.BaseTracker.update`.

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
        :func:`~shipvision.tracking.association.appearance.pairwise_appearance`, and "does this
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


@TRACKERS.register("sort", backend=NATIVE)
class NativeSortTracker(_NativeTracker):
    """SORT with its per-frame work in C++. See
    :class:`~shipvision.tracking.core.sort.tracker.SortTracker` for the algorithm.

    The keyword arguments are the numpy tracker's, exactly — same names, same defaults, same
    validation. That is not politeness: :mod:`shipvision.tune` validates a search space against
    whichever class the registry resolves, so a native tracker that quietly accepted a
    different set would let a study tune parameters the tracker it actually ran did not have.
    """

    def __init__(
        self,
        *,
        det_threshold: float = 0.5,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
        gate: bool = True,
    ) -> None:
        """
        Raises:
            BackendUnavailableError: the extension is not built, or is older than the trackers.
            ConfigurationError: an argument cannot work. Checked here as well as in the C++,
                because a caller who wrote ``iou_threshold=1.5`` deserves the same typed
                failure whichever backend the registry resolved.
        """
        _require_extension("SortTracker")
        _validate_lifecycle(max_age, min_hits)
        if not 0.0 <= det_threshold <= 1.0:
            raise ConfigurationError(f"det_threshold must be in [0, 1], got {det_threshold}")
        if not 0.0 < iou_threshold <= 1.0:
            raise ConfigurationError(f"iou_threshold must be in (0, 1], got {iou_threshold}")
        super().__init__(
            _C.SortTracker(
                det_threshold=float(det_threshold),
                iou_threshold=float(iou_threshold),
                max_age=int(max_age),
                min_hits=int(min_hits),
                gate=bool(gate),
            )
        )

    def describe(self) -> str:
        return "SORT: Kalman + IoU + one assignment per frame, in C++"


@TRACKERS.register("bytetrack", backend=NATIVE)
class NativeByteTrackTracker(_NativeTracker):
    """ByteTrack with its two association stages in C++. See
    :class:`~shipvision.tracking.core.bytetrack.tracker.ByteTrackTracker` for the algorithm.

    ``embedding_momentum`` is handled on this side rather than passed to the C++ session, for
    the reason in the module docstring — but it is still a constructor keyword with the same
    name and default, because the registry may hand this class to anything that was written
    against the numpy one.
    """

    def __init__(
        self,
        *,
        track_threshold: float = 0.5,
        low_threshold: float = 0.1,
        match_threshold: float = 0.2,
        second_match_threshold: float = 0.5,
        max_age: int = 30,
        min_hits: int = 3,
        gate: bool = True,
        embedding_momentum: float = 0.9,
    ) -> None:
        """
        Raises:
            BackendUnavailableError: the extension is not built, or is older than the trackers.
            ConfigurationError: the two score thresholds are the wrong way round, which would
                leave the high tier empty and mean no track is ever born.
        """
        _require_extension("ByteTrackTracker")
        _validate_lifecycle(max_age, min_hits)
        if not low_threshold < track_threshold <= 1.0:
            raise ConfigurationError(
                f"need 0 <= low_threshold ({low_threshold}) < track_threshold "
                f"({track_threshold}) <= 1"
            )
        super().__init__(
            _C.ByteTrackTracker(
                track_threshold=float(track_threshold),
                low_threshold=float(low_threshold),
                match_threshold=float(match_threshold),
                second_match_threshold=float(second_match_threshold),
                max_age=int(max_age),
                min_hits=int(min_hits),
                gate=bool(gate),
            ),
            embedding_momentum=embedding_momentum,
        )

    def describe(self) -> str:
        return (
            "ByteTrack: high-score association, then a second pass over the low-score "
            "leftovers, in C++"
        )


@TRACKERS.register("ocsort", backend=NATIVE)
class NativeOcSortTracker(_NativeTracker):
    """OC-SORT with its two association stages and its re-update in C++. See
    :class:`~shipvision.tracking.core.ocsort.tracker.OcSortTracker` for the algorithm.

    Nothing crosses the boundary that does not cross for SORT: OC-SORT's three fixes are all
    about the *filter and the observations*, which live in the C++ pool, and none of them is
    about appearance.
    """

    def __init__(
        self,
        *,
        det_threshold: float = 0.5,
        iou_threshold: float = 0.3,
        recovery_iou_threshold: float = 0.5,
        delta_t: int = 3,
        momentum_weight: float = 0.2,
        max_age: int = 30,
        min_hits: int = 3,
        gate: bool = True,
        re_update: bool = True,
        recover: bool = True,
    ) -> None:
        """
        Raises:
            BackendUnavailableError: the extension is not built, or is older than the trackers.
            ConfigurationError: an argument cannot work. Checked here as well as in the C++,
                because a caller who wrote ``delta_t=0`` deserves the same typed failure
                whichever backend the registry resolved.
        """
        _require_extension("OcSortTracker")
        _validate_lifecycle(max_age, min_hits)
        if not 0.0 <= det_threshold <= 1.0:
            raise ConfigurationError(f"det_threshold must be in [0, 1], got {det_threshold}")
        if not 0.0 < iou_threshold <= 1.0:
            raise ConfigurationError(f"iou_threshold must be in (0, 1], got {iou_threshold}")
        if not 0.0 < recovery_iou_threshold <= 1.0:
            raise ConfigurationError(
                f"recovery_iou_threshold must be in (0, 1], got {recovery_iou_threshold}"
            )
        if delta_t < 1:
            raise ConfigurationError(f"delta_t must be >= 1, got {delta_t}")
        if not 0.0 <= momentum_weight <= 1.0:
            raise ConfigurationError(
                f"momentum_weight must be in [0, 1], got {momentum_weight}"
            )
        super().__init__(
            _C.OcSortTracker(
                det_threshold=float(det_threshold),
                iou_threshold=float(iou_threshold),
                recovery_iou_threshold=float(recovery_iou_threshold),
                delta_t=int(delta_t),
                momentum_weight=float(momentum_weight),
                max_age=int(max_age),
                min_hits=int(min_hits),
                gate=bool(gate),
                re_update=bool(re_update),
                recover=bool(recover),
            )
        )

    def describe(self) -> str:
        return "OC-SORT: observation-centric momentum, recovery and re-update over SORT, in C++"


@TRACKERS.register("botsort", backend=NATIVE)
class NativeBotSortTracker(_NativeTracker):
    """BoT-SORT with ByteTrack's two stages in C++. See
    :class:`~shipvision.tracking.core.botsort.tracker.BotSortTracker` for the algorithm.

    The camera-motion estimator stays here rather than being reimplemented in C++, and that is
    the interesting half of this class. ``cmc="sparse_flow"`` needs OpenCV and pixels;
    ``cmc="external"`` takes the affine from PTZ telemetry, which beats any estimate made from
    the image. Both are registered Python objects, and what the binding receives is the ``(2,
    3)`` matrix they produce — so the compiled tracker gains a new motion model whenever the
    registry does, without a rebuild.
    """

    def __init__(
        self,
        *,
        cmc: str = "none",
        cmc_options: dict[str, object] | None = None,
        appearance_gate: float = 0.25,
        appearance_weight: float = 0.5,
        track_threshold: float = 0.5,
        low_threshold: float = 0.1,
        match_threshold: float = 0.2,
        second_match_threshold: float = 0.5,
        max_age: int = 30,
        min_hits: int = 3,
        gate: bool = True,
        embedding_momentum: float = 0.9,
    ) -> None:
        """
        Raises:
            BackendUnavailableError: the extension is not built, or is older than the trackers.
            ConfigurationError: an argument cannot work, including an ``appearance_gate``
                outside the range a cosine distance can take.
        """
        _require_extension("BotSortTracker")
        _validate_lifecycle(max_age, min_hits)
        if not low_threshold < track_threshold <= 1.0:
            raise ConfigurationError(
                f"need 0 <= low_threshold ({low_threshold}) < track_threshold "
                f"({track_threshold}) <= 1"
            )
        if not 0.0 < appearance_gate <= 2.0:
            raise ConfigurationError(
                f"appearance_gate is a cosine distance and must be in (0, 2], got "
                f"{appearance_gate}"
            )
        if not 0.0 < appearance_weight <= 1.0:
            raise ConfigurationError(
                f"appearance_weight must be in (0, 1], got {appearance_weight}"
            )
        super().__init__(
            _C.BotSortTracker(
                track_threshold=float(track_threshold),
                low_threshold=float(low_threshold),
                match_threshold=float(match_threshold),
                second_match_threshold=float(second_match_threshold),
                max_age=int(max_age),
                min_hits=int(min_hits),
                gate=bool(gate),
                appearance_gate=float(appearance_gate),
                appearance_weight=float(appearance_weight),
            ),
            embedding_momentum=embedding_momentum,
        )
        self._motion: CameraMotionEstimator = CAMERA_MOTION.build(cmc, **(cmc_options or {}))
        self._track_threshold = float(track_threshold)

    @property
    def camera_motion(self) -> CameraMotionEstimator:
        """The estimator, so a caller with PTZ telemetry can push into it."""
        return self._motion

    def _advance(
        self,
        detections: Detections,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        image: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        # The high tier, because that is the only stage BoT-SORT lets appearance into — and it
        # is the same set the numpy tracker asks `pairwise_appearance` about, so the two agree
        # on the frames where appearance evidence exists at all.
        appearance = self._appearance(
            detections, _columns_above(detections, self._track_threshold)
        )
        affine = np.ascontiguousarray(self._motion.estimate(image), dtype=np.float32)
        return self._session.update(boxes, scores, class_ids, appearance, affine)

    def reset(self) -> None:
        super().reset()
        self._motion.reset()

    def describe(self) -> str:
        return (
            f"BoT-SORT: ByteTrack + camera-motion compensation ({self._motion.name}) + "
            f"min-fused appearance, in C++"
        )


@TRACKERS.register("deepsortv2", backend=NATIVE)
class NativeDeepSortV2Tracker(_NativeTracker):
    """DeepSORTv2's four-stage cascade in C++. See
    :class:`~shipvision.tracking.core.deepsortv2.tracker.DeepSortV2Tracker` for the algorithm.

    Two things stay on this side, and both for the same reason: they are about the appearance
    *vector*, which never crosses. The cosine distances the cascade reads are built here, and
    so is the per-detection EMA rate that decides how far a crop moves a track's memory —
    :func:`~shipvision.tracking.core.deepsortv2.utils.dynamic_momentum`, the same function the
    numpy tracker calls, so a track's gallery vector is identical on both backends.
    """

    def __init__(
        self,
        *,
        det_threshold: float = 0.5,
        appearance_weight: float = 0.9,
        appearance_gate: float = 0.15,
        giou_gate: float = 1.2,
        stage_a_max_cost: float = 0.45,
        cascade_stride: int = 5,
        stage_b_max_cost: float = 0.55,
        stage_b_max_age: int = 6,
        stage_c_max_cost: float = 0.65,
        recover: bool = True,
        stage_d_max_cost: float = 0.8,
        border_fraction: float = 0.05,
        skip_border_recovery: bool = True,
        max_age: int = 30,
        min_hits: int = 3,
        re_update: bool = True,
        appearance_momentum: tuple[float, float] = (0.9, 0.95),
        dynamic_appearance: bool = True,
    ) -> None:
        """
        Raises:
            BackendUnavailableError: the extension is not built, or is older than the trackers.
            ConfigurationError: an argument cannot work.
        """
        _require_extension("DeepSortV2Tracker")
        _validate_lifecycle(max_age, min_hits)
        if not 0.0 <= det_threshold <= 1.0:
            raise ConfigurationError(f"det_threshold must be in [0, 1], got {det_threshold}")
        if not 0.0 <= appearance_weight <= 1.0:
            raise ConfigurationError(
                f"appearance_weight must be in [0, 1], got {appearance_weight}"
            )
        if cascade_stride < 1:
            raise ConfigurationError(f"cascade_stride must be >= 1, got {cascade_stride}")
        if not 0.0 <= border_fraction < 0.5:
            raise ConfigurationError(
                f"border_fraction must be in [0, 0.5), got {border_fraction}"
            )
        low, high = appearance_momentum
        if not 0.0 <= low <= high < 1.0:
            raise ConfigurationError(
                f"appearance_momentum must be an increasing pair inside [0, 1), got "
                f"{appearance_momentum}"
            )
        super().__init__(
            _C.DeepSortV2Tracker(
                det_threshold=float(det_threshold),
                appearance_weight=float(appearance_weight),
                appearance_gate=float(appearance_gate),
                giou_gate=float(giou_gate),
                stage_a_max_cost=float(stage_a_max_cost),
                cascade_stride=int(cascade_stride),
                stage_b_max_cost=float(stage_b_max_cost),
                stage_b_max_age=int(stage_b_max_age),
                stage_c_max_cost=float(stage_c_max_cost),
                recover=bool(recover),
                stage_d_max_cost=float(stage_d_max_cost),
                border_fraction=float(border_fraction),
                skip_border_recovery=bool(skip_border_recovery),
                max_age=int(max_age),
                min_hits=int(min_hits),
                re_update=bool(re_update),
            ),
            # The floor of the dynamic range, not a fixed rate: it is what a frame that gives
            # no reason to distrust a detection uses, so a frame with no dynamic rate degrades
            # to "update normally" rather than to "barely update" — which would freeze every
            # gallery vector the first time a frame came back empty.
            embedding_momentum=low,
        )
        self._det_threshold = float(det_threshold)
        self._momentum_bounds = (float(low), float(high))
        self._dynamic_appearance = bool(dynamic_appearance)

    def _advance(
        self,
        detections: Detections,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        image: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        appearance = self._appearance(
            detections, _columns_above(detections, self._det_threshold)
        )
        # The frame size, or zero when the caller does not know it. Zero means the border rule
        # that keeps stage C off half-visible objects is skipped rather than guessed.
        return self._session.update(
            boxes, scores, class_ids, appearance, int(detections.height), int(detections.width)
        )

    def _blend_rates(self, detections: Detections) -> float | np.ndarray:
        if not self._dynamic_appearance:
            return self._momentum
        columns = _columns_above(detections, self._det_threshold)
        rates = dynamic_momentum(
            [detections.items[column] for column in columns], bounds=self._momentum_bounds
        )
        if rates is None:
            return self._momentum
        # Widened to the frame's full detection list, because `Track.last_match` indexes it
        # that way. The detections below the threshold keep the floor and are never read: the
        # C++ tracker cannot match one, so it can never be a `last_match`.
        widened = np.full(len(detections.items), self._momentum, dtype=np.float32)
        widened[list(columns)] = rates
        return widened

    def describe(self) -> str:
        return (
            "DeepSORTv2: four-stage cascade (fused / IoU / observation-centric recovery / "
            "tentative) with re-update and a dynamic appearance EMA, in C++"
        )


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
