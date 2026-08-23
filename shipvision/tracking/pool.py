"""The Kalman-backed bookkeeping every tracker in this package shares.

Split out because the five trackers here differ only in *how they associate*. Everything
else — predicting, promoting out of TENTATIVE, ageing, killing, blending appearance — is
identical, and duplicating it is how five trackers in one codebase drift until only one of
them is correct.

Two capabilities are off by default and switched on by the trackers that need them, rather
than living in those trackers:

* ``observation_history`` keeps a **bounded** ring of past measurements per track, which is
  what "observation-centric" association reads instead of the filter's extrapolation.
* ``re_update`` enables OC-SORT's re-update along a virtual trajectory when a track is
  re-found after a gap.

Both are here rather than in ``ocsort.py`` because the state they need is the filter state,
and a tracker reaching into another object's covariance to rewind it is how a shared
component stops being shared.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence

import numpy as np

from shipvision.errors import ConfigurationError
from shipvision.tracking.base import next_track_id
from shipvision.tracking.motion.kalman import KalmanFilter
from shipvision.types import (
    Detection,
    FrameTag,
    Track,
    TrackState,
    cxcyah_to_xyxy,
    xyxy_to_cxcyah,
)

__all__ = ["TrackPool"]


class TrackPool:
    """Holds every live track's filter state as dense arrays.

    Dense rather than one filter object per track: the predict step is then a single matrix
    multiply for the whole set instead of N Python calls, and at fifty cameras that
    difference is the frame budget.

    The invariant the whole class exists to keep is that row ``i`` of every array describes
    ``tracks[i]``. :meth:`sweep` rebuilds all of them from one mask for exactly that reason.

    Args:
        max_age: frames a confirmed track survives without a match before it is dropped.
        min_hits: matches before a track is published.
        embedding_momentum: default EMA retention for a track's appearance vector. Callers
            may override it per detection — see :meth:`apply_matches`.
        observation_history: how many past measurements to remember per track. ``0`` keeps
            only the most recent, which is all an IoU-against-last-observation recovery
            needs; a momentum term needs ``delta_t + 1``. Bounded because a process here
            runs for weeks.
        re_update: rebuild the filter along a virtual trajectory when a gapped track is
            re-found, instead of feeding one distant measurement to a filter whose
            covariance has been inflating for the whole gap.
    """

    def __init__(
        self,
        *,
        max_age: int,
        min_hits: int,
        embedding_momentum: float = 0.9,
        observation_history: int = 0,
        re_update: bool = False,
    ) -> None:
        if max_age < 1 or min_hits < 1:
            raise ConfigurationError(
                f"max_age ({max_age}) and min_hits ({min_hits}) must both be >= 1"
            )
        if not 0.0 <= embedding_momentum < 1.0:
            raise ConfigurationError(
                f"embedding_momentum must be in [0, 1), got {embedding_momentum}"
            )
        if observation_history < 0:
            raise ConfigurationError(
                f"observation_history must be >= 0, got {observation_history}"
            )

        self._filter = KalmanFilter()
        self._max_age = max_age
        self._min_hits = min_hits
        self._momentum = embedding_momentum
        self._history = observation_history
        self._re_update = re_update

        self._tracks: list[Track] = []
        self._means = np.zeros((0, 8), dtype=np.float32)
        self._covs = np.zeros((0, 8, 8), dtype=np.float32)
        # The state at the last real observation, kept so a gapped track can be re-derived
        # from a measurement rather than from its own extrapolation.
        self._observed_means = np.zeros((0, 8), dtype=np.float32)
        self._observed_covs = np.zeros((0, 8, 8), dtype=np.float32)
        self._observed = np.zeros((0, 4), dtype=np.float32)
        self._observations: list[deque[tuple[int, np.ndarray]]] = []
        self._tag: FrameTag | None = None

    # -- views ---------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._tracks)

    @property
    def tracks(self) -> list[Track]:
        return self._tracks

    @property
    def means(self) -> np.ndarray:
        return self._means

    @property
    def covariances(self) -> np.ndarray:
        return self._covs

    @property
    def tag(self) -> FrameTag | None:
        """The tag of the frame currently being processed."""
        return self._tag

    def boxes(self) -> np.ndarray:
        """``(n, 4)`` predicted xyxy, one row per live track."""
        if not self._tracks:
            return np.zeros((0, 4), dtype=np.float32)
        return np.stack([t.box for t in self._tracks])

    def observed_boxes(self) -> np.ndarray:
        """``(n, 4)`` xyxy of each track's **last real observation**.

        Not the prediction. Association against this is what recovers a track whose object
        stopped moving while hidden: the filter kept extrapolating the old velocity and its
        prediction has walked away, while the object is still sitting where it was last seen.
        """
        if not self._tracks:
            return np.zeros((0, 4), dtype=np.float32)
        return cxcyah_to_xyxy(self._observed)

    def ages(self) -> np.ndarray:
        """``(n,)`` ``time_since_update`` per track, for a cascade to band on."""
        return np.array([t.time_since_update for t in self._tracks], dtype=np.int32)

    def embeddings(self) -> np.ndarray | None:
        """``(n, d)`` if *every* live track has one, else `None`.

        All-or-nothing, matching :attr:`shipvision.types.Detections.embeddings`. A cost
        matrix whose rows are half appearance-based and half not is not a matrix anyone can
        threshold.
        """
        if not self._tracks or any(t.embedding is None for t in self._tracks):
            return None
        return np.stack([t.embedding for t in self._tracks])  # type: ignore[misc]

    def indices_where(self, predicate: Callable[[Track], bool]) -> list[int]:
        return [i for i, track in enumerate(self._tracks) if predicate(track)]

    def directions(self, delta_t: int) -> np.ndarray:
        """``(n, 2)`` unit heading per track, measured between two real observations.

        The heading is taken from the observation roughly ``delta_t`` frames back to the most
        recent one. Measured over a span rather than between consecutive frames because a
        single frame's displacement is mostly detector jitter — at 20 fps a person moves a
        few pixels and the box wobbles by a few pixels, so a one-frame heading is noise.

        A track with too little history gets ``(0, 0)``, which every consumer must treat as
        "no information" rather than "not moving".
        """
        directions = np.zeros((len(self._tracks), 2), dtype=np.float32)
        for row, history in enumerate(self._observations):
            if len(history) < 2:
                continue
            latest_age, latest = history[-1]
            previous = history[0][1]
            for age, measurement in reversed(history):
                if latest_age - age >= delta_t:
                    previous = measurement
                    break
            offset = latest[:2] - previous[:2]
            norm = float(np.linalg.norm(offset))
            if norm > 1e-6:
                directions[row] = offset / norm
        return directions

    # -- the frame cycle -----------------------------------------------------------------

    def predict(self, tag: FrameTag) -> None:
        """Open a frame: stamp its tag on every live track, age them, advance the filters.

        The tag is stamped here rather than on output because a track's tag must be right
        even on the frames it is not published — a LOST track that is later re-found carries
        the tag of the frame it was re-found on, and reconstructing that afterwards from a
        counter is how a result ends up attributed to the wrong frame.
        """
        self._tag = tag
        for track in self._tracks:
            track.age += 1
            track.tag = tag
        if not self._tracks:
            return
        self._means, self._covs = self._filter.predict(self._means, self._covs)
        predicted = cxcyah_to_xyxy(self._means[:, :4])
        for track, box in zip(self._tracks, predicted, strict=True):
            track.box = box

    def apply_camera_motion(self, affine: np.ndarray) -> None:
        """Warp every predicted state by a ``(2, 3)`` image-to-image affine.

        BoT-SORT's camera-motion compensation. The affine maps a point in the previous frame
        to where it appears in this one, so applying it to the predictions puts them in the
        same coordinate frame as this frame's detections. Without it, a camera that pans
        makes every track's prediction wrong by the pan, the association fails for all of
        them at once, and the tracker re-births the entire scene.

        Rotation and scale are applied to the centre and to the centre velocity; the height
        is scaled and the aspect ratio is left alone, which is exact for a similarity
        transform and the reason the state is parameterised as ``(cx, cy, a, h)``.

        **Everything the pool remembers about image positions is warped, not only the
        prediction.** The last-observation array and the observation ring are image
        coordinates too, and leaving them in the previous frame's system would give the
        observation-centric stages a stale frame of reference: recovery would score a
        detection against a box measured before the camera moved, and a heading measured
        across a pan would be mostly the pan. That combination is unreachable today — only
        BoT-SORT compensates and it has no recovery stage — which is exactly why it is done
        here rather than left as a trap for whoever combines the two.
        """
        matrix = np.asarray(affine, dtype=np.float32)
        if matrix.shape != (2, 3):
            raise ConfigurationError(
                f"a camera-motion affine must be (2, 3), got {matrix.shape}"
            )
        if not self._tracks:
            return

        rotation = matrix[:, :2]
        translation = matrix[:, 2]
        scale = float(np.sqrt(abs(np.linalg.det(rotation))))

        # (cx, cy) rotate and translate; aspect is invariant under a similarity; height
        # scales. The same 4x4 block applies to a state and to a measurement, which is why
        # the 8x8 is two copies of it.
        measurement_transform = np.zeros((4, 4), dtype=np.float32)
        measurement_transform[0:2, 0:2] = rotation
        measurement_transform[2, 2] = 1.0
        measurement_transform[3, 3] = scale

        transform = np.zeros((8, 8), dtype=np.float32)
        transform[:4, :4] = measurement_transform
        # The velocity block gets the same rotation and scale but no translation: a constant
        # offset shifts where a thing is, not how fast it is going.
        transform[4:, 4:] = measurement_transform

        self._means = self._means @ transform.T
        self._means[:, 0:2] += translation
        self._covs = transform @ self._covs @ transform.T

        self._observed_means = self._observed_means @ transform.T
        self._observed_means[:, 0:2] += translation
        self._observed_covs = transform @ self._observed_covs @ transform.T
        self._observed = self._warp_measurements(
            self._observed, measurement_transform, translation
        )
        for history in self._observations:
            for index, (age, measurement) in enumerate(history):
                history[index] = (
                    age,
                    self._warp_measurements(
                        measurement[None, :], measurement_transform, translation
                    )[0],
                )

        warped = cxcyah_to_xyxy(self._means[:, :4])
        for track, box in zip(self._tracks, warped, strict=True):
            track.box = box

    @staticmethod
    def _warp_measurements(
        measurements: np.ndarray, transform: np.ndarray, translation: np.ndarray
    ) -> np.ndarray:
        """``(n, 4)`` ``(cx, cy, a, h)`` through a similarity transform."""
        warped = measurements @ transform.T
        warped[:, 0:2] += translation
        return warped.astype(np.float32)

    def gating_distance(self, boxes: np.ndarray, rows: Sequence[int]) -> np.ndarray:
        """Squared Mahalanobis distance between the given rows' filters and ``boxes``."""
        boxes = np.atleast_2d(np.asarray(boxes, dtype=np.float32))
        if not len(rows) or boxes.shape[0] == 0:
            return np.zeros((len(rows), boxes.shape[0]), dtype=np.float32)
        return self._filter.gating_distance(
            self._means[rows], self._covs[rows], xyxy_to_cxcyah(boxes)
        )

    def apply_matches(
        self,
        matches: Sequence[tuple[int, int]],
        detections: Sequence[Detection],
        *,
        embedding_momentum: np.ndarray | None = None,
    ) -> None:
        """Correct the matched filters, promote anything that has earned it, blend appearance.

        Args:
            matches: ``(track_row, detection_column)`` pairs. Each row may appear once.
            detections: this frame's detections, indexed by the columns in ``matches``.
            embedding_momentum: optional per-detection EMA retention, indexed the same way
                as ``detections``. DeepSORTv2 derives it from confidence and crowding so a
                clean, isolated crop moves a track's appearance further than a half-occluded
                one does.
        """
        if not matches:
            return
        rows = [row for row, _ in matches]
        if len(set(rows)) != len(rows):
            raise ConfigurationError(
                f"a track may match at most one detection per frame; got rows {rows}"
            )
        measurements = xyxy_to_cxcyah(np.stack([detections[col].box for _, col in matches]))

        if self._re_update:
            self._re_update_gaps(rows, measurements)

        means, covs = self._filter.update(self._means[rows], self._covs[rows], measurements)
        self._means[rows] = means
        self._covs[rows] = covs
        self._observed_means[rows] = means
        self._observed_covs[rows] = covs
        self._observed[rows] = measurements

        updated = cxcyah_to_xyxy(means[:, :4])
        for index, ((row, col), box) in enumerate(zip(matches, updated, strict=True)):
            track = self._tracks[row]
            detection = detections[col]
            track.box = box
            track.score = detection.score
            track.class_id = detection.class_id
            track.hits += 1
            track.time_since_update = 0
            self._observations[row].append((track.age, measurements[index].copy()))
            if track.state == TrackState.LOST:
                track.state = TrackState.CONFIRMED
            else:
                self._promote_if_earned(track)
            momentum = (
                self._momentum if embedding_momentum is None else float(embedding_momentum[col])
            )
            self._blend_embedding(track, detection.embedding, momentum)

    def _re_update_gaps(self, rows: Sequence[int], measurements: np.ndarray) -> None:
        """OC-SORT's observation-centric re-update, applied to the gapped rows only.

        A track that has coasted for a gap is holding two things: a position that is a pure
        extrapolation and a covariance that has been inflated once per frame. Handing that
        filter a single distant measurement produces a *velocity* correction proportional to
        the whole accumulated position error, so the next prediction overshoots, misses, and
        the track is lost again — this time for good, and a new identity is born.

        The fix is to stop pretending the gap was observed. Rewind to the last real
        observation, invent the measurements the detector would have produced had it not
        blinked (a straight line between the two observations — the only interpolation
        justified by two points), and run the filter through them. The velocity that comes
        out is the average velocity actually travelled rather than a spike.

        The loop is per track and per gap frame on purpose. It runs only for tracks that were
        actually lost, which is a small minority of a small number, and vectorising a
        variable-length recursion would cost more in bookkeeping than it saves.
        """
        for index, row in enumerate(rows):
            gap = self._tracks[row].time_since_update
            if gap < 1:
                continue
            mean = self._observed_means[row][None, :].copy()
            cov = self._observed_covs[row][None, ...].copy()
            start = self._observed[row]
            end = measurements[index]
            for step in range(1, gap + 1):
                virtual = start + (end - start) * (step / (gap + 1))
                mean, cov = self._filter.predict(mean, cov)
                mean, cov = self._filter.update(mean, cov, virtual[None, :])
            mean, cov = self._filter.predict(mean, cov)
            self._means[row] = mean[0]
            self._covs[row] = cov[0]

    def _promote_if_earned(self, track: Track) -> None:
        """One rule, applied everywhere ``hits`` changes.

        Checked on spawn as well as on match, because with ``min_hits=1`` a brand-new track
        has already met the bar — and a caller who asked for immediate publication and got
        silence would reasonably call that a bug rather than a policy.
        """
        if track.state == TrackState.TENTATIVE and track.hits >= self._min_hits:
            track.state = TrackState.CONFIRMED

    def _blend_embedding(
        self, track: Track, embedding: np.ndarray | None, momentum: float
    ) -> None:
        """Exponential moving average, renormalised.

        A track's appearance should follow the object slowly. Replacing it outright means one
        badly-cropped frame — a person half behind a pillar — becomes the reference for every
        future match, and the identity walks away from itself.
        """
        if embedding is None:
            return
        if track.embedding is None:
            track.embedding = embedding.astype(np.float32)
            return
        blended = momentum * track.embedding + (1.0 - momentum) * embedding
        norm = float(np.linalg.norm(blended))
        track.embedding = (
            (blended / norm).astype(np.float32) if norm > 1e-9 else track.embedding
        )

    def mark_missed(self, rows: Sequence[int]) -> None:
        """Age the tracks that found nothing this frame."""
        for row in rows:
            track = self._tracks[row]
            track.time_since_update += 1
            if track.state == TrackState.TENTATIVE:
                # An unconfirmed track that misses even once was probably a false positive.
                # Keeping it alive costs an identity slot and invites a wrong association.
                track.state = TrackState.REMOVED
            elif track.time_since_update > self._max_age:
                track.state = TrackState.REMOVED
            elif track.state == TrackState.CONFIRMED:
                track.state = TrackState.LOST

    def spawn(self, detections: Sequence[Detection], columns: Sequence[int]) -> None:
        """Start a track for each of the given unmatched detections."""
        if self._tag is None:
            raise ConfigurationError("predict() must open the frame before spawn()")
        for col in columns:
            detection = detections[col]
            measurement = xyxy_to_cxcyah(detection.box[None, :])[0]
            mean, cov = self._filter.initiate(measurement)
            self._means = np.concatenate([self._means, mean[None, :]])
            self._covs = np.concatenate([self._covs, cov[None, ...]])
            self._observed_means = np.concatenate([self._observed_means, mean[None, :]])
            self._observed_covs = np.concatenate([self._observed_covs, cov[None, ...]])
            self._observed = np.concatenate([self._observed, measurement[None, :]])
            self._observations.append(
                deque([(1, measurement.copy())], maxlen=max(self._history, 1))
            )
            self._tracks.append(
                Track(
                    track_id=next_track_id(),
                    box=detection.box.astype(np.float32),
                    tag=self._tag,
                    state=TrackState.TENTATIVE,
                    score=detection.score,
                    class_id=detection.class_id,
                    age=1,
                    hits=1,
                    time_since_update=0,
                    embedding=(
                        None
                        if detection.embedding is None
                        else detection.embedding.astype(np.float32)
                    ),
                )
            )
            self._promote_if_earned(self._tracks[-1])

    def sweep(self) -> None:
        """Drop removed tracks, keeping every dense array aligned with the list.

        The alignment is the whole hazard here: ``tracks[i]`` must always describe
        ``means[i]``. Rebuilding everything from one mask is the only version of this that
        cannot drift.
        """
        keep = [i for i, track in enumerate(self._tracks) if track.state != TrackState.REMOVED]
        if len(keep) == len(self._tracks):
            return
        self._tracks = [self._tracks[i] for i in keep]
        self._observations = [self._observations[i] for i in keep]
        if keep:
            self._means = self._means[keep]
            self._covs = self._covs[keep]
            self._observed_means = self._observed_means[keep]
            self._observed_covs = self._observed_covs[keep]
            self._observed = self._observed[keep]
        else:
            self.reset(keep_tag=True)

    def reset(self, *, keep_tag: bool = False) -> None:
        """Forget every track. Called when a camera reconnects and continuity is broken."""
        self._tracks = []
        self._observations = []
        self._means = np.zeros((0, 8), dtype=np.float32)
        self._covs = np.zeros((0, 8, 8), dtype=np.float32)
        self._observed_means = np.zeros((0, 8), dtype=np.float32)
        self._observed_covs = np.zeros((0, 8, 8), dtype=np.float32)
        self._observed = np.zeros((0, 4), dtype=np.float32)
        if not keep_tag:
            self._tag = None

    def output(self) -> list[Track]:
        """What a consumer should see: confirmed tracks seen this frame.

        Tentative tracks are withheld deliberately — emitting one means publishing an
        identity for what may be a false positive, and downstream cannot tell the difference.
        LOST tracks are withheld for a stronger reason: their box is a prediction no detector
        saw, and publishing it is how a phantom object drifts across a scene.
        """
        return [t for t in self._tracks if t.is_publishable]
