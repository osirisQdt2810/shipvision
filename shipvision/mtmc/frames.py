"""The input unit: every camera's tracks at one synchronised instant.

Cross-camera tracking is not tracklet-to-tracklet association over a recording. It runs
*once per time-synchronised group of frames*, on the tracks that exist in that instant, and
it has to answer before the next group arrives. The unit it consumes therefore has to be the
group, not a frame and not a tracklet — which is what :class:`FrameTrackCluster` is.

Two things in here are load-bearing rather than convenient.

**The cross-camera key is a type, not a convention.** Within a camera a track is identified
by ``track_id``; across cameras that integer means nothing on its own, because camera A's
track 7 and camera B's track 7 are unrelated. Every map in this package is therefore keyed
on :class:`TrackKey`, a ``(camera_id, track_id)`` pair. The reference implementation this
ports used a formatted string, ``f"{camera}_{track_id}"``, and paid for it twice: it split
the string back apart whenever it needed the camera, and a camera id containing an
underscore silently produced a key that parsed to something else.

**Frame dimensions are required, not optional.** Three separate decisions need them — the
height gate, the test for a bounding box clipped by the bottom of the frame, and scaling a
point into the domain a homography was calibrated in. A default of zero would make all three
silently wrong rather than loudly absent, so a view without dimensions is rejected at
construction.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from shipvision.errors import ConfigurationError
from shipvision.types import FrameTag, Track

__all__ = ["CameraTracks", "FrameTrackCluster", "TrackKey", "TrackObservation"]


class TrackKey(NamedTuple):
    """Identifies one single-camera track across the whole camera group.

    A :class:`typing.NamedTuple` rather than a dataclass, deliberately: it *is* a
    ``tuple[str, int]``, so it is hashable, orderable and comparable for free, and it drops
    straight into :attr:`shipvision.types.GlobalTrack.members` — which is declared as
    ``tuple[tuple[str, int], ...]`` — with no conversion step to get wrong.

    Ordering is ``(camera_id, track_id)`` lexicographically. Nothing in the algorithm
    depends on the order being meaningful; several things depend on it being *stable*, which
    is how a tie between two equally good candidates resolves the same way twice.
    """

    camera_id: str
    track_id: int

    def __str__(self) -> str:
        return f"{self.camera_id}#{self.track_id}"


@dataclass(slots=True, frozen=True)
class TrackObservation:
    """One track, at this instant, with the frame context the geometry needs.

    A thin wrapper over :class:`shipvision.types.Track` rather than a replacement for it:
    what it adds is the :class:`TrackKey` and the size of the frame the box was measured in.
    The track itself is passed through untouched so that whatever produced it — box, score,
    embedding, metadata — reaches the caller of MTMC unchanged.
    """

    key: TrackKey
    track: Track
    frame_height: int
    frame_width: int

    @property
    def camera_id(self) -> str:
        return self.key.camera_id

    @property
    def track_id(self) -> int:
        return self.key.track_id

    @property
    def tag(self) -> FrameTag:
        return self.track.tag

    @property
    def box(self) -> np.ndarray:
        """``(4,)`` float32 xyxy in absolute pixels of ``frame_width`` x ``frame_height``."""
        return self.track.box

    @property
    def embedding(self) -> np.ndarray | None:
        return self.track.embedding

    @property
    def height_fraction(self) -> float:
        """Box height as a fraction of frame height — a cheap proxy for how far away it is.

        The quantity the height gate thresholds. A fraction rather than pixels because the
        same physical distance is a different pixel count on a 1080p and a 4K camera, and a
        threshold in pixels therefore has to be retuned per camera model.
        """
        return float(self.track.box[3] - self.track.box[1]) / float(self.frame_height)

    def __str__(self) -> str:
        return f"{self.key} {self.track.box.tolist()} state={self.track.state}"


@dataclass(slots=True, frozen=True)
class CameraTracks:
    """One camera's contribution to a synchronised group: its tag, its tracks, its size."""

    tag: FrameTag
    tracks: tuple[Track, ...] = ()
    height: int = 0
    width: int = 0

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise ConfigurationError(
                f"camera {self.tag.camera_id!r} needs positive frame dimensions, got "
                f"{self.width}x{self.height}. MTMC needs them for the height gate, for the "
                f"bottom-truncated-box test and to scale into the homography's domain; a "
                f"zero default would make all three silently wrong"
            )
        object.__setattr__(self, "tracks", tuple(self.tracks))
        seen: set[int] = set()
        for track in self.tracks:
            if track.camera_id != self.tag.camera_id:
                raise ConfigurationError(
                    f"track {track.track_id} is tagged camera {track.camera_id!r} but was "
                    f"handed in under {self.tag.camera_id!r}. A mis-tagged track becomes a "
                    f"real-looking identity on a camera where nothing happened"
                )
            if track.track_id in seen:
                raise ConfigurationError(
                    f"camera {self.tag.camera_id!r} has track_id {track.track_id} twice in "
                    f"one frame; (camera_id, track_id) is the key every map here uses"
                )
            seen.add(track.track_id)

    @property
    def camera_id(self) -> str:
        return self.tag.camera_id

    def __len__(self) -> int:
        return len(self.tracks)

    def __iter__(self) -> Iterator[Track]:
        return iter(self.tracks)


@dataclass(slots=True, frozen=True)
class FrameTrackCluster:
    """The tracks from every camera in a group, at one synchronised instant.

    "Cluster" is the reference implementation's word for a set of cameras that overlap enough
    to share a ground plane — a berth, a gate, a quay — and it is kept because the unit of
    deployment really is that set: one tracker instance per camera group, because two groups
    that never see each other must not compete for the same identity space.

    Treated as immutable input. The tracks inside are the caller's objects and are not
    copied: at 50 cameras and 20 fps this is built a thousand times a second, and copying a
    few hundred boxes and embeddings each time would cost more than the association it
    exists to feed.
    """

    views: tuple[CameraTracks, ...] = ()
    observations: tuple[TrackObservation, ...] = field(default=(), init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "views", tuple(self.views))
        seen: set[str] = set()
        for view in self.views:
            if view.camera_id in seen:
                raise ConfigurationError(
                    f"camera {view.camera_id!r} appears twice in one synchronised group. Two "
                    f"frames from one camera in one instant are two instants, and merging "
                    f"them makes the same-camera exclusion mask leak"
                )
            seen.add(view.camera_id)

        flat = tuple(
            TrackObservation(
                key=TrackKey(view.camera_id, track.track_id),
                track=track,
                frame_height=view.height,
                frame_width=view.width,
            )
            for view in self.views
            for track in view.tracks
        )
        object.__setattr__(self, "observations", flat)

    # -- construction -------------------------------------------------------------------

    @classmethod
    def from_views(cls, views: Iterable[CameraTracks]) -> FrameTrackCluster:
        """Build from an iterable of per-camera views."""
        return cls(views=tuple(views))

    @classmethod
    def from_tracks(
        cls,
        tracks: Iterable[Track],
        *,
        height: int,
        width: int,
        timestamp: float = 0.0,
    ) -> FrameTrackCluster:
        """Group a flat iterable of tracks by camera, assuming one frame size for all.

        A convenience for tests and for the common deployment where every camera in a group
        is the same model. It groups by ``track.camera_id`` and takes each camera's
        ``frame_id`` from its first track, which is exactly the reconstruction-by-convention
        that :class:`shipvision.types.Detections` exists to prevent — so it is a named
        constructor a caller has to choose, not the default path.
        """
        by_camera: dict[str, list[Track]] = {}
        for track in tracks:
            by_camera.setdefault(track.camera_id, []).append(track)
        views = [
            CameraTracks(
                tag=FrameTag(
                    camera_id=camera_id,
                    frame_id=group[0].tag.frame_id,
                    timestamp=timestamp or group[0].tag.timestamp,
                ),
                tracks=tuple(group),
                height=height,
                width=width,
            )
            for camera_id, group in by_camera.items()
        ]
        return cls(views=tuple(views))

    # -- reading ------------------------------------------------------------------------

    @property
    def cameras(self) -> tuple[str, ...]:
        return tuple(view.camera_id for view in self.views)

    @property
    def keys(self) -> tuple[TrackKey, ...]:
        return tuple(observation.key for observation in self.observations)

    def __len__(self) -> int:
        """How many tracks in total — not how many cameras."""
        return len(self.observations)

    def __iter__(self) -> Iterator[TrackObservation]:
        return iter(self.observations)

    def filter(self, keep: Sequence[bool]) -> tuple[TrackObservation, ...]:
        """The observations where ``keep`` is true, in input order."""
        if len(keep) != len(self.observations):
            raise ConfigurationError(
                f"keep has {len(keep)} entries for {len(self.observations)} observations"
            )
        return tuple(obs for obs, take in zip(self.observations, keep, strict=True) if take)

    def __str__(self) -> str:
        return f"<FrameTrackCluster cameras={len(self.views)} tracks={len(self.observations)}>"
