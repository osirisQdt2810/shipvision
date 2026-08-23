"""The tracker contract.

One method. A tracker is handed this frame's :class:`~shipvision.types.Detections` and
returns the tracks that are currently publishable; everything else — how it associates, what
it remembers, when it gives up on a track — is the implementation's business. Keeping the
interface this narrow is what makes five trackers drop-in alternatives rather than five
different APIs, and it is what lets a deployment A/B two of them on one stream from config.

The argument is the tagged container, not a list plus a frame number. That is the one
interface decision in this file that is not negotiable: a tracker is the stage where a
mis-tagged result becomes indistinguishable from a real detection, because its output is an
*identity* that downstream will trust across cameras. Carrying ``(camera_id, frame_id)``
inside the input means the tag on the output cannot disagree with the tag on the input, and
:meth:`BaseTracker.begin` makes a camera swap on a live instance fail loudly instead of
silently associating one camera's objects with another's.
"""

from __future__ import annotations

import abc
import itertools
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from shipvision.errors import TrackingError
from shipvision.registry import Registry
from shipvision.types import Detections, FrameTag, Track

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance, not a runtime dependency
    from shipvision.tracking.pool import TrackPool

__all__ = ["TRACKERS", "BaseTracker", "next_track_id"]

_TRACK_IDS = itertools.count(1)


def next_track_id() -> int:
    """A process-unique track id. ``itertools.count`` is atomic under the GIL.

    One counter for the whole process, not one per tracker. Per-instance counters mean camera
    3's track 7 and camera 9's track 7 collide the moment two cameras' output meets
    downstream — and making that meeting possible is the entire point of the cross-camera
    tier. A track id is therefore globally unique and says nothing about which camera produced
    it; the camera lives in :class:`~shipvision.types.FrameTag`, where it can be read.
    """
    return next(_TRACK_IDS)


class BaseTracker(abc.ABC):
    """Consumes per-frame detections, produces stable track identities.

    Stateful and **single-camera by construction**: a tracker's state is per camera, so one
    instance serves one camera and sharding is the caller's job. Feeding one instance two
    cameras used to be a silent disaster — it associates one camera's objects with another's
    — so :meth:`begin` refuses it.

    Subclasses build the :class:`~shipvision.tracking.pool.TrackPool` they want and hand it
    up, which is what makes ``reset``, ``pool_size`` and the tag discipline shared rather
    than re-derived five times.
    """

    name: ClassVar[str] = "abstract"
    backend: ClassVar[str] = "python"

    def __init__(self, pool: TrackPool) -> None:
        self._pool = pool
        self._last_tag: FrameTag | None = None

    # -- the contract --------------------------------------------------------------------

    @abc.abstractmethod
    def update(self, detections: Detections, *, image: np.ndarray | None = None) -> list[Track]:
        """Advance one frame.

        Args:
            detections: this frame's detections *and its tag*. May be empty — that is
                information, not a reason to skip the update: tracks still age and eventually
                die, and a tracker that treats an empty frame as a no-op keeps dead objects
                alive forever.
            image: the decoded frame, for the trackers that compensate for camera motion.
                Keyword-only and optional because only BoT-SORT can use it, and a library
                that demanded pixels from every caller would be unusable by one that only has
                boxes — an evaluation over an MOT ground-truth file, for instance.

        Returns:
            The tracks that are confirmed and were seen on this frame, in no guaranteed
            order. Every one carries ``detections.tag``.

        Raises:
            TrackingError: the tag contradicts the sequence seen so far.
        """

    # -- shared machinery ----------------------------------------------------------------

    def begin(self, detections: Detections) -> FrameTag:
        """Validate the tag, advance the filters, and return the tag to stamp on the output.

        Two refusals, both of which used to be silent corruption:

        A **camera change** on a live instance means someone shared one tracker across
        cameras. The result is not degraded tracking, it is objects from camera A being given
        camera B's identities — which downstream reads as a real detection somewhere nothing
        happened. There is no recovery worth attempting, so it raises.

        A **frame_id that does not advance** means a duplicate or reordered frame. Accepting
        it double-ages every track and double-counts the hit that promotes one, so a replayed
        frame quietly changes which identities exist. A caller whose stream legitimately
        restarts calls :meth:`reset`.
        """
        tag = detections.tag
        previous = self._last_tag
        if previous is not None:
            if tag.camera_id != previous.camera_id:
                raise TrackingError(
                    f"{type(self).__name__} was built for camera {previous.camera_id!r} and "
                    f"handed a frame from {tag.camera_id!r}. One tracker serves one camera: "
                    f"sharing an instance associates one camera's objects with another's"
                )
            if tag.frame_id <= previous.frame_id:
                raise TrackingError(
                    f"frame_id must advance; got {tag.frame_id} after {previous.frame_id} on "
                    f"camera {tag.camera_id!r}. Call reset() when continuity is broken"
                )
        self._last_tag = tag
        self._pool.predict(tag)
        return tag

    def reset(self) -> None:
        """Forget everything. Called when a camera reconnects and continuity is broken."""
        self._pool.reset()
        self._last_tag = None

    @property
    def pool_size(self) -> int:
        """How many tracks are alive, published or not.

        Exposed because "the pool is empty" is the only way to assert that a tracker actually
        releases memory rather than merely stopping publication, and a process here runs for
        weeks.
        """
        return len(self._pool)

    @property
    def tracks(self) -> list[Track]:
        """Every live track, including tentative and lost ones. Read-only in spirit."""
        return self._pool.tracks

    def describe(self) -> str:
        return type(self).__name__

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}/{self.backend} tracks={self.pool_size}>"


TRACKERS: Registry[BaseTracker] = Registry("tracker")
"""Every tracker, keyed on ``(name, backend)``.

A new tracker is a new file plus a decorator — never an edit to a switch statement. Selecting
one by name from config is what lets a deployment A/B two association strategies on the same
stream without a code change, which is the only honest way to decide between them. Every
reference implementation this library replaces picks its tracker with a hand-written
``if/elif``, and every one has the same consequence: the tracker that shipped first wins by
default rather than by measurement.

There is no ``native`` tracker yet. When there is, it registers here under the same name as
its numpy twin and a parity test compares the two on one sequence — a compiled association
loop nobody can compare against is a compiled association loop nobody can trust.
"""
