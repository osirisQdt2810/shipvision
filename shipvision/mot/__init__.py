"""Single-camera multi-object tracking: detections in, stable identities out.

This is the **stateful** stage of a perception pipeline. A tracker's state is per camera by
definition, so one instance serves one camera and sharding is the caller's job — and because
getting that wrong is silent corruption rather than an error,
:meth:`~shipvision.mot.base.BaseTracker.begin` refuses a frame whose ``camera_id``
disagrees with the one the instance has been serving.

    from shipvision.mot import TRACKERS
    from shipvision.types import Detection, Detections, FrameTag

    tracker = TRACKERS.build("bytetrack", track_threshold=0.5)
    for frame_id, boxes, scores in stream:
        detections = Detections(
            tag=FrameTag(camera_id="quay-3", frame_id=frame_id),
            items=[Detection(box=b, score=s) for b, s in zip(boxes, scores)],
        )
        for track in tracker.update(detections):
            publish(track)  # track.tag is detections.tag, always

Six trackers ship, and they are deliberately a chain of one-idea-at-a-time differences so
that a claim about any of them can be tested against the one below it:

``sort``
    The baseline. Kalman prediction, IoU, one Hungarian assignment.
``bytetrack``
    Adds a second association over the low-confidence detections everyone else discards,
    which is what carries an identity through a partial occlusion.
``ocsort``
    Stops trusting the filter's extrapolation: re-updates along a virtual trajectory after a
    gap, recovers against the last observation, and adds a heading-consistency term.
``botsort``
    ByteTrack plus camera-motion compensation and minimum-fused appearance — the two things
    that matter once the camera is on a PTZ head or a moving hull.
``mcbyte``
    BoT-SORT that locks the pairs nothing else was bidding for before the solve, so the
    Hungarian total cannot trade an unambiguous match away for two it will then throw out.
``deepsortv2``
    The internal C++ tracker's four-stage cascade, with OC-SORT's re-update and recovery and
    a dynamic appearance EMA.

Layout, and the one reason each directory exists:

``base.py``
    The contract and the process-wide id counter.
``registry.py``
    ``TRACKERS``. A leaf module, so that ``core/<algorithm>/tracker.py`` can take the decorator
    without dragging the whole contract in behind it.
``pool.py``
    :class:`~shipvision.mot.pool.TrackPool` — the lifecycle every tracker shares. A
    tracker that re-derives it is a tracker that will disagree with the other four about when
    a track dies.
``motion/``
    How a track moves (the Kalman filter) and how the camera moves under it (``cmc/``).
``association/``
    Which detection is which track: the costs, the solver, and the appearance policy. Anything
    two algorithms both use lives here, which is why the five ``tracker.py`` files are short.
``core/``
    One **package** per algorithm: the tracker class, what it asks of the shared track state,
    and the helpers it alone uses. Adding an algorithm is a new package plus a decorator.
``backends/``
    What *runs* an algorithm, as opposed to which algorithm it is. ``core/`` is numpy;
    :mod:`shipvision.mot.backends.native` is the C++ association loops in
    ``shipvision._C``, and five of the six have a compiled twin. Both backends register
    under the same name, so ``TRACKERS.build("sort")`` takes the fastest one this machine can
    actually build and ``TRACKERS.build("sort", backend="python")`` pins the reference.
``trackers/``
    A compatibility shim for the flat ``trackers/<name>.py`` layout this replaced. Nothing new
    goes there.

Everything above is re-exported from this module, and that flat surface is a compatibility
shim too: ``from shipvision.mot import TRACKERS, SortTracker`` predates ``core/`` and
must keep working. The registries are the supported way in — ``TRACKERS.build("sort")`` — and
importing a tracker class by name is only for the callers that already do.
"""

from shipvision.mot.association import (
    INFEASIBLE,
    appearance_cost,
    associate,
    associate_subset,
    cascade_associate,
    direction_cost,
    dynamic_appearance_momentum,
    fuse_score,
    gate_cost,
    gated_iou_cost,
    giou_cost,
    giou_matrix,
    iou_cost,
    isolation,
    min_fuse,
    pairwise_appearance,
)
from shipvision.mot.backends.native import native_available
from shipvision.mot.base import BaseTracker, next_track_id
from shipvision.mot.motion import (
    CAMERA_MOTION,
    CHI2_INV_95_4DOF,
    IDENTITY_AFFINE,
    CameraMotionEstimator,
    ExternalCameraMotion,
    KalmanFilter,
    NoCameraMotion,
    SparseOpticalFlowCameraMotion,
)
from shipvision.mot.pool import TrackPool
from shipvision.mot.registry import TRACKERS

# Each compiled tracker now lives in its own algorithm package beside the readable one — one
# algorithm, two implementations — so importing `mot.trackers` registers both backends and
# there is no separate list of native classes to keep in step with the real one. That list was
# the thing that let three of the five go a release with no compiled version and nothing
# saying so.
from shipvision.mot.trackers.botsort.tracker import BotSortTracker, NativeBotSortTracker
from shipvision.mot.trackers.bytetrack.tracker import ByteTrackTracker, NativeByteTrackTracker
from shipvision.mot.trackers.deepsortv2.tracker import (
    DeepSortV2Tracker,
    NativeDeepSortV2Tracker,
)
from shipvision.mot.trackers.mcbyte.tracker import McByteTracker
from shipvision.mot.trackers.ocsort.tracker import NativeOcSortTracker, OcSortTracker
from shipvision.mot.trackers.sort.tracker import NativeSortTracker, SortTracker

__all__ = [
    "CAMERA_MOTION",
    "CHI2_INV_95_4DOF",
    "IDENTITY_AFFINE",
    "INFEASIBLE",
    "TRACKERS",
    "BaseTracker",
    "BotSortTracker",
    "ByteTrackTracker",
    "CameraMotionEstimator",
    "DeepSortV2Tracker",
    "ExternalCameraMotion",
    "KalmanFilter",
    "McByteTracker",
    "NativeBotSortTracker",
    "NativeByteTrackTracker",
    "NativeDeepSortV2Tracker",
    "NativeOcSortTracker",
    "NativeSortTracker",
    "NoCameraMotion",
    "OcSortTracker",
    "SortTracker",
    "SparseOpticalFlowCameraMotion",
    "TrackPool",
    "appearance_cost",
    "associate",
    "associate_subset",
    "cascade_associate",
    "direction_cost",
    "dynamic_appearance_momentum",
    "fuse_score",
    "gate_cost",
    "gated_iou_cost",
    "giou_cost",
    "giou_matrix",
    "iou_cost",
    "isolation",
    "min_fuse",
    "native_available",
    "next_track_id",
    "pairwise_appearance",
]
