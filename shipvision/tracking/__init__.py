"""Single-camera multi-object tracking: detections in, stable identities out.

This is the **stateful** stage of a perception pipeline. A tracker's state is per camera by
definition, so one instance serves one camera and sharding is the caller's job — and because
getting that wrong is silent corruption rather than an error,
:meth:`~shipvision.tracking.base.BaseTracker.begin` refuses a frame whose ``camera_id``
disagrees with the one the instance has been serving.

    from shipvision.tracking import TRACKERS
    from shipvision.types import Detection, Detections, FrameTag

    tracker = TRACKERS.build("bytetrack", track_threshold=0.5)
    for frame_id, boxes, scores in stream:
        detections = Detections(
            tag=FrameTag(camera_id="quay-3", frame_id=frame_id),
            items=[Detection(box=b, score=s) for b, s in zip(boxes, scores)],
        )
        for track in tracker.update(detections):
            publish(track)  # track.tag is detections.tag, always

Five trackers ship, and they are deliberately a chain of one-idea-at-a-time differences so
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
``deepsortv2``
    The internal C++ tracker's four-stage cascade, with OC-SORT's re-update and recovery and
    a dynamic appearance EMA.

Layout, and the one reason each directory exists:

``base.py``
    The contract, the process-wide id counter, and the ``TRACKERS`` registry.
``pool.py``
    :class:`~shipvision.tracking.pool.TrackPool` — the lifecycle every tracker shares. A
    tracker that re-derives it is a tracker that will disagree with the other four about when
    a track dies.
``motion/``
    How a track moves (the Kalman filter) and how the camera moves under it (``cmc/``).
``association/``
    Which detection is which track: the costs, the solver, and the appearance policy. The
    only part that differs between algorithms, which is why the five tracker files are short.
``trackers/``
    One algorithm per file.
"""

from shipvision.tracking.association import (
    INFEASIBLE,
    appearance_cost,
    associate,
    associate_subset,
    cascade_associate,
    direction_cost,
    dynamic_appearance_momentum,
    fuse_score,
    gate_cost,
    giou_cost,
    giou_matrix,
    iou_cost,
    isolation,
    min_fuse,
)
from shipvision.tracking.base import TRACKERS, BaseTracker, next_track_id
from shipvision.tracking.motion import (
    CAMERA_MOTION,
    CHI2_INV_95_4DOF,
    IDENTITY_AFFINE,
    CameraMotionEstimator,
    ExternalCameraMotion,
    KalmanFilter,
    NoCameraMotion,
    SparseOpticalFlowCameraMotion,
)
from shipvision.tracking.pool import TrackPool
from shipvision.tracking.trackers import (
    BotSortTracker,
    ByteTrackTracker,
    DeepSortV2Tracker,
    OcSortTracker,
    SortTracker,
)

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
    "giou_cost",
    "giou_matrix",
    "iou_cost",
    "isolation",
    "min_fuse",
    "next_track_id",
]
