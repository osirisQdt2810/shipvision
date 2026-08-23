"""BoT-SORT: ByteTrack that knows the camera can move, and fuses appearance by minimum.

Aharon, Orfaig and Bobrovsky, "BoT-SORT: Robust Associations Multi-Pedestrian Tracking",
2022. Written from the paper.

The paper's contribution list is short and this class is a faithful reading of it: take
ByteTrack and change exactly two things.

**Camera-motion compensation.** ByteTrack's Kalman prediction is in the previous frame's
coordinate system. On a fixed camera those are the same system; on a panning one they differ
by the pan, so *every* track's prediction is wrong by the same amount on the same frame, the
whole association fails at once, and the tracker re-births the entire scene. Warping the
predictions by the estimated frame-to-frame affine before association fixes it. How the affine
is obtained is a separate, pluggable question — see :mod:`shipvision.tracking.motion.cmc`.

**Minimum fusion instead of a weighted sum.** DeepSORT adds an appearance distance to a motion
distance, so a pair that is unambiguous on one signal can be dragged over the threshold by the
other. BoT-SORT takes the element-wise minimum of two independently gated costs, so either
signal on its own suffices. See :func:`shipvision.tracking.association.costs.min_fuse`.

Deliberately not implemented: the paper's third, smaller change — inflating the Kalman noise
model's width/height terms — because this library's state is ``(cx, cy, aspect, height)``
rather than ``(cx, cy, w, h)`` and the corresponding term is already height-scaled here. Nor
the paper's separate re-ID feature extractor: this library keeps extraction in
:mod:`shipvision.reid` and a tracker consumes ``Detection.embedding``, so a BoT-SORT with no
embeddings degrades to CMC-plus-ByteTrack rather than refusing to run.
"""

from __future__ import annotations

import numpy as np

from shipvision.errors import ConfigurationError
from shipvision.registry import PYTHON
from shipvision.tracking.association import appearance_cost, gate_cost, iou_cost, min_fuse
from shipvision.tracking.base import TRACKERS
from shipvision.tracking.motion.cmc import CAMERA_MOTION, CameraMotionEstimator
from shipvision.tracking.motion.kalman import CHI2_INV_95_4DOF
from shipvision.tracking.trackers.bytetrack import ByteTrackTracker
from shipvision.types import Detection

__all__ = ["BotSortTracker"]


@TRACKERS.register("botsort", backend=PYTHON, aliases=("bot", "bot_sort"))
class BotSortTracker(ByteTrackTracker):
    """ByteTrack plus camera-motion compensation and minimum appearance fusion.

    A subclass rather than a copy because the paper *is* a two-line diff against ByteTrack,
    and expressing it as two overridden methods keeps that true. If the two ever diverge
    structurally, the shared two-stage loop moves into a helper — but pre-emptively splitting
    it would hide the fact that they are the same algorithm.

    Args:
        cmc: name of a registered camera-motion estimator. The default assumes a fixed camera,
            which is right for most of a fifty-camera installation and is why turning this on
            has to be a deliberate act. ``"sparse_flow"`` needs OpenCV and needs ``image=``
            passed to :meth:`update`; ``"external"`` takes the affine from PTZ telemetry,
            which beats any estimate made from pixels.
        cmc_options: keyword arguments for the estimator.
        appearance_gate: cosine distance above which appearance contributes nothing.
        appearance_weight: the paper halves the cosine distance before the minimum, because
            ``1 - IoU`` and a cosine distance are not on the same scale.
        **byte: everything :class:`~shipvision.tracking.trackers.bytetrack.ByteTrackTracker` takes,
            including ``embedding_momentum``.
    """

    def __init__(
        self,
        *,
        cmc: str = "none",
        cmc_options: dict[str, object] | None = None,
        appearance_gate: float = 0.25,
        appearance_weight: float = 0.5,
        **byte: object,
    ) -> None:
        super().__init__(**byte)  # type: ignore[arg-type]
        if not 0.0 < appearance_gate <= 2.0:
            raise ConfigurationError(
                f"appearance_gate is a cosine distance and must be in (0, 2], got "
                f"{appearance_gate}"
            )
        if not 0.0 < appearance_weight <= 1.0:
            raise ConfigurationError(
                f"appearance_weight must be in (0, 1], got {appearance_weight}"
            )
        self._motion: CameraMotionEstimator = CAMERA_MOTION.build(cmc, **(cmc_options or {}))
        self._appearance_gate = appearance_gate
        self._appearance_weight = appearance_weight

    @property
    def camera_motion(self) -> CameraMotionEstimator:
        """The estimator, so a caller with PTZ telemetry can push into it."""
        return self._motion

    def _compensate(self, image: np.ndarray | None) -> None:
        self._pool.apply_camera_motion(self._motion.estimate(image))

    def _first_cost(
        self, rows: list[int], columns: list[int], high: list[Detection]
    ) -> np.ndarray:
        """Minimum of the gated IoU cost and the gated, halved appearance cost.

        Note what is *not* here: :func:`~shipvision.tracking.association.costs.fuse_score`, which
        ByteTrack uses to scale similarity by detector confidence. Folding confidence into a
        cost that is already a minimum of two gated terms double-counts it — the high-score
        stage has by definition already filtered on confidence — and it pushes the fused cost
        above the appearance gate for exactly the medium-confidence detections appearance is
        supposed to rescue.
        """
        boxes = np.stack([high[c].box for c in columns])
        motion = iou_cost(self._pool.boxes()[rows], boxes)

        track_embeddings = self._pool.embeddings()
        detection_embeddings = [high[c].embedding for c in columns]
        if track_embeddings is not None and all(e is not None for e in detection_embeddings):
            cost = min_fuse(
                motion,
                appearance_cost(track_embeddings[rows], np.stack(detection_embeddings)),
                motion_gate=self._max_cost,
                appearance_gate=self._appearance_gate,
                appearance_weight=self._appearance_weight,
            )
        else:
            # No embeddings on this frame: fall back to geometry alone rather than treating a
            # missing appearance distance as zero. A zero would mean "identical appearance",
            # which is the strongest possible claim, made on no evidence.
            cost = motion

        if self._gate:
            cost = gate_cost(cost, self._pool.gating_distance(boxes, rows), CHI2_INV_95_4DOF)
        return cost

    def reset(self) -> None:
        super().reset()
        self._motion.reset()

    def describe(self) -> str:
        return (
            f"BoT-SORT: ByteTrack + camera-motion compensation ({self._motion.name}) + "
            f"min-fused appearance"
        )
