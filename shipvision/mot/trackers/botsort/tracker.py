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
is obtained is a separate, pluggable question — see :mod:`shipvision.mot.motion.cmc`.

**Minimum fusion instead of a weighted sum.** DeepSORT adds an appearance distance to a motion
distance, so a pair that is unambiguous on one signal can be dragged over the threshold by the
other. BoT-SORT takes the element-wise minimum of two independently gated costs, so either
signal on its own suffices. See :func:`shipvision.mot.association.costs.min_fuse`.

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
from shipvision.mot.association import pairwise_appearance
from shipvision.mot.backends.native import (
    NativeTracker,
    _columns_above,
    require_extension,
    validate_lifecycle,
)
from shipvision.mot.motion.cmc import CAMERA_MOTION, CameraMotionEstimator
from shipvision.mot.registry import TRACKERS
from shipvision.mot.trackers.botsort.utils import first_cost
from shipvision.mot.trackers.bytetrack import ByteTrackTracker
from shipvision.registry import NATIVE, PYTHON
from shipvision.types import Detection, Detections

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
        **byte: everything :class:`~shipvision.mot.trackers.bytetrack.tracker.ByteTrackTracker`
            takes, including ``embedding_momentum``.
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
        boxes = np.stack([high[c].box for c in columns])
        return first_cost(
            self._pool.boxes()[rows],
            boxes,
            appearance=pairwise_appearance(
                self._pool.embeddings(), rows, [high[c].embedding for c in columns]
            ),
            motion_gate=self._max_cost,
            appearance_gate=self._appearance_gate,
            appearance_weight=self._appearance_weight,
            gating_distances=(self._pool.gating_distance(boxes, rows) if self._gate else None),
        )

    def reset(self) -> None:
        super().reset()
        self._motion.reset()

    def describe(self) -> str:
        return (
            f"BoT-SORT: ByteTrack + camera-motion compensation ({self._motion.name}) + "
            f"min-fused appearance"
        )


# -- the compiled implementation ------------------------------------------------------------
#
# Same algorithm, same registry name, `native` backend, and in this file rather than beside
# it: botsort is one tracker with two implementations, and splitting them by language splits
# them by the least interesting thing about them. The readable class above is the
# specification; the parity tests assert the two agree.
#
# Only `deepsortv2` is what `motservice` actually runs — its README says "currently supports
# only deepsort". This one is kept because it is written and tested (V50), not because
# anything downstream selects it. New compiled work goes to what the services use.
#
# The extension probe and the marshalling are in `mot/backends/base.py`: not per algorithm,
# and five copies would be five places to disagree about what an empty detection set is.


@TRACKERS.register("botsort", backend=NATIVE)
class NativeBotSortTracker(NativeTracker):
    """BoT-SORT with ByteTrack's two stages in C++. See
    :class:`~shipvision.mot.trackers.botsort.tracker.BotSortTracker` for the algorithm.

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
        native = require_extension("BotSortTracker")
        validate_lifecycle(max_age, min_hits)
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
            native.BotSortTracker(
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
