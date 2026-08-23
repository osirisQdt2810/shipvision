"""DeepSORTv2: the four-stage cascade from the internal C++ tracker, ported.

Ported from ``gitea-generic-multi-object-tracking-cpp``
(``src/tracker/models/deepsortv2/``) and its Python twin in the multi-camera service. Both are
first-party, so this is a port rather than a clean-room rewrite — and where the sources
disagree with themselves the *paper* each stage comes from decided it.

It is DeepSORT (Wojke et al., 2017) with three additions the reference had already made:

* OC-SORT's **ORU** and **OCR**, which is what makes stage C exist at all.
* A **dynamic appearance** EMA, whose rate depends on how confident and how isolated the
  detection was. See :mod:`shipvision.tracking.association.appearance`.
* A four-stage cascade that separates "how confident are we in this track" from "how good is
  the evidence", instead of running one assignment over everything.

The four stages, in order, each consuming what the last one could not match:

======  ============================================  ==============================================
Stage   Tracks                                        Cost
======  ============================================  ==============================================
A       confirmed and lost, banded by age             GIoU fused with appearance, both gated
B       stage-A leftovers seen recently               IoU, gated by appearance
C       everything still unmatched                    IoU against the **last observation** (OCR)
D       tentative                                     IoU, nothing else
======  ============================================  ==============================================

The ordering is the design. Stage A gives well-supported tracks first refusal on the good
evidence. Stage B relaxes to geometry for tracks whose appearance has gone stale but which
were seen recently enough for the prediction to be worth something. Stage C throws away the
prediction entirely, which is the only thing that recovers an object that stopped moving while
hidden. Stage D runs last because a tentative track is the weakest claim in the pool and must
never outbid a confirmed one.

Three defects in the C++ reference were found and **not** ported; they are named here so the
next person to compare the two does not think this file is the one that is wrong:

* ``Cost.cpp:178`` — the pure-loop Euclidean distance reads ``featuresB[b](90, i)`` where it
  means ``(0, i)``. The features are ``1 x D`` row vectors, so row 90 is out of bounds; it is
  a live out-of-bounds read on any build that selects that optimisation path.
* ``Cost.cpp:303`` — ``minEmbeddingCost`` is an empty stub that returns an *uninitialised*
  Eigen matrix. Selecting ``EMBEDDING_METHOD: min`` therefore associates on uninitialised
  memory rather than failing.
* ``Assignment.cpp:113`` — the ``LAPJV`` case has no ``break`` and falls into ``UNKNOWN``.
  Benign only because ``UNKNOWN`` happens to do nothing; add a case between them and it stops
  being benign.
* ``Cost.cpp:396`` vs ``:406`` — the two optimisation paths of ``gatedFusionGiouEmbeddingCost``
  do not agree. The loop keeps a pair when *both* gates pass; the vectorised branch keeps it
  when *either* does. This file implements the conjunction, because a gate that can be
  satisfied by ignoring it is not a gate.
"""

from __future__ import annotations

import numpy as np

from shipvision.errors import ConfigurationError
from shipvision.registry import PYTHON
from shipvision.tracking.association import (
    INFEASIBLE,
    appearance_cost,
    associate_subset,
    cascade_associate,
    gate_cost,
    giou_cost,
    iou_cost,
)
from shipvision.tracking.association.appearance import dynamic_appearance_momentum
from shipvision.tracking.base import TRACKERS, BaseTracker
from shipvision.tracking.motion.kalman import CHI2_INV_95_4DOF
from shipvision.tracking.pool import TrackPool
from shipvision.types import Detection, Detections, Track, TrackState

__all__ = ["DeepSortV2Tracker"]


@TRACKERS.register("deepsortv2", backend=PYTHON, aliases=("deepsort2", "dsv2"))
class DeepSortV2Tracker(BaseTracker):
    """Four association stages, an observation-centric re-update, and a dynamic appearance EMA.

    Args:
        det_threshold: detections below this are discarded.
        appearance_weight: how much of stage A's fused cost is appearance rather than GIoU.
            The reference uses 0.9, which reads as extreme until you remember the gates: GIoU
            has already vetoed anything geometrically impossible, so what is left for the cost
            to decide *is* an appearance question.
        appearance_gate: cosine distance above which a stage-A or stage-B pair is forbidden.
        giou_gate: GIoU cost above which a stage-A pair is forbidden. Remember the range is
            ``[0, 2]``, so 1.2 admits pairs that do not overlap at all but are close.
        stage_a_max_cost: per-pair threshold inside each cascade band.
        cascade_stride: band width for the cascade. One is DeepSORT's original formulation;
            the reference uses five, trading a little precedence for a fifth of the solves.
        stage_b_max_cost: threshold for the geometric fallback.
        stage_b_max_age: a track older than this does not get a stage-B chance. Its prediction
            has been extrapolating too long for IoU against it to mean anything, and stage C
            is where it belongs.
        stage_c_max_cost: threshold for the observation-centric recovery.
        recover: run stage C at all. Off is not a configuration anyone should deploy — it is
            how the stage's contribution gets measured, and a stage nobody can switch off is
            a stage nobody can show is earning its keep.
        stage_d_max_cost: threshold for tentative tracks. Loosest of the four, because a
            tentative track has no history to be judged against and the cost of getting it
            wrong is one frame of a track nobody has published yet.
        border_fraction: how close to the frame edge counts as "near the border", as a
            fraction of the smaller frame dimension. Requires ``Detections.height`` and
            ``.width``; without them the border rule is skipped rather than guessed.
        skip_border_recovery: exclude near-border tracks from stage C. An object leaving the
            frame is half out of it, so its last observation is a truncated box that overlaps
            whatever else is at the edge, and recovering on that evidence swaps identities
            between everything entering and leaving.
        max_age: frames a confirmed track survives without a match.
        min_hits: matches before a track is published.
        re_update: enable ORU.
        appearance_momentum: ``(min, max)`` bounds for the dynamic EMA retention.
        dynamic_appearance: when off, use ``min`` of the bounds as a fixed EMA rate. Provided
            because it is the only way to measure whether the dynamic rule is earning its keep.
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
            TrackPool(
                max_age=max_age,
                min_hits=min_hits,
                embedding_momentum=low,
                re_update=re_update,
            )
        )
        self._det_threshold = det_threshold
        self._appearance_weight = appearance_weight
        self._appearance_gate = appearance_gate
        self._giou_gate = giou_gate
        self._stage_a_max_cost = stage_a_max_cost
        self._cascade_stride = cascade_stride
        self._stage_b_max_cost = stage_b_max_cost
        self._stage_b_max_age = stage_b_max_age
        self._stage_c_max_cost = stage_c_max_cost
        self._recover = recover
        self._stage_d_max_cost = stage_d_max_cost
        self._border_fraction = border_fraction
        self._skip_border_recovery = skip_border_recovery
        self._max_age = max_age
        self._momentum_bounds = (low, high)
        self._dynamic_appearance = dynamic_appearance

    def update(self, detections: Detections, *, image: np.ndarray | None = None) -> list[Track]:
        self.begin(detections)
        kept = detections.filter(min_score=self._det_threshold).items
        columns = list(range(len(kept)))
        momentum = self._momentum(kept)

        tentative = self._pool.indices_where(lambda t: t.state == TrackState.TENTATIVE)
        established = self._pool.indices_where(lambda t: t.state != TrackState.TENTATIVE)
        ages = self._pool.ages()

        # -- A: the cascade, on fused appearance and geometry ----------------------------
        matched_a, unmatched_a, columns = cascade_associate(
            lambda r, c: self._stage_a_cost(r, c, kept),
            self._stage_a_max_cost,
            established,
            columns,
            ages,
            stride=self._cascade_stride,
            max_depth=self._max_age + 1,
        )

        # -- B: geometry alone, for tracks whose prediction is still worth something ------
        recent = [row for row in unmatched_a if ages[row] <= self._stage_b_max_age]
        stale = [row for row in unmatched_a if ages[row] > self._stage_b_max_age]
        matched_b, unmatched_b, columns = associate_subset(
            lambda r, c: self._stage_b_cost(r, c, kept),
            self._stage_b_max_cost,
            recent,
            columns,
        )

        # -- C: OCR, against the last observation instead of the prediction ---------------
        candidates = self._recoverable(stale + unmatched_b, detections) if self._recover else []
        matched_c, unmatched_c, columns = associate_subset(
            lambda r, c: self._stage_c_cost(r, c, kept),
            self._stage_c_max_cost,
            candidates,
            columns,
        )

        # -- D: tentative tracks, last and on the weakest evidence ------------------------
        matched_d, unmatched_d, columns = associate_subset(
            lambda r, c: self._stage_d_cost(r, c, kept),
            self._stage_d_max_cost,
            tentative,
            columns,
        )

        self._pool.apply_matches(
            [*matched_a, *matched_b, *matched_c, *matched_d], kept, embedding_momentum=momentum
        )
        excluded = [row for row in stale + unmatched_b if row not in candidates]
        self._pool.mark_missed([*unmatched_c, *unmatched_d, *excluded])
        self._pool.spawn(kept, columns)
        self._pool.sweep()
        return self._pool.output()

    # -- costs ---------------------------------------------------------------------------

    def _stage_a_cost(
        self, rows: list[int], columns: list[int], kept: list[Detection]
    ) -> np.ndarray:
        """GIoU blended with appearance, then gated on **both** independently.

        The conjunction is deliberate and is where the C++ reference contradicts itself: its
        loop path requires both gates and its vectorised path requires either. A cost matrix
        whose gates can each be satisfied by ignoring the other is not gated.
        """
        boxes = np.stack([kept[c].box for c in columns])
        geometry = giou_cost(self._pool.boxes()[rows], boxes)
        appearance = self._appearance(rows, columns, kept)
        if appearance is None:
            # Nothing to fuse. Falling back to geometry alone is the honest degradation; the
            # alternative — treating a missing appearance distance as zero — asserts that
            # every pair looks identical, which is the strongest claim available and made on
            # no evidence at all.
            return gate_cost(
                geometry, self._pool.gating_distance(boxes, rows), CHI2_INV_95_4DOF
            )

        fused = (
            self._appearance_weight * appearance + (1.0 - self._appearance_weight) * geometry
        )
        forbidden = (geometry > self._giou_gate) | (appearance > self._appearance_gate)
        fused = np.where(forbidden, INFEASIBLE, fused).astype(np.float32)
        return gate_cost(fused, self._pool.gating_distance(boxes, rows), CHI2_INV_95_4DOF)

    def _stage_b_cost(
        self, rows: list[int], columns: list[int], kept: list[Detection]
    ) -> np.ndarray:
        """IoU against the prediction, with appearance demoted to a veto."""
        boxes = np.stack([kept[c].box for c in columns])
        cost = iou_cost(self._pool.boxes()[rows], boxes)
        appearance = self._appearance(rows, columns, kept)
        if appearance is not None:
            cost = np.where(appearance > self._appearance_gate, INFEASIBLE, cost).astype(
                np.float32
            )
        return gate_cost(cost, self._pool.gating_distance(boxes, rows), CHI2_INV_95_4DOF)

    def _stage_c_cost(
        self, rows: list[int], columns: list[int], kept: list[Detection]
    ) -> np.ndarray:
        """IoU against the last observation. No motion gate, by design.

        The prediction is what already failed in stages A and B, and the filter's covariance
        after a gap is wide enough that its gate admits almost anything. Both are reasons to
        leave the filter out of this stage entirely rather than to consult it more carefully.
        """
        boxes = np.stack([kept[c].box for c in columns])
        return iou_cost(self._pool.observed_boxes()[rows], boxes)

    def _stage_d_cost(
        self, rows: list[int], columns: list[int], kept: list[Detection]
    ) -> np.ndarray:
        """IoU, and nothing else. A tentative track has no history worth gating on."""
        boxes = np.stack([kept[c].box for c in columns])
        return iou_cost(self._pool.boxes()[rows], boxes)

    def _appearance(
        self, rows: list[int], columns: list[int], kept: list[Detection]
    ) -> np.ndarray | None:
        """``(len(rows), len(columns))`` cosine distance, or `None` if either side lacks one."""
        track_embeddings = self._pool.embeddings()
        if track_embeddings is None:
            return None
        detection_embeddings = [kept[c].embedding for c in columns]
        if any(e is None for e in detection_embeddings):
            return None
        return appearance_cost(track_embeddings[rows], np.stack(detection_embeddings))

    # -- policy --------------------------------------------------------------------------

    def _momentum(self, kept: list[Detection]) -> np.ndarray | None:
        if not self._dynamic_appearance or not kept:
            return None
        boxes = np.stack([d.box for d in kept])
        scores = np.array([d.score for d in kept], dtype=np.float32)
        low, high = self._momentum_bounds
        return dynamic_appearance_momentum(boxes, scores, min_momentum=low, max_momentum=high)

    def _recoverable(self, rows: list[int], detections: Detections) -> list[int]:
        """The subset of ``rows`` that stage C is allowed to try.

        Only confirmed and lost tracks, and — when the frame size is known — only those whose
        last observation was not against the frame edge. A box clipped by the border is a
        partial view of its object, so its IoU against anything else at that edge is high for
        a reason that has nothing to do with identity.
        """
        eligible = [
            row
            for row in rows
            if self._pool.tracks[row].state in (TrackState.CONFIRMED, TrackState.LOST)
        ]
        if not self._skip_border_recovery or not eligible:
            return eligible
        height, width = detections.height, detections.width
        if height <= 0 or width <= 0:
            # The frame size was not supplied. Guessing it from the boxes would make the rule
            # depend on where the objects happen to be, so the rule is skipped instead.
            return eligible
        margin = self._border_fraction * min(height, width)
        boxes = self._pool.observed_boxes()
        return [
            row
            for row in eligible
            if not (
                boxes[row][0] < margin
                or boxes[row][1] < margin
                or width - boxes[row][2] < margin
                or height - boxes[row][3] < margin
            )
        ]

    def describe(self) -> str:
        return (
            "DeepSORTv2: four-stage cascade (fused / IoU / observation-centric recovery / "
            "tentative) with re-update and a dynamic appearance EMA"
        )
