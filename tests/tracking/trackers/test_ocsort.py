"""OC-SORT's three mechanisms, each isolated, each against a baseline that is asserted to fail.

The three are switchable independently (``recover``, ``re_update``, ``momentum_weight``) and
these tests use that, because it is the only way to say which of them won a scenario. A test
that turned all three on and beat ByteTrack would prove that *something* in OC-SORT helps,
which is not a statement anyone can act on.

ByteTrack is the named baseline and is built with its paper's thresholds
(``track_threshold=0.5``, ``match_threshold=0.2`` — looser than OC-SORT's
``iou_threshold=0.3``, so if anything it has the advantage). Every scenario here occludes the
object *completely*, with no detections at all, so ByteTrack's low-score second pass has
nothing to work with by construction. That is stated rather than hidden: it is what makes the
comparison about the observation-centric mechanisms rather than about confidence thresholds.

ORU is tested in ``tests/tracking/test_pool.py`` rather than here, and the docstring there
says why: with this library's Kalman filter its effect on the *state* is exact but small, so
the honest test is the exactness rather than a scenario tuned to a threshold.
"""

from __future__ import annotations

import numpy as np

from shipvision.tracking import TRACKERS
from shipvision.types import Detection, Detections, FrameTag

# A ship: wide, and it moves a good fraction of its own width every frame. Deliberately not
# person-shaped — a fast, wide object is the case where a drifted prediction stays
# geometrically plausible for a long time, which is where observation-centric ideas earn their
# keep.
SHIP_W, SHIP_H = 220.0, 130.0
LANE_Y = 500.0
CAMERA = "approach-channel"


def _ship(cx: float) -> Detection:
    return Detection(
        box=np.array(
            [cx - SHIP_W / 2, LANE_Y - SHIP_H / 2, cx + SHIP_W / 2, LANE_Y + SHIP_H / 2],
            np.float32,
        ),
        score=0.95,
    )


def _run(name: str, path: list[float | None], **options: object) -> list[set[int]]:
    """Feed one detection per frame; `None` means the detector saw nothing at all."""
    tracker = TRACKERS.build(name, min_hits=2, max_age=40, **options)
    published = []
    for frame_id, cx in enumerate(path):
        items = [] if cx is None else [_ship(cx)]
        tracks = tracker.update(
            Detections(tag=FrameTag(CAMERA, frame_id), items=items, height=1080, width=1920)
        )
        published.append({t.track_id for t in tracks})
    return published


def _kept(published: list[set[int]], before: slice, after: slice) -> bool:
    """Did the identity published before the disruption survive to after it?"""
    first = set().union(*published[before]) if published[before] else set()
    last = set().union(*published[after]) if published[after] else set()
    return bool(first) and bool(last) and first == last


# --------------------------------------------------------------- OCR: it moored while hidden

_LEAD, _SPEED, _GAP = 12, 22.0, 6
_LAST_SEEN = 200.0 + _SPEED * (_LEAD - 1)
_MOORED_PATH: list[float | None] = (
    [200.0 + _SPEED * f for f in range(_LEAD)] + [None] * _GAP + [_LAST_SEEN] * 12
)
_BEFORE = slice(0, _LEAD)
_AFTER = slice(_LEAD + _GAP + 2, None)


class TestObservationCentricRecovery:
    """Matching against the last observation instead of the prediction, which is what recovers
    an object that stopped moving while it was hidden.
    """

    def test_observation_centric_recovery_keeps_an_object_that_stopped_while_hidden(
        self,
    ) -> None:
        """The case OCR exists for, and the one the brief describes: occluded while moving fast.

        A ship doing 22 px/frame goes behind a crane and moors there. Six frames later it is
        still exactly where it was last seen, but the filter has spent seven frames extrapolating
        22 px/frame, so its prediction sits 154 px downstream — three quarters of the hull — and
        IoU against the reappearing detection is 0.18. Every prediction-based association refuses
        that, and the ship comes back as a new vessel.

        OC-SORT's second association ignores the prediction and scores the detection against the
        track's *last observation*, which is the very box the ship is sitting in. IoU 1.0.

        All three baseline halves are asserted rather than assumed. ByteTrack and SORT must fail,
        and so must OC-SORT with ``recover=False`` — if that one succeeded, the scenario would be
        being won by the momentum term or by the re-update and this test would be mislabelled.
        """
        assert _kept(_run("ocsort", _MOORED_PATH), _BEFORE, _AFTER)

        for baseline, options in (
            ("bytetrack", {}),
            ("sort", {}),
            ("ocsort", {"recover": False}),
        ):
            assert not _kept(_run(baseline, _MOORED_PATH, **options), _BEFORE, _AFTER), (
                f"{baseline} {options} kept the identity; if the baseline succeeds, this test "
                f"measures nothing about observation-centric recovery"
            )

    def test_recovery_publishes_from_the_first_frame_the_object_is_back(self) -> None:
        """Not just eventually. The recovered track is CONFIRMED, so it publishes immediately.

        A tracker that recovers the identity but withholds it for ``min_hits`` frames has
        converted an identity error into a latency error, which for a berthing alarm is the same
        error.
        """
        published = _run("ocsort", _MOORED_PATH)
        first_frame_back = _LEAD + _GAP
        assert published[first_frame_back], "nothing published on the frame the ship reappeared"
        assert published[first_frame_back] == published[_LEAD - 1]

    def test_recovery_is_not_a_licence_to_reattach_anything_nearby(self) -> None:
        """The same six-frame gap, but a *different* vessel has taken the berth.

        Matching the stale box to whatever is closest would give the newcomer the moored ship's
        identity, which is exactly what a looser recovery threshold buys. The reappearing box is
        three hull-lengths past the last observation, so it must not be recovered.
        """
        path: list[float | None] = (
            [200.0 + _SPEED * f for f in range(_LEAD)]
            + [None] * _GAP
            + [_LAST_SEEN + 3 * SHIP_W] * 12
        )
        published = _run("ocsort", path)
        before = set().union(*published[_BEFORE])
        after = set().union(*published[_AFTER])
        assert before and after
        assert before.isdisjoint(after), "recovery reattached an identity to a different object"


class TestObservationCentricMomentum:
    """The heading term, and where it sits relative to the motion gate."""

    def test_momentum_refuses_a_candidate_the_object_could_not_have_reached(self) -> None:
        """OCM against the same tracker with the term switched off, and nothing else changed.

        Without the heading term the track takes the geometrically nearer box and jumps
        backwards. With it, the backward candidate carries the full ``arccos(-1) / pi = 1``
        penalty, weighted 0.2, which is more than the 0.09 of IoU cost it was winning by — so the
        forward box wins and the ship keeps going the way ships go.

        ``gate=False`` here, and that is the finding rather than a convenience. With the
        Mahalanobis gate on, *both* candidates are refused before the momentum term is ever
        consulted: an 80 px jump is many sigma for a filter that has observed this object on ten
        consecutive frames. So on a well-supported track the gate is the first line of defence
        and momentum is the second, and the second only speaks when the first has been widened by
        a gap. Asserting the gate's behaviour too is what stops someone later reading this test as
        evidence that momentum is what protects a healthy track.
        """
        without = _which_way(0.0, gate=False)
        with_momentum = _which_way(0.2, gate=False)

        assert len(without) == 1 and len(with_momentum) == 1
        assert without[0] < _OCM_PREDICTED, (
            f"the baseline was expected to jump backwards towards {_OCM_BACKWARD}; it went to "
            f"{without[0]:.0f}, so this frame is not ambiguous and the comparison is empty"
        )
        assert with_momentum[0] > _OCM_PREDICTED, (
            f"momentum should have chosen the forward box near {_OCM_FORWARD}; it chose "
            f"{with_momentum[0]:.0f}"
        )

    def test_the_motion_gate_refuses_both_candidates_before_momentum_is_consulted(self) -> None:
        """The other half of the finding above, asserted so it cannot be forgotten.

        A track observed on ten consecutive frames has a tight covariance, so an 80 px
        displacement is implausible at *any* cost. The correct outcome is that neither candidate
        is taken and the track ages — not that the tracker picks the least bad one.
        """
        for momentum_weight in (0.0, 0.2):
            assert _which_way(momentum_weight, gate=True) == []


# --------------------------------------------------- OCM: the candidate has to be plausible

# One track heading right at 30 px/frame. On the disputed frame the detector offers two boxes:
# one 80 px *behind* the prediction and one 100 px *ahead* of it. On IoU alone the nearer,
# backward box wins outright (0.467 against 0.375) — there is no tie to break, so the baseline
# fails for a stated reason rather than by coin toss. Only the heading says the ship cannot
# have gone backwards.
_OCM_LEAD, _OCM_SPEED = 10, 30.0
_OCM_LAST = 200.0 + _OCM_SPEED * (_OCM_LEAD - 1)
_OCM_PREDICTED = _OCM_LAST + _OCM_SPEED
_OCM_BACKWARD = _OCM_PREDICTED - 80.0
_OCM_FORWARD = _OCM_PREDICTED + 100.0


def _which_way(momentum_weight: float, *, gate: bool) -> list[float]:
    """Run the ambiguous frame and return the centres of whatever was published."""
    tracker = TRACKERS.build(
        "ocsort",
        min_hits=2,
        max_age=40,
        momentum_weight=momentum_weight,
        recover=False,
        re_update=False,
        gate=gate,
    )
    for frame_id in range(_OCM_LEAD):
        tracker.update(
            Detections(
                tag=FrameTag(CAMERA, frame_id),
                items=[_ship(200.0 + _OCM_SPEED * frame_id)],
                height=1080,
                width=1920,
            )
        )
    tracks = tracker.update(
        Detections(
            tag=FrameTag(CAMERA, _OCM_LEAD),
            # Backward first, so the solver's own column order also favours the wrong answer.
            items=[_ship(_OCM_BACKWARD), _ship(_OCM_FORWARD)],
            height=1080,
            width=1920,
        )
    )
    return [float(t.box[0] + t.box[2]) / 2.0 for t in tracks]
