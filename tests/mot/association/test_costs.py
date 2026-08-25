"""The cost matrices, on inputs whose right answer can be worked out by hand.

A cost function is the one part of a tracker that can be checked exactly, so it is checked
exactly here rather than inferred from a scenario. Everything downstream — five trackers and
every gate — is a composition of these, so a wrong sign or a wrong normalisation shows up as
"tracking is a bit worse", which is not a debuggable symptom.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipvision.mot.association import (
    INFEASIBLE,
    appearance_cost,
    direction_cost,
    fuse_score,
    gate_cost,
    gated_iou_cost,
    giou_cost,
    giou_matrix,
    iou_cost,
    min_fuse,
)
from shipvision.mot.motion.kalman import CHI2_INV_95_4DOF


def _box(x1: float, y1: float, x2: float, y2: float) -> list[float]:
    return [x1, y1, x2, y2]


class TestIouCost:
    """``1 - IoU``: the workhorse, and the only cost three of the five trackers use."""

    def test_it_is_one_minus_the_overlap(self) -> None:
        tracks = np.array([_box(0, 0, 10, 10)], np.float32)
        detections = np.array(
            [_box(0, 0, 10, 10), _box(5, 0, 15, 10), _box(40, 40, 50, 50)], np.float32
        )
        np.testing.assert_allclose(
            iou_cost(tracks, detections)[0], [0.0, 1 - 1 / 3, 1.0], atol=1e-6
        )

    def test_an_empty_side_gives_a_correctly_shaped_matrix(self) -> None:
        """An empty frame is normal input, and ``(0,)`` instead of ``(0, m)`` turns it into an
        IndexError two functions away."""
        assert iou_cost(np.zeros((0, 4), np.float32), np.ones((3, 4), np.float32)).shape == (
            0,
            3,
        )


class TestGiou:
    """Generalised IoU, which exists because IoU is flat at zero for everything disjoint."""

    def test_identical_boxes_score_one(self) -> None:
        box = np.array([_box(10, 10, 30, 40)], np.float32)
        assert giou_matrix(box, box)[0, 0] == pytest.approx(1.0)

    def test_it_keeps_falling_as_boxes_separate_where_iou_does_not(self) -> None:
        """The property the cascade depends on: two candidates that both miss are still
        ranked. On IoU they would both score exactly zero and the assignment would be
        deciding between them on nothing."""
        track = np.array([_box(0, 0, 10, 10)], np.float32)
        near = np.array([_box(20, 0, 30, 10)], np.float32)
        far = np.array([_box(200, 0, 210, 10)], np.float32)
        assert giou_matrix(track, near)[0, 0] > giou_matrix(track, far)[0, 0]
        assert giou_cost(track, near)[0, 0] < giou_cost(track, far)[0, 0]

    def test_the_cost_range_is_zero_to_two(self) -> None:
        """Worth pinning down because it decides what a threshold means: a GIoU *cost* of 1.0
        is "touching but not overlapping", not "no overlap at all"."""
        track = np.array([_box(0, 0, 10, 10)], np.float32)
        touching = np.array([_box(10, 0, 20, 10)], np.float32)
        assert giou_cost(track, track)[0, 0] == pytest.approx(0.0)
        assert giou_cost(track, touching)[0, 0] == pytest.approx(1.0, abs=1e-5)
        assert giou_cost(track, np.array([_box(1e5, 1e5, 1e5 + 10, 1e5 + 10)], np.float32))[
            0, 0
        ] == pytest.approx(2.0, abs=1e-3)

    def test_giou_never_exceeds_iou(self) -> None:
        rng = np.random.default_rng(0)
        corners = rng.uniform(0, 100, size=(8, 2))
        sizes = rng.uniform(5, 40, size=(8, 2))
        boxes = np.concatenate([corners, corners + sizes], axis=1).astype(np.float32)
        from shipvision.types import iou_matrix

        assert np.all(giou_matrix(boxes, boxes) <= iou_matrix(boxes, boxes) + 1e-5)


class TestAppearanceCost:
    """Cosine distance, and the contract that it does not renormalise."""

    def test_identical_vectors_cost_nothing_and_opposites_cost_two(self) -> None:
        a = np.array([[1.0, 0.0]], np.float32)
        b = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], np.float32)
        np.testing.assert_allclose(appearance_cost(a, b)[0], [0.0, 1.0, 2.0], atol=1e-6)

    def test_it_does_not_renormalise_its_inputs(self) -> None:
        """A contract, not laziness: silently normalising here would hide a caller feeding
        raw logits, and the cost would look plausible while meaning nothing."""
        unit = np.array([[1.0, 0.0]], np.float32)
        scaled = np.array([[3.0, 0.0]], np.float32)
        assert appearance_cost(unit, scaled)[0, 0] == pytest.approx(0.0)  # clipped at 0
        assert appearance_cost(unit, -scaled)[0, 0] == pytest.approx(2.0)  # clipped at 2


class TestFuseScore:
    """ByteTrack's confidence folding."""

    def test_a_weak_detection_makes_the_same_overlap_cost_more(self) -> None:
        cost = np.array([[0.2, 0.2]], np.float32)  # IoU 0.8 for both
        fused = fuse_score(cost, np.array([0.9, 0.3], np.float32))
        assert fused[0, 0] < fused[0, 1]
        np.testing.assert_allclose(fused[0], [1 - 0.8 * 0.9, 1 - 0.8 * 0.3], atol=1e-6)

    def test_an_empty_matrix_passes_through(self) -> None:
        empty = np.zeros((0, 3), np.float32)
        assert fuse_score(empty, np.ones(3, np.float32)).shape == (0, 3)


class TestGateCost:
    """Forbidding, not penalising."""

    def test_a_gated_pair_becomes_unselectable_but_finite(self) -> None:
        """Finite because ``linear_sum_assignment`` raises on an all-infinite matrix, and a
        frame where every pair is implausible is a normal frame."""
        cost = np.array([[0.1, 0.2]], np.float32)
        distances = np.array([[1.0, 99.0]], np.float32)
        gated = gate_cost(cost, distances, 9.4877)
        assert gated[0, 0] == pytest.approx(0.1)
        assert gated[0, 1] == INFEASIBLE
        assert np.isfinite(gated).all()

    def test_the_original_matrix_is_not_modified(self) -> None:
        """Several trackers gate the same cost twice with different thresholds; in-place
        would make the second call depend on the first."""
        cost = np.array([[0.1]], np.float32)
        gate_cost(cost, np.array([[99.0]], np.float32), 1.0)
        assert cost[0, 0] == pytest.approx(0.1)

    def test_a_weight_breaks_ties_towards_what_the_filter_expected(self) -> None:
        cost = np.array([[0.4, 0.4]], np.float32)
        distances = np.array([[1.0, 8.0]], np.float32)
        blended = gate_cost(cost, distances, 9.4877, weight=0.1)
        assert blended[0, 0] < blended[0, 1]


class TestMinFuse:
    """BoT-SORT's fusion, and the specific thing it does that a weighted sum does not."""

    def test_either_signal_on_its_own_is_enough(self) -> None:
        """The point of the minimum. A weighted sum would drag both of these over a
        threshold; the minimum lets each be decided by whichever signal is confident."""
        motion = np.array([[0.7, 0.05]], np.float32)  # poor overlap, then excellent
        appearance = np.array([[0.02, 0.9]], np.float32)  # excellent, then poor
        fused = min_fuse(
            motion, appearance, motion_gate=0.8, appearance_gate=0.25, appearance_weight=0.5
        )
        assert fused[0, 0] == pytest.approx(0.01)  # decided by appearance
        assert fused[0, 1] == pytest.approx(0.05)  # decided by geometry

    def test_a_weighted_sum_would_reject_what_the_minimum_accepts(self) -> None:
        """Stated as a comparison so the choice is documented rather than assumed.

        Two people in identical uniforms rounding a corner: appearance says 0.02, geometry
        says 0.7. A 50/50 sum gives 0.36, which a 0.25 threshold refuses; the minimum gives
        0.01, which it accepts. The pair is the same pair either way — only the fusion
        differs.
        """
        motion = np.array([[0.7]], np.float32)
        appearance = np.array([[0.02]], np.float32)
        weighted_sum = 0.5 * motion + 0.5 * appearance
        fused = min_fuse(motion, appearance, motion_gate=0.8, appearance_gate=0.25)
        assert weighted_sum[0, 0] > 0.25
        assert fused[0, 0] < 0.25

    def test_appearance_cannot_rescue_a_pair_the_geometry_gate_refused(self) -> None:
        """The paper gates appearance by IoU as well, and this pins that down.

        Without it, a matching shirt would be able to attach an identity to a box on the
        other side of the frame — which is the failure a re-ID-heavy tracker has, and it is
        much worse than a fragmented track.
        """
        motion = np.array([[0.95]], np.float32)  # over the gate
        appearance = np.array([[0.0]], np.float32)  # perfect
        fused = min_fuse(motion, appearance, motion_gate=0.8, appearance_gate=0.25)
        assert fused[0, 0] == pytest.approx(1.0)

    def test_a_pair_failing_both_gates_is_one(self) -> None:
        fused = min_fuse(
            np.array([[0.99]], np.float32),
            np.array([[1.4]], np.float32),
            motion_gate=0.8,
            appearance_gate=0.25,
        )
        assert fused[0, 0] == pytest.approx(1.0)


class TestDirectionCost:
    """OC-SORT's momentum term, whose value is fully determined by two angles."""

    def test_straight_ahead_is_free_and_straight_back_is_maximal(self) -> None:
        heading = np.array([[1.0, 0.0]], np.float32)
        origin = np.array([_box(0, 0, 10, 10)], np.float32)
        candidates = np.array(
            [_box(100, 0, 110, 10), _box(-100, 0, -90, 10), _box(0, 100, 10, 110)], np.float32
        )
        cost = direction_cost(heading, origin, candidates)[0]
        assert cost[0] == pytest.approx(0.0, abs=1e-5)
        assert cost[1] == pytest.approx(1.0, abs=1e-5)
        assert cost[2] == pytest.approx(0.5, abs=1e-5)

    def test_a_track_with_no_measured_heading_is_neutral_not_punished(self) -> None:
        """``(0, 0)`` means unmeasured. Scoring it 0.5 — the mean — would quietly penalise
        every newly born track for the crime of being new."""
        cost = direction_cost(
            np.array([[0.0, 0.0]], np.float32),
            np.array([_box(0, 0, 10, 10)], np.float32),
            np.array([_box(-500, 0, -490, 10)], np.float32),
        )
        assert cost[0, 0] == pytest.approx(0.0)

    def test_a_detection_on_top_of_the_origin_carries_no_direction(self) -> None:
        """Its offset has no direction to compare, so any value but zero would be invented."""
        box = np.array([_box(0, 0, 10, 10)], np.float32)
        cost = direction_cost(np.array([[1.0, 0.0]], np.float32), box, box)
        assert cost[0, 0] == pytest.approx(0.0)

    def test_the_result_is_always_in_zero_one(self) -> None:
        rng = np.random.default_rng(4)
        headings = rng.normal(size=(6, 2)).astype(np.float32)
        headings /= np.linalg.norm(headings, axis=1, keepdims=True)
        origins = rng.uniform(0, 500, size=(6, 4)).astype(np.float32)
        origins[:, 2:] += origins[:, :2]
        dets = rng.uniform(0, 500, size=(9, 4)).astype(np.float32)
        dets[:, 2:] += dets[:, :2]
        cost = direction_cost(headings, origins, dets)
        assert cost.shape == (6, 9)
        assert np.all((cost >= 0.0) & (cost <= 1.0))

    def test_an_empty_side_gives_a_correctly_shaped_matrix(self) -> None:
        assert direction_cost(
            np.zeros((0, 2), np.float32),
            np.zeros((0, 4), np.float32),
            np.ones((3, 4), np.float32),
        ).shape == (0, 3)


class TestGatedIouCost:
    """``1 - IoU`` plus an optional motion veto: the composition SORT and ByteTrack's second
    stage both are.

    It lives in ``association/`` rather than in either algorithm's ``utils.py`` because two
    copies of it is how the baseline and the tracker measured against it drift apart: someone
    tightens a gate in one, both keep passing every test they have, and the comparison the two
    exist to support quietly stops being a comparison.
    """

    def test_without_a_gate_it_is_exactly_the_iou_cost(self) -> None:
        tracks = np.array([_box(0, 0, 10, 10), _box(20, 20, 30, 30)], np.float32)
        detections = np.array([_box(0, 0, 10, 10), _box(5, 0, 15, 10)], np.float32)

        np.testing.assert_array_equal(
            gated_iou_cost(tracks, detections, threshold=9.4877),
            iou_cost(tracks, detections),
        )

    def test_a_pair_the_filter_refuses_becomes_unselectable(self) -> None:
        """A perfect overlap the motion model calls impossible must not be selectable at any
        price. Leaving it selectable is how one crowded frame hands an identity to the wrong
        object, and an ID switch does not recover the way a missed frame does."""
        tracks = np.array([_box(0, 0, 10, 10)], np.float32)
        detections = np.array([_box(0, 0, 10, 10), _box(0, 0, 10, 10)], np.float32)
        distances = np.array([[0.5, 99.0]], np.float32)

        cost = gated_iou_cost(tracks, detections, gating_distances=distances, threshold=9.4877)

        assert cost[0, 0] == pytest.approx(0.0)
        assert cost[0, 1] == INFEASIBLE

    def test_no_gate_and_an_all_passing_gate_are_different_calls_with_the_same_answer(
        self,
    ) -> None:
        """``None`` means "the gate is off", which is a different state from "the gate passed
        everything" — the caller who switched gating off should not have to compute the
        distances first. The two agree numerically, and that is what makes the cheaper call
        safe to make."""
        tracks = np.array([_box(0, 0, 10, 10)], np.float32)
        detections = np.array([_box(0, 0, 10, 10), _box(5, 0, 15, 10)], np.float32)
        passing = np.zeros((1, 2), np.float32)

        np.testing.assert_allclose(
            gated_iou_cost(tracks, detections, threshold=9.4877),
            gated_iou_cost(tracks, detections, gating_distances=passing, threshold=9.4877),
        )

    def test_the_threshold_has_no_default(self) -> None:
        """The chi-square value belongs to the filter that produced the distances, so
        ``association`` does not carry one — a default here would outlive a change to the
        state dimension and gate on the wrong number without saying so."""
        tracks = np.array([_box(0, 0, 10, 10)], np.float32)

        with pytest.raises(TypeError, match="threshold"):
            gated_iou_cost(tracks, tracks)  # type: ignore[call-arg]

    def test_it_is_what_both_callers_actually_call(self) -> None:
        """The claim that the two algorithms share one implementation, checked rather than
        asserted in a docstring."""
        from shipvision.mot.trackers.bytetrack.utils import low_score_cost
        from shipvision.mot.trackers.sort.utils import association_cost

        tracks = np.array([_box(0, 0, 10, 10), _box(20, 20, 30, 30)], np.float32)
        detections = np.array([_box(2, 0, 12, 10), _box(21, 20, 31, 30)], np.float32)
        distances = np.array([[0.5, 99.0], [99.0, 0.5]], np.float32)

        expected = gated_iou_cost(
            tracks, detections, gating_distances=distances, threshold=CHI2_INV_95_4DOF
        )
        np.testing.assert_array_equal(
            association_cost(tracks, detections, gating_distances=distances), expected
        )
        np.testing.assert_array_equal(
            low_score_cost(tracks, detections, gating_distances=distances), expected
        )
