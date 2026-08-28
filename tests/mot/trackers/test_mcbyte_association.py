"""McByte's pre-assignment helpers, against the reference they were ported from.

The oracle is ``data/mcbyte_association_golden.json``, dumped from roboflow/trackers' own
``mask_association`` **before** this port existed and never regenerated from it. The reference
maximises a similarity and this library minimises a cost, so every case is read through one
conversion — ``cost = 1 - similarity``, ``max_cost = 1 - minimum_similarity`` — and a sign flip
anywhere in the port makes a golden case disagree rather than quietly agreeing with itself.

The hand-built cases beside it are all asymmetric and mostly non-square, because a transposed
or inverted comparison passes every square symmetric test that exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from shipvision.mot import TRACKERS
from shipvision.mot.association import INFEASIBLE, associate, fuse_score
from shipvision.mot.trackers.mcbyte.utils import (
    ambiguous_candidates,
    clear_matches,
    isolated_candidates,
    reduce_problem,
)

GOLDEN = json.loads(
    (Path(__file__).parent / "data" / "mcbyte_association_golden.json").read_text()
)
CASES = sorted(GOLDEN["cases"])


def case(name: str) -> tuple[np.ndarray, np.ndarray, float]:
    """``(cost, iou_cost, max_cost)`` for a golden case, in this library's minimising space."""
    inputs = GOLDEN["cases"][name]["inputs"]
    return (
        1.0 - np.asarray(inputs["similarity"], dtype=np.float32),
        1.0 - np.asarray(inputs["raw_iou_similarity"], dtype=np.float32),
        1.0 - inputs["minimum_similarity"],
    )


class TestClearMatches:
    """What locks is what has no alternative on either side. Everything else waits."""

    def test_a_pair_alone_in_its_row_and_its_column_locks(self) -> None:
        cost = np.array([[0.9, 0.2, 0.9, 0.4], [0.9, 0.9, 0.1, 0.9]], np.float32)

        assert clear_matches(cost, 0.5) == [(1, 2)]

    def test_a_row_with_two_affordable_candidates_locks_nothing(self) -> None:
        """Two eligible candidates in one row is exactly the situation an assignment is for,
        and deciding it early would be deciding it on no evidence."""
        cost = np.array([[0.2, 0.4, 0.9]], np.float32)

        assert clear_matches(cost, 0.5) == []

    def test_a_column_with_two_affordable_candidates_locks_nothing(self) -> None:
        """The column half of the rule, asserted separately: an implementation that reduced
        over the wrong axis passes the row test and loses an identity on a crowded frame."""
        cost = np.array([[0.2], [0.4]], np.float32)

        assert clear_matches(cost, 0.5) == []

    def test_a_gate_that_forbade_every_pair_locks_nothing(self) -> None:
        """``INFEASIBLE`` is a large finite cost, not a missing entry. Locking one would turn
        the motion model's "impossible" into this frame's certainty."""
        cost = np.full((2, 3), INFEASIBLE, np.float32)
        cost[0, 1] = 0.2

        assert clear_matches(cost, 0.5) == [(0, 1)]
        assert clear_matches(np.full((2, 3), INFEASIBLE, np.float32), 0.5) == []

    def test_a_pair_costing_exactly_the_threshold_is_affordable_to_the_lock_and_the_solver(
        self,
    ) -> None:
        """``cost <= max_cost`` here is the same question as ``cost > max_cost: continue``
        in :func:`~shipvision.mot.association.solver.associate`, and the two have to answer
        it identically at the boundary. A pair the solver would accept but locking calls
        unaffordable is a pair whose fate depends on which of the two saw it first — and the
        conversion this file lives in, ``max_cost = 1 - minimum_similarity``, lands on the
        boundary exactly whenever the threshold is a round number."""
        cost = np.array([[0.5, 0.9, 0.9], [0.9, 0.9, 0.62]], np.float32)

        assert clear_matches(cost, 0.5) == [(0, 0)]
        assert associate(cost, 0.5)[0] == [(0, 0)]

    @pytest.mark.parametrize("name", CASES)
    def test_it_agrees_with_the_reference(self, name: str) -> None:
        cost, _, max_cost = case(name)
        expected = [tuple(pair) for pair in GOLDEN["cases"][name]["helpers"]["clear_matches"]]

        assert clear_matches(cost, max_cost) == expected


class TestTheReducedProblemMapsBackToCallerIndices:
    """Two reductions happen per stage — the caller's subset, then the locked rows — so two
    translations happen on the way back. Applying one of them associates the wrong objects
    while every shape still agrees, which is the bug this class exists for.
    """

    #: Lock at ``(1, 2)``: not row 0 and not the last column, so a submatrix index used as a
    #: caller index lands on a different real object rather than on itself.
    COST = np.array(
        [[0.9, 0.2, 0.9, 0.4], [0.9, 0.9, 0.1, 0.9], [0.9, 0.45, 0.9, 0.3]], np.float32
    )

    def test_the_locked_row_and_column_leave_the_problem(self) -> None:
        reduced, kept_rows, kept_columns = reduce_problem(self.COST, [(1, 2)])

        assert kept_rows == [0, 2]
        assert kept_columns == [0, 1, 3]
        np.testing.assert_allclose(reduced, [[0.9, 0.2, 0.4], [0.9, 0.45, 0.3]], atol=1e-6)

    def test_a_stage_returns_the_callers_own_track_and_detection_indices(self) -> None:
        """Driven through the association hook itself, with rows and columns that are not
        ``range(n)``: that is the shape a second association stage always has, and it is where
        an unmapped submatrix index is indistinguishable from a match."""
        tracker = TRACKERS.build("mcbyte", match_threshold=0.5)

        matches, unmatched_rows, unmatched_columns = tracker._associate(
            lambda rows, columns: self.COST, 0.5, [3, 5, 7], [10, 11, 12, 13], []
        )

        assert matches == [(3, 11), (5, 12), (7, 13)]
        assert unmatched_rows == []
        assert unmatched_columns == [10]

    @pytest.mark.parametrize("name", CASES)
    def test_it_agrees_with_the_reference(self, name: str) -> None:
        cost, _, max_cost = case(name)
        helpers = GOLDEN["cases"][name]["helpers"]
        reduced, kept_rows, kept_columns = reduce_problem(cost, clear_matches(cost, max_cost))

        assert [kept_rows, kept_columns] == helpers["remaining"]
        if GOLDEN["cases"][name]["inputs"]["mask_spec"] is None:
            # No masks means the reference applied no boost, so its returned matrix is the
            # reduction and nothing else — a real oracle for the extraction.
            expected = GOLDEN["cases"][name]["result"]["conditioned_similarity"]
            np.testing.assert_allclose(
                reduced,
                1.0 - np.asarray(expected, np.float32).reshape(reduced.shape),
                atol=1e-6,
            )


class TestAmbiguousAndIsolatedCandidates:
    """The two predicates the mask half of the paper conditions on. Neither is used by the
    association yet; both are ported and pinned now, because they are where the sign flip is.
    """

    def test_a_pair_is_ambiguous_when_its_row_or_its_column_has_a_rival(self) -> None:
        cost = np.array([[0.2, 0.4, 0.9, 0.9], [0.9, 0.3, 0.9, 0.9]], np.float32)

        assert ambiguous_candidates(cost, 0.5).tolist() == [
            [True, True, False, False],
            [False, True, False, False],
        ]

    def test_a_clear_pair_is_not_ambiguous(self) -> None:
        """The two predicates partition the affordable pairs, which is what lets the paper
        lock one group and condition the other."""
        cost = np.array([[0.2, 0.9, 0.9], [0.9, 0.9, 0.1]], np.float32)

        assert clear_matches(cost, 0.5) == [(0, 0), (1, 2)]
        assert not ambiguous_candidates(cost, 0.5).any()

    def test_isolation_reads_raw_overlap_and_not_a_score_fused_cost(self) -> None:
        """Fusing the detector's confidence moves a pair across the threshold for reasons that
        have nothing to do with pixels, and can erase the *rival* that made a pair crowded —
        inventing an isolation the geometry never supported."""
        raw = np.array([[0.6, 0.9]], np.float32)
        fused = fuse_score(raw, np.array([1.0, 0.0], np.float32))

        assert isolated_candidates(raw, 0.5).tolist() == [[False, False]]
        assert isolated_candidates(fused, 0.5).tolist() == [[True, False]]

    def test_a_pair_good_enough_to_match_is_not_isolated(self) -> None:
        """Isolation is a rescue for pairs *below* the threshold. Above it the ordinary
        assignment already has the pair, and marking it would double-count the evidence."""
        cost = np.array([[0.2, 1.0], [1.0, 1.0]], np.float32)

        assert not isolated_candidates(cost, 0.5).any()

    def test_a_gated_pair_never_counts_as_overlap(self) -> None:
        cost = np.array([[0.6, INFEASIBLE], [INFEASIBLE, INFEASIBLE]], np.float32)

        assert isolated_candidates(cost, 0.5).tolist() == [[True, False], [False, False]]

    @pytest.mark.parametrize("name", CASES)
    def test_they_agree_with_the_reference(self, name: str) -> None:
        cost, iou_cost, max_cost = case(name)
        helpers = GOLDEN["cases"][name]["helpers"]

        assert ambiguous_candidates(cost, max_cost).tolist() == helpers["ambiguous"]
        assert isolated_candidates(iou_cost, max_cost).tolist() == helpers["isolated"]


class TestTheGoldenFixtureIsTheReferencesAndNotOurs:
    """A fixture regenerated from the port is a test of nothing. The provenance block is the
    only thing standing between those two states, so it is asserted rather than trusted.
    """

    def test_it_records_the_upstream_commit_it_came_from(self) -> None:
        provenance = GOLDEN["provenance"]

        assert provenance["upstream"] == "roboflow/trackers (Apache-2.0)"
        assert len(provenance["commit"]) == 40
        assert "1 - similarity" in provenance["conversion"]

    def test_every_case_is_exercised(self) -> None:
        """A case added to the fixture and named nowhere is a case that proves nothing."""
        assert len(CASES) == 11
