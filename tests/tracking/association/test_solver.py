"""The assignment: one optimal solve, one threshold, and the index bookkeeping around it.

The three things tested here are the three that multi-stage trackers get wrong, and none of
them is about the Hungarian algorithm — scipy's is correct and is not retested. They are about
*when* the threshold is applied, *whose* indices come back, and *which* tracks are allowed to
bid first.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipvision.tracking.association import associate, associate_subset, cascade_associate


class TestAssociate:
    """The single solve."""

    def test_it_finds_the_globally_cheapest_pairing_not_the_greedy_one(self) -> None:
        """Greedy would take (0, 0) at 0.10 and then be forced into (1, 1) at 0.90, total
        1.00. The optimum is (0, 1) + (1, 0) at 0.20 + 0.15 = 0.35."""
        cost = np.array([[0.10, 0.20], [0.15, 0.90]], np.float32)
        matches, rows, cols = associate(cost, max_cost=0.95)
        assert sorted(matches) == [(0, 1), (1, 0)]
        assert rows == [] and cols == []

    def test_the_threshold_is_applied_after_the_solve(self) -> None:
        """Not a cosmetic ordering: the two orders give different matches.

        Three tracks, three detections, and track 0 has no plausible candidate at all. The
        solver must produce a full assignment, so it hands track 0 the cheapest thing left
        (0.70) — and the reason that is *correct* is that doing so lets tracks 1 and 2 keep
        the pairs they actually want. Dropping track 0's over-threshold pair afterwards
        leaves those two alone.

        Masking first instead forbids track 0 everything, so the solver reshuffles: it gives
        detection 0 to **track 1**, which is not where track 1 belongs, and detection 2 —
        track 1's real match — goes unclaimed. One frame of that is an ID switch, and an ID
        switch does not heal the way a missed frame does.
        """
        from shipvision.tracking.association import INFEASIBLE

        cost = np.array(
            [[0.70, 0.75, 0.80], [0.15, 0.95, 0.20], [0.90, 0.30, 0.95]], np.float32
        )
        matches, rows, cols = associate(cost, max_cost=0.5)
        assert sorted(matches) == [(1, 2), (2, 1)]
        assert rows == [0] and cols == [0]

        # The same matrix, gated before the solve, reaches a different and worse answer.
        masked = np.where(cost > 0.5, INFEASIBLE, cost)
        early_matches, _, early_cols = associate(masked, max_cost=0.5)
        assert sorted(early_matches) == [(1, 0), (2, 1)]
        assert early_cols == [2]

    def test_an_all_gated_matrix_terminates_with_no_matches(self) -> None:
        """A frame where every pair is implausible is a normal frame; the solver must not
        raise, which is why the gate uses a large finite cost rather than infinity."""
        from shipvision.tracking.association import INFEASIBLE

        cost = np.full((3, 2), INFEASIBLE, np.float32)
        matches, rows, cols = associate(cost, max_cost=0.7)
        assert matches == []
        assert rows == [0, 1, 2] and cols == [0, 1]

    @pytest.mark.parametrize("shape", [(0, 0), (0, 4), (5, 0)])
    def test_an_empty_side_returns_everything_unmatched(self, shape: tuple[int, int]) -> None:
        matches, rows, cols = associate(np.zeros(shape, np.float32), max_cost=0.5)
        assert matches == []
        assert rows == list(range(shape[0]))
        assert cols == list(range(shape[1]))


class TestAssociateSubset:
    """The index translation that every multi-stage tracker gets wrong at least once."""

    def test_positions_come_back_as_the_caller_s_own_indices(self) -> None:
        """The failure this prevents is silent: the solver returns positions *within* the
        submatrix, and using those as track indices associates the wrong objects while
        everything still runs."""
        calls: list[tuple[list[int], list[int]]] = []

        def build(rows: list[int], cols: list[int]) -> np.ndarray:
            calls.append((list(rows), list(cols)))
            # Row 7 wants column 30; row 4 wants column 12.
            return np.array([[0.9, 0.1], [0.1, 0.9]], np.float32)

        matches, rows, cols = associate_subset(build, 0.5, [4, 7], [12, 30])
        assert calls == [([4, 7], [12, 30])]
        assert sorted(matches) == [(4, 30), (7, 12)]
        assert rows == [] and cols == []

    def test_unmatched_rows_and_columns_are_also_translated(self) -> None:
        def build(rows: list[int], cols: list[int]) -> np.ndarray:
            return np.array([[0.05, 0.99, 0.99]], np.float32)

        matches, rows, cols = associate_subset(build, 0.5, [9], [100, 200, 300])
        assert matches == [(9, 100)]
        assert rows == []
        assert cols == [200, 300]

    def test_an_empty_side_never_calls_the_cost_builder(self) -> None:
        """Cost builders here do ``np.stack`` over the selected columns, which raises on an
        empty selection. Guarding once, here, is why no tracker has to."""

        def build(rows: list[int], cols: list[int]) -> np.ndarray:
            raise AssertionError("must not be called")

        assert associate_subset(build, 0.5, [], [1, 2]) == ([], [], [1, 2])
        assert associate_subset(build, 0.5, [3], []) == ([], [3], [])


class TestCascadeAssociate:
    """Recently-seen tracks choose first."""

    def test_a_fresh_track_gets_first_refusal_over_a_stale_one(self) -> None:
        """One detection, two tracks that both want it, and the stale one wants it *more*
        cheaply — which is exactly the situation a widening gate creates. Banding by age is
        what stops the older track winning on the strength of having been guessing longer.
        """
        ages = np.array([0, 9], np.int32)

        def build(rows: list[int], cols: list[int]) -> np.ndarray:
            # Track 1 (stale) would score 0.05; track 0 (fresh) scores 0.30.
            return np.array([[0.30]] if rows == [0] else [[0.05]], np.float32)

        matches, rows, cols = cascade_associate(
            build, 0.5, [0, 1], [0], ages, stride=1, max_depth=10
        )
        assert matches == [(0, 0)]
        assert rows == [1] and cols == []

    def test_a_wider_stride_puts_both_in_one_band_and_the_cheaper_one_wins(self) -> None:
        """The trade the stride buys, made explicit: fewer solves, less precedence. The
        internal reference uses five, so this is the behaviour it chose."""
        ages = np.array([0, 4], np.int32)

        def build(rows: list[int], cols: list[int]) -> np.ndarray:
            return np.array([[0.30], [0.05]], np.float32)

        matches, _rows, _cols = cascade_associate(
            build, 0.5, [0, 1], [0], ages, stride=5, max_depth=10
        )
        assert matches == [(1, 0)]

    def test_it_stops_once_the_detections_are_used_up(self) -> None:
        bands: list[list[int]] = []
        ages = np.array([0, 1, 2], np.int32)

        def build(rows: list[int], cols: list[int]) -> np.ndarray:
            bands.append(list(rows))
            return np.zeros((len(rows), len(cols)), np.float32)

        cascade_associate(build, 0.5, [0, 1, 2], [0], ages, stride=1, max_depth=10)
        assert bands == [[0]], "later bands were solved with nothing left to assign"

    def test_a_track_older_than_max_depth_is_never_offered_anything(self) -> None:
        ages = np.array([50], np.int32)

        def build(rows: list[int], cols: list[int]) -> np.ndarray:
            raise AssertionError("must not be called")

        matches, rows, cols = cascade_associate(
            build, 0.5, [0], [0], ages, stride=1, max_depth=10
        )
        assert matches == [] and rows == [0] and cols == [0]

    def test_a_stride_below_one_is_refused(self) -> None:
        """It would loop forever. Failing at the call is better than a hung worker thread."""
        with pytest.raises(ValueError, match="stride"):
            cascade_associate(
                lambda r, c: np.zeros((1, 1)),
                0.5,
                [0],
                [0],
                np.zeros(1, np.int32),
                stride=0,
                max_depth=4,
            )

    def test_an_empty_side_returns_everything_unmatched(self) -> None:
        def build(rows: list[int], cols: list[int]) -> np.ndarray:
            raise AssertionError("must not be called")

        assert cascade_associate(
            build, 0.5, [], [1], np.zeros(0, np.int32), stride=1, max_depth=4
        ) == ([], [], [1])
