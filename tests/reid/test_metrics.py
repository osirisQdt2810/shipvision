from __future__ import annotations

import numpy as np
import pytest

from shipvision.errors import ConfigurationError
from shipvision.reid import evaluate_ranking


def sim(rows: list[list[float]]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float32)


def test_a_perfect_ranking_scores_one_everywhere() -> None:
    result = evaluate_ranking(
        sim([[0.9, 0.1, 0.1], [0.1, 0.9, 0.1]]),
        query_identities=["a", "b"],
        gallery_identities=["a", "b", "c"],
    )

    assert result.rank(1) == 1.0
    assert result.mean_ap == 1.0
    assert result.evaluated == 2
    assert result.skipped == 0


def test_rank_five_catches_what_rank_one_misses() -> None:
    """The whole point of a CMC curve: a model that is never first but always third is a
    usable candidate generator, and rank-1 alone reports it as a total failure."""
    result = evaluate_ranking(
        sim([[0.1, 0.9, 0.8, 0.7]]),
        query_identities=["a"],
        gallery_identities=["a", "b", "c", "d"],
    )

    assert result.rank(1) == 0.0
    assert result.rank(4) == 1.0


def test_map_and_rank_one_disagree_and_both_are_right() -> None:
    """An identity with one easy view and several hard ones scores rank-1 = 1 and a poor
    AP. Reporting only rank-1 is how a regression in the hard cases goes unnoticed."""
    result = evaluate_ranking(
        sim([[0.99, 0.5, 0.4, 0.3, 0.2]]),
        query_identities=["a"],
        gallery_identities=["a", "x", "y", "z", "a"],
    )

    assert result.rank(1) == 1.0
    # Positives at ranks 1 and 5: (1/1 + 2/5) / 2.
    assert result.mean_ap == pytest.approx((1.0 + 2.0 / 5.0) / 2.0)


def test_a_same_camera_hit_is_discarded_and_the_ranks_below_it_move_up() -> None:
    """The protocol's central rule, and the one that decides whether a reported number
    means anything. The easy same-camera match must not be counted right — but it must
    also not be counted wrong, which would punish the model for a match nobody asked for.
    """
    similarity = sim([[0.99, 0.80, 0.10]])
    ids = ["a", "a", "b"]
    cams = ["cam-1", "cam-2", "cam-1"]

    counted = evaluate_ranking(
        similarity, ["a"], ids, query_cameras=["cam-1"], gallery_cameras=cams
    )

    # Entry 0 is the query's own camera and identity: dropped. Entry 1 is the same identity
    # from another camera and is now rank 1.
    assert counted.rank(1) == 1.0
    assert counted.mean_ap == 1.0


def test_without_the_camera_filter_the_same_data_flatters_the_model() -> None:
    """Kept as a pair with the test above: the difference between them is exactly the
    inflation the protocol exists to remove."""
    similarity = sim([[0.99, 0.10, 0.95]])
    ids = ["a", "a", "b"]
    cams = ["cam-1", "cam-2", "cam-1"]

    unfiltered = evaluate_ranking(similarity, ["a"], ids)
    filtered = evaluate_ranking(
        similarity, ["a"], ids, query_cameras=["cam-1"], gallery_cameras=cams
    )

    assert unfiltered.rank(1) == 1.0, "the trivial self-match wins"
    assert filtered.rank(1) == 0.0, "with it gone, the model is actually wrong here"


def test_a_query_with_no_valid_ground_truth_is_skipped_not_scored_zero() -> None:
    """Averaging in a zero for an unanswerable query reports the composition of the test
    set as if it were the model's accuracy."""
    result = evaluate_ranking(
        sim([[0.9, 0.1], [0.1, 0.9]]),
        query_identities=["a", "ghost"],
        gallery_identities=["a", "b"],
    )

    assert result.evaluated == 1
    assert result.skipped == 1
    assert result.rank(1) == 1.0, "the one answerable query was answered correctly"


def test_a_query_whose_only_positive_is_filtered_out_is_also_skipped() -> None:
    result = evaluate_ranking(
        sim([[0.9, 0.1]]),
        query_identities=["a"],
        gallery_identities=["a", "b"],
        query_cameras=["cam-1"],
        gallery_cameras=["cam-1", "cam-1"],
    )

    assert result.evaluated == 0
    assert result.skipped == 1
    assert result.mean_ap == 0.0


def test_the_cmc_curve_is_monotone_non_decreasing() -> None:
    rng = np.random.default_rng(7)
    ids = [f"id-{i % 10}" for i in range(60)]
    result = evaluate_ranking(
        rng.random((20, 60), dtype=np.float32),
        query_identities=[f"id-{i % 10}" for i in range(20)],
        gallery_identities=ids,
    )

    assert np.all(np.diff(result.cmc) >= 0.0)
    assert result.cmc[-1] <= 1.0


def test_max_rank_is_clamped_to_the_gallery() -> None:
    result = evaluate_ranking(sim([[0.9, 0.1]]), ["a"], ["a", "b"], max_rank=500)

    assert len(result.cmc) == 2
    assert result.rank(500) == result.rank(2), "asking beyond the end returns the end"


def test_a_shape_disagreement_says_which_side_is_wrong() -> None:
    with pytest.raises(ConfigurationError, match="3 queries"):
        evaluate_ranking(sim([[0.1, 0.2]]), ["a", "b", "c"], ["x", "y"])


def test_rank_is_one_indexed_as_everyone_writes_it() -> None:
    result = evaluate_ranking(sim([[0.9, 0.1]]), ["a"], ["a", "b"])

    assert result.rank(1) == result.cmc[0]
    with pytest.raises(ConfigurationError, match="1-indexed"):
        result.rank(0)


class TestTheCameraFilterCannotBeHalfSupplied:
    """The protocol filter needs both camera lists, and forgetting one used to disable it.

    Silently, and in the flattering direction. On the data below, both lists give rank-1
    0.0000 and mAP 0.5000; either one alone gives 1.0000 and 0.8333 — the trivial
    same-camera self-match counted as a hit, which the module docstring names as how an
    implementation reports rank-1 in the high nineties and fails in the field. One missing
    keyword argument is a plausible mistake, and its symptom is a better number.
    """

    IDS = ["a", "a", "b"]
    CAMS = ["cam-1", "cam-2", "cam-1"]
    SIM = np.asarray([[0.99, 0.10, 0.95]], dtype=np.float32)

    def test_query_cameras_without_gallery_cameras_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="both"):
            evaluate_ranking(self.SIM, ["a"], self.IDS, query_cameras=["cam-1"])

    def test_gallery_cameras_without_query_cameras_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="both"):
            evaluate_ranking(self.SIM, ["a"], self.IDS, gallery_cameras=self.CAMS)

    def test_both_together_filter_and_neither_is_an_explicit_choice(self) -> None:
        """Omitting both is still allowed: a single-camera dataset has nothing to exclude,
        and the caller has said so by passing neither rather than by forgetting one."""
        filtered = evaluate_ranking(
            self.SIM, ["a"], self.IDS, query_cameras=["cam-1"], gallery_cameras=self.CAMS
        )
        unfiltered = evaluate_ranking(self.SIM, ["a"], self.IDS)

        assert filtered.rank(1) == 0.0
        assert filtered.mean_ap == pytest.approx(0.5)
        assert unfiltered.rank(1) == 1.0, "the trivial self-match, counted on purpose"
