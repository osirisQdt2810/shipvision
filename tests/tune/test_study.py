"""Running a study: the baseline is reported, the result is resumable, and nothing is invented.

Skipped when optuna is absent — which is the whole point of the dependency being optional, and
is asserted separately in ``test_dependency.py``. Everything here runs on five-frame synthetic
sequences, so the assertions are about the machinery: what is counted, what is reported, and
what happens when the study finds nothing.
"""

from __future__ import annotations

import pytest

from shipvision.errors import ConfigurationError

optuna = pytest.importorskip(
    "optuna", reason="optuna is an optional extra; pip install 'shipvision[tune]'"
)
optuna.logging.set_verbosity(optuna.logging.WARNING)

from shipvision.tune import run_study  # noqa: E402
from shipvision.tune.spaces import FloatRange, IntRange, SearchSpace  # noqa: E402


@pytest.fixture
def small_space() -> SearchSpace:
    """Two parameters, so a four-trial study is a real search rather than a formality."""
    return SearchSpace(
        "sort",
        (FloatRange("iou_threshold", 0.15, 0.45), IntRange("min_hits", 1, 3)),
    )


class TestTheResultReportsTheBaseline:
    def test_both_scores_are_present_and_the_verdict_is_spelled_out(
        self, case, small_space
    ) -> None:
        """ "Best HOTA 0.62" is not a finding. "0.62 against a default of 0.61" is, and so is
        "0.62 against a default of 0.63" — which happens, and has to be sayable."""
        result = run_study("sort", [case], space=small_space, trials=4, seed=1)

        assert result.baseline.name == "baseline"
        assert result.best.name == "best"
        summary = result.summary()
        assert "baseline HOTA" in summary
        assert "best HOTA" in summary
        assert ("tuning gained" in summary) or ("did NOT beat the default" in summary)

    def test_the_improvement_is_signed_so_positive_always_means_better(
        self, case, small_space
    ) -> None:
        result = run_study("sort", [case], space=small_space, trials=4, seed=1)

        assert result.improvement == pytest.approx(result.best_score - result.baseline_score)
        assert result.beat_the_baseline == (result.improvement > 0.0)

    def test_for_a_cost_metric_the_sign_flips_so_lower_is_still_positive(
        self, case, small_space
    ) -> None:
        """A report where "improvement: -0.4" sometimes means better and sometimes worse is a
        report that will be read wrongly."""
        result = run_study(
            "sort", [case], space=small_space, metric="ms_per_frame", trials=4, seed=1
        )

        assert result.direction == "minimize"
        assert result.improvement == pytest.approx(result.baseline_score - result.best_score)

    def test_the_best_trial_is_rescored_so_every_metric_is_available(
        self, case, small_space
    ) -> None:
        """A report that showed only the objective would hide a HOTA win bought by doubling
        ms/frame."""
        result = run_study("sort", [case], space=small_space, trials=4, seed=1)

        scores = result.best.scores()
        assert {"HOTA", "IDF1", "MOTA", "IDSW", "ms_per_frame"} <= set(scores)
        assert scores["HOTA"] == pytest.approx(result.best_score)

    def test_the_best_parameters_are_names_the_space_declared(self, case, small_space) -> None:
        result = run_study("sort", [case], space=small_space, trials=4, seed=1)

        assert set(result.best_params) == set(small_space.names)


class TestReproducibility:
    def test_the_same_seed_gives_the_same_best_parameters(self, case, small_space) -> None:
        """Two runs of the same command producing two answers means neither can be quoted."""
        first = run_study("sort", [case], space=small_space, trials=5, seed=11)
        second = run_study("sort", [case], space=small_space, trials=5, seed=11)

        assert first.best_params == second.best_params
        assert first.best_score == pytest.approx(second.best_score)

    def test_a_different_seed_may_explore_differently(self, case, small_space) -> None:
        """Not a strict inequality — a five-frame case has plateaus — but the sampled points
        must not be identical, or the seed is being ignored."""
        first = run_study("sort", [case], space=small_space, trials=5, seed=11)
        second = run_study("sort", [case], space=small_space, trials=5, seed=12)

        assert first.best_params != second.best_params or first.best_score != second.best_score


class TestPersistenceAndResuming:
    def test_a_second_call_adds_trials_to_the_same_study(
        self, case, small_space, tmp_path
    ) -> None:
        """A study that cannot be resumed will not be run long enough to matter."""
        url = f"sqlite:///{tmp_path / 'tune.db'}"

        first = run_study("sort", [case], space=small_space, trials=4, seed=3, storage=url)
        second = run_study("sort", [case], space=small_space, trials=4, seed=3, storage=url)

        assert second.trials + second.pruned + second.invalid > first.trials + first.pruned
        assert second.study_name == first.study_name

    def test_the_default_study_name_is_what_makes_the_resume_work(
        self, case, small_space, tmp_path
    ) -> None:
        """Without a stable default the caller has to remember a name, and a forgotten name
        silently starts a fresh study that looks like a continuation."""
        url = f"sqlite:///{tmp_path / 'tune.db'}"

        result = run_study("sort", [case], space=small_space, trials=2, seed=3, storage=url)

        assert result.study_name == "sort-HOTA"

    def test_two_metrics_do_not_collide_in_one_database(
        self, case, small_space, tmp_path
    ) -> None:
        url = f"sqlite:///{tmp_path / 'tune.db'}"

        hota = run_study("sort", [case], space=small_space, trials=2, seed=3, storage=url)
        mota = run_study(
            "sort", [case], space=small_space, metric="MOTA", trials=2, seed=3, storage=url
        )

        assert hota.study_name != mota.study_name


class TestPruningAndCounting:
    """A pruned trial, an invalid configuration and a completed trial are three different
    events. A study that reports only "50 trials" cannot tell you that forty-eight of them were
    configurations the constructor refused."""

    def test_pruning_abandons_a_hopeless_configuration_before_the_later_sequences(
        self, two_cases, small_space
    ) -> None:
        """At 900 frames a real sequence is most of a trial's cost, so a configuration that is
        hopeless on the first one should not be given the other four."""
        result = run_study(
            "sort", list(two_cases), space=small_space, trials=8, seed=5, prune=True
        )

        assert result.trials >= 1
        assert result.trials + result.pruned + result.invalid >= 8

    def test_pruning_can_be_switched_off_so_every_trial_costs_the_same(
        self, two_cases, small_space
    ) -> None:
        """Which is what a fair timing comparison between samplers needs."""
        result = run_study(
            "sort", list(two_cases), space=small_space, trials=4, seed=5, prune=False
        )

        assert result.pruned == 0
        assert result.trials == 4

    def test_a_configuration_the_constructor_refuses_is_counted_as_invalid(self, case) -> None:
        """Not as a failure and not silently: a space that is mostly invalid must not hide
        behind a best trial found in its usable sliver."""
        mostly_invalid = SearchSpace(
            "bytetrack",
            (FloatRange("low_threshold", 0.05, 0.9),),
            constants={"track_threshold": 0.5, "min_hits": 1},
        )

        result = run_study(
            "bytetrack", [case], space=mostly_invalid, trials=10, seed=2, prune=False
        )

        assert result.invalid > 0
        assert result.trials > 0

    def test_a_study_where_nothing_completed_raises_rather_than_reporting_the_baseline(
        self, case
    ) -> None:
        """A study with no completed trial has no finding. Returning the baseline as the best
        would report the default as if it had won a search it never entered."""
        always_invalid = SearchSpace(
            "bytetrack",
            (FloatRange("low_threshold", 0.6, 0.9),),
            constants={"track_threshold": 0.5},
        )

        with pytest.raises(ConfigurationError, match="no trial of"):
            run_study("bytetrack", [case], space=always_invalid, trials=4, seed=2, prune=False)


class TestRefusals:
    def test_a_non_positive_trial_count_is_rejected(self, case, small_space) -> None:
        with pytest.raises(ConfigurationError, match="trials must be positive"):
            run_study("sort", [case], space=small_space, trials=0)

    def test_no_sequences_is_rejected_by_the_objective(self, small_space) -> None:
        with pytest.raises(ConfigurationError, match="score of nothing"):
            run_study("sort", [], space=small_space, trials=2)

    def test_an_unknown_metric_is_rejected_before_any_trial_runs(
        self, case, small_space
    ) -> None:
        with pytest.raises(ConfigurationError, match="unknown metric"):
            run_study("sort", [case], space=small_space, metric="HOTA1", trials=2)


class TestParallelTrials:
    def test_two_threads_produce_the_same_number_of_trials(self, case, small_space) -> None:
        """Threads rather than processes: a trial spends its time inside numpy, which releases
        the GIL, and a process pool would reload every sequence per worker. The assertion is
        only that the accounting survives concurrency — the *best* trial legitimately differs,
        because a parallel TPE sampler sees a different history."""
        result = run_study("sort", [case], space=small_space, trials=6, seed=4, jobs=2)

        assert result.trials + result.pruned + result.invalid == 6
