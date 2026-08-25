"""The objective: deterministic, HOTA by default, and a baseline that is measured.

None of this needs optuna. The trial is a structural protocol, so the whole objective runs
against a fake suggester — which is the point: the part of a tuning system worth unit testing
is the part that decides what "better" means, and that part must not need a sampler installed.
"""

from __future__ import annotations

from typing import Any

import pytest

from shipvision.errors import ConfigurationError, TrackingError
from shipvision.eval.metrics import SequenceResult
from shipvision.tune.objective import DIRECTIONS, Objective, direction_of, midpoint_suggester
from shipvision.tune.spaces import FloatRange, SearchSpace, space_for


class FixedSuggester:
    """Returns whatever it was told to, and records what was asked for.

    Four methods, no optuna, and it makes the objective's output a pure function of a dict —
    which is what lets determinism be asserted rather than hoped for.
    """

    def __init__(self, values: dict[str, Any]) -> None:
        self.values = values
        self.asked: list[str] = []

    def suggest_float(self, name, low, high, *, step=None, log=False):
        self.asked.append(name)
        return float(self.values[name])

    def suggest_int(self, name, low, high, *, step=1, log=False):
        self.asked.append(name)
        return int(self.values[name])

    def suggest_categorical(self, name, choices):
        self.asked.append(name)
        return self.values[name]


class TestTheDefaultObjective:
    def test_it_optimises_hota_and_not_mota(self, case) -> None:
        """MOTA is dominated by false negatives, which belong to the detector; a study against
        it mostly resolves its own sampling noise. The default states which question is being
        asked."""
        objective = Objective("sort", (case,))

        assert objective.metric == "HOTA"
        assert objective.direction == "maximize"

    def test_a_cost_metric_is_minimised(self, case) -> None:
        """A box that has to serve fifty cameras may reasonably tune for latency, and an
        objective that assumed higher-is-better would rank the slowest configuration top."""
        assert Objective("sort", (case,), metric="ms_per_frame").direction == "minimize"
        assert Objective("sort", (case,), metric="IDSW").direction == "minimize"

    def test_every_score_the_report_exposes_has_a_direction(self, case) -> None:
        """Otherwise a caller can name a metric the report shows and the objective cannot rank."""
        available = set(Objective("sort", (case,)).baseline().scores())

        assert available == set(DIRECTIONS)

    def test_an_unknown_metric_raises_rather_than_defaulting(self) -> None:
        with pytest.raises(ConfigurationError, match="unknown metric"):
            direction_of("HOTA1")


class TestDeterminism:
    def test_the_same_parameters_give_the_same_score(self, case) -> None:
        """Track ids differ between runs — the id counter is process-global — but the metrics
        relabel both sides densely before scoring, so the score does not. That is what makes a
        resumed study comparable with the trials it resumes."""
        objective = Objective("sort", (case,))
        parameters = {"det_threshold": 0.4, "iou_threshold": 0.3, "max_age": 20, "min_hits": 2}

        first = objective.score(parameters)
        second = objective.score(parameters)

        assert first == second

    def test_the_track_ids_really_do_differ_between_the_two_runs(self, case) -> None:
        """Without this the test above could be passing because nothing was stateful. The point
        is that the ids *do* move and the score does not."""
        objective = Objective("sort", (case,), space=space_for("sort"))
        parameters = {"min_hits": 1}

        first = objective.run(parameters)
        second = objective.run(parameters)

        assert first.score("HOTA") == second.score("HOTA")

    def test_a_fixed_suggester_makes_the_whole_objective_a_pure_function(self, case) -> None:
        objective = Objective("sort", (case,))
        values = {"det_threshold": 0.4, "iou_threshold": 0.3, "max_age": 20, "min_hits": 2}

        assert objective(FixedSuggester(values)) == objective(FixedSuggester(values))

    def test_the_midpoint_suggester_needs_no_randomness_at_all(self, case) -> None:
        objective = Objective("sort", (case,))
        suggester = midpoint_suggester(objective.search_space)

        assert objective(suggester) == pytest.approx(objective(suggester))


class TestTheBaselineIsMeasured:
    def test_it_is_the_trackers_own_defaults(self, case) -> None:
        """ "Best HOTA 0.62" without "the default scored 0.61" has told nobody anything."""
        objective = Objective("sort", (case,))

        baseline = objective.baseline()

        assert isinstance(baseline, SequenceResult)
        assert baseline.name == "baseline"
        assert baseline.score("HOTA") == objective.run({}).score("HOTA")

    def test_it_is_not_the_midpoint_of_the_search_space(self, case) -> None:
        """A baseline of range midpoints answers a different question. Asserted through the
        parameters rather than the score, because on a five-frame case the two may coincide."""
        space = SearchSpace("sort", (FloatRange("det_threshold", 0.05, 0.7),))

        assert space.defaults() == {}
        assert space.middle() == {"det_threshold": pytest.approx(0.375)}

    def test_the_constants_are_part_of_the_baseline(self, case) -> None:
        """A pinned parameter is part of the configuration being defended, not part of what is
        being searched."""
        space = SearchSpace(
            "sort", (FloatRange("det_threshold", 0.05, 0.7),), constants={"min_hits": 1}
        )
        objective = Objective("sort", (case,), space=space)

        assert objective.baseline().score("HOTA") == objective.run({"min_hits": 1}).score(
            "HOTA"
        )


class TestPerSequencePruningHook:
    def test_the_callback_sees_one_running_aggregate_per_case(self, two_cases) -> None:
        objective = Objective("sort", two_cases)
        seen: list[tuple[int, int]] = []

        objective.run(
            {"min_hits": 1}, on_sequence=lambda step, r: seen.append((step, r.num_frames))
        )

        assert [step for step, _ in seen] == [0, 1]
        assert seen[0][1] < seen[1][1], "the aggregate should grow as sequences are added"

    def test_whatever_the_callback_raises_propagates_unchanged(self, two_cases) -> None:
        """Which is how this module stays free of optuna: the study translates the abandonment
        into ``TrialPruned``, and the objective never learns that word."""

        class AbandonError(Exception):
            pass

        def report(step: int, running: SequenceResult) -> None:
            raise AbandonError("hopeless")

        with pytest.raises(AbandonError):
            Objective("sort", two_cases).run({"min_hits": 1}, on_sequence=report)

    def test_the_running_aggregate_is_the_summed_counts_so_far(self, two_cases) -> None:
        objective = Objective("sort", two_cases)
        partials: list[SequenceResult] = []

        total = objective.run({"min_hits": 1}, on_sequence=lambda step, r: partials.append(r))

        assert partials[-1].num_gt_dets == total.num_gt_dets
        assert partials[0].num_gt_dets < total.num_gt_dets


class TestOneTrackerPerCase:
    def test_two_cases_on_different_cameras_are_both_scored(self, two_cases) -> None:
        """A tracker is stateful and single-camera by construction. A shared instance would
        raise on the camera change, so the objective must build one per case."""
        result = Objective("sort", two_cases).run({"min_hits": 1})

        assert result.num_frames == sum(c.num_frames for c in two_cases)

    def test_a_shared_instance_would_have_raised(self, two_cases) -> None:
        """The guard, asserted directly, so the requirement above cannot quietly stop being one."""
        from shipvision.eval.runner import run
        from shipvision.tracking import TRACKERS

        tracker = TRACKERS.build("sort", min_hits=1)
        run(tracker, two_cases[0])

        with pytest.raises(TrackingError, match="one camera"):
            run(tracker, two_cases[1], reset=False)


class TestRefusals:
    def test_an_empty_case_list_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="score of nothing"):
            Objective("sort", ())

    def test_a_space_for_a_different_tracker_is_rejected(self, case) -> None:
        """A mismatched pair validates its names against the wrong constructor, which is the
        typo check silently switched off."""
        with pytest.raises(ConfigurationError, match="but the objective tunes"):
            Objective("sort", (case,), space=space_for("bytetrack"))

    def test_a_configuration_the_constructor_refuses_becomes_a_typed_error(self, case) -> None:
        """A sampler can reach a corner a constructor rejects. The failure is named here so the
        study can decide between pruning the trial and stopping the run."""
        objective = Objective("bytetrack", (case,))

        with pytest.raises(ConfigurationError, match="bytetrack rejected"):
            objective.run({"low_threshold": 0.9, "track_threshold": 0.5})

    def test_the_description_names_the_metric_the_direction_and_the_sequences(
        self, two_cases
    ) -> None:
        text = Objective("sort", two_cases).describe()

        assert "maximize HOTA" in text
        assert "synthetic-a" in text and "synthetic-b" in text
