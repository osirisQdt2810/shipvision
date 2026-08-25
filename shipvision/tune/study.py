"""Running a study: pruning, persistence, and a result that reports the baseline.

Three things this module insists on, each because its absence has produced a study nobody
should have believed.

**The baseline is measured, not remembered.** Every :class:`StudyResult` carries the score of
the tracker as it ships, produced by the same objective over the same sequences. "Best HOTA
0.62" is not a finding; "0.62 against a default of 0.61, over five sequences" is, and so is
"0.62 against a default of 0.63", which happens often enough — a sampler exploring a
seven-dimensional space in fifty trials frequently never revisits the shipped configuration.

**A study that cannot be resumed will not be run long enough to matter.** Pass a ``storage``
URL and a ``study_name`` and the trials land in SQLite; run the same call again and it picks up
where it stopped. That is what turns "I'll run 200 trials overnight" into something that
survives a laptop lid.

**A pruned trial is counted and reported separately from a failed one.** A configuration the
constructor refuses, a configuration the pruner abandoned after one sequence, and a
configuration that crashed are three different events, and a study that reports only "50
trials" cannot tell you that forty-eight of them were invalid.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from shipvision.errors import ConfigurationError
from shipvision.eval.metrics import SequenceResult
from shipvision.eval.sequence import EvaluationCase
from shipvision.tune._optuna import require_optuna
from shipvision.tune.objective import Objective
from shipvision.tune.spaces import SearchSpace

optuna = require_optuna()

__all__ = ["StudyResult", "run_study"]


@dataclass(frozen=True, slots=True)
class StudyResult:
    """What a study found, next to what it started from.

    Attributes:
        tracker: the registry name that was tuned.
        metric: the score that was optimised, and its direction.
        baseline: the shipped configuration, scored by the same objective.
        best: the winning trial, re-scored so that every metric is available and not only the
            objective. Re-scored rather than cached from the trial because a study loaded from
            storage has the parameters but not the full result, and a report that showed HOTA
            for the best trial and nothing else would hide a win bought by doubling ms/frame.
        best_params: the winning keywords.
        trials: how many trials completed.
        pruned: how many the pruner abandoned.
        invalid: how many the tracker's constructor refused.
    """

    tracker: str
    metric: str
    baseline: SequenceResult
    best: SequenceResult
    best_params: Mapping[str, Any] = field(default_factory=dict)
    trials: int = 0
    pruned: int = 0
    invalid: int = 0
    study_name: str = ""

    @property
    def baseline_score(self) -> float:
        return self.baseline.score(self.metric)

    @property
    def best_score(self) -> float:
        return self.best.score(self.metric)

    @property
    def improvement(self) -> float:
        """Best minus baseline, signed so that positive always means better.

        Signed by *direction*, not by subtraction order: for ``ms_per_frame`` a lower number is
        the win, and a report where "improvement: -0.4" sometimes means better and sometimes
        worse is a report that will be read wrongly.
        """
        delta = self.best_score - self.baseline_score
        return delta if self.direction == "maximize" else -delta

    @property
    def direction(self) -> str:
        from shipvision.tune.objective import direction_of

        return direction_of(self.metric)

    @property
    def beat_the_baseline(self) -> bool:
        return self.improvement > 0.0

    def summary(self) -> str:
        """The three lines a report needs, with the verdict spelled out."""
        verdict = (
            f"tuning gained {self.improvement:+.4f}"
            if self.beat_the_baseline
            else f"tuning did NOT beat the default ({self.improvement:+.4f})"
        )
        lines = [
            f"{self.tracker}: {self.direction} {self.metric} over "
            f"{self.trials} completed trial(s) "
            f"({self.pruned} pruned, {self.invalid} invalid)",
            f"  baseline {self.metric} = {self.baseline_score:.4f}   "
            f"best {self.metric} = {self.best_score:.4f}   ->  {verdict}",
            f"  best params: {dict(self.best_params)}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"<StudyResult {self.tracker} {self.metric} baseline={self.baseline_score:.4f} "
            f"best={self.best_score:.4f} trials={self.trials}>"
        )


def run_study(
    tracker: str,
    cases: Sequence[EvaluationCase],
    *,
    space: SearchSpace | None = None,
    metric: str = "HOTA",
    trials: int = 30,
    seed: int | None = None,
    storage: str | None = None,
    study_name: str | None = None,
    jobs: int = 1,
    prune: bool = True,
    threshold: float = 0.5,
    backend: str | None = None,
    timeout: float | None = None,
) -> StudyResult:
    """Tune ``tracker`` over ``cases`` and return the best trial *and* the baseline.

    Args:
        tracker: the registry name.
        cases: the sequences, in a fixed order. Report per sequence elsewhere; the objective
            sums their counts, which is the only aggregation that means anything.
        space: what to vary. Defaults to the declared space for this tracker.
        metric: what to optimise. HOTA by default — see :mod:`shipvision.tune.objective` for
            why not MOTA.
        trials: how many to run *in this call*. With ``storage`` set, a second call adds
            another ``trials`` to the same study rather than starting over.
        seed: the sampler's seed. Set it, or two runs of the same command produce two answers
            and neither can be quoted.
        storage: an Optuna storage URL, e.g. ``sqlite:///tune.db``. Without it the study lives
            in memory and dies with the process.
        study_name: needed to resume. Defaults to ``"<tracker>-<metric>"``, which is what makes
            the resume work without the caller having to remember a name.
        jobs: parallel trials. Threads, not processes: each trial spends its time inside numpy,
            which releases the GIL, and a process pool would reload every sequence per worker.
        prune: report the running score after each sequence so a hopeless configuration is
            abandoned before the expensive ones. Off makes every trial cost the same, which is
            what a fair timing comparison between samplers needs.
        threshold: the IoU cliff for CLEAR and IDF1.
        backend: which tracker implementation to build.
        timeout: seconds, passed to Optuna. A wall-clock budget rather than a trial budget.

    Raises:
        BackendUnavailableError: optuna is not installed. Raised on import of this module.
        ConfigurationError: nothing usable came out — every trial was pruned or invalid. Not
            returned as a result with a NaN best: a study that produced no completed trial has
            no finding, and reporting one would be reporting the baseline as if it had won.
    """
    if trials < 1:
        raise ConfigurationError(f"trials must be positive, got {trials}")

    objective = Objective(
        tracker=tracker,
        cases=tuple(cases),
        space=space,
        metric=metric,
        threshold=threshold,
        backend=backend,
    )
    counters = {"pruned": 0, "invalid": 0}

    def wrapped(trial: Any) -> float:
        def report(step: int, running: SequenceResult) -> None:
            trial.report(running.score(metric), step)
            if trial.should_prune():
                raise optuna.TrialPruned(
                    f"abandoned after {step + 1} of {len(objective.cases)} sequence(s)"
                )

        parameters = objective.parameters(trial)
        try:
            result = objective.run(parameters, on_sequence=report if prune else None)
        except ConfigurationError as error:
            # A configuration the tracker refuses is not a failed study, it is a corner of the
            # space. Counted, so a space that is mostly invalid cannot hide behind a best
            # trial found in its usable sliver.
            counters["invalid"] += 1
            raise optuna.TrialPruned(str(error)) from error
        return result.score(metric)

    study = optuna.create_study(
        direction=objective.direction,
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=(
            optuna.pruners.MedianPruner(n_warmup_steps=1)
            if prune
            else optuna.pruners.NopPruner()
        ),
        storage=storage,
        study_name=study_name or f"{tracker}-{metric}",
        load_if_exists=True,
    )
    study.optimize(wrapped, n_trials=trials, n_jobs=jobs, timeout=timeout, gc_after_trial=True)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    counters["pruned"] = (
        sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
        - counters["invalid"]
    )
    if not completed:
        raise ConfigurationError(
            f"no trial of {tracker!r} completed: {counters['invalid']} were configurations the "
            f"constructor refused and {max(0, counters['pruned'])} were pruned. A study with no "
            f"completed trial has no finding — check the search space rather than the tracker"
        )

    best = study.best_trial
    return StudyResult(
        tracker=tracker,
        metric=metric,
        baseline=objective.baseline(),
        best=objective.run(dict(best.params)).renamed("best"),
        best_params=dict(best.params),
        trials=len(completed),
        pruned=max(0, counters["pruned"]),
        invalid=counters["invalid"],
        study_name=study.study_name,
    )
