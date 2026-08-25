"""Trial parameters in, one number out — and the number is HOTA by default.

**Why HOTA and not MOTA.** MOTA is ``1 - (FN + FP + IDSW) / GT``, an unweighted sum of three
error types, and on a public-detection benchmark FN dominates by an order of magnitude: on
MOT17-09 with SORT it is 2464 misses against 16 false positives and 15 identity switches.
Misses are the detector's, and no amount of tuning a tracker changes them. So a study that
optimises MOTA spends its budget resolving differences of ten or twenty errors inside a total
of two and a half thousand — which is measuring its own sampler. HOTA's association half
responds to exactly the thing a tracker controls, so the same budget buys a real answer. The
metric is selectable because a deployment whose problem genuinely is duplicate boxes should be
able to say so, but the default states which question is usually being asked.

**Determinism is a property, not an accident.** Every tracker here is deterministic given its
input, and the input is a fixed list of :class:`~shipvision.eval.sequence.EvaluationCase`
loaded once. Track *ids* differ between runs — the id counter is process-global — but the
metrics relabel both sides to a dense range before scoring, so the score does not. That is
what makes a resumed study comparable with the trials it is resuming, and there is a test that
asserts it rather than assuming it.

**This module does not import optuna.** The trial is a
:class:`~shipvision.tune.spaces.Suggester`, which Optuna's ``Trial`` satisfies structurally, so
the whole objective can be exercised offline with a four-line fake. Pruning is expressed as a
callback the caller supplies; translating it into ``optuna.TrialPruned`` is
:mod:`shipvision.tune.study`'s job, because that is the module that already owns the
dependency.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from shipvision.errors import ConfigurationError
from shipvision.eval.metrics import SequenceResult, combine
from shipvision.eval.runner import evaluate
from shipvision.eval.sequence import EvaluationCase
from shipvision.mot import TRACKERS
from shipvision.tune.spaces import SearchSpace, Suggester, space_for

__all__ = ["DIRECTIONS", "Objective", "direction_of", "midpoint_suggester"]

#: Which way is better, per metric name from
#: :meth:`~shipvision.eval.metrics.base.SequenceResult.scores`.
#:
#: Needed because an objective that assumed "higher is better" would rank a tracker that emits
#: nothing top of a study minimising nothing — and because ``ms_per_frame`` is a perfectly
#: reasonable thing to optimise on a box that has to serve fifty cameras.
DIRECTIONS: dict[str, str] = {
    "HOTA": "maximize",
    "DetA": "maximize",
    "AssA": "maximize",
    "AssRe": "maximize",
    "AssPr": "maximize",
    "LocA": "maximize",
    "IDF1": "maximize",
    "IDP": "maximize",
    "IDR": "maximize",
    "MOTA": "maximize",
    "MOTP": "maximize",
    "MT": "maximize",
    "IDSW": "minimize",
    "FP": "minimize",
    "FN": "minimize",
    "ML": "minimize",
    "Frag": "minimize",
    "ms_per_frame": "minimize",
}


def direction_of(metric: str) -> str:
    """``"maximize"`` or ``"minimize"``. Raises on an unknown name.

    Not defaulted to "maximize": a typo'd metric would then be optimised in whichever
    direction happened to be wrong for it, and the study would still finish and report a best
    trial.
    """
    if metric not in DIRECTIONS:
        raise ConfigurationError(f"unknown metric {metric!r}; known: {sorted(DIRECTIONS)}")
    return DIRECTIONS[metric]


@dataclass(frozen=True, slots=True)
class Objective:
    """One tracker, one metric, a fixed set of sequences. Callable with a trial.

    Attributes:
        tracker: the registry name.
        cases: the sequences to score, in a fixed order. Fixed because the pruner compares
            trial *n*'s score after two sequences with trial *m*'s after two sequences, and
            that is only a comparison if they were the same two.
        space: what to vary. Defaults to :func:`~shipvision.tune.spaces.space_for`.
        metric: the name from ``SequenceResult.scores()`` to return.
        threshold: the IoU cliff CLEAR and IDF1 use. HOTA sweeps its own nineteen.
        backend: which tracker implementation to build.
    """

    tracker: str
    cases: tuple[EvaluationCase, ...]
    space: SearchSpace | None = None
    metric: str = "HOTA"
    threshold: float = 0.5
    backend: str | None = None
    _resolved: SearchSpace = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "cases", tuple(self.cases))
        if not self.cases:
            raise ConfigurationError(
                f"no sequences to tune {self.tracker!r} on; a study over an empty case list "
                f"would compare every configuration's score of nothing"
            )
        direction_of(self.metric)
        space = self.space if self.space is not None else space_for(self.tracker)
        if space.tracker != self.tracker:
            raise ConfigurationError(
                f"the space is for {space.tracker!r} but the objective tunes {self.tracker!r}; "
                f"a mismatched pair validates its names against the wrong constructor"
            )
        object.__setattr__(self, "_resolved", space)

    @property
    def search_space(self) -> SearchSpace:
        return self._resolved

    @property
    def direction(self) -> str:
        return direction_of(self.metric)

    @property
    def sequence_names(self) -> tuple[str, ...]:
        return tuple(case.name for case in self.cases)

    # -- evaluation ----------------------------------------------------------------------

    def build(self, parameters: Mapping[str, Any]) -> Any:
        """Construct the tracker, turning a rejected configuration into a typed error.

        A sampler can reach a corner the constructor refuses — ByteTrack requires
        ``low_threshold < track_threshold``, for instance. The declared spaces are built so
        that cannot happen, but a caller's own space may not be, so the failure is named here
        and :mod:`shipvision.tune.study` decides whether to prune the trial or stop.
        """
        try:
            return TRACKERS.build(self.tracker, backend=self.backend, **dict(parameters))
        except ConfigurationError as error:
            raise ConfigurationError(
                f"{self.tracker} rejected {dict(parameters)}: {error}"
            ) from error

    def run(
        self,
        parameters: Mapping[str, Any],
        *,
        on_sequence: Callable[[int, SequenceResult], None] | None = None,
    ) -> SequenceResult:
        """Score one configuration over every case, summing the counts.

        Args:
            parameters: keywords for the tracker's constructor.
            on_sequence: called after each case with ``(step, running_aggregate)``. This is the
                pruning hook: a configuration that is hopeless on the first sequence does not
                need the other four, and at 900 frames a sequence that is most of the trial's
                cost. The callback may raise to abandon the trial, and whatever it raises
                propagates unchanged — which is how this module stays free of optuna.

        A **fresh tracker per case**, because a tracker is stateful and single-camera by
        construction: one instance across two sequences either raises on the camera change or
        carries the first sequence's pool into the second.
        """
        results: list[SequenceResult] = []
        for step, case in enumerate(self.cases):
            results.append(evaluate(self.build(parameters), case, threshold=self.threshold))
            if on_sequence is not None:
                on_sequence(step, combine(results, name=f"after-{step + 1}"))
        return combine(results, name=self.tracker)

    def baseline(self) -> SequenceResult:
        """The tracker as it ships, plus the space's constants.

        Reported next to every result this module produces. "Best HOTA 0.62" without "the
        default scored 0.61" has told nobody anything, and the case where tuning *loses* — the
        sampler never finding the shipped configuration inside a 200-trial budget — is common
        enough that the comparison has to be automatic rather than remembered.
        """
        return self.run(self._resolved.defaults()).renamed("baseline")

    def score(self, parameters: Mapping[str, Any]) -> float:
        """The single number a sampler optimises."""
        return self.run(parameters).score(self.metric)

    # -- the trial interface -------------------------------------------------------------

    def parameters(self, trial: Suggester) -> dict[str, Any]:
        return self._resolved.suggest(trial)

    def __call__(
        self,
        trial: Suggester,
        *,
        on_sequence: Callable[[int, SequenceResult], None] | None = None,
    ) -> float:
        """Sample, run, and return the metric. The signature Optuna's ``optimize`` wants."""
        parameters = self.parameters(trial)
        return self.run(parameters, on_sequence=on_sequence).score(self.metric)

    def describe(self) -> str:
        return (
            f"{self.tracker}: {self.direction} {self.metric} over "
            f"{len(self.cases)} sequence(s) ({', '.join(self.sequence_names)})\n"
            f"{self._resolved.describe()}"
        )


def midpoint_suggester(space: SearchSpace) -> Suggester:
    """A :class:`Suggester` that returns the middle of every range, for a smoke run.

    Exists so a caller — and the offline test tier — can exercise the whole objective with no
    optuna installed and no randomness at all. Every study result should be reproducible; this
    is the degenerate case of that.
    """

    class _Midpoint:
        def __init__(self) -> None:
            self._middles = space.middle()

        def suggest_float(
            self,
            name: str,
            low: float,
            high: float,
            *,
            step: float | None = None,
            log: bool = False,
        ) -> float:
            return float(self._middles[name])

        def suggest_int(
            self, name: str, low: int, high: int, *, step: int = 1, log: bool = False
        ) -> int:
            return int(self._middles[name])

        def suggest_categorical(self, name: str, choices: Sequence[Any]) -> Any:
            return self._middles[name]

    return _Midpoint()
