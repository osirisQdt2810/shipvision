"""One search space per registered tracker, declared as data and validated against the code.

A search space is a table: a parameter name, a kind, and a range. Declaring it as data rather
than as a function full of ``trial.suggest_float`` calls is what makes it inspectable — a
reviewer can read the whole space of one tracker in ten lines, and a test can assert a
property of every space in the registry without running a study.

**Every name is checked against the tracker's constructor at construction time.** This is the
single most important thing in the module. A misspelled hyperparameter is accepted silently by
a ``**kwargs`` constructor, or raises deep inside trial 1 of 200 — and in the first case the
study runs to completion, reports an improvement, and has tuned nothing. The improvement is
real noise: the same fixed configuration evaluated two hundred times over a deterministic
benchmark gives one number, so any spread a study reports for a parameter the tracker ignores
is a bug in the study rather than a property of the tracker. So the space refuses to exist:

    >>> SearchSpace("sort", (FloatRange("iou_treshold", 0.1, 0.5),))
    Traceback (most recent call last):
    shipvision.errors.ConfigurationError: sort does not accept 'iou_treshold' ...

**Names that the tracker accepts but cannot act on are excluded on purpose**, and the exclusion
is documented next to each space. BoT-SORT's ``appearance_weight`` is a real parameter that does
nothing at all on a box-only benchmark, because there are no embeddings to weigh — tuning it
is the same failure as a typo, with a correctly spelled name. :data:`APPEARANCE_PARAMETERS`
holds them for a caller who has a re-ID extractor in the pipeline and can therefore measure
them.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from shipvision.errors import ConfigurationError
from shipvision.mot import TRACKERS

__all__ = [
    "APPEARANCE_PARAMETERS",
    "SPACES",
    "CategoricalChoice",
    "FloatRange",
    "IntRange",
    "ParameterRange",
    "SearchSpace",
    "Suggester",
    "accepted_parameters",
    "all_spaces",
    "space_for",
]


@runtime_checkable
class Suggester(Protocol):
    """What a search space needs from a trial. Optuna's ``Trial`` satisfies it structurally.

    A protocol rather than an import because it is what lets the whole of this module and
    :mod:`shipvision.tune.objective` be tested with no optuna installed — a fake suggester that
    returns the midpoint of every range is four lines and makes the objective deterministic.
    """

    def suggest_float(
        self, name: str, low: float, high: float, *, step: float | None = ..., log: bool = ...
    ) -> float: ...

    def suggest_int(
        self, name: str, low: int, high: int, *, step: int = ..., log: bool = ...
    ) -> int: ...

    def suggest_categorical(self, name: str, choices: Sequence[Any]) -> Any: ...


# ------------------------------------------------------------------------ parameters


@dataclass(frozen=True, slots=True)
class ParameterRange:
    """Base of the three kinds. Carries the name and nothing else."""

    name: str

    def suggest(self, trial: Suggester) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def middle(self) -> Any:  # pragma: no cover - abstract
        """A representative value, for a deterministic smoke run with no sampler."""
        raise NotImplementedError

    def describe(self) -> str:  # pragma: no cover - abstract
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class FloatRange(ParameterRange):
    """A continuous parameter. ``log=True`` for one whose useful range spans decades."""

    low: float = 0.0
    high: float = 1.0
    log: bool = False
    step: float | None = None

    def __post_init__(self) -> None:
        if not self.low < self.high:
            raise ConfigurationError(
                f"{self.name}: low ({self.low}) must be below high ({self.high}); a collapsed "
                f"range is a constant wearing a parameter's name"
            )
        if self.log and self.low <= 0.0:
            raise ConfigurationError(
                f"{self.name}: a log range needs a positive low, got {self.low}"
            )

    def suggest(self, trial: Suggester) -> float:
        return float(
            trial.suggest_float(self.name, self.low, self.high, step=self.step, log=self.log)
        )

    def middle(self) -> float:
        return float(self.low + (self.high - self.low) / 2)

    def describe(self) -> str:
        return f"float[{self.low:g}, {self.high:g}]" + (" log" if self.log else "")


@dataclass(frozen=True, slots=True)
class IntRange(ParameterRange):
    """A discrete parameter. Inclusive at both ends, as Optuna's ``suggest_int`` is."""

    low: int = 0
    high: int = 1
    step: int = 1

    def __post_init__(self) -> None:
        if not self.low < self.high:
            raise ConfigurationError(
                f"{self.name}: low ({self.low}) must be below high ({self.high})"
            )
        if self.step < 1:
            raise ConfigurationError(f"{self.name}: step must be positive, got {self.step}")

    def suggest(self, trial: Suggester) -> int:
        return int(trial.suggest_int(self.name, self.low, self.high, step=self.step))

    def middle(self) -> int:
        return int(self.low + (self.high - self.low) // 2)

    def describe(self) -> str:
        return f"int[{self.low}, {self.high}]" + (
            f" step {self.step}" if self.step != 1 else ""
        )


@dataclass(frozen=True, slots=True)
class CategoricalChoice(ParameterRange):
    """A choice from a fixed set — a flag, or a name from another registry."""

    choices: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if len(self.choices) < 2:
            raise ConfigurationError(
                f"{self.name}: a categorical needs at least two choices, got {self.choices}. "
                f"One choice is a constant, and a study that samples it reports its own noise"
            )

    def suggest(self, trial: Suggester) -> Any:
        return trial.suggest_categorical(self.name, list(self.choices))

    def middle(self) -> Any:
        return self.choices[0]

    def describe(self) -> str:
        return f"choice{list(self.choices)}"


# ----------------------------------------------------------------------- validation


def accepted_parameters(cls: type) -> frozenset[str]:
    """Every keyword ``cls(...)`` accepts, following ``**kwargs`` up the MRO.

    The walk is what makes the validation useful rather than decorative. BoT-SORT's
    constructor is ``(*, cmc, cmc_options, appearance_gate, appearance_weight, **byte)`` and
    forwards ``byte`` to ByteTrack — so a check that only read ``BotSortTracker.__init__``
    would reject ``track_threshold``, which is real, and a check that saw the ``**kwargs`` and
    gave up would accept anything at all. Both failures are worse than no check: the first
    makes the correct space unwritable, the second makes the typo silent again.

    The walk stops at the first ``__init__`` with no ``**kwargs``, because that is where the
    forwarding chain ends.
    """
    names: set[str] = set()
    for base in inspect.getmro(cls):
        initialiser = base.__dict__.get("__init__")
        if initialiser is None:
            continue
        signature = inspect.signature(initialiser)
        forwards = False
        for parameter in signature.parameters.values():
            if parameter.name == "self":
                continue
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                forwards = True
            elif parameter.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                names.add(parameter.name)
        if not forwards:
            break
    return frozenset(names)


# ---------------------------------------------------------------------- the space


@dataclass(frozen=True, slots=True)
class SearchSpace:
    """What a study is allowed to vary, for one registered tracker.

    Attributes:
        tracker: the name in :data:`~shipvision.tracking.TRACKERS`. Resolved at construction,
            so an unregistered tracker fails here rather than in the first trial.
        parameters: the ranges to sample.
        constants: keywords pinned for every trial — used to hold a parameter still while
            another is tuned, which is the only way to attribute an improvement to one of them.
        backend: which implementation of the tracker to build.
    """

    tracker: str
    parameters: tuple[ParameterRange, ...] = ()
    constants: Mapping[str, Any] = field(default_factory=dict)
    backend: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", tuple(self.parameters))
        object.__setattr__(self, "constants", dict(self.constants))
        if not self.parameters:
            raise ConfigurationError(
                f"the search space for {self.tracker!r} is empty; a study over no parameters "
                f"evaluates one configuration many times and reports the spread as progress"
            )

        accepted = accepted_parameters(TRACKERS.get(self.tracker, self.backend))
        names = [p.name for p in self.parameters]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ConfigurationError(
                f"{self.tracker}: {duplicates} appear twice in the space; the second range "
                f"would silently win and the first would be dead documentation"
            )
        unknown = sorted((set(names) | set(self.constants)) - accepted)
        if unknown:
            raise ConfigurationError(
                f"{self.tracker} does not accept {unknown}; it accepts {sorted(accepted)}. A "
                f"name the constructor ignores tunes nothing while the study still reports an "
                f"improvement, which is the study measuring its own sampling noise"
            )

    def __len__(self) -> int:
        return len(self.parameters)

    def __iter__(self) -> Iterable[ParameterRange]:
        return iter(self.parameters)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.parameters)

    def suggest(self, trial: Suggester) -> dict[str, Any]:
        """Sample every parameter, then add the constants. Constants win on a clash."""
        sampled = {p.name: p.suggest(trial) for p in self.parameters}
        sampled.update(self.constants)
        return sampled

    def middle(self) -> dict[str, Any]:
        """The midpoint of every range. For a smoke run with no sampler at all."""
        return {**{p.name: p.middle() for p in self.parameters}, **dict(self.constants)}

    def defaults(self) -> dict[str, Any]:
        """The constants alone, so the tracker's own defaults stand in for everything else.

        This is what a baseline is built from, and it must not be the midpoint of the ranges:
        the question a study answers is "did tuning beat the shipped configuration", and a
        baseline of range midpoints answers a different and less interesting one.
        """
        return dict(self.constants)

    def with_parameters(self, *extra: ParameterRange) -> SearchSpace:
        """A copy with more parameters — for a pipeline where the extra ones can act.

        See :data:`APPEARANCE_PARAMETERS`.
        """
        return SearchSpace(
            tracker=self.tracker,
            parameters=(*self.parameters, *extra),
            constants=self.constants,
            backend=self.backend,
        )

    def describe(self) -> str:
        lines = [f"{self.tracker}: {len(self.parameters)} parameter(s)"]
        lines.extend(f"  {p.name:<24} {p.describe()}" for p in self.parameters)
        if self.constants:
            lines.append(f"  fixed: {dict(self.constants)}")
        return "\n".join(lines)


#: Appearance parameters, held out of the default spaces below.
#:
#: They are real constructor arguments and they do nothing when the detections carry no
#: embedding: every cost function that uses them checks first and falls back to geometry. On a
#: public-detection MOT17 run — the benchmark this package is built around — that is always the
#: case, so a study that sampled them would report the spread of its own sampler as an
#: improvement. Add them with :meth:`SearchSpace.with_parameters` when a re-ID extractor is in
#: the pipeline and they can be measured.
APPEARANCE_PARAMETERS: dict[str, tuple[ParameterRange, ...]] = {
    "botsort": (
        FloatRange("appearance_gate", 0.1, 0.6),
        FloatRange("appearance_weight", 0.2, 1.0),
        FloatRange("embedding_momentum", 0.7, 0.99),
    ),
    "deepsortv2": (
        FloatRange("appearance_gate", 0.05, 0.4),
        FloatRange("appearance_weight", 0.3, 1.0),
    ),
}


def _sort() -> SearchSpace:
    """SORT: the baseline. Four knobs, and every one of them is geometry or lifecycle.

    ``det_threshold`` matters more here than anywhere else because SORT has no second
    association pass to rescue a low-scoring box, so the threshold is the whole of its
    detection policy.
    """
    return SearchSpace(
        "sort",
        (
            FloatRange("det_threshold", 0.05, 0.7),
            FloatRange("iou_threshold", 0.1, 0.5),
            IntRange("max_age", 5, 60),
            IntRange("min_hits", 1, 5),
        ),
    )


def _bytetrack() -> SearchSpace:
    """ByteTrack. The two threshold ranges are deliberately disjoint.

    The constructor requires ``low_threshold < track_threshold``, and a space that can sample
    a violation turns a study into a lottery over which trials survive. Keeping the ranges
    apart — low in [0.02, 0.2], track in [0.3, 0.8] — makes every sample valid by construction
    rather than by rejection, which is worth more than the sliver of space it gives up.
    """
    return SearchSpace(
        "bytetrack",
        (
            FloatRange("track_threshold", 0.3, 0.8),
            FloatRange("low_threshold", 0.02, 0.2),
            FloatRange("match_threshold", 0.1, 0.5),
            FloatRange("second_match_threshold", 0.2, 0.7),
            IntRange("max_age", 5, 60),
            IntRange("min_hits", 1, 5),
        ),
    )


def _ocsort() -> SearchSpace:
    """OC-SORT. ``momentum_weight`` may go to zero, which is the point.

    Zero disables the heading-consistency term entirely, so the study can answer "does
    observation-centric momentum help on this footage" rather than only "how much of it".
    A range that started at 0.05 could not express the negative answer.
    """
    return SearchSpace(
        "ocsort",
        (
            FloatRange("det_threshold", 0.05, 0.7),
            FloatRange("iou_threshold", 0.1, 0.5),
            FloatRange("recovery_iou_threshold", 0.1, 0.6),
            IntRange("delta_t", 1, 6),
            FloatRange("momentum_weight", 0.0, 0.5),
            IntRange("max_age", 5, 60),
            IntRange("min_hits", 1, 5),
            CategoricalChoice("recover", (True, False)),
        ),
    )


def _botsort() -> SearchSpace:
    """BoT-SORT, minus the two things it cannot demonstrate on a box-only benchmark.

    ``cmc`` is excluded because camera-motion compensation needs pixels and an evaluation over
    a ground-truth file has none — every estimator would return the identity transform and the
    choice would be free. The appearance parameters are excluded for the same reason, and both
    exclusions are why a report that compares BoT-SORT with ByteTrack on MOT17 public
    detections is comparing ByteTrack with itself plus overhead. That has to be said out loud
    rather than left for the reader to infer from a table where they score the same.
    """
    return SearchSpace(
        "botsort",
        (
            FloatRange("track_threshold", 0.3, 0.8),
            FloatRange("low_threshold", 0.02, 0.2),
            FloatRange("match_threshold", 0.1, 0.5),
            FloatRange("second_match_threshold", 0.2, 0.7),
            IntRange("max_age", 5, 60),
            IntRange("min_hits", 1, 5),
        ),
    )


def _deepsortv2() -> SearchSpace:
    """DeepSORT v2's four-stage cascade, on the stage costs that geometry can move.

    The four ``stage_*_max_cost`` gates are the interesting part and they are ordered by
    intent — A strictest, D loosest — but the space does not enforce the ordering. A study that
    finds the ordering inverted has found something worth reading, and a constraint that made
    it unrepresentable would have hidden it.
    """
    return SearchSpace(
        "deepsortv2",
        (
            FloatRange("det_threshold", 0.05, 0.7),
            FloatRange("stage_a_max_cost", 0.2, 0.8),
            FloatRange("stage_b_max_cost", 0.2, 0.8),
            FloatRange("stage_c_max_cost", 0.3, 0.9),
            FloatRange("stage_d_max_cost", 0.4, 1.0),
            FloatRange("giou_gate", 0.6, 1.8),
            IntRange("cascade_stride", 1, 10),
            IntRange("stage_b_max_age", 2, 20),
            IntRange("max_age", 5, 60),
            IntRange("min_hits", 1, 5),
        ),
    )


#: The default space for every registered tracker, keyed by its registry name.
#:
#: Built lazily on first access rather than at import: constructing a space resolves the
#: tracker class out of the registry to validate the names against it, and importing
#: ``shipvision.tune`` should not drag in every tracker module for a caller who only wants to
#: read a search space's shape.
_BUILDERS = {
    "sort": _sort,
    "bytetrack": _bytetrack,
    "ocsort": _ocsort,
    "botsort": _botsort,
    "deepsortv2": _deepsortv2,
}

SPACES: dict[str, SearchSpace] = {}


def space_for(tracker: str) -> SearchSpace:
    """The default search space for ``tracker``.

    Raises:
        ConfigurationError: no space is declared for that name. Deliberately not a silent
            fallback to an empty or generic space: a study over the wrong parameters is worse
            than no study, because it produces a number.
    """
    if tracker in SPACES:
        return SPACES[tracker]
    builder = _BUILDERS.get(tracker)
    if builder is None:
        raise ConfigurationError(
            f"no search space declared for tracker {tracker!r}; declared: "
            f"{sorted(_BUILDERS)}. Add one in shipvision/tune/spaces.py rather than tuning "
            f"a guess"
        )
    SPACES[tracker] = builder()
    return SPACES[tracker]


def all_spaces() -> dict[str, SearchSpace]:
    """Every declared space, built. For a test that asserts a property of all of them."""
    return {name: space_for(name) for name in _BUILDERS}
