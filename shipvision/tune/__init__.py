"""Hyperparameter search over the trackers, with Optuna and against HOTA.

    from shipvision.eval import load_cases
    from shipvision.tune import run_study

    cases = load_cases("data/mot17/train", sequences=["MOT17-09-FRCNN", "MOT17-11-FRCNN"])
    result = run_study("bytetrack", cases, trials=50, seed=7, storage="sqlite:///tune.db")
    print(result.summary())   # baseline and best, side by side, with the verdict

Three decisions worth knowing before quoting a number this package produced.

**HOTA is the default objective, not MOTA.** MOTA is dominated by false negatives, which are
the detector's and which tuning a tracker cannot change; a study against it mostly resolves
its own sampling noise. See :mod:`shipvision.tune.objective`.

**Every search space is validated against the tracker's constructor when it is built.** A
misspelled hyperparameter tunes nothing while the study still reports an improvement, and the
improvement is entirely noise. See :mod:`shipvision.tune.spaces`.

**Optuna is optional, and its absence is a typed failure.** ``import shipvision.tune`` works
without it — the search spaces and the objective are pure numpy and are the parts worth unit
testing — but touching :func:`run_study` or :class:`StudyResult` raises
:class:`~shipvision.errors.BackendUnavailableError` with the install line, not an ImportError
from four frames down. The two are resolved through a module ``__getattr__`` so that
``from shipvision.tune import run_study`` is the thing that fails, which is the line a caller
actually wrote.
"""

from __future__ import annotations

from typing import Any

from shipvision.tune.objective import DIRECTIONS, Objective, direction_of, midpoint_suggester
from shipvision.tune.spaces import (
    APPEARANCE_PARAMETERS,
    SPACES,
    CategoricalChoice,
    FloatRange,
    IntRange,
    ParameterRange,
    SearchSpace,
    Suggester,
    accepted_parameters,
    all_spaces,
    space_for,
)

#: Names that live in :mod:`shipvision.tune.study` and therefore need optuna.
_NEEDS_OPTUNA = ("StudyResult", "run_study")

__all__ = [
    "APPEARANCE_PARAMETERS",
    "DIRECTIONS",
    "SPACES",
    "CategoricalChoice",
    "FloatRange",
    "IntRange",
    "Objective",
    "ParameterRange",
    "SearchSpace",
    "StudyResult",
    "Suggester",
    "accepted_parameters",
    "all_spaces",
    "direction_of",
    "midpoint_suggester",
    "run_study",
    "space_for",
]


def __getattr__(name: str) -> Any:
    """Resolve the two optuna-dependent names on first access.

    PEP 562. A non-``AttributeError`` raised here propagates unchanged, which is exactly the
    behaviour wanted: ``from shipvision.tune import run_study`` on a machine with no optuna
    raises :class:`~shipvision.errors.BackendUnavailableError` naming the install command.
    """
    if name not in _NEEDS_OPTUNA:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from shipvision.tune import study

    value = getattr(study, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_NEEDS_OPTUNA))
