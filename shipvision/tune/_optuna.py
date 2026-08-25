"""The one place optuna is imported, so its absence is a typed failure.

``pip install shipvision[tune]`` brings it; a machine without it must not see an ImportError
raised from four frames down inside a study. A caller cannot tell "optuna is not installed"
from "optuna is installed and broken" from "the tracker refused this configuration" if all
three arrive as different exception types from different modules, and those are three different
operational events.
"""

from __future__ import annotations

from types import ModuleType

from shipvision.errors import BackendUnavailableError

__all__ = ["require_optuna"]

_INSTALL = "pip install 'shipvision[tune]'"


def require_optuna() -> ModuleType:
    """Import optuna, or raise :class:`~shipvision.errors.BackendUnavailableError`.

    Called at the top of :mod:`shipvision.tune.study`'s module body rather than inside each
    function, so ``from shipvision.tune import run_study`` fails immediately and with the
    install line rather than at the end of a long data load.
    """
    try:
        import optuna
    except ImportError as error:  # pragma: no cover - exercised by a patched importer
        raise BackendUnavailableError(
            f"optuna is not installed, so no study can be run. {_INSTALL}. The search spaces "
            f"in shipvision.tune.spaces and the objective in shipvision.tune.objective do not "
            f"need it and remain importable"
        ) from error
    return optuna
