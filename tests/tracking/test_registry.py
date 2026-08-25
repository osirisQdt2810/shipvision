"""The ``TRACKERS`` registry: the directory *is* the list of trackers.

The generic ``Registry`` mechanics — aliases, backend resolution, lazy targets, the error
messages — are tested once in ``tests/test_registry.py`` and are not repeated here. What this
file asserts is the thing the repackaging is for: that adding an algorithm is a new package
under ``core/`` plus a decorator, and that nothing anywhere keeps a second, hand-maintained
list of what exists. A second list is the failure being prevented, and it is a quiet one — it
does not break, it just goes stale, and the tracker somebody added last week is missing from
``TRACKERS.names()`` with no error to point at.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

from shipvision.errors import ConfigurationError
from shipvision.registry import PYTHON, Registry
from shipvision.tracking import TRACKERS
from shipvision.tracking.base import BaseTracker

CORE = Path(__file__).resolve().parents[2] / "shipvision" / "tracking" / "core"

#: Name to the aliases it must answer to. Written out rather than read from the registry,
#: because a test that asks the registry what it contains and then checks it contains that
#: cannot fail. These strings appear in deployed config files; renaming one is a breaking
#: change and should have to be made here too.
PUBLISHED = {
    "sort": (),
    "bytetrack": ("byte",),
    "ocsort": ("oc", "oc_sort"),
    "botsort": ("bot", "bot_sort"),
    "deepsortv2": ("deepsort2", "dsv2"),
}


def core_packages() -> list[str]:
    return sorted(m.name for m in pkgutil.iter_modules([str(CORE)]) if m.ispkg)


class TestTheRegistryAndTheDirectoryAgree:
    """``core/`` and ``TRACKERS.names()`` are two views of one fact, so they cannot disagree.

    If they ever can, one of them is a hand-maintained list — which is exactly the ``if/elif``
    the registry replaced, wearing a different hat.
    """

    def test_every_core_package_is_registered(self) -> None:
        missing = [name for name in core_packages() if name not in TRACKERS.names()]
        assert not missing, (
            f"{missing} is a package under core/ but is not in TRACKERS. Its __init__.py "
            f"must import its tracker module, and core/__init__.py must import the package"
        )

    def test_every_registered_tracker_has_a_core_package(self) -> None:
        packages = core_packages()
        stray = [name for name in TRACKERS.names() if name not in packages]
        assert not stray, (
            f"{stray} is registered but has no package under core/. A tracker defined "
            f"outside core/ is one nobody will find when they go looking for it"
        )

    def test_the_published_names_are_exactly_what_ships(self) -> None:
        assert sorted(TRACKERS.names()) == sorted(PUBLISHED)

    @pytest.mark.parametrize("name", sorted(PUBLISHED))
    def test_a_tracker_resolves_to_a_class_in_its_own_package(self, name: str) -> None:
        cls = TRACKERS.get(name, PYTHON)

        assert issubclass(cls, BaseTracker)
        assert cls.__module__ == f"shipvision.tracking.core.{name}.tracker", (
            f"{cls.__name__} lives in {cls.__module__}; tracker.py holds the tracker class "
            f"and each algorithm's class lives in its own package"
        )

    @pytest.mark.parametrize(("name", "aliases"), sorted(PUBLISHED.items()))
    def test_every_documented_alias_still_answers(self, name: str, aliases: tuple) -> None:
        for alias in aliases:
            assert TRACKERS.get(alias) is TRACKERS.get(name), (
                f"alias {alias!r} stopped resolving to {name!r}; these strings are in "
                f"deployed config files"
            )

    def test_the_class_carries_the_name_it_was_registered_under(self) -> None:
        """The decorator stamps it, so a log line can say what a tracker is without the
        caller having to remember what it asked for."""
        for name in PUBLISHED:
            built = TRACKERS.build(name, min_hits=1)

            assert built.name == name
            assert built.backend == PYTHON


class TestAddingATrackerIsAPackagePlusADecorator:
    """The mechanism, demonstrated on a throwaway registry.

    Deliberately not on the real ``TRACKERS``: registering into a process-global registry from
    a test leaks into every test that runs after it, and ``tests/tracking/test_contract.py``
    parametrises over ``TRACKERS.names()``. A fake registered there would be held to the whole
    tracker contract by a test that never mentioned it.
    """

    def test_a_decorator_is_the_whole_of_registration(self) -> None:
        registry: Registry[BaseTracker] = Registry("tracker")

        @registry.register("harbourtrack", backend=PYTHON, aliases=("harbour",))
        class HarbourTracker(BaseTracker):
            def update(self, detections, *, image=None):
                return []

        assert registry.names() == ["harbourtrack"]
        assert registry.get("harbour") is HarbourTracker
        assert isinstance(registry.build("harbourtrack", pool=None), HarbourTracker)

    def test_two_trackers_cannot_claim_one_name(self) -> None:
        """The registry refuses rather than letting the second import win silently, because
        which one wins would then depend on import order."""
        registry: Registry[BaseTracker] = Registry("tracker")
        registry.register("harbourtrack")(type("A", (BaseTracker,), {"update": lambda *a: []}))

        with pytest.raises(ConfigurationError, match="already registered"):
            registry.register("harbourtrack")(
                type("B", (BaseTracker,), {"update": lambda *a: []})
            )


class TestTheRegistryModuleStaysALeaf:
    """``registry.py`` exists so the decorator is reachable without the contract behind it.

    That is only true while the module has no runtime import from ``shipvision.tracking``, so
    the property is checked rather than trusted.
    """

    def test_it_has_no_runtime_import_from_the_tracking_package(self) -> None:
        path = CORE.parent / "registry.py"
        tree = ast.parse(path.read_text())
        guarded = {
            node
            for parent in ast.walk(tree)
            if isinstance(parent, ast.If)
            for node in ast.walk(parent)
            if isinstance(node, ast.ImportFrom)
        }

        offenders = [
            f"line {node.lineno}: {node.module}"
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and (node.module or "").startswith("shipvision.tracking")
            and node not in guarded
        ]

        assert not offenders, (
            "shipvision/tracking/registry.py imports from its own package at runtime; it is "
            "the one module every tracker imports, so it must stay importable in any "
            "order:\n  " + "\n  ".join(offenders)
        )

    def test_one_registry_object_is_shared_by_every_path_that_exposes_it(self) -> None:
        """Two registry objects would mean ``build`` succeeding or failing depending on which
        module the caller imported — the least debuggable failure in the package."""
        import shipvision
        import shipvision.tracking.base as base
        import shipvision.tracking.registry as leaf

        assert TRACKERS is leaf.TRACKERS
        assert TRACKERS is base.TRACKERS
        assert TRACKERS is shipvision.TRACKERS


class TestRegistrationHappensOnImport:
    """Every path into the package leaves the registry fully populated.

    Note what is *not* claimed: importing one algorithm's package does **not** register only
    that algorithm. ``import shipvision.tracking.core.sort`` runs the parent ``__init__`` files
    first, and ``tracking/__init__.py`` imports ``core``, which imports all five. The split
    into packages buys readability and a clean place to add the sixth; it does not buy import
    isolation, and a test asserting otherwise would be asserting a wish.

    Subprocesses, because the interesting claim is about a *fresh* interpreter: in this one
    everything is imported already and every assertion would pass for the wrong reason.
    """

    @staticmethod
    def names_after(statement: str) -> list[str]:
        code = (
            f"{statement}\n"
            "from shipvision.tracking.registry import TRACKERS\n"
            "print(TRACKERS.names())"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        return ast.literal_eval(result.stdout.strip())

    def test_importing_the_family_registers_all_five(self) -> None:
        assert self.names_after("import shipvision.tracking") == sorted(PUBLISHED)

    def test_importing_a_single_algorithm_package_is_enough(self) -> None:
        """Enough, and in fact more than enough — see the class docstring. The claim worth
        having is that ``TRACKERS.build`` works afterwards, not that the registry is small."""
        assert "sort" in self.names_after("import shipvision.tracking.core.sort")

    def test_importing_the_compatibility_shim_still_registers_them(self) -> None:
        """The reason this one matters: a caller who has always written
        ``import shipvision.tracking.trackers`` to make ``TRACKERS.build("sort")`` work must
        not find an empty registry after a rename they did not make."""
        assert self.names_after("import shipvision.tracking.trackers") == sorted(PUBLISHED)

    def test_reaching_only_the_leaf_registry_module_still_registers_them(self) -> None:
        """``registry.py`` holds no tracker, so on its own it would hand back an empty
        registry; it is the parent package's eager import of ``core`` that fills it. Asserted
        because making ``tracking/__init__.py`` lazy would break ``TRACKERS.build`` for
        exactly this caller and for nobody else, which is a hard bug to find."""
        assert self.names_after("import shipvision.tracking.registry") == sorted(PUBLISHED)


class TestNoTrackerIsSelectedByABranch:
    """The registry is worth nothing if a caller still picks the algorithm with an ``if``."""

    def test_no_module_in_the_package_compares_against_a_tracker_name(self) -> None:
        package = CORE.parent
        offenders: list[str] = []
        for path in sorted(package.rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node, ast.Compare):
                    continue
                literals = {
                    operand.value
                    for operand in (node.left, *node.comparators)
                    if isinstance(operand, ast.Constant) and isinstance(operand.value, str)
                }
                named = literals & set(PUBLISHED)
                if named:
                    offenders.append(
                        f"{path.relative_to(package)}:{node.lineno} compares against {named}"
                    )

        assert not offenders, (
            "a tracker is being selected or special-cased by name; that is the if/elif the "
            "registry replaced:\n  " + "\n  ".join(offenders)
        )


class TestTheDocumentedEntryPointStillWorks:
    """The three-line example in CLAUDE.md and the README is the API most people meet first."""

    def test_build_by_name_returns_a_working_tracker(self) -> None:
        module = importlib.import_module("shipvision.tracking")

        tracker = module.TRACKERS.build("bytetrack", track_threshold=0.5)

        assert isinstance(tracker, BaseTracker)
        assert tracker.pool_size == 0

    def test_pinning_the_backend_is_accepted_even_though_there_is_one(self) -> None:
        """``backend="python"`` has to keep working now, or the day a native tracker lands
        every config that pinned the reference implementation breaks at once."""
        assert TRACKERS.build("sort", backend=PYTHON).backend == PYTHON
