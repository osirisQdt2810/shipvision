"""The package shape, and the compatibility shims that hide it from existing callers.

Two claims, and they pull in opposite directions on purpose.

The **shims** say the move was invisible: every import path that worked before ``core/``
existed still resolves, to the same objects. Those paths are in a shipped library, a README
and a deployed service's imports, and a repackaging that breaks one of them has not saved
anybody anything.

The **shape** says the move was worth making: one package per algorithm, a tracker class in
``tracker.py`` and nowhere else, and no helper copied into two algorithms' ``utils.py``. That
last one is the failure this restructuring exists to prevent, and it is silent — two copies of
a cost function do not break, they get fixed on one side, and two trackers that are supposed to
be comparable quietly stop being.
"""

from __future__ import annotations

import ast
import pkgutil
from pathlib import Path

import pytest

from shipvision.tracking.base import BaseTracker

PACKAGE = Path(__file__).resolve().parents[2] / "shipvision" / "tracking"
CORE = PACKAGE / "core"

ALGORITHMS = sorted(m.name for m in pkgutil.iter_modules([str(CORE)]) if m.ispkg)

#: The five class names, as every existing caller spells them.
CLASSES = (
    "BotSortTracker",
    "ByteTrackTracker",
    "DeepSortV2Tracker",
    "OcSortTracker",
    "SortTracker",
)


def module_all(path: Path) -> set[str]:
    """``__all__`` as a set of strings, read from source rather than by importing.

    Reading the source keeps the check honest about *which file declares a name*: importing
    would resolve a re-export to wherever it originally came from, which is precisely the
    distinction the duplication test below is trying to make.
    """
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
        ):
            return {
                element.value
                for element in node.value.elts  # type: ignore[attr-defined]
                if isinstance(element, ast.Constant)
            }
    return set()


class TestTheOldImportPathsStillResolve:
    """Nothing a caller wrote before ``core/`` existed may have to change."""

    @pytest.mark.parametrize("name", CLASSES)
    def test_the_flat_package_surface_still_exports_every_tracker(self, name: str) -> None:
        import shipvision.tracking as tracking

        assert hasattr(tracking, name)
        assert name in tracking.__all__

    @pytest.mark.parametrize("name", CLASSES)
    def test_the_trackers_shim_still_exports_every_tracker(self, name: str) -> None:
        import shipvision.tracking.trackers as shim

        assert hasattr(shim, name)
        assert name in shim.__all__

    @pytest.mark.parametrize("name", CLASSES)
    def test_all_three_paths_hand_back_the_same_class(self, name: str) -> None:
        """The same object, not merely an equal one. Two classes with one name is how
        ``isinstance`` starts returning False for an object that plainly is one."""
        import shipvision.tracking as tracking
        import shipvision.tracking.core as core
        import shipvision.tracking.trackers as shim

        assert getattr(tracking, name) is getattr(core, name)
        assert getattr(shim, name) is getattr(core, name)

    def test_the_contract_is_still_importable_from_base(self) -> None:
        """``shipvision/eval/runner.py`` types its argument on this import."""
        from shipvision.tracking.base import TRACKERS, BaseTracker, next_track_id

        assert issubclass(TRACKERS.get("sort"), BaseTracker)
        assert next_track_id() > 0

    def test_everything_the_package_advertises_actually_resolves(self) -> None:
        """A stale ``__all__`` entry is an ``ImportError`` for a caller who read the docs."""
        import shipvision.tracking as tracking

        missing = [name for name in tracking.__all__ if not hasattr(tracking, name)]

        assert not missing, f"shipvision.tracking.__all__ advertises {missing}"

    def test_the_shim_says_it_is_one(self) -> None:
        """A forwarding module with no note saying so is a module somebody adds a tracker to."""
        source = (PACKAGE / "trackers" / "__init__.py").read_text()

        assert "shim" in source.lower()
        assert "shipvision.tracking.core" in source


class TestEachAlgorithmIsAPackageOfThreeFiles:
    """``tracker.py`` / ``tracklet.py`` / ``utils.py``, everywhere, with no exceptions.

    Uniform because the point of the split is that a reader can open the same file in two
    algorithms and diff them. One package that arranges itself differently costs more than it
    saves.
    """

    def test_there_are_five_algorithms(self) -> None:
        assert ALGORITHMS == ["botsort", "bytetrack", "deepsortv2", "ocsort", "sort"]

    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_the_package_holds_exactly_the_four_expected_modules(self, algorithm: str) -> None:
        present = sorted(p.name for p in (CORE / algorithm).glob("*.py"))

        assert present == ["__init__.py", "tracker.py", "tracklet.py", "utils.py"]

    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_the_tracker_class_lives_in_tracker_py_and_nowhere_else(
        self, algorithm: str
    ) -> None:
        package = CORE / algorithm
        classes = {
            path.name: [
                node.name
                for node in ast.walk(ast.parse(path.read_text()))
                if isinstance(node, ast.ClassDef)
            ]
            for path in package.glob("*.py")
        }

        assert len(classes["tracker.py"]) == 1, (
            f"core/{algorithm}/tracker.py defines {classes['tracker.py']}; it holds the "
            f"tracker class and only the tracker class"
        )
        assert not classes["tracklet.py"] and not classes["utils.py"], (
            f"core/{algorithm} defines a class outside tracker.py: "
            f"tracklet={classes['tracklet.py']} utils={classes['utils.py']}"
        )

    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_the_registered_class_is_the_one_in_tracker_py(self, algorithm: str) -> None:
        from shipvision.tracking import TRACKERS

        cls = TRACKERS.get(algorithm)
        declared = [
            node.name
            for node in ast.walk(ast.parse((CORE / algorithm / "tracker.py").read_text()))
            if isinstance(node, ast.ClassDef)
        ]

        assert cls.__name__ in declared
        assert issubclass(cls, BaseTracker)

    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_the_package_states_what_it_asks_of_the_shared_track_state(
        self, algorithm: str
    ) -> None:
        """``tracklet.py`` is where "which optional pool capabilities does this one need"
        lives. There is no per-algorithm tracklet *class* — the pool is struct-of-arrays for
        every track — so what has to be true is that the file exists and declares a factory,
        which is what makes the five readable as a diff."""
        assert "new_pool" in module_all(CORE / algorithm / "tracklet.py")


class TestNoHelperIsCopiedBetweenAlgorithms:
    """The rule the split exists to enforce: shared code lives in ``association/`` or
    ``motion/``, never in two ``utils.py`` files.
    """

    def test_no_two_algorithms_declare_a_helper_of_the_same_name(self) -> None:
        owners: dict[str, list[str]] = {}
        for algorithm in ALGORITHMS:
            for name in module_all(CORE / algorithm / "utils.py"):
                owners.setdefault(name, []).append(algorithm)

        shared = {name: who for name, who in owners.items() if len(who) > 1}

        assert not shared, (
            f"{shared} appears in more than one algorithm's utils.py. A helper two "
            f"algorithms need belongs in association/ or motion/: two copies get fixed on "
            f"one side, and the trackers that are supposed to be comparable stop being"
        )

    def test_utils_declares_something_for_every_algorithm(self) -> None:
        """An empty ``utils.py`` means the split was cosmetic for that algorithm."""
        empty = [a for a in ALGORITHMS if not module_all(CORE / a / "utils.py")]

        assert not empty, f"{empty} have a utils.py that declares nothing"

    def test_the_only_dependency_between_two_algorithms_is_botsort_on_bytetrack(self) -> None:
        """BoT-SORT subclasses ByteTrack because the paper is a two-line diff against it.
        Any *other* edge between two algorithms means one is reaching into another's private
        helpers, which is the copy-by-reference version of the failure above.
        """
        allowed = {("botsort", "bytetrack")}
        offenders: list[str] = []
        for algorithm in ALGORITHMS:
            for path in sorted((CORE / algorithm).glob("*.py")):
                for node in ast.walk(ast.parse(path.read_text())):
                    if not isinstance(node, ast.ImportFrom):
                        continue
                    module = node.module or ""
                    prefix = "shipvision.tracking.core."
                    if not module.startswith(prefix):
                        continue
                    other = module[len(prefix) :].split(".")[0]
                    if other != algorithm and (algorithm, other) not in allowed:
                        offenders.append(
                            f"core/{algorithm}/{path.name}:{node.lineno} imports {module}"
                        )

        assert not offenders, "unexpected coupling between algorithms:\n  " + "\n  ".join(
            offenders
        )

    def test_the_shared_helpers_the_algorithms_lean_on_are_exported_centrally(self) -> None:
        """The two extractions this split produced. If either stops being re-exported, the
        next person to need it copies it instead, which is where this started."""
        from shipvision.tracking import association

        for name in ("gated_iou_cost", "pairwise_appearance"):
            assert name in association.__all__
            assert hasattr(association, name)
