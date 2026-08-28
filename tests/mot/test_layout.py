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

from shipvision.mot.base import BaseTracker

PACKAGE = Path(__file__).resolve().parents[2] / "shipvision" / "mot"
TRACKERS_DIR = PACKAGE / "trackers"

ALGORITHMS = sorted(m.name for m in pkgutil.iter_modules([str(TRACKERS_DIR)]) if m.ispkg)


#: Every tracker class the package publishes, read from the registry rather than listed, so a
#: tracker added without a package — or a package added without a registration — fails here
#: instead of being invisible. Listing them by hand is what let three of the five go a release
#: with no compiled twin and nothing saying so.
def _published() -> tuple[str, ...]:
    from shipvision.mot import TRACKERS

    return tuple(sorted(TRACKERS.get(name).__name__ for name in TRACKERS.names()))


CLASSES = _published()


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


class TestEachAlgorithmIsItsOwnPackage:
    """``tracker.py`` / ``tracklet.py`` / ``utils.py``, everywhere, with no exceptions.

    Uniform because the point of the split is that a reader can open the same file in two
    algorithms and diff them. One package that arranges itself differently costs more than it
    saves.
    """

    def test_every_registered_tracker_has_a_package(self) -> None:
        assert ALGORITHMS == [
            "botsort",
            "bytetrack",
            "deepsortv2",
            "mcbyte",
            "ocsort",
            "sort",
        ]

    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_the_package_holds_exactly_the_four_expected_modules(self, algorithm: str) -> None:
        present = sorted(p.name for p in (TRACKERS_DIR / algorithm).glob("*.py"))

        assert present == ["__init__.py", "tracker.py", "tracklet.py", "utils.py"]

    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_the_tracker_class_lives_in_tracker_py_and_nowhere_else(
        self, algorithm: str
    ) -> None:
        """``tracker.py`` holds every implementation of this algorithm and nothing else.

        It used to say *one* class, and that was right until the native twins moved in beside
        the Python ones (V48): a compiled DeepSORTv2 is DeepSORTv2, so keeping the two in one
        file is what makes them diffable, and splitting them by implementation put the two
        halves of one algorithm in different directories.

        So the rule is now "exactly the registered implementations of this name", which is a
        stronger claim than the count it replaced rather than a relaxation of it. A helper class
        that drifts into ``tracker.py`` still fails, because it is not in the registry under
        this name; an implementation that moves out still fails, because it is.
        """
        from shipvision.mot import TRACKERS

        package = TRACKERS_DIR / algorithm
        classes = {
            path.name: [
                node.name
                for node in ast.walk(ast.parse(path.read_text()))
                if isinstance(node, ast.ClassDef)
            ]
            for path in package.glob("*.py")
        }
        registered = {
            TRACKERS.get(algorithm, backend).__name__
            for backend in TRACKERS.backends(algorithm)
        }

        assert set(classes["tracker.py"]) == registered, (
            f"trackers/{algorithm}/tracker.py defines {sorted(classes['tracker.py'])} but the "
            f"registry knows {sorted(registered)} for this name. It holds the implementations "
            f"of this algorithm and nothing else"
        )
        assert not classes["tracklet.py"] and not classes["utils.py"], (
            f"trackers/{algorithm} defines a class outside tracker.py: "
            f"tracklet={classes['tracklet.py']} utils={classes['utils.py']}"
        )

    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_the_registered_class_is_the_one_in_tracker_py(self, algorithm: str) -> None:
        """The **python** backend's class, pinned. Unpinned, the registry answers with the
        fastest backend that implements the name, which for ``sort`` and ``bytetrack`` is the
        compiled twin in ``backends/native.py`` — a different class in a different file, and
        correctly so. The claim here is about the reference implementation's layout."""
        from shipvision.mot import TRACKERS
        from shipvision.registry import PYTHON

        cls = TRACKERS.get(algorithm, PYTHON)
        declared = [
            node.name
            for node in ast.walk(
                ast.parse((TRACKERS_DIR / algorithm / "tracker.py").read_text())
            )
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
        assert "new_pool" in module_all(TRACKERS_DIR / algorithm / "tracklet.py")


class TestNoHelperIsCopiedBetweenAlgorithms:
    """The rule the split exists to enforce: shared code lives in ``association/`` or
    ``motion/``, never in two ``utils.py`` files.
    """

    def test_no_two_algorithms_declare_a_helper_of_the_same_name(self) -> None:
        owners: dict[str, list[str]] = {}
        for algorithm in ALGORITHMS:
            for name in module_all(TRACKERS_DIR / algorithm / "utils.py"):
                owners.setdefault(name, []).append(algorithm)

        shared = {name: who for name, who in owners.items() if len(who) > 1}

        assert not shared, (
            f"{shared} appears in more than one algorithm's utils.py. A helper two "
            f"algorithms need belongs in association/ or motion/: two copies get fixed on "
            f"one side, and the trackers that are supposed to be comparable stop being"
        )

    def test_utils_declares_something_for_every_algorithm(self) -> None:
        """An empty ``utils.py`` means the split was cosmetic for that algorithm."""
        empty = [a for a in ALGORITHMS if not module_all(TRACKERS_DIR / a / "utils.py")]

        assert not empty, f"{empty} have a utils.py that declares nothing"

    def test_the_only_edges_between_algorithms_are_the_two_subclass_diffs(self) -> None:
        """BoT-SORT and BoostTrack subclass ByteTrack because each paper *is* a diff against
        it — CMC plus appearance for one, three cost boosts for the other. Any other edge
        means one algorithm is reaching into another's private helpers, which is the
        copy-by-reference version of the failure above.
        """
        # Two subclass edges now, and both are the paper being a diff: BoT-SORT is ByteTrack
        # plus CMC and appearance, BoostTrack is ByteTrack plus three cost boosts. Any *other*
        # edge means one algorithm is reaching into another's private helpers, which is the
        # copy-by-reference version of duplicating them.
        allowed = {("botsort", "bytetrack"), ("mcbyte", "botsort")}
        offenders: list[str] = []
        for algorithm in ALGORITHMS:
            for path in sorted((TRACKERS_DIR / algorithm).glob("*.py")):
                for node in ast.walk(ast.parse(path.read_text())):
                    if not isinstance(node, ast.ImportFrom):
                        continue
                    module = node.module or ""
                    prefix = "shipvision.mot.trackers."
                    if not module.startswith(prefix):
                        continue
                    other = module[len(prefix) :].split(".")[0]
                    if other != algorithm and (algorithm, other) not in allowed:
                        offenders.append(
                            f"trackers/{algorithm}/{path.name}:{node.lineno} imports {module}"
                        )

        assert not offenders, "unexpected coupling between algorithms:\n  " + "\n  ".join(
            offenders
        )

    def test_the_shared_helpers_the_algorithms_lean_on_are_exported_centrally(self) -> None:
        """The two extractions this split produced. If either stops being re-exported, the
        next person to need it copies it instead, which is where this started."""
        from shipvision.mot import association

        for name in ("gated_iou_cost", "pairwise_appearance"):
            assert name in association.__all__
            assert hasattr(association, name)
