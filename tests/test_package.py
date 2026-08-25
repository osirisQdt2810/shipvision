"""The top-level surface: what `import shipvision` costs, and what it gives you.

Both halves are load-bearing. `from shipvision import TRACKERS` is the documented entry
point in CLAUDE.md and the README, so it has to work — and `import shipvision` has to stay
free of torch, scipy, cv2 and TensorRT, because the offline test tier and every evaluation
run on a laptop depend on the library importing where none of those exist.

Those two requirements pull against each other: reading a family's registry means importing
the family, and importing `shipvision.detection` runs the module body that registers the
TensorRT backend. PEP 562's module `__getattr__` is what reconciles them, and these tests
are what stop a later `from shipvision.detection import DETECTORS` at the top of
`__init__.py` from quietly undoing it.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

import shipvision

HEAVY = ("torch", "scipy", "cv2", "tensorrt")

#: Read from the package rather than duplicated here, so a family added to the lookup table
#: is tested automatically and a family declared before it exists fails immediately. A
#: hand-maintained copy would drift, and the drift would be silent in the direction that
#: matters: a registry nobody tests.
REGISTRIES = tuple(shipvision._REGISTRY_HOMES)


class TestImportCost:
    """`import shipvision` must not drag in an accelerator stack."""

    def test_importing_the_package_loads_nothing_heavy(self) -> None:
        """Run in a fresh interpreter: this test process has already imported plenty, so
        checking `sys.modules` in-process would prove nothing."""
        code = (
            "import sys, shipvision; "
            f"print(sorted(m for m in {HEAVY!r} if m in sys.modules))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )

        assert result.stdout.strip() == "[]", (
            f"import shipvision pulled in {result.stdout.strip()} — something in "
            f"__init__.py is importing a family eagerly"
        )

    def test_the_contract_types_are_available_without_touching_a_family(self) -> None:
        code = (
            "import sys, shipvision; "
            "shipvision.Detection(box=[0, 0, 1, 1]); "
            "shipvision.FrameTag(camera_id='c', frame_id=0); "
            f"print(sorted(m for m in {HEAVY!r} if m in sys.modules))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )

        assert result.stdout.strip() == "[]"


class TestRegistryExports:
    """Every family's registry is reachable from the top level, resolved on first access."""

    @pytest.mark.parametrize("name", REGISTRIES)
    def test_a_registry_is_reachable_from_the_top_level(self, name: str) -> None:
        registry = getattr(shipvision, name)

        assert hasattr(registry, "names")
        assert hasattr(registry, "build")

    @pytest.mark.parametrize("name", REGISTRIES)
    def test_a_registry_is_declared_in_all(self, name: str) -> None:
        """Otherwise `from shipvision import *` and a documentation build disagree with what
        actually resolves."""
        assert name in shipvision.__all__

    @pytest.mark.parametrize("name", REGISTRIES)
    def test_the_second_access_is_cached(self, name: str) -> None:
        first = getattr(shipvision, name)
        second = getattr(shipvision, name)

        assert first is second
        assert name in vars(shipvision), "resolution should memoise into module globals"

    def test_every_declared_family_actually_resolves(self) -> None:
        """The lookup table is a promise. An entry naming a module that does not exist, or a
        module that does not define the registry, must fail here rather than at the first
        call site that trusts the documented entry point."""
        for name, home in shipvision._REGISTRY_HOMES.items():
            resolved = getattr(shipvision, name)
            assert type(resolved).__name__ == "Registry", f"{name} in {home} is not a Registry"

    def test_the_registries_are_not_all_the_same_object(self) -> None:
        """A copy-paste slip in the lookup table would make every family share one registry,
        and `TRACKERS.build("flat")` would then succeed."""
        if len(REGISTRIES) < 2:
            pytest.skip(
                f"needs at least two families registered; this build declares {len(REGISTRIES)}"
            )
        resolved = [getattr(shipvision, name) for name in REGISTRIES]

        assert len({id(r) for r in resolved}) > 1

    def test_dir_lists_the_lazy_names(self) -> None:
        """Tab completion and `help()` read `__dir__`; a lazy attribute absent from it is
        invisible to anyone exploring the package."""
        listed = dir(shipvision)

        for name in REGISTRIES:
            assert name in listed

    def test_an_unknown_attribute_raises_attribute_error(self) -> None:
        """Not KeyError, and not a silent None — `getattr(shipvision, x, default)` and
        `hasattr` both depend on this being AttributeError."""
        with pytest.raises(AttributeError, match="no attribute 'NotAThing'"):
            shipvision.NotAThing  # noqa: B018

        assert not hasattr(shipvision, "NotAThing")
        assert getattr(shipvision, "NotAThing", "fallback") == "fallback"


def requires(family: str) -> None:
    """Skip when a family is not part of this build, naming it.

    CLAUDE.md and README.md describe the finished library, while a branch or a slimmed
    install may ship a subset — the packages register themselves into
    `_REGISTRY_HOMES` as they land. Skipping is not a weakening: if the family IS present
    and the documented line is wrong, the test still fails. What is refused is a test that
    fails for the absence of something it is not testing.
    """
    if family not in shipvision._REGISTRY_HOMES:
        pytest.skip(f"{family} is not part of this build")


class TestDocumentedEntryPoints:
    """The exact lines that appear in CLAUDE.md and README.md must actually run."""

    def test_the_claude_md_tracker_example_works(self) -> None:
        requires("TRACKERS")
        from shipvision import TRACKERS

        assert "bytetrack" in TRACKERS.names()
        assert TRACKERS.build("bytetrack", track_threshold=0.5) is not None

    def test_the_readme_gallery_example_works(self) -> None:
        requires("GALLERIES")
        import numpy as np

        from shipvision import GALLERIES, Embedding

        gallery = GALLERIES.build("flat", per_identity=8)
        gallery.add(
            Embedding(vector=np.ones(32, np.float32), identity="ship-14", camera_id="cam-03")
        )

        assert len(gallery) == 1


class TestTheLazyMachineryItself:
    """The half `REGISTRIES` cannot reach, because the shipped table is empty.

    `_REGISTRY_HOMES` starts empty on purpose — a family is added to it when it lands — so
    every test above parametrized over `REGISTRIES` collects as an empty parameter set and
    passes without running. That is fine as a *shape* check for whatever ships later, but it
    means the machinery those tests exist to protect is currently exercised by nothing:
    `globals()[name] = value`, the `ImportError` → `AttributeError` translation, and `__dir__`
    could each be wrong today with the suite green.

    So this class installs a real family — a real `Registry` in a real module — for the
    duration of one test, and takes it out again. The memoised global has to be deleted in
    teardown too: `__getattr__` caches into `globals()`, so a leftover entry would make a later
    test pass for the wrong reason.
    """

    HOME = "tests.lazy_widget"

    @pytest.fixture(autouse=True)
    def _leave_no_trace(self):
        """Undo the two global effects a resolution has, after *every* test in this class.

        Autouse rather than attached to `declared`, because the leak came from the one test
        here that names `HOME` without asking for that fixture — so its teardown never ran, and
        `test_registry.py::test_a_lazy_entry_is_not_imported_until_it_is_built` failed from
        this file. Test pollution reported as a bug in the code under test is the most
        expensive kind of green-to-red there is.

        Two things to undo: `__getattr__` memoises into `globals()`, and resolving the family
        imports `tests/lazy_widget.py`, which is meant to be imported only by the test that
        builds it.
        """
        yield
        vars(shipvision).pop("WIDGETS", None)
        sys.modules.pop("tests.lazy_widget", None)

    @pytest.fixture
    def declared(self, monkeypatch):
        """`WIDGETS` declared in the table for the duration of one test."""
        monkeypatch.setitem(shipvision._REGISTRY_HOMES, "WIDGETS", self.HOME)
        return "WIDGETS"

    def test_a_declared_family_resolves_on_first_access(self, declared: str) -> None:
        from tests.lazy_widget import WIDGETS

        assert getattr(shipvision, declared) is WIDGETS

    def test_it_was_not_a_global_until_it_was_asked_for(self, declared: str) -> None:
        """The point of the mechanism: declaring a family must not import it."""
        assert declared not in vars(shipvision)

        getattr(shipvision, declared)

        assert declared in vars(shipvision), "resolution should memoise into module globals"

    def test_the_second_access_is_the_memoised_global(self, declared: str) -> None:
        first = getattr(shipvision, declared)
        second = getattr(shipvision, declared)

        assert first is second

    def test_dir_lists_it_before_it_has_been_resolved(self, declared: str) -> None:
        """Tab completion and `help()` read `__dir__`. A lazy attribute absent from it is
        invisible to anyone exploring the package, which is when it matters most."""
        assert declared not in vars(shipvision)

        assert declared in dir(shipvision)

    def test_a_home_that_cannot_be_imported_surfaces_as_attribute_error(
        self, monkeypatch
    ) -> None:
        """Not ImportError. `hasattr` and `getattr(x, name, default)` both swallow only
        AttributeError, so a family whose own dependencies are missing must arrive as an
        absent attribute rather than as an exception from an unrelated package."""
        monkeypatch.setitem(shipvision._REGISTRY_HOMES, "ABSENT", "shipvision.no_such_module")

        with pytest.raises(AttributeError, match="could not be imported"):
            _ = shipvision.ABSENT

    def test_a_home_that_lacks_the_registry_is_not_silently_none(self, monkeypatch) -> None:
        """The other half of the promise: the module imports, but does not define the name."""
        monkeypatch.setitem(shipvision._REGISTRY_HOMES, "MISSING", self.HOME)

        with pytest.raises(AttributeError):
            _ = shipvision.MISSING
