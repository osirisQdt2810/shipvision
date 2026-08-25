"""The registry is the seam every algorithm family hangs off, so its edges are pinned here."""

from __future__ import annotations

import sys

import pytest

from shipvision import (
    NATIVE,
    PYTHON,
    TORCH,
    BackendUnavailableError,
    ConfigurationError,
    Registry,
)


class Base:
    pass


@pytest.fixture()
def registry() -> Registry:
    return Registry[Base]("widget")


class TestRegistration:
    """Claiming a name: who gets it, who is refused, and what an alias may point at."""

    def test_a_registered_class_is_built_by_name(self, registry: Registry) -> None:
        @registry.register("alpha")
        class Alpha(Base):
            def __init__(self, size: int = 1) -> None:
                self.size = size

        built = registry.build("alpha", size=7)

        assert isinstance(built, Alpha)
        assert built.size == 7
        assert built.name == "alpha"
        assert built.backend == PYTHON

    def test_registering_the_same_name_and_backend_twice_is_refused(
        self, registry: Registry
    ) -> None:
        """Silently replacing the first would make which implementation runs depend on import
        order, which is the least debuggable failure a registry can have."""

        @registry.register("alpha")
        class First(Base):
            pass

        with pytest.raises(ConfigurationError, match="already registered"):

            @registry.register("alpha")
            class Second(Base):
                pass

    def test_the_same_name_on_a_different_backend_is_fine(self, registry: Registry) -> None:
        @registry.register("alpha", backend=PYTHON)
        class Py(Base):
            pass

        @registry.register("alpha", backend=NATIVE)
        class Native(Base):
            pass

        assert len(registry.backends("alpha")) == 2

    def test_aliases_resolve(self, registry: Registry) -> None:
        @registry.register("bytetrack", aliases=("byte", "bt"))
        class ByteTrack(Base):
            pass

        assert isinstance(registry.build("byte"), ByteTrack)
        assert isinstance(registry.build("bt"), ByteTrack)
        assert registry.names() == ["bytetrack"], "an alias is not a separate algorithm"

    def test_an_alias_cannot_be_stolen(self, registry: Registry) -> None:
        @registry.register("alpha", aliases=("a",))
        class Alpha(Base):
            pass

        with pytest.raises(ConfigurationError, match="already points at"):

            @registry.register("beta", aliases=("a",))
            class Beta(Base):
                pass


class TestBackendResolution:
    """Which implementation you get when you name an algorithm but not a backend."""

    def test_the_same_name_may_have_several_backends(self, registry: Registry) -> None:
        """The whole point. A compiled implementation and a readable one are the same algorithm,
        and a test that compares them needs to reach both by name."""

        @registry.register("alpha", backend=PYTHON)
        class AlphaPy(Base):
            pass

        @registry.register("alpha", backend=NATIVE)
        class AlphaNative(Base):
            pass

        assert registry.names() == ["alpha"]
        assert registry.backends("alpha") == [NATIVE, PYTHON]
        assert isinstance(registry.build("alpha", backend=PYTHON), AlphaPy)
        assert isinstance(registry.build("alpha", backend=NATIVE), AlphaNative)

    def test_an_unnamed_backend_resolves_to_the_fastest_available(
        self, registry: Registry
    ) -> None:
        @registry.register("alpha", backend=PYTHON)
        class AlphaPy(Base):
            pass

        assert isinstance(
            registry.build("alpha"), AlphaPy
        ), "numpy only — it must still resolve"

        @registry.register("alpha", backend=TORCH)
        class AlphaTorch(Base):
            pass

        assert isinstance(registry.build("alpha"), AlphaTorch), "torch outranks numpy"

        @registry.register("alpha", backend=NATIVE)
        class AlphaNative(Base):
            pass

        assert isinstance(registry.build("alpha"), AlphaNative), "compiled outranks both"

    def test_numpy_is_the_fallback_not_the_default(self, registry: Registry) -> None:
        """Both halves of that sentence are load-bearing: it must never be *chosen* over a
        faster backend, and it must always be *there* — that is what lets the offline test tier
        run with no GPU and no build."""

        @registry.register("alpha", backend=NATIVE)
        class AlphaNative(Base):
            pass

        @registry.register("alpha", backend=PYTHON)
        class AlphaPy(Base):
            pass

        assert registry.backends("alpha")[-1] == PYTHON
        assert isinstance(registry.build("alpha"), AlphaNative)
        assert isinstance(registry.build("alpha", backend=PYTHON), AlphaPy)

    def test_a_missing_backend_says_which_ones_exist(self, registry: Registry) -> None:
        """An operator who set backend=native on a machine with no build needs to be told that,
        not handed a numpy implementation that quietly runs at a tenth of the speed."""

        @registry.register("alpha", backend=PYTHON)
        class Alpha(Base):
            pass

        with pytest.raises(
            ConfigurationError, match=r"has no 'native' backend; it has \['python'\]"
        ):
            registry.build("alpha", backend=NATIVE)


class TestAvailabilityIsNotRegistration:
    """A compiled backend registers on a machine that cannot build it, deliberately: making
    registration conditional would mean :meth:`Registry.backends` answering differently per
    host, and an error message losing the ability to say "native exists, it just is not built
    here". Only the constructor knows the truth, so an unpinned ``build`` has to act on it —
    which is what makes "fastest available, numpy as the floor" a run-time promise rather than
    an import-time one."""

    @staticmethod
    def register_absent(registry: Registry, backend: str) -> None:
        """Register a backend under ``alpha`` whose constructor says it cannot run here."""

        @registry.register("alpha", backend=backend)
        class Absent(Base):
            def __init__(self, **kwargs: object) -> None:
                raise BackendUnavailableError(f"{backend} is not built on this machine")

    def test_an_unpinned_build_falls_back_past_a_backend_that_cannot_be_built(
        self, registry: Registry
    ) -> None:
        self.register_absent(registry, NATIVE)

        @registry.register("alpha", backend=PYTHON)
        class AlphaPy(Base):
            pass

        assert registry.backends("alpha") == [NATIVE, PYTHON], "it is still registered"
        assert isinstance(registry.build("alpha"), AlphaPy)

    def test_the_fallback_walks_the_whole_preference_order(self, registry: Registry) -> None:
        """The floor, reached for real: every faster backend refuses at once. That is the state
        of a plain CI runner, and the answer must be a working object rather than an
        exception — it is what lets the offline test tier exist."""
        self.register_absent(registry, NATIVE)
        self.register_absent(registry, TORCH)

        @registry.register("alpha", backend=PYTHON)
        class AlphaPy(Base):
            pass

        assert isinstance(registry.build("alpha"), AlphaPy)

    def test_a_pinned_backend_never_falls_back(self, registry: Registry) -> None:
        """Asking for ``native`` and silently getting numpy would be a large throughput
        regression reported as a successful start-up."""
        self.register_absent(registry, NATIVE)

        @registry.register("alpha", backend=PYTHON)
        class AlphaPy(Base):
            pass

        with pytest.raises(BackendUnavailableError, match="not built on this machine"):
            registry.build("alpha", backend=NATIVE)

    def test_when_nothing_can_be_built_the_failure_lists_every_attempt(
        self, registry: Registry
    ) -> None:
        """One exception naming both refusals, not the last one. An operator debugging a
        deployment needs to know that ``torch`` was tried too."""
        self.register_absent(registry, NATIVE)
        self.register_absent(registry, TORCH)

        with pytest.raises(BackendUnavailableError) as raised:
            registry.build("alpha")

        assert "native" in str(raised.value)
        assert "torch" in str(raised.value)
        assert "widget" in str(raised.value), "the family, so the message says what failed"

    def test_only_an_unavailable_runtime_is_treated_as_try_the_next(
        self, registry: Registry
    ) -> None:
        """A backend that refuses the *arguments* must not be papered over by a slower one
        accepting them. "This machine has no CUDA" and "0.3 is not a valid threshold" are
        different events, and silently answering the second with a different implementation
        is how a study tunes a tracker nobody configured."""

        @registry.register("alpha", backend=NATIVE)
        class AlphaNative(Base):
            def __init__(self, **kwargs: object) -> None:
                raise ConfigurationError("threshold must be in [0, 1]")

        @registry.register("alpha", backend=PYTHON)
        class AlphaPy(Base):
            def __init__(self, **kwargs: object) -> None:
                pass

        with pytest.raises(ConfigurationError, match="threshold"):
            registry.build("alpha", threshold=1.5)

    def test_an_unknown_name_still_says_so_rather_than_reporting_no_backends(
        self, registry: Registry
    ) -> None:
        """The fallback loop must not swallow the "you asked for something that does not
        exist" case into a vaguer one."""

        @registry.register("alpha")
        class Alpha(Base):
            pass

        with pytest.raises(ConfigurationError, match=r"available: \['alpha'\]"):
            registry.build("omega")


class TestLazyRegistration:
    """A deferred import must be indistinguishable from an eager one once resolved."""

    def test_a_lazy_entry_is_not_imported_until_it_is_built(self, registry: Registry) -> None:
        """Importing the TensorRT backend to discover TensorRT is absent costs a second and an
        exception on every start-up of every process that never uses it."""
        # Established rather than assumed. The property is "register_lazy does not import", and
        # inheriting an unimported state from whichever files ran first makes this test a
        # report on collection order — it failed exactly that way once, from `test_package.py`.
        sys.modules.pop("tests.lazy_widget", None)

        registry.register_lazy("late", "tests.lazy_widget:LazyWidget", backend=NATIVE)

        assert "late" in registry
        assert registry.backends("late") == [NATIVE]
        assert "tests.lazy_widget" not in sys.modules

        built = registry.build("late")

        assert type(built).__name__ == "LazyWidget"
        assert "tests.lazy_widget" in sys.modules

    def test_a_lazy_class_reports_the_same_name_and_backend_as_an_eager_one(
        self,
        registry: Registry,
    ) -> None:
        """The decorator could not stamp these — the class did not exist yet. Without stamping
        them on import, a log line says "python" for a TensorRT implementation, and the two
        registration paths become distinguishable to everything downstream."""
        registry.register_lazy("late", "tests.lazy_widget:LazyWidget", backend=NATIVE)

        built = registry.build("late")

        assert built.name == "late"
        assert built.backend == NATIVE

    def test_a_lazy_target_must_name_a_class(self, registry: Registry) -> None:
        registry.register_lazy("bad", "tests.lazy_widget", backend=NATIVE)

        with pytest.raises(ConfigurationError, match="module:Class"):
            registry.build("bad")


class TestConfigurationErrors:
    """A wrong config must fail loudly at start-up, never quietly at frame 40 000."""

    def test_an_unknown_name_lists_what_there_is(self, registry: Registry) -> None:
        @registry.register("alpha")
        class Alpha(Base):
            pass

        with pytest.raises(ConfigurationError, match=r"available: \['alpha'\]"):
            registry.build("omega")

    def test_a_typo_in_a_config_key_raises_rather_than_being_dropped(
        self, registry: Registry
    ) -> None:
        """A dropped keyword means the algorithm runs with a default nobody chose, and the run
        looks successful."""

        @registry.register("alpha")
        class Alpha(Base):
            def __init__(self, threshold: float = 0.5) -> None:
                self.threshold = threshold

        with pytest.raises(TypeError):
            registry.build("alpha", threshhold=0.9)

    def test_a_registered_class_must_be_reachable_by_the_family_it_claims(
        self,
        registry: Registry,
    ) -> None:
        @registry.register("alpha")
        class Alpha(Base):
            pass

        # It registered a class and asserted nothing — a test that passed with `_claim` deleted
        # and with `register` writing to nothing at all. Its name states the claim; now so does
        # its body: the family lists the name, resolves it to this class, and builds it.
        assert "alpha" in registry.names()
        assert registry.get("alpha") is Alpha
        assert isinstance(registry.build("alpha"), Alpha)


class TestIntrospection:
    """Reading the registry without building anything."""

    def test_membership_and_length_read_naturally(self, registry: Registry) -> None:
        @registry.register("alpha")
        class Alpha(Base):
            pass

        assert "alpha" in registry
        assert "omega" not in registry
        assert 42 not in registry
        assert len(registry) == 1

    def test_a_registered_class_must_be_reachable_by_the_family_it_claims(
        self,
        registry: Registry,
    ) -> None:
        @registry.register("alpha")
        class Alpha(Base):
            pass

        # It registered a class and asserted nothing — a test that passed with `_claim` deleted
        # and with `register` writing to nothing at all. Its name states the claim; now so does
        # its body: the family lists the name, resolves it to this class, and builds it.
        assert "alpha" in registry.names()
        assert registry.get("alpha") is Alpha
        assert isinstance(registry.build("alpha"), Alpha)

        assert registry.get("alpha") is Alpha
        assert "widget" in repr(registry)


class TestAnAliasMayNotShadowARealName:
    """The lookup order is what makes this dangerous, and it is invisible at the call site.

    `get` and `backends` both consult the alias table *before* the name table. So an alias that
    happens to spell a real algorithm's name outranks it: the shadowed algorithm is registered,
    listed by `names()`, and reachable by nothing. A config saying `tracker: sort` then either
    raises `TypeError` on a keyword the other constructor does not take, or — where the two
    happen to overlap — runs the wrong tracker, and the A/B measurement this registry exists to
    enable compares an algorithm against itself. Which of the two happens is decided by import
    order, so it can differ between a laptop and a container.

    Fixed in review, and untested until now, which is the half worth stating: the fix was
    verified by running it once by hand, and a fix verified that way is one nobody will notice
    losing.
    """

    def test_a_name_may_not_be_registered_over_an_existing_alias(
        self, registry: Registry
    ) -> None:
        @registry.register("botsort", aliases=("sort",))
        class BotSort(Base):
            pass

        with pytest.raises(ConfigurationError, match="already an alias"):

            @registry.register("sort")
            class Sort(Base):
                pass

    def test_an_alias_may_not_be_claimed_over_an_existing_name(
        self, registry: Registry
    ) -> None:
        """The other arrival order. Symmetric on purpose — which registration happens first is
        an import-order accident, and an accident must not decide whether this is an error."""

        @registry.register("sort")
        class Sort(Base):
            pass

        with pytest.raises(ConfigurationError, match="would shadow"):

            @registry.register("botsort", aliases=("sort",))
            class BotSort(Base):
                pass

    def test_a_name_may_alias_itself(self, registry: Registry) -> None:
        """Redundant but harmless, and refusing it would be a rule about spelling rather than
        about reachability — nothing is shadowed when the alias and the name are the same."""

        @registry.register("sort", aliases=("sort",))
        class Sort(Base):
            pass

        assert isinstance(registry.build("sort"), Sort)

    def test_a_rejected_registration_leaves_no_alias_behind(self, registry: Registry) -> None:
        """The old loop bound each alias as it went, so a registration refused on its third
        alias left its first two pointing at a class that is not registered — a state no single
        call could have produced, and one that makes the *next* registration fail confusingly.
        """

        @registry.register("sort")
        class Sort(Base):
            pass

        with pytest.raises(ConfigurationError):

            @registry.register("botsort", aliases=("bot", "sort"))
            class BotSort(Base):
                pass

        # `bot` was the alias before the one that failed. It must not have been bound.
        @registry.register("bytetrack", aliases=("bot",))
        class ByteTrack(Base):
            pass

        assert isinstance(registry.build("bot"), ByteTrack)
