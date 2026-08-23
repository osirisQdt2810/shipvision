"""The registry is the seam every algorithm family hangs off, so its edges are pinned here."""

from __future__ import annotations

import pytest

from shipvision import NATIVE, PYTHON, TORCH, ConfigurationError, Registry


class Base:
    pass


@pytest.fixture()
def registry() -> Registry:
    return Registry[Base]("widget")


def test_a_registered_class_is_built_by_name(registry: Registry) -> None:
    @registry.register("alpha")
    class Alpha(Base):
        def __init__(self, size: int = 1) -> None:
            self.size = size

    built = registry.build("alpha", size=7)

    assert isinstance(built, Alpha)
    assert built.size == 7
    assert built.name == "alpha"
    assert built.backend == PYTHON


def test_the_same_name_may_have_several_backends(registry: Registry) -> None:
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


def test_an_unnamed_backend_resolves_to_the_fastest_available(registry: Registry) -> None:
    @registry.register("alpha", backend=PYTHON)
    class AlphaPy(Base):
        pass

    assert isinstance(registry.build("alpha"), AlphaPy), "numpy only — it must still resolve"

    @registry.register("alpha", backend=TORCH)
    class AlphaTorch(Base):
        pass

    assert isinstance(registry.build("alpha"), AlphaTorch), "torch outranks numpy"

    @registry.register("alpha", backend=NATIVE)
    class AlphaNative(Base):
        pass

    assert isinstance(registry.build("alpha"), AlphaNative), "compiled outranks both"


def test_numpy_is_the_fallback_not_the_default(registry: Registry) -> None:
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


def test_aliases_resolve(registry: Registry) -> None:
    @registry.register("bytetrack", aliases=("byte", "bt"))
    class ByteTrack(Base):
        pass

    assert isinstance(registry.build("byte"), ByteTrack)
    assert isinstance(registry.build("bt"), ByteTrack)
    assert registry.names() == ["bytetrack"], "an alias is not a separate algorithm"


def test_registering_the_same_name_and_backend_twice_is_refused(registry: Registry) -> None:
    """Silently replacing the first would make which implementation runs depend on import
    order, which is the least debuggable failure a registry can have."""

    @registry.register("alpha")
    class First(Base):
        pass

    with pytest.raises(ConfigurationError, match="already registered"):

        @registry.register("alpha")
        class Second(Base):
            pass


def test_the_same_name_on_a_different_backend_is_fine(registry: Registry) -> None:
    @registry.register("alpha", backend=PYTHON)
    class Py(Base):
        pass

    @registry.register("alpha", backend=NATIVE)
    class Native(Base):
        pass

    assert len(registry.backends("alpha")) == 2


def test_an_alias_cannot_be_stolen(registry: Registry) -> None:
    @registry.register("alpha", aliases=("a",))
    class Alpha(Base):
        pass

    with pytest.raises(ConfigurationError, match="already points at"):

        @registry.register("beta", aliases=("a",))
        class Beta(Base):
            pass


def test_an_unknown_name_lists_what_there_is(registry: Registry) -> None:
    @registry.register("alpha")
    class Alpha(Base):
        pass

    with pytest.raises(ConfigurationError, match=r"available: \['alpha'\]"):
        registry.build("omega")


def test_a_missing_backend_says_which_ones_exist(registry: Registry) -> None:
    """An operator who set backend=native on a machine with no build needs to be told that,
    not handed a numpy implementation that quietly runs at a tenth of the speed."""

    @registry.register("alpha", backend=PYTHON)
    class Alpha(Base):
        pass

    with pytest.raises(
        ConfigurationError, match=r"has no 'native' backend; it has \['python'\]"
    ):
        registry.build("alpha", backend=NATIVE)


def test_a_lazy_entry_is_not_imported_until_it_is_built(registry: Registry) -> None:
    """Importing the TensorRT backend to discover TensorRT is absent costs a second and an
    exception on every start-up of every process that never uses it."""
    registry.register_lazy("late", "tests.lazy_widget:LazyWidget", backend=NATIVE)

    assert "late" in registry
    assert registry.backends("late") == [NATIVE]
    assert "tests.lazy_widget" not in __import__("sys").modules

    built = registry.build("late")

    assert type(built).__name__ == "LazyWidget"
    assert "tests.lazy_widget" in __import__("sys").modules


def test_a_lazy_target_must_name_a_class(registry: Registry) -> None:
    registry.register_lazy("bad", "tests.lazy_widget", backend=NATIVE)

    with pytest.raises(ConfigurationError, match="module:Class"):
        registry.build("bad")


def test_a_typo_in_a_config_key_raises_rather_than_being_dropped(registry: Registry) -> None:
    """A dropped keyword means the algorithm runs with a default nobody chose, and the run
    looks successful."""

    @registry.register("alpha")
    class Alpha(Base):
        def __init__(self, threshold: float = 0.5) -> None:
            self.threshold = threshold

    with pytest.raises(TypeError):
        registry.build("alpha", threshhold=0.9)


def test_membership_and_length_read_naturally(registry: Registry) -> None:
    @registry.register("alpha")
    class Alpha(Base):
        pass

    assert "alpha" in registry
    assert "omega" not in registry
    assert 42 not in registry
    assert len(registry) == 1


def test_a_registered_class_must_be_reachable_by_the_family_it_claims(
    registry: Registry,
) -> None:
    @registry.register("alpha")
    class Alpha(Base):
        pass

    assert registry.get("alpha") is Alpha
    assert "widget" in repr(registry)
