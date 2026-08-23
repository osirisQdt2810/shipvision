"""One registry primitive, used by every family of algorithm.

Adding an implementation is a new file and a decorator — never an edit to a switch
statement. That is not tidiness: the reference implementations this library replaces all
select algorithms with a hand-written ``if/elif`` on a string, and every one of them has the
same consequence, which is that a deployment cannot A/B two trackers on the same stream
without a code change. If the choice cannot be made from config, it does not get made, and
the algorithm that shipped first wins by default rather than by measurement.

**Backends are part of the key, not a separate concept.** A tracker written in C++ and a
tracker written in numpy are the same algorithm answering the same question at different
speeds, and both need to exist: the readable one is how the fast one is checked (a fused
kernel nobody can compare against is a fused kernel nobody can trust), and the readable one
is also what runs on a machine with no build. So an implementation registers under
``(name, backend)``, and ``build("bytetrack")`` resolves the backend by preference order
while ``build("bytetrack", backend="python")`` pins it.

    TRACKERS = Registry[BaseTracker]("tracker")

    @TRACKERS.register("bytetrack", backend="python", aliases=("byte",))
    class ByteTrackTracker(BaseTracker): ...

    tracker = TRACKERS.build("bytetrack", track_threshold=0.5)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Generic, TypeVar

from shipvision.errors import ConfigurationError

__all__ = ["NATIVE", "PYTHON", "TENSORRT", "TORCH", "Registry"]

T = TypeVar("T")

#: Backend names. Not an enum: a third-party package must be able to register a backend
#: this library has never heard of, and an enum would make that an edit here.
NATIVE = "native"
"""The compiled C++/CUDA/HIP path in ``shipvision._C``."""
PYTHON = "python"
"""Pure numpy. Always available, always the reference the others are checked against."""
TORCH = "torch"
"""Implemented with torch ops — GPU-resident without leaving Python."""
TENSORRT = "tensorrt"
"""A built engine. Fastest and least portable."""

#: Resolution order when a caller names an algorithm but not a backend. Fastest first, with
#: ``python`` last so it is the fallback rather than the default — but it *is* always a
#: fallback, which is what lets the offline test tier run with no GPU and no build.
DEFAULT_PREFERENCE: tuple[str, ...] = (TENSORRT, NATIVE, TORCH, PYTHON)


class Registry(Generic[T]):
    """A named family of interchangeable implementations, keyed on ``(name, backend)``."""

    def __init__(self, family: str, *, preference: Iterable[str] = DEFAULT_PREFERENCE) -> None:
        self.family = family
        self._preference = tuple(preference)
        self._entries: dict[tuple[str, str], type[T]] = {}
        self._aliases: dict[str, str] = {}
        self._lazy: dict[tuple[str, str], str] = {}

    # -- registration -----------------------------------------------------------------

    def register(
        self,
        name: str,
        *,
        backend: str = PYTHON,
        aliases: Iterable[str] = (),
    ) -> Callable[[type[T]], type[T]]:
        """Class decorator. ``@TRACKERS.register("bytetrack", backend="native")``."""

        def decorator(cls: type[T]) -> type[T]:
            self._claim(name, backend, aliases)
            self._entries[(name, backend)] = cls
            # Stamped on the class so an instance can say what it is without the caller
            # having to remember what it asked for — which matters in a log line.
            cls.name = name  # type: ignore[attr-defined]
            cls.backend = backend  # type: ignore[attr-defined]
            return cls

        return decorator

    def register_lazy(
        self,
        name: str,
        target: str,
        *,
        backend: str = PYTHON,
        aliases: Iterable[str] = (),
    ) -> None:
        """Register ``"module:Class"`` without importing it.

        For implementations whose import is expensive or whose runtime may be absent —
        importing the TensorRT backend to discover that TensorRT is not installed costs a
        second and an exception on every start-up of every process that never uses it.
        """
        self._claim(name, backend, aliases)
        self._lazy[(name, backend)] = target

    def _claim(self, name: str, backend: str, aliases: Iterable[str]) -> None:
        key = (name, backend)
        if key in self._entries or key in self._lazy:
            raise ConfigurationError(
                f"{self.family} {name!r} is already registered for backend {backend!r}"
            )
        for alias in aliases:
            existing = self._aliases.get(alias)
            if existing is not None and existing != name:
                raise ConfigurationError(
                    f"alias {alias!r} already points at {self.family} {existing!r}"
                )
            self._aliases[alias] = name

    # -- lookup -----------------------------------------------------------------------

    def names(self) -> list[str]:
        """Every algorithm name, regardless of which backends implement it."""
        return sorted({name for name, _ in (*self._entries, *self._lazy)})

    def backends(self, name: str) -> list[str]:
        """Which backends implement ``name``, in preference order."""
        resolved = self._aliases.get(name, name)
        found = {b for n, b in (*self._entries, *self._lazy) if n == resolved}
        ordered = [b for b in self._preference if b in found]
        return ordered + sorted(found.difference(ordered))

    def get(self, name: str, backend: str | None = None) -> type[T]:
        """The class for ``name``, resolving the backend if one is not named."""
        resolved = self._aliases.get(name, name)
        available = self.backends(resolved)
        if not available:
            raise ConfigurationError(
                f"unknown {self.family} {name!r}; available: {self.names()}"
            )
        if backend is None:
            chosen = available[0]
        elif backend in available:
            chosen = backend
        else:
            raise ConfigurationError(
                f"{self.family} {resolved!r} has no {backend!r} backend; it has " f"{available}"
            )

        key = (resolved, chosen)
        entry = self._entries.get(key)
        if entry is None:
            entry = _import_target(self._lazy[key])
            # Stamp the same attributes `register` stamps. The decorator cannot have done it
            # — the class did not exist yet — and without this a lazily-registered class
            # reports whatever `name`/`backend` it happens to inherit, so a log line says
            # "python" for a TensorRT extractor. Doing it here rather than asking each lazy
            # class to declare them keeps the two registration paths indistinguishable to
            # everything downstream.
            entry.name = resolved  # type: ignore[attr-defined]
            entry.backend = chosen  # type: ignore[attr-defined]
            self._entries[key] = entry
        return entry

    def build(self, name: str, *, backend: str | None = None, **kwargs: object) -> T:
        """Instantiate. Unknown keyword arguments raise from the constructor.

        Deliberately not swallowed: a typo in a config key that is silently dropped means
        the algorithm runs with a default nobody chose, and the run looks successful.
        """
        return self.get(name, backend)(**kwargs)  # type: ignore[call-arg]

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and bool(self.backends(name))

    def __len__(self) -> int:
        return len(self.names())

    def __repr__(self) -> str:
        return f"<Registry {self.family} {self.names()}>"


def _import_target(target: str) -> type:
    module_name, _, attribute = target.partition(":")
    if not attribute:
        raise ConfigurationError(f"lazy target {target!r} must be 'module:Class'")
    from importlib import import_module

    return getattr(import_module(module_name), attribute)
