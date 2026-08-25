"""Imported only by the registry's lazy-loading test, and only when that test builds it."""

from __future__ import annotations

from shipvision.registry import Registry


class LazyWidget:
    pass


#: A real registry in a real module, so the lazy-loading machinery in ``shipvision/__init__``
#: has something to resolve. The shipped ``_REGISTRY_HOMES`` is empty by design — a family is
#: added when it lands — but that left every test of the machinery parametrized over ``()``,
#: which pytest collects as an empty parameter set and reports as a pass. ``globals()[name] =
#: value``, the ``ImportError`` translation and ``__dir__`` could each have been wrong.
WIDGETS: Registry[LazyWidget] = Registry("widget")
