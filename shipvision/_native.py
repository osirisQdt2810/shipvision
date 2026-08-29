"""Finding the compiled extension — and refusing one that belongs to another checkout.

An editable install resolves ``shipvision._C`` through its own finder, which happily hands
back a ``.so`` built in a *different* working tree while ``shipvision`` itself comes from
this one. Nothing errors: the trackers silently run the other checkout's C++, a review's
mutation results mean nothing, and the numpy path everyone believes is under test is never
executed. That has already cost one review round, so the import lives here, once, and says
where the extension came from.
"""

from __future__ import annotations

import functools
import os
import warnings
from pathlib import Path
from typing import Any

__all__ = ["ALLOW_FOREIGN", "load_extension", "provenance"]

#: Set to keep a foreign extension anyway — a deliberate cross-tree build, and the operator
#: has said so. Anything else is the accident this module exists to catch.
ALLOW_FOREIGN = "SHIPVISION_ALLOW_FOREIGN_C"

#: Spellings that mean "no". Presence alone would make ``ALLOW_FOREIGN=0`` keep the foreign
#: extension *and* silence the warning — an operator turning the escape hatch back off by the
#: obvious means would land in exactly the state this module exists to prevent.
_FALSY = frozenset({"", "0", "false", "no", "off"})


def _opted_in() -> bool:
    return os.environ.get(ALLOW_FOREIGN, "").strip().lower() not in _FALSY


def _package_root() -> Path:
    import shipvision

    return Path(shipvision.__file__).resolve().parent


def _is_local(module: Any, here: Path) -> bool:
    """Whether ``_C`` was built inside this package rather than another checkout's."""
    there = Path(getattr(module, "__file__", "") or "").resolve()
    return there.is_relative_to(here)


@functools.cache
def load_extension() -> tuple[Any | None, str | None]:
    """``(module, reason it is absent)`` — exactly one of the two is ``None``.

    Cached: three backends and the test header all ask, and without it the warning fires once
    per caller and the four answers are only *probably* the same. Call ``cache_clear()`` in a
    test that moves the extension under it.
    """
    try:
        from shipvision import _C
    except ImportError as exc:
        return None, str(exc)

    here = _package_root()
    if _is_local(_C, here) or _opted_in():
        return _C, None

    reason = (
        f"shipvision._C at {getattr(_C, '__file__', '?')} was built in a different checkout "
        f"from shipvision itself ({here}); treating it as absent so this tree's own code is "
        f"what runs. Set {ALLOW_FOREIGN} to any non-empty, non-false value to keep it."
    )
    warnings.warn(reason, RuntimeWarning, stacklevel=2)
    return None, reason


def provenance() -> str:
    """One line saying which extension is live — what a test header should print."""
    module, reason = load_extension()
    if module is None:
        return f"shipvision._C: absent ({reason})"
    return f"shipvision._C: {getattr(module, '__file__', '?')}"
