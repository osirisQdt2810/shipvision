"""The tracker registry: a name resolves to a class, and nothing dispatches on a string.

Split out of :mod:`shipvision.tracking.base` when the algorithms became packages under
``core/``. The reason is the direction of the dependencies, not tidiness.

``base.py`` is the *contract* — the ABC, the process-wide id counter, the tag discipline —
and every ``core/<algorithm>/tracker.py`` needs the registry for one thing only: the decorator
on its class. With both in one module, a module that wants a decorator imports the whole
contract, and — the part that actually bites — the registry becomes unreachable from anything
``base`` itself imports. This module imports nothing from :mod:`shipvision.tracking` at
runtime (the ABC below is a typing-only import), so it can be imported from any point in the
package in whatever order a caller arrives, which is the property all five decorators rely on.

There is no runtime import cycle here today, and this module is not claiming to prevent one:
the cycle that *is* real in this package is ``base`` <-> ``pool``, and it is handled where it
lives, in :mod:`shipvision.tracking.base`.

``base.py`` re-exports ``TRACKERS`` so ``from shipvision.tracking.base import TRACKERS`` keeps
working — see the note there.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from shipvision.registry import Registry

if TYPE_CHECKING:  # typing only; the runtime graph is a tree because of this line
    from shipvision.tracking.base import BaseTracker

__all__ = ["TRACKERS"]

TRACKERS: Registry[BaseTracker] = Registry("tracker")
"""Every tracker, keyed on ``(name, backend)``.

A new tracker is a new **package** under :mod:`shipvision.tracking.core` plus a decorator —
never an edit to a switch statement. Selecting one by name from config is what lets a
deployment A/B two association strategies on the same stream without a code change, which is
the only honest way to decide between them. Every reference implementation this library
replaces picks its tracker with a hand-written ``if/elif``, and every one has the same
consequence: the tracker that shipped first wins by default rather than by measurement.

There is no ``native`` tracker yet. When there is, it registers here under the same name as
its numpy twin and a parity test compares the two on one sequence — a compiled association
loop nobody can compare against is a compiled association loop nobody can trust.
"""
