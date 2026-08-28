"""The tracker registry: a name resolves to a class, and nothing dispatches on a string.

Split out of :mod:`shipvision.mot.base` when the algorithms became packages under
``core/``. The reason is the direction of the dependencies, not tidiness.

``base.py`` is the *contract* — the ABC, the process-wide id counter, the tag discipline —
and every ``core/<algorithm>/tracker.py`` needs the registry for one thing only: the decorator
on its class. With both in one module, a module that wants a decorator imports the whole
contract, and — the part that actually bites — the registry becomes unreachable from anything
``base`` itself imports. This module imports nothing from :mod:`shipvision.mot` at
runtime (the ABC below is a typing-only import), so it can be imported from any point in the
package in whatever order a caller arrives, which is the property all five decorators rely on.

There is no runtime import cycle here today, and this module is not claiming to prevent one:
the cycle that *is* real in this package is ``base`` <-> ``pool``, and it is handled where it
lives, in :mod:`shipvision.mot.base`.

``base.py`` re-exports ``TRACKERS`` so ``from shipvision.mot.base import TRACKERS`` keeps
working — see the note there.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from shipvision.registry import Registry

if TYPE_CHECKING:  # typing only; the runtime graph is a tree because of this line
    from shipvision.mot.base import BaseTracker

__all__ = ["TRACKERS"]

TRACKERS: Registry[BaseTracker] = Registry("tracker")
"""Every tracker, keyed on ``(name, backend)``.

A new tracker is a new **package** under :mod:`shipvision.mot.core` plus a decorator —
never an edit to a switch statement. Selecting one by name from config is what lets a
deployment A/B two association strategies on the same stream without a code change, which is
the only honest way to decide between them. Every reference implementation this library
replaces picks its tracker with a hand-written ``if/elif``, and every one has the same
consequence: the tracker that shipped first wins by default rather than by measurement.

Five of the six have a ``native`` twin — :mod:`shipvision.mot.backends.native`, over the C++
association loops in ``shipvision._C``. Both backends register under the same name, so a config
that says ``sort`` keeps saying ``sort``, and ``tests/mot/backends/test_parity.py``
enumerates the pairs *from this registry* and runs each over one sequence, comparing
identities: a compiled association loop nobody can compare against is a compiled association
loop nobody can trust. Enumerating rather than listing is what makes that true by construction
— a compiled tracker registered without its numpy oracle would join the parity suite the day
it was added, rather than the day somebody remembered to add it to a list.
"""
