"""One module per execution backend for the trackers.

Deliberately empty of imports, exactly like :mod:`shipvision.imgproc.backends`. Importing a
backend module is what runs its registration decorator, so *which* backends get imported is a
policy decision and it belongs in :mod:`shipvision.tracking`, which owns the public surface,
rather than being an invisible side effect of touching this directory.

The reference implementations are not here. They live one per package under
:mod:`shipvision.tracking.core`, because the algorithm and the backend are different axes: the
five algorithms differ in *how they associate*, and a backend differs in *what runs the
association*. Putting the numpy trackers here would mean five files that are the algorithms and
one that is a translator, all claiming to be the same kind of thing.

:mod:`~shipvision.tracking.backends.native`
    The C++ association loops in ``shipvision._C``, one translator class per algorithm.
    Registered unconditionally; construction is what fails when there is no build.
"""

from __future__ import annotations
