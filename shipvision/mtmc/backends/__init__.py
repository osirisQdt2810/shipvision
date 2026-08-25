"""One module per execution backend for the cross-camera matchers.

Empty of imports for the same reason as :mod:`shipvision.tracking.backends`: importing a
backend module runs its registration decorator, so which backends get imported is a policy
decision that belongs in :mod:`shipvision.mtmc`, not an invisible side effect of touching this
directory.

The three matchers themselves — appearance, spatial, and appearance gated by geometry — are
algorithms and live one package each under :mod:`shipvision.mtmc.core`. What lives here is what
*runs* them.

:mod:`~shipvision.mtmc.backends.native`
    The fused ``(n, n)`` passes in ``shipvision._C``. Registered unconditionally; construction
    is what fails when there is no build.
"""

from __future__ import annotations
