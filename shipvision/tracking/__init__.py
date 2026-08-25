"""The old name of :mod:`shipvision.mot`, kept importable for one release.

The package moved to ``mot`` — one package per algorithm, three files each — and the parent
(`shipinfer`'s tracking stage) still imports ``shipvision.tracking`` until its own sync lands
(the two-planes rule applies to the library seam too). Everything public resolves through here;
new code imports :mod:`shipvision.mot`.
"""

from __future__ import annotations

import warnings

from shipvision.mot import *  # noqa: F403 — the re-export is the point
from shipvision.mot import __all__

warnings.warn(
    "shipvision.tracking is now shipvision.mot; the old name is kept for one release",
    DeprecationWarning,
    stacklevel=2,
)
