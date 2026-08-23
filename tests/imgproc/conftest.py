"""Which image-ops backends this machine can actually run.

Availability is decided here, once, so the parity tests read as "compare every backend
against the oracle" rather than as three near-identical skip conditions. The offline tier
must pass with neither torch nor the compiled extension installed, so the numpy backend is
the only one that is never skipped.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from shipvision.imgproc import IMGPROC, native_available
from shipvision.registry import NATIVE, PYTHON, TORCH

TORCH_INSTALLED = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("torchvision") is not None
)
NATIVE_BUILT = native_available()


def backend_params() -> list:
    """``pytest.param`` per backend, each carrying its own skip and marker."""
    return [
        pytest.param(PYTHON, id="python"),
        pytest.param(
            TORCH,
            id="torch",
            marks=pytest.mark.skipif(
                not TORCH_INSTALLED, reason="torch and torchvision are not installed"
            ),
        ),
        pytest.param(
            NATIVE,
            id="native",
            marks=[
                pytest.mark.native,
                pytest.mark.skipif(
                    not NATIVE_BUILT, reason="shipvision._C is not built, or there is no GPU"
                ),
            ],
        ),
    ]


@pytest.fixture()
def oracle():
    """The numpy backend. Every other one is judged against this."""
    return IMGPROC.build("default", backend=PYTHON)


@pytest.fixture()
def bgr_image() -> np.ndarray:
    """A deterministic 480x640 uint8 BGR frame with structure in all three channels.

    Random rather than smooth: a smooth image hides a half-pixel error, because the value it
    would have read is almost the value it did read. Random noise makes a shifted sample
    differ by roughly the full dynamic range.
    """
    rng = np.random.default_rng(20260823)
    return rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
