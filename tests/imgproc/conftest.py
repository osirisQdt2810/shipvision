"""Which image-ops backends this machine can actually run, plus a stand-in for the one that
needs a GPU.

Availability is decided here, once, so the parity tests read as "compare every backend
against the oracle" rather than as three near-identical skip conditions. The offline tier
must pass with neither torch nor the compiled extension installed, so the numpy backend is
the only one that is never skipped.

:class:`FakeExtension` is here for the other half of that: most of the native backend is a
*translator* — it plans the geometry, checks the extension's answer against it, and owns the
thread affinity — and none of that needs a device to exercise. Substituting the fake for
``shipvision._C`` is what lets those guards be tested in the offline tier, which is where a
bug in them should be caught.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from shipvision.imgproc import IMGPROC, native_available
from shipvision.imgproc.geometry import LetterboxGeometry
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


class FakeExtension:
    """Stands in for ``shipvision._C``, reporting whatever geometry a test wants.

    A fake rather than a mock of the whole call: the point is to drive the *translator* — the
    Python side that plans the geometry and compares — with a plausible answer from the other
    side of the binding, which is exactly what a drifting build would produce.
    """

    def __init__(self, *, scale_delta=0.0, pad_delta=0, extent_delta=0) -> None:
        self.scale_delta = scale_delta
        self.pad_delta = pad_delta
        self.extent_delta = extent_delta
        self.calls = 0

    def ImageOps(self, *, device_index: int):  # mirrors the extension's own name
        self.device_index = device_index
        return self

    def scratch_bytes(self) -> dict[str, int]:
        return {"staging_ring": 0, "output": 0, "nms": 0}

    def letterbox_batch(self, images, dst_h, dst_w, mean, std, swap_rb, pad_value, stream):
        self.calls += 1
        tensor = np.zeros((len(images), 3, dst_h, dst_w), dtype=np.float32)
        return (tensor, *self._geometry(images, dst_h, dst_w))

    def letterbox_into(
        self, images, out_ptr, out_bytes, dst_h, dst_w, mean, std, swap_rb, pad_value, stream
    ):
        self.calls += 1
        return self._geometry(images, dst_h, dst_w)

    def _geometry(self, images, dst_h, dst_w):
        """The truth, plus whatever drift the test asked for."""
        scales = np.zeros(len(images), dtype=np.float32)
        pads = np.zeros((len(images), 2), dtype=np.float32)
        extents = np.zeros((len(images), 2), dtype=np.int32)
        for index, image in enumerate(images):
            plan = LetterboxGeometry.plan(image.shape[:2], (dst_h, dst_w))
            scales[index] = plan.scale + self.scale_delta
            pads[index] = (plan.pad_left + self.pad_delta, plan.pad_top + self.pad_delta)
            extents[index] = (
                plan.resized_height + self.extent_delta,
                plan.resized_width,
            )
        return scales, pads, extents


def build_native(monkeypatch, fake: FakeExtension):
    """A native backend whose extension is the fake. No device involved."""
    from shipvision.imgproc.backends import native_ops

    monkeypatch.setattr(native_ops, "_C", fake)
    return native_ops.NativeImageOps(device_index=0)


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
