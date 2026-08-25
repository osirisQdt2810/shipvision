"""`ImageOps` binds its device before any member that creates a CUDA resource.

Members are constructed in declaration order, *before* the constructor body. The staging ring
creates three CUDA events in its constructor, and an event belongs to whichever device is
current when it is created. With `gpuSetDevice` in the constructor body, every event was
created on the thread's default device and then recorded on this instance's stream — which
CUDA reports as `invalid resource handle` on any device other than 0.

That is a construction-order bug wearing a letterbox costume: `crop_batch` and `nms` never
record the slot event, so they worked, and the failure looked specific to letterbox. It cost the
first native letterbox measurement on this project. The fix is a `BoundDevice` member declared
before the ring, so the ordering is a guarantee of the language and not of a comment.

Two tests, because the two things that can go wrong are different: the source can lose the
ordering (checked everywhere, no GPU), and a build can still misbehave on a second device
(checked where there is one).
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

#: Self-contained rather than imported from `tests/imgproc/conftest.py`, because this file
#: ships with the csrc piece and that conftest ships with the imgproc piece; the two are
#: separate pull requests and this must collect on a branch that has only one of them.
NATIVE_BUILT = importlib.util.find_spec("shipvision._C") is not None

REPO = Path(__file__).resolve().parents[2]
BINDINGS = REPO / "csrc" / "bindings" / "module.cpp"


class TestTheDeviceIsBoundBeforeTheRingIsBuilt:
    def test_bound_device_is_declared_before_the_staging_ring(self) -> None:
        source = BINDINGS.read_text()
        bound = source.index("BoundDevice bound_;")
        ring = source.index("StagingRing ring_;")

        assert bound < ring, (
            "BoundDevice must be declared before ring_: members are constructed in "
            "declaration order, and the ring creates CUDA events on whichever device is "
            "current at that moment"
        )

    def test_the_constructor_body_does_not_set_the_device_itself(self) -> None:
        """If it did, someone could later 'simplify' by removing the member and keeping the
        body call — which is exactly the shape that was broken."""
        source = BINDINGS.read_text()
        ctor = re.search(r"explicit ImageOps\(int device_index\)(.*?)\{\}", source, re.S)

        assert ctor is not None, "the ImageOps constructor should have an empty body"
        assert "gpuSetDevice" not in ctor.group(0)


@pytest.mark.gpu
@pytest.mark.skipif(not NATIVE_BUILT, reason="shipvision._C is not built")
class TestASecondDeviceIsUsableFromAThreadOnTheFirst:
    def test_letterbox_on_device_one_from_a_device_zero_thread(self) -> None:
        import numpy as np

        from shipvision import _C

        if _C.device_count() < 2:
            pytest.skip("needs two devices to tell a device-0 event from a device-1 stream")
        # The calling thread's current device is 0 — the default — which is the condition
        # under which the old code created the ring's events on the wrong device.
        ops = _C.ImageOps(1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        tensor, _scales, _pads, extents = ops.letterbox_batch(
            [frame], 320, 320, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], False, 114, 0
        )

        assert tensor.shape == (1, 3, 320, 320)
        assert extents.shape == (1, 2)


class TestTheNmsMaskIsStagedThroughPinnedMemory:
    """One allocation was the whole gap between this kernel and torchvision's.

    `nms()` used to download the `(n, ceil(n/64))` overlap mask into a *fresh*
    `std::vector<unsigned long long>` on every call — up to 78 MB of pageable memory whose
    pages the OS faulted in while the DMA was writing them. Measured on the dev box for the
    18 764-candidate shape: 30.8 ms for that copy, 5.6 ms into reused pageable memory, 1.7 ms
    into pinned. Native NMS benchmarked at 33 ms against torchvision's 2 ms and it was never the
    mask kernel or the sweep. Through the instance's pinned download buffer it is 6.1 ms at
    25 000 candidates against torchvision's 12.1 ms on realistic boxes — 2x faster at every n
    tried, from 0.14 ms at 2 000.

    Pinned structurally rather than by a timing assertion, because a timing test on a shared
    box is a coin. The two facts that must stay true: the library function accepts a pinned
    destination, and the binding hands it one.
    """

    IMAGE_OPS_CU = REPO / "csrc" / "shipvision" / "imgproc" / "image_ops.cu"

    @staticmethod
    def _code(path: Path) -> str:
        """Source with `//` comment lines dropped, so prose about the old shape is not evidence
        of it. The comment explaining this fix quotes the very line the test forbids — and the
        first run of this test failed on that quotation, which is the trap exactly."""
        return "\n".join(
            line for line in path.read_text().splitlines() if not line.strip().startswith("//")
        )

    def test_the_library_downloads_into_the_callers_pinned_buffer(self) -> None:
        source = self._code(self.IMAGE_OPS_CU)
        body = source[source.index("std::vector<int64_t> nms(") :]

        assert "scratch.host_mask" in body, "nms() ignores the pinned destination"
        assert (
            "std::vector<unsigned long long> mask(mask_words)" not in body
        ), "the fresh pageable allocation is back; it cost 31 ms per call at 18k candidates"

    def test_the_binding_supplies_one_from_the_pinned_download_scratch(self) -> None:
        source = self._code(BINDINGS)
        run_nms = source[source.index("std::vector<int64_t> run_nms(") :]
        run_nms = run_nms[: run_nms.index("\n            }\n")]

        assert "scratch.host_mask" in run_nms
        assert "pinned_download_.reserve" in run_nms, (
            "run_nms must stage the mask through pinned memory, not leave host_mask null and "
            "fall back to the pageable path"
        )
