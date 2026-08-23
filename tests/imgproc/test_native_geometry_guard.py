"""The guard that catches C++ and Python rounding the letterbox differently.

``NativeImageOps`` computes the scale, the pads *and* the resized extent through
:meth:`~shipvision.imgproc.geometry.LetterboxGeometry.plan` and checks them against what the
extension reports, because two implementations of a rounding rule eventually drift and the
symptom — boxes off by a pixel on the cameras whose resolution lands on a boundary — is close
to undebuggable after the fact.

It was checking the wrong numbers. It compared ``scale``, ``pad_left`` and ``pad_top``, and the
number that decides the sampling ratio is the resized extent: the kernel samples
``(y + 0.5) * view.height / view.out_h - 0.5``. A one-pixel disagreement in ``out_h`` hides
behind an equal pad, because ``pad = (T - r) // 2`` is the same for ``r`` and ``r + 1``
whenever ``T - r`` is even — a 7-row source at scale 0.5 into a 100-row canvas gives Python
``r = 4`` and a truncating C++ ``r = 3``, and both pad to 48. The guard passed and every row
of that camera was sampled from the wrong ratio.

The whole file runs with no GPU. The translator is Python, the guard is Python, and the only
thing that needed a device was the extension itself — so a fake stands in for it, which is how
this class of bug gets caught on a laptop instead of on a release candidate.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipvision.errors import InferenceError
from shipvision.imgproc.backends import native_ops
from shipvision.imgproc.geometry import LetterboxGeometry
from tests.imgproc.conftest import NATIVE_BUILT

DRIFT_SOURCE = (7, 200)
"""7 rows at scale 0.5 is 3.5 rows, which rounds half up to 4 — the boundary case."""

DRIFT_TARGET = (100, 100)
"""100 - 4 is even, so ``r`` and ``r - 1`` pad identically and the old guard saw nothing."""


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


@pytest.fixture()
def frame() -> np.ndarray:
    return np.zeros((*DRIFT_SOURCE, 3), dtype=np.uint8)


def build(monkeypatch, fake: FakeExtension) -> native_ops.NativeImageOps:
    """A native backend whose extension is the fake. No device involved."""
    monkeypatch.setattr(native_ops, "_C", fake)
    return native_ops.NativeImageOps(device_index=0)


class TestTheGuardComparesTheResizedExtent:
    """A one-pixel drift in ``out_h`` must be a loud failure, not a silent resample."""

    def test_a_drifted_extent_is_refused(self, monkeypatch, frame) -> None:
        ops = build(monkeypatch, FakeExtension(extent_delta=-1))

        with pytest.raises(InferenceError, match="resized extent"):
            ops.letterbox(frame, DRIFT_TARGET)

    def test_the_scale_and_the_pads_still_agree_in_that_case(self, monkeypatch, frame) -> None:
        """The evidence that the extent is doing the work.

        If this case were caught by the old scale-and-pad comparison there would be nothing to
        fix, so the test asserts that those two numbers match while the extent does not — the
        exact blind spot, written down.
        """
        fake = FakeExtension(extent_delta=-1)
        geometry = LetterboxGeometry.plan(DRIFT_SOURCE, DRIFT_TARGET)
        scales, pads, extents = fake._geometry([np.zeros((*DRIFT_SOURCE, 3))], *DRIFT_TARGET)

        assert float(scales[0]) == pytest.approx(geometry.scale)
        assert (int(pads[0][0]), int(pads[0][1])) == (geometry.pad_left, geometry.pad_top)
        assert geometry.resized_height == 4
        assert int(extents[0][0]) == 3
        assert (DRIFT_TARGET[0] - 4) // 2 == (DRIFT_TARGET[0] - 3) // 2 == 48

    def test_a_drifted_extent_is_refused_on_the_device_output_path_too(
        self, monkeypatch, frame
    ) -> None:
        """``letterbox_into`` is the production path, so it cannot be the unguarded one."""
        from shipvision.imgproc.base import DeviceBuffer

        ops = build(monkeypatch, FakeExtension(extent_delta=1))
        buffer = DeviceBuffer(pointer=0x1000, nbytes=1 << 24, device_index=0)

        with pytest.raises(InferenceError, match="resized extent"):
            ops.letterbox_into(frame, DRIFT_TARGET, buffer)

    def test_an_agreeing_extension_is_accepted(self, monkeypatch, frame) -> None:
        """The guard must not be a tripwire on the ordinary case."""
        fake = FakeExtension()
        ops = build(monkeypatch, fake)

        batch, geometries = ops.letterbox(frame, DRIFT_TARGET)

        assert batch.shape == (1, 3, *DRIFT_TARGET)
        assert geometries[0].resized_height == 4
        assert fake.calls == 1


class TestTheOlderChecksAreStillThere:
    """Adding a comparison must not remove the two that were already right."""

    def test_a_drifted_scale_is_refused(self, monkeypatch, frame) -> None:
        ops = build(monkeypatch, FakeExtension(scale_delta=0.01))

        with pytest.raises(InferenceError, match="scale"):
            ops.letterbox(frame, DRIFT_TARGET)

    def test_a_drifted_pad_is_refused(self, monkeypatch, frame) -> None:
        ops = build(monkeypatch, FakeExtension(pad_delta=1))

        with pytest.raises(InferenceError, match="pad"):
            ops.letterbox(frame, DRIFT_TARGET)

    def test_the_message_names_the_image(self, monkeypatch) -> None:
        """A ragged batch of fifty cameras needs to say *which* one, because the answer is
        almost always one resolution rather than the build."""
        ops = build(monkeypatch, FakeExtension(extent_delta=-1))
        frames = [np.zeros((16, 16, 3), dtype=np.uint8), np.zeros((*DRIFT_SOURCE, 3), np.uint8)]

        with pytest.raises(InferenceError, match="image 0"):
            ops.letterbox(frames, DRIFT_TARGET)


@pytest.mark.native
@pytest.mark.skipif(not NATIVE_BUILT, reason="shipvision._C is not built, or there is no GPU")
class TestTheRealExtensionAgrees:
    """And the real one, on the shapes where rounding is least obvious.

    The fake proves the guard fires; this proves there is nothing for it to fire on. 1077x1920
    is the case that rounds (287.2 rows), 1079x1919 the case whose byte count is odd, and 7x200
    the half-way case the fake uses.
    """

    @pytest.mark.parametrize(
        "source_hw", [(1080, 1920), (1077, 1920), (1079, 1919), (7, 200), (1, 1), (37, 53)]
    )
    def test_the_extension_reports_the_extent_python_derives(self, source_hw) -> None:
        from shipvision import _C

        ops = _C.ImageOps(device_index=0)
        image = np.zeros((*source_hw, 3), dtype=np.uint8)
        target = (100, 100)

        _, scales, pads, extents = ops.letterbox_batch(
            [image], *target, [0.0] * 3, [255.0] * 3, True, 114, 0
        )

        expected = LetterboxGeometry.plan(source_hw, target)
        assert int(extents[0][0]) == expected.resized_height
        assert int(extents[0][1]) == expected.resized_width
        assert float(scales[0]) == pytest.approx(expected.scale, abs=1e-6)
        assert (int(pads[0][0]), int(pads[0][1])) == (expected.pad_left, expected.pad_top)
