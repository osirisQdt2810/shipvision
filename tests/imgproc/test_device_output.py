"""The device-output seam: asking a backend for a batch that never touches host memory.

``shipvision._C`` has had ``letterbox_into`` and ``crop_into`` since the beginning, and
``csrc/bindings/module.cpp`` calls the first of them "the production path" — but
:class:`~shipvision.imgproc.base.ImageOps` only had numpy-returning methods, so every consumer
went through the convenience entry points, which pay a device-to-host copy plus a
``gpuStreamSynchronize`` per call. ``grep -rn "letterbox_into" --include=*.py .`` returned
nothing at all. Measured on this box:

===========================================  ==========  ================
path                                         8x1080p     15 crops from
                                             -> 640^2    1080p -> 256x128
===========================================  ==========  ================
``ImageOps.letterbox`` / ``crop_batch``      44.7 ms     2.17 ms
``_C.letterbox_into`` / ``crop_into``         8.6 ms     0.76 ms
===========================================  ==========  ================

The fix is a seam, not a plumbing change: a capability a consumer can *ask* about
(:attr:`~shipvision.imgproc.base.ImageOps.supports_device_output`), two entry points that take
a caller-owned buffer, and a
:class:`~shipvision.imgproc.base.DeviceBuffer` descriptor that a detector can build from
whatever it already owns — a torch tensor or a TensorRT binding — without knowing which
backend is on the other side. The numpy backend refuses with a typed error rather than
pretending.

Most of this file runs with no GPU, deliberately: the descriptor's validation is where the
mistakes are (a buffer on the wrong device, a buffer too small, a strided tensor), and none of
them need a device to test.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from shipvision.errors import BackendUnavailableError, ConfigurationError
from shipvision.imgproc import IMGPROC
from shipvision.imgproc.base import DeviceBuffer, nchw_nbytes
from shipvision.registry import NATIVE, PYTHON, TORCH
from tests.imgproc.conftest import NATIVE_BUILT, TORCH_INSTALLED, backend_params

TARGET = (64, 64)


def fake_tensor(
    *,
    pointer: int = 0x1000,
    numel: int = 1024,
    element_size: int = 4,
    device_type: str = "cuda",
    device_index: int = 0,
    contiguous: bool = True,
    dtype: str = "torch.float32",
) -> object:
    """Something that quacks like a device tensor, without importing torch.

    :meth:`~shipvision.imgproc.base.DeviceBuffer.from_tensor` is duck-typed on purpose — the
    imgproc package must not import torch, and a TensorRT binding is not a torch tensor either
    — so the tests for it are duck-typed too, and run on a laptop.
    """
    return SimpleNamespace(
        data_ptr=lambda: pointer,
        numel=lambda: numel,
        element_size=lambda: element_size,
        is_contiguous=lambda: contiguous,
        device=SimpleNamespace(type=device_type, index=device_index),
        dtype=dtype,
    )


@pytest.fixture(params=backend_params())
def candidate(request):
    """One image-ops backend that this machine can actually build."""
    return IMGPROC.build("default", backend=request.param)


class TestTheCapabilityIsAskable:
    """A consumer asks the object, not the class name.

    This is the whole point of putting it on the ABC: ``if ops.supports_device_output`` is a
    question the detection package can ask about a backend it was handed by the registry,
    which ``isinstance(ops, NativeImageOps)`` is not — that would put the backend list back
    into every consumer.
    """

    def test_every_backend_answers(self, candidate) -> None:
        assert isinstance(candidate.supports_device_output, bool)

    def test_numpy_says_no(self) -> None:
        ops = IMGPROC.build("default", backend=PYTHON)

        assert ops.supports_device_output is False

    @pytest.mark.native
    @pytest.mark.skipif(not NATIVE_BUILT, reason="shipvision._C is not built, or no GPU")
    def test_the_native_backend_says_yes(self) -> None:
        ops = IMGPROC.build("default", backend=NATIVE)

        assert ops.supports_device_output is True

    @pytest.mark.skipif(not TORCH_INSTALLED, reason="torch and torchvision are not installed")
    def test_the_torch_backend_says_no_on_the_cpu(self) -> None:
        """A CPU tensor's pointer is not a device pointer, so the honest answer is no."""
        ops = IMGPROC.build("default", backend=TORCH, device="cpu")

        assert ops.supports_device_output is False


class TestABackendWithoutItRefusesTypedly:
    """Not by returning a numpy array, and not by raising ``NotImplementedError``.

    An untyped failure is the thing this library's error vocabulary exists to prevent: a caller
    has to be able to tell "this backend cannot do that" from "the GPU is gone", because one is
    a fall-back-to-numpy and the other is take-the-worker-out-of-rotation.
    """

    def test_letterbox_into_is_refused(self) -> None:
        ops = IMGPROC.build("default", backend=PYTHON)
        buffer = DeviceBuffer(pointer=0x1000, nbytes=1 << 20, device_index=0)

        with pytest.raises(BackendUnavailableError, match="device output"):
            ops.letterbox_into([np.zeros((8, 8, 3), dtype=np.uint8)], TARGET, buffer)

    def test_crop_batch_into_is_refused(self) -> None:
        ops = IMGPROC.build("default", backend=PYTHON)
        buffer = DeviceBuffer(pointer=0x1000, nbytes=1 << 20, device_index=0)

        with pytest.raises(BackendUnavailableError, match="device output"):
            ops.crop_batch_into(
                np.zeros((8, 8, 3), dtype=np.uint8),
                np.zeros((1, 4), dtype=np.float32),
                TARGET,
                buffer,
            )


class TestDeviceBuffer:
    """The descriptor, and the four ways a caller gets it wrong.

    All of them are silent if unchecked. A buffer that is too small overruns into whatever the
    allocator handed out next; a buffer on another device is a cross-device write that either
    faults or corrupts; a strided tensor is written as if it were contiguous; a float16 binding
    of the right byte count comes back as noise. None of these need a GPU to refuse.
    """

    def test_it_holds_what_a_kernel_needs(self) -> None:
        buffer = DeviceBuffer(pointer=0x2000, nbytes=4096, device_index=3)

        assert (buffer.pointer, buffer.nbytes, buffer.device_index) == (0x2000, 4096, 3)

    @pytest.mark.parametrize(
        ("pointer", "nbytes", "device_index"),
        [(0, 4096, 0), (-1, 4096, 0), (0x1000, 0, 0), (0x1000, -4, 0), (0x1000, 4096, -1)],
    )
    def test_a_nonsense_descriptor_is_refused(self, pointer, nbytes, device_index) -> None:
        with pytest.raises(ConfigurationError):
            DeviceBuffer(pointer=pointer, nbytes=nbytes, device_index=device_index)

    def test_it_is_built_from_anything_that_quacks_like_a_tensor(self) -> None:
        buffer = DeviceBuffer.from_tensor(fake_tensor(pointer=0x4000, numel=256))

        assert buffer.pointer == 0x4000
        assert buffer.nbytes == 256 * 4
        assert buffer.device_index == 0
        assert buffer.owner is not None

    def test_a_host_tensor_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="device"):
            DeviceBuffer.from_tensor(fake_tensor(device_type="cpu"))

    def test_a_strided_tensor_is_refused(self) -> None:
        """The kernel writes a dense NCHW block; a strided destination would be scrambled."""
        with pytest.raises(ConfigurationError, match="contiguous"):
            DeviceBuffer.from_tensor(fake_tensor(contiguous=False))

    def test_a_non_float32_tensor_is_refused(self) -> None:
        """A float16 binding with the right byte count is the dangerous case: the write
        succeeds and every value is garbage."""
        with pytest.raises(ConfigurationError, match="float32"):
            DeviceBuffer.from_tensor(fake_tensor(dtype="torch.float16", element_size=2))

    def test_an_object_that_is_not_a_tensor_at_all_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="data_ptr"):
            DeviceBuffer.from_tensor(np.zeros(4, dtype=np.float32))

    def test_require_accepts_an_exact_fit(self) -> None:
        buffer = DeviceBuffer(pointer=0x1000, nbytes=4096, device_index=1)

        buffer.require(4096, 1)

    def test_require_refuses_a_buffer_that_is_too_small(self) -> None:
        buffer = DeviceBuffer(pointer=0x1000, nbytes=4095, device_index=1)

        with pytest.raises(ConfigurationError, match="too small"):
            buffer.require(4096, 1)

    def test_require_refuses_a_buffer_on_another_device(self) -> None:
        """On a 16-GPU box this is the mistake that happens: the engine was built on device 5
        and the image ops were bound to device 0."""
        buffer = DeviceBuffer(pointer=0x1000, nbytes=4096, device_index=5)

        with pytest.raises(ConfigurationError, match="device 5"):
            buffer.require(4096, 0)


class TestOutputSizing:
    """A consumer has to be able to size the buffer before it asks for the work."""

    def test_the_byte_count_is_nchw_float32(self) -> None:
        assert nchw_nbytes(8, (640, 640)) == 8 * 3 * 640 * 640 * 4

    def test_an_empty_batch_needs_no_bytes(self) -> None:
        assert nchw_nbytes(0, (640, 640)) == 0

    def test_a_negative_count_is_refused(self) -> None:
        with pytest.raises(ConfigurationError):
            nchw_nbytes(-1, (64, 64))


@pytest.mark.native
@pytest.mark.skipif(not NATIVE_BUILT, reason="shipvision._C is not built, or there is no GPU")
@pytest.mark.skipif(not TORCH_INSTALLED, reason="the test needs torch to allocate the buffer")
class TestNativeDeviceOutput:
    """The fast path writes the same numbers as the slow one.

    Same numbers, not merely a plausible tensor: the whole risk of a second entry point is that
    it diverges from the one the parity suite checks, and then only the production path is
    wrong. So this compares against ``letterbox``/``crop_batch`` on the same backend, exactly.
    """

    def test_letterbox_into_matches_the_host_returning_path(self) -> None:
        import torch

        ops = IMGPROC.build("default", backend=NATIVE, device_index=0)
        rng = np.random.default_rng(5)
        frames = [
            rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
            for h, w in [(480, 640), (1077, 1920), (37, 53)]
        ]
        expected, expected_geometry = ops.letterbox(frames, TARGET)
        out = torch.empty(
            nchw_nbytes(len(frames), TARGET) // 4, dtype=torch.float32, device="cuda:0"
        )

        geometry = ops.letterbox_into(frames, TARGET, DeviceBuffer.from_tensor(out))

        assert geometry == expected_geometry
        actual = out.cpu().numpy().reshape(len(frames), 3, *TARGET)
        assert np.array_equal(actual, expected)

    def test_crop_batch_into_matches_the_host_returning_path(self) -> None:
        import torch

        ops = IMGPROC.build("default", backend=NATIVE, device_index=0)
        rng = np.random.default_rng(6)
        frame = rng.integers(0, 256, size=(720, 1280, 3), dtype=np.uint8)
        boxes = np.array(
            [[10.0, 20.0, 110.0, 220.0], [-5.0, -5.0, 60.0, 60.0], [7.0, 7.0, 7.0, 7.0]],
            dtype=np.float32,
        )
        expected = ops.crop_batch(frame, boxes, TARGET)
        out = torch.empty(
            nchw_nbytes(len(boxes), TARGET) // 4, dtype=torch.float32, device="cuda:0"
        )

        ops.crop_batch_into(frame, boxes, TARGET, DeviceBuffer.from_tensor(out))

        actual = out.cpu().numpy().reshape(len(boxes), 3, *TARGET)
        assert np.array_equal(actual, expected)

    def test_a_buffer_that_is_too_small_is_refused_before_the_launch(self) -> None:
        """Before, not during: an overrun inside the kernel is an illegal access, which is
        sticky and takes the worker down for the life of the process."""
        import torch

        ops = IMGPROC.build("default", backend=NATIVE, device_index=0)
        frames = [np.zeros((16, 16, 3), dtype=np.uint8)] * 2
        out = torch.empty(16, dtype=torch.float32, device="cuda:0")

        with pytest.raises(ConfigurationError, match="too small"):
            ops.letterbox_into(frames, TARGET, DeviceBuffer.from_tensor(out))

        assert np.isfinite(ops.letterbox(frames, TARGET)[0]).all()

    def test_a_buffer_on_another_device_is_refused(self) -> None:
        """The 16-GPU mistake: the engine was built on one device and the ops bound to
        another. Skipped rather than marked, because ``multigpu`` is not a marker this
        repository declares and adding one is a change to the test configuration."""
        import torch

        if torch.cuda.device_count() < 2:
            pytest.skip("needs two devices")
        ops = IMGPROC.build("default", backend=NATIVE, device_index=0)
        out = torch.empty(nchw_nbytes(1, TARGET) // 4, dtype=torch.float32, device="cuda:1")

        with pytest.raises(ConfigurationError, match="device 1"):
            ops.letterbox_into(
                [np.zeros((16, 16, 3), dtype=np.uint8)], TARGET, DeviceBuffer.from_tensor(out)
            )

    def test_the_empty_crop_batch_writes_nothing_and_does_not_raise(self) -> None:
        import torch

        ops = IMGPROC.build("default", backend=NATIVE, device_index=0)
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        out = torch.zeros(8, dtype=torch.float32, device="cuda:0")

        ops.crop_batch_into(frame, np.zeros((0, 4), dtype=np.float32), TARGET, out_buffer(out))

        assert out.cpu().numpy().tolist() == [0.0] * 8


def out_buffer(tensor) -> DeviceBuffer:
    """``DeviceBuffer.from_tensor``, named for what the call site is doing."""
    return DeviceBuffer.from_tensor(tensor)


@pytest.mark.gpu
@pytest.mark.skipif(not TORCH_INSTALLED, reason="torch and torchvision are not installed")
class TestTorchDeviceOutput:
    """The torch backend supports it too, through the tensor rather than the raw pointer.

    It cannot build a tensor over a foreign device pointer in pure Python, so it writes through
    the object the descriptor came from — which is why
    :meth:`~shipvision.imgproc.base.DeviceBuffer.from_tensor` keeps a reference to it. A caller
    holding only a TensorRT pointer gets a typed refusal that says so, rather than a host round
    trip nobody asked for.
    """

    def test_it_matches_its_own_host_returning_path(self) -> None:
        import torch

        if not torch.cuda.is_available():
            pytest.skip("no CUDA device for torch")
        ops = IMGPROC.build("default", backend=TORCH, device="cuda:0")
        rng = np.random.default_rng(8)
        frames = [rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)]
        expected, expected_geometry = ops.letterbox(frames, TARGET)
        out = torch.empty(nchw_nbytes(1, TARGET) // 4, dtype=torch.float32, device="cuda:0")

        geometry = ops.letterbox_into(frames, TARGET, DeviceBuffer.from_tensor(out))

        assert geometry == expected_geometry
        assert np.array_equal(out.cpu().numpy().reshape(1, 3, *TARGET), expected)

    def test_a_raw_pointer_without_its_tensor_is_refused(self) -> None:
        import torch

        if not torch.cuda.is_available():
            pytest.skip("no CUDA device for torch")
        ops = IMGPROC.build("default", backend=TORCH, device="cuda:0")
        out = torch.empty(nchw_nbytes(1, TARGET) // 4, dtype=torch.float32, device="cuda:0")
        pointer_only = DeviceBuffer(
            pointer=out.data_ptr(), nbytes=out.numel() * 4, device_index=0
        )

        with pytest.raises(BackendUnavailableError, match="from_tensor"):
            ops.letterbox_into([np.zeros((16, 16, 3), dtype=np.uint8)], TARGET, pointer_only)


class TestAConsumerCanAskThroughTheRegistry:
    """The adoption pattern, written out as the detection package would use it.

    No ``isinstance``, no backend name, no ``try: import _C``. The consumer asks the capability
    and takes one of two routes, and both routes produce the same tensor — which is what makes
    it safe for a detector to prefer the fast one without a per-backend branch anywhere in its
    own code.
    """

    @staticmethod
    def preprocess(ops, frames, target_hw, buffer=None):
        """What a detector's ``__call__`` would do."""
        if ops.supports_device_output and buffer is not None:
            return ops.letterbox_into(frames, target_hw, buffer), None
        batch, geometry = ops.letterbox(frames, target_hw)
        return geometry, batch

    def test_a_numpy_only_host_takes_the_host_route(self, candidate) -> None:
        frames = [np.zeros((16, 24, 3), dtype=np.uint8)]

        geometry, batch = self.preprocess(candidate, frames, TARGET)

        assert len(geometry) == 1
        assert batch is not None
        assert batch.shape == (1, 3, *TARGET)

    @pytest.mark.native
    @pytest.mark.skipif(
        not NATIVE_BUILT, reason="shipvision._C is not built, or there is no GPU"
    )
    @pytest.mark.skipif(not TORCH_INSTALLED, reason="the test needs torch to allocate")
    def test_a_device_capable_backend_takes_the_device_route(self) -> None:
        import torch

        ops = IMGPROC.build("default", backend=NATIVE, device_index=0)
        frames = [np.zeros((16, 24, 3), dtype=np.uint8)]
        out = torch.empty(nchw_nbytes(1, TARGET) // 4, dtype=torch.float32, device="cuda:0")

        geometry, batch = self.preprocess(ops, frames, TARGET, DeviceBuffer.from_tensor(out))

        assert batch is None, "the device route must not materialise a host tensor"
        assert len(geometry) == 1
        assert torch.isfinite(out).all()
