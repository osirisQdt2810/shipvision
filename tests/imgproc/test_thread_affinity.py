"""One native instance serves one thread, and says so instead of racing.

``NativeImageOps`` documents it: "one instance owns one device index and one staging ring, so
it belongs to one worker thread for the life of the process. Sharing it between threads races
the ring, and the result is plausible-looking output from the wrong frame". That was prose
only. Nothing enforced it, and the failure is the worst kind this library has a name for — a
mis-tagged result rather than a dropped one, a real-looking detection on a camera where
nothing happened.

Measured on this box with two threads sharing one instance, 300 frames each from cameras
filled with 17 and 200: every single call failed with ``GpuError: gpuEventSynchronize failed:
an illegal memory access``, and that error is sticky, so the process was finished. The review
that found this observed the other outcome on its own box — 11 of 300 and 242 of 300 frames
carrying the *other* camera's pixels, with no error at all. Same race, two symptoms, and the
silent one is worse.

Ownership is claimed on first use rather than at construction, which is a deliberate
difference: what has to be protected is the staging ring, and the ring is only touched when a
method runs. Claiming in ``__init__`` would break the ordinary pattern of building a component
on the thread that assembles the pipeline and running it on the worker — a false positive on
every frame — while catching nothing extra.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from shipvision.errors import InferenceError
from shipvision.imgproc import IMGPROC, DeviceBuffer
from shipvision.registry import NATIVE
from tests.imgproc.conftest import NATIVE_BUILT, FakeExtension, build_native

TARGET = (32, 32)


def frame(fill: int = 0) -> np.ndarray:
    return np.full((16, 24, 3), fill, dtype=np.uint8)


def call_from_a_new_thread(work) -> BaseException | None:
    """Run ``work`` on another thread and hand back whatever it raised, or ``None``."""
    captured: list[BaseException] = []

    def target() -> None:
        try:
            work()
        except BaseException as exc:  # the exception *is* the result this helper returns
            captured.append(exc)

    thread = threading.Thread(target=target, name="foreign-worker")
    thread.start()
    thread.join()
    return captured[0] if captured else None


class TestOneInstanceOneThread:
    """A foreign call is a typed failure, not a race. Tested with no device at all.

    The affinity check is pure Python in the translator, so the fake extension is enough — and
    that matters, because this is a bug the offline tier can now catch on a laptop instead of
    it needing sixteen GPUs and a stress loop.
    """

    def test_a_second_thread_is_refused(self, monkeypatch) -> None:
        ops = build_native(monkeypatch, FakeExtension())
        ops.letterbox(frame(), TARGET)

        error = call_from_a_new_thread(lambda: ops.letterbox(frame(), TARGET))

        assert isinstance(error, InferenceError)

    def test_the_message_names_both_threads(self, monkeypatch) -> None:
        """ "Wrong thread" is only actionable if it says which one owns it: on a 50-camera box
        the answer is which worker leaked its ops, and thread names are how that is found."""
        ops = build_native(monkeypatch, FakeExtension())
        ops.letterbox(frame(), TARGET)

        error = call_from_a_new_thread(lambda: ops.letterbox(frame(), TARGET))

        assert "foreign-worker" in str(error)
        assert threading.current_thread().name in str(error)

    def test_the_owner_can_keep_calling(self, monkeypatch) -> None:
        ops = build_native(monkeypatch, FakeExtension())

        for _ in range(5):
            batch, _ = ops.letterbox(frame(), TARGET)

        assert batch.shape == (1, 3, *TARGET)

    def test_the_first_thread_to_use_it_owns_it_not_the_one_that_built_it(
        self, monkeypatch
    ) -> None:
        """The construct-here-run-there pattern is legitimate and must keep working.

        A pipeline is assembled on one thread and run on a worker; the ring is untouched until
        the first call, so the first caller is the owner. This is the case that would fail if
        the thread were recorded in ``__init__``.
        """
        ops = build_native(monkeypatch, FakeExtension())

        assert call_from_a_new_thread(lambda: ops.letterbox(frame(), TARGET)) is None
        with pytest.raises(InferenceError):
            ops.letterbox(frame(), TARGET)

    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(lambda ops: ops.letterbox(frame(), TARGET), id="letterbox"),
            pytest.param(
                lambda ops: ops.crop_batch(
                    frame(), np.array([[0.0, 0.0, 8.0, 8.0]], dtype=np.float32), TARGET
                ),
                id="crop_batch",
            ),
            pytest.param(
                lambda ops: ops.nms(
                    np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
                    np.array([0.9], dtype=np.float32),
                    iou_threshold=0.5,
                ),
                id="nms",
            ),
            pytest.param(
                lambda ops: ops.letterbox_into(
                    frame(),
                    TARGET,
                    DeviceBuffer(pointer=0x1000, nbytes=1 << 20, device_index=0),
                ),
                id="letterbox_into",
            ),
            pytest.param(
                lambda ops: ops.crop_batch_into(
                    frame(),
                    np.array([[0.0, 0.0, 8.0, 8.0]], dtype=np.float32),
                    TARGET,
                    DeviceBuffer(pointer=0x1000, nbytes=1 << 20, device_index=0),
                ),
                id="crop_batch_into",
            ),
        ],
    )
    def test_every_entry_point_that_touches_the_scratch_is_guarded(
        self, monkeypatch, call
    ) -> None:
        """Including NMS, whose scratch is also per-instance, and both ``_into`` paths.

        One guarded method and one unguarded one is worse than none: the unguarded one becomes
        the way the race gets in, and it is the fast path a tuned pipeline reaches for.
        """
        ops = build_native(monkeypatch, FakeExtension())
        ops.letterbox(frame(), TARGET)

        assert isinstance(call_from_a_new_thread(lambda: call(ops)), InferenceError)


@pytest.mark.native
@pytest.mark.skipif(not NATIVE_BUILT, reason="shipvision._C is not built, or there is no GPU")
class TestTheRealBackendRefusesTheRace:
    """The same guard against the real extension, on the workload that used to break it."""

    def test_two_threads_sharing_one_instance_get_a_typed_failure(self) -> None:
        """Not a ``GpuError``, and not another camera's pixels.

        Two cameras with distinct uniform fills, so any frame carrying the wrong one is
        obvious from a single pixel. The assertion is that the shared instance refuses the
        second thread rather than producing anything at all — and that whatever *is* produced
        is that thread's own camera.
        """
        ops = IMGPROC.build("default", backend=NATIVE, device_index=0)
        fills = (17, 200)
        refused: list[int] = []
        wrong_camera: list[int] = []
        lock = threading.Lock()

        def worker(fill: int) -> None:
            image = np.full((240, 320, 3), fill, dtype=np.uint8)
            for _ in range(20):
                try:
                    batch, _ = ops.letterbox(image, TARGET, pad_value=fill)
                except InferenceError:
                    with lock:
                        refused.append(fill)
                    continue
                if round(float(batch[0, 0, 16, 16]) * 255) != fill:
                    with lock:
                        wrong_camera.append(fill)

        threads = [threading.Thread(target=worker, args=(fill,)) for fill in fills]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert wrong_camera == [], "a frame came back holding the other camera's pixels"
        assert len(refused) == 20, "the non-owning thread must be refused on every call"

    def test_one_instance_per_thread_is_the_supported_shape(self) -> None:
        """The fix must not make the correct pattern harder: two threads, two instances, two
        cameras, and each one sees its own frames."""
        results: dict[int, list[int]] = {17: [], 200: []}
        lock = threading.Lock()

        def worker(fill: int) -> None:
            ops = IMGPROC.build("default", backend=NATIVE, device_index=0)
            image = np.full((240, 320, 3), fill, dtype=np.uint8)
            seen = [
                round(
                    float(ops.letterbox(image, TARGET, pad_value=fill)[0][0, 0, 16, 16]) * 255
                )
                for _ in range(20)
            ]
            with lock:
                results[fill] = seen

        threads = [threading.Thread(target=worker, args=(fill,)) for fill in (17, 200)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert results[17] == [17] * 20
        assert results[200] == [200] * 20
