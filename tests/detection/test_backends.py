"""The runtimes: what happens when they are absent, and what happens when they are there.

Two claims matter more than the rest. A missing runtime must be a
:class:`~shipvision.errors.BackendUnavailableError` and not an ``ImportError``, because the two
say different things to whoever is on call: one is "install this", the other is a stack trace
they have to read the source to interpret. And the package must import on a machine with
neither, or the registry could not list a backend that machine cannot run.
"""

from __future__ import annotations

import sys
import textwrap

import numpy as np
import pytest

from shipvision.detection import DETECTORS
from shipvision.errors import (
    BackendUnavailableError,
    ConfigurationError,
    ModelLoadError,
)
from shipvision.registry import TENSORRT, TORCH

from .conftest import geometry, grey_frame, to_network_space

SMALL_SOURCE = (270, 480)
SMALL_NETWORK = (64, 64)

#: Boxes chosen in source pixels. Every end-to-end assertion below is that these come back.
WANTED = np.array([[40.0, 30.0, 160.0, 190.0], [300.0, 100.0, 460.0, 260.0]], dtype=np.float32)


class TestPackageImports:
    """``import shipvision.detection`` costs nothing and needs nothing."""

    def test_importing_the_package_does_not_import_torch_or_tensorrt(self) -> None:
        """Checked in a clean interpreter, because this one has already imported plenty."""
        script = textwrap.dedent(
            """
            import sys
            import shipvision.detection as detection
            assert detection.DETECTORS.names() == ["mock", "yolo26"], detection.DETECTORS.names()
            assert "torch" not in sys.modules, "importing the package imported torch"
            assert "tensorrt" not in sys.modules, "importing the package imported tensorrt"
            print("ok")
            """
        )
        result = _run(script)

        assert result.stdout.strip().endswith("ok"), result.stderr

    def test_the_package_imports_with_both_runtimes_unimportable(self) -> None:
        """`None` in ``sys.modules`` is exactly what an unimportable module looks like to
        ``import``, so this is the real path on a machine that happens to have them."""
        script = textwrap.dedent(
            """
            import sys
            sys.modules["torch"] = None
            sys.modules["tensorrt"] = None

            import shipvision.detection as detection
            from shipvision.errors import BackendUnavailableError

            # The classes still resolve: registration is not availability.
            assert detection.DETECTORS.get("yolo26", "tensorrt").__name__ == "TensorRTDetector"
            assert detection.DETECTORS.get("yolo26", "torch").__name__ == "TorchDetector"

            for backend, extra in (("tensorrt", {}), ("torch", {"input_hw": (64, 64)})):
                try:
                    detection.DETECTORS.build(
                        "yolo26", backend=backend, path="absent.bin", **extra
                    )
                except BackendUnavailableError:
                    pass
                else:
                    raise AssertionError(backend + " did not report itself unavailable")
            print("ok")
            """
        )
        result = _run(script)

        assert result.stdout.strip().endswith("ok"), result.stderr

    def test_the_tensorrt_modules_import_with_no_tensorrt_installed(self) -> None:
        """They must, or the registry could not list a backend this machine cannot run."""
        import importlib

        for module in (
            "shipvision.detection.backends.tensorrt.bindings",
            "shipvision.detection.backends.tensorrt.calibration",
            "shipvision.detection.backends.tensorrt.engine",
            "shipvision.detection.engine_build",
        ):
            assert importlib.import_module(module) is not None


class TestBackendAvailability:
    """A missing runtime is a typed error naming what to install."""

    @pytest.mark.parametrize(
        ("backend", "absent", "extra"),
        [
            (TENSORRT, "tensorrt", {}),
            (TORCH, "torch", {"input_hw": SMALL_NETWORK}),
        ],
    )
    def test_a_missing_runtime_is_a_typed_error_not_an_import_error(
        self, backend, absent, extra, monkeypatch, tmp_path
    ) -> None:
        monkeypatch.setitem(sys.modules, absent, None)

        with pytest.raises(BackendUnavailableError, match=absent):
            DETECTORS.build(
                "yolo26", backend=backend, path=tmp_path / "absent.engine", **extra
            )

    def test_building_an_engine_without_tensorrt_says_so(self, monkeypatch, tmp_path) -> None:
        from shipvision.detection import OptimisationProfile, build_engine

        monkeypatch.setitem(sys.modules, "tensorrt", None)
        onnx = tmp_path / "model.onnx"
        onnx.write_bytes(b"not really an onnx")

        with pytest.raises(BackendUnavailableError, match="tensorrt"):
            build_engine(
                onnx,
                tmp_path / "model.engine",
                profiles=[OptimisationProfile.for_batch(input_hw=(64, 64), max_batch=2)],
            )

    def test_yolo26_seg_is_an_alias_of_yolo26_rather_than_a_second_entry(self) -> None:
        """Which head runs is read off the artefact, so the name does not have to say."""
        assert DETECTORS.get("yolo26_seg", TENSORRT) is DETECTORS.get("yolo26", TENSORRT)


@pytest.fixture(scope="module")
def torch_module():
    return pytest.importorskip("torch", reason="the torch detector needs torch")


@pytest.fixture
def scripted_detector(torch_module, tmp_path):
    """A TorchScript artefact returning boxes we chose, in network space.

    Fixed output rather than a real network on purpose: what is under test is the path — the
    letterbox, the execution, the decode and the inverse — and a real network's boxes would have
    no known-correct answer to assert against.
    """
    torch = torch_module
    network_boxes = to_network_space(WANTED, geometry(SMALL_SOURCE, SMALL_NETWORK))

    class FixedDetector(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            proposals = torch.zeros(1, 2, 6)
            proposals[0, :, :4] = torch.from_numpy(network_boxes)
            proposals[0, :, 4] = torch.tensor([0.9, 0.75])
            proposals[0, :, 5] = torch.tensor([0.0, 2.0])
            self.register_buffer("proposals", proposals)

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            # Touches the input so a wrong-shaped batch is still a wrong-shaped batch, then
            # returns the fixed proposals once per row.
            scale = images.mean() * 0.0 + 1.0
            return self.proposals.expand(images.shape[0], -1, -1) * scale

    path = tmp_path / "fixed.ts"
    torch.jit.script(FixedDetector()).save(str(path))
    return path


@pytest.fixture
def scripted_segmenter(torch_module, tmp_path):
    """An artefact returning ``(detections, prototypes)`` — a nested tuple, as exporters do."""
    torch = torch_module

    class FixedSegmenter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("proposals", torch.zeros(1, 2, 6 + 4))
            self.register_buffer("prototypes", torch.zeros(1, 4, 16, 16))

        def forward(self, images: torch.Tensor):
            rows = images.shape[0]
            return self.proposals.expand(rows, -1, -1), (self.prototypes,)

    path = tmp_path / "seg.ts"
    torch.jit.script(FixedSegmenter()).save(str(path))
    return path


@pytest.mark.slow
class TestTorchDetector:
    """The whole path, on the CPU: letterbox, execute, decode, invert."""

    def build(self, path, **kwargs):
        return DETECTORS.build(
            "yolo26",
            backend=TORCH,
            path=path,
            input_hw=SMALL_NETWORK,
            image_ops_backend="python",
            head_options={"conf_threshold": 0.1},
            **kwargs,
        )

    def test_the_boxes_come_back_in_source_pixels(self, scripted_detector) -> None:
        detector = self.build(scripted_detector)

        result = detector.detect_one(grey_frame("cam-01", 4, SMALL_SOURCE))

        assert result.tag.camera_id == "cam-01"
        assert (result.height, result.width) == SMALL_SOURCE
        assert np.abs(result.boxes - WANTED).max() < 1e-2
        assert result.class_ids.tolist() == [0, 2]

    def test_it_reports_the_extent_it_was_probed_at(self, scripted_detector) -> None:
        assert self.build(scripted_detector).input_hw == SMALL_NETWORK

    def test_it_names_itself_after_the_registry_entry(self, scripted_detector) -> None:
        detector = self.build(scripted_detector)

        assert (detector.name, detector.backend) == ("yolo26", "torch")

    def test_a_batch_larger_than_max_batch_is_chunked_and_stays_aligned(
        self, scripted_detector
    ) -> None:
        """Chunking happens above the backend, so a caller may hand over any number of frames
        while the device buffers stay sized once."""
        detector = self.build(scripted_detector, max_batch=2)
        frames = [grey_frame(f"cam-{i:02d}", i, SMALL_SOURCE) for i in range(5)]

        results = detector.detect(frames)

        assert [r.tag for r in results] == [f.tag for f in frames]
        assert all(len(r) == 2 for r in results)

    def test_an_empty_batch_is_not_an_error(self, scripted_detector) -> None:
        assert self.build(scripted_detector).detect([]) == []

    def test_the_head_is_discovered_from_the_probes_outputs(self, scripted_segmenter) -> None:
        """A nested ``(detections, (prototypes,))`` return, flattened and identified by rank."""
        detector = DETECTORS.build(
            "yolo26",
            backend=TORCH,
            path=scripted_segmenter,
            input_hw=SMALL_NETWORK,
            image_ops_backend="python",
        )

        assert detector.head.name == "yolo26_seg"
        assert detector.head.produces_masks is True

    def test_a_named_head_that_the_artefact_cannot_feed_is_refused_at_load(
        self, scripted_segmenter
    ) -> None:
        with pytest.raises(ModelLoadError, match="decodes 1 output"):
            DETECTORS.build(
                "yolo26",
                backend=TORCH,
                path=scripted_segmenter,
                input_hw=SMALL_NETWORK,
                image_ops_backend="python",
                head="yolo26",
            )

    def test_an_input_extent_the_artefact_refuses_fails_at_load(
        self, torch_module, tmp_path
    ) -> None:
        """The probe is the only check available for a TorchScript artefact, and it catches the
        case that matters: a fixed-size head told the wrong extent."""
        torch = torch_module

        class FixedSize(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = torch.nn.Linear(3 * 32 * 32, 6)

            def forward(self, images: torch.Tensor) -> torch.Tensor:
                return self.fc(images.flatten(1)).reshape(-1, 1, 6)

        path = tmp_path / "fixed_size.ts"
        torch.jit.script(FixedSize()).save(str(path))

        with pytest.raises(ModelLoadError, match="probe"):
            DETECTORS.build(
                "yolo26",
                backend=TORCH,
                path=path,
                input_hw=(64, 64),
                image_ops_backend="python",
            )

    def test_an_artefact_that_is_not_a_detector_is_refused(
        self, torch_module, tmp_path
    ) -> None:
        torch = torch_module

        class NotADetector(torch.nn.Module):
            def forward(self, images: torch.Tensor) -> torch.Tensor:
                return images.mean(dim=(1, 2, 3))

        path = tmp_path / "not_a_detector.ts"
        torch.jit.script(NotADetector()).save(str(path))

        with pytest.raises(ModelLoadError, match="cannot tell which head"):
            DETECTORS.build(
                "yolo26",
                backend=TORCH,
                path=path,
                input_hw=SMALL_NETWORK,
                image_ops_backend="python",
            )

    def test_a_missing_or_unreadable_artefact_is_a_model_load_error(
        self, torch_module, tmp_path
    ) -> None:
        """Distinct from BackendUnavailableError: "there is no torch here" is a deployment
        problem and "this file is not a model" is an artefact problem."""
        with pytest.raises(ModelLoadError, match="no TorchScript artefact"):
            DETECTORS.build(
                "yolo26", backend=TORCH, path=tmp_path / "absent.ts", input_hw=SMALL_NETWORK
            )

        junk = tmp_path / "junk.ts"
        junk.write_text("this is not a model")
        with pytest.raises(ModelLoadError, match="not a loadable TorchScript module"):
            DETECTORS.build(
                "yolo26", backend=TORCH, path=junk, input_hw=SMALL_NETWORK
            )

    def test_a_frame_with_no_pixels_is_refused_with_its_tag_named(
        self, scripted_detector
    ) -> None:
        """This backend preprocesses on the host, and a message that does not say which camera
        sent the frame is not actionable."""
        from shipvision.types import Frame, FrameTag

        detector = self.build(scripted_detector)
        frame = Frame(FrameTag("cam-13", 2), image=None, height=270, width=480)

        with pytest.raises(ConfigurationError, match="cam-13#2"):
            detector.detect_one(frame)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"input_hw": (0, 64)},
            {"input_hw": (64,)},
            {"max_batch": 0},
            {"pad_value": 300},
        ],
    )
    def test_an_impossible_argument_is_refused(self, scripted_detector, kwargs) -> None:
        arguments = {"input_hw": SMALL_NETWORK, "image_ops_backend": "python", **kwargs}

        with pytest.raises(ConfigurationError):
            DETECTORS.build("yolo26", backend=TORCH, path=scripted_detector, **arguments)

    def test_the_default_image_ops_resolution_also_works(self, scripted_detector) -> None:
        """No pinned backend: whatever is fastest on this machine, with numpy as the floor."""
        detector = DETECTORS.build(
            "yolo26",
            backend=TORCH,
            path=scripted_detector,
            input_hw=SMALL_NETWORK,
            head_options={"conf_threshold": 0.1},
        )

        result = detector.detect_one(grey_frame("cam-01", 0, SMALL_SOURCE))

        assert np.abs(result.boxes - WANTED).max() < 1.0


@pytest.mark.gpu
class TestTensorRTDetector:
    """The real engine path. Needs tensorrt, a CUDA device and an ONNX to build from.

    Not run in the offline tier and not run on the machine this was written on — the offline
    coverage of this path is :mod:`tests.detection.test_bindings`, which exercises every
    decision the loader makes against a stub engine.
    """

    @pytest.fixture
    def engine(self, tmp_path):
        torch = pytest.importorskip("torch")
        pytest.importorskip("tensorrt")
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")

        from shipvision.detection import OptimisationProfile, build_engine

        network_boxes = to_network_space(WANTED, geometry(SMALL_SOURCE, SMALL_NETWORK))

        class FixedDetector(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                proposals = torch.zeros(1, 2, 6)
                proposals[0, :, :4] = torch.from_numpy(network_boxes)
                proposals[0, :, 4] = torch.tensor([0.9, 0.75])
                proposals[0, :, 5] = torch.tensor([0.0, 2.0])
                self.register_buffer("proposals", proposals)

            def forward(self, images: torch.Tensor) -> torch.Tensor:
                return self.proposals.expand(images.shape[0], -1, -1) + images.mean() * 0.0

        onnx = tmp_path / "fixed.onnx"
        torch.onnx.export(
            FixedDetector().eval(),
            torch.zeros(1, 3, *SMALL_NETWORK),
            str(onnx),
            input_names=["images"],
            output_names=["output0"],
            dynamic_axes={"images": {0: "batch"}, "output0": {0: "batch"}},
        )
        return build_engine(
            onnx,
            tmp_path / "fixed.engine",
            profiles=[
                OptimisationProfile.for_batch(input_hw=SMALL_NETWORK, max_batch=4, opt_batch=2)
            ],
        )

    def test_the_extent_and_the_batch_bound_come_from_the_engine(self, engine) -> None:
        detector = DETECTORS.build("yolo26", backend=TENSORRT, path=engine)

        assert detector.input_hw == SMALL_NETWORK
        assert detector.max_batch == 4
        assert detector.bindings.dynamic_batch is True

    def test_a_configured_extent_that_contradicts_the_engine_is_refused(self, engine) -> None:
        with pytest.raises(ModelLoadError, match="The engine is the artefact and wins"):
            DETECTORS.build(
                "yolo26", backend=TENSORRT, path=engine, input_hw=(128, 128)
            )

    def test_the_boxes_come_back_in_source_pixels(self, engine) -> None:
        detector = DETECTORS.build(
            "yolo26", backend=TENSORRT, path=engine, head_options={"conf_threshold": 0.1}
        )

        results = detector.detect(
            [grey_frame(f"cam-{i:02d}", i, SMALL_SOURCE) for i in range(3)]
        )

        assert [r.tag.camera_id for r in results] == ["cam-00", "cam-01", "cam-02"]
        for result in results:
            assert np.abs(result.boxes - WANTED).max() < 1.0

    def test_a_missing_engine_is_a_model_load_error(self, tmp_path) -> None:
        pytest.importorskip("tensorrt")

        with pytest.raises(ModelLoadError, match="no TensorRT engine"):
            DETECTORS.build("yolo26", backend=TENSORRT, path=tmp_path / "absent.engine")

    def test_a_corrupt_engine_is_a_model_load_error(self, tmp_path) -> None:
        pytest.importorskip("tensorrt")
        junk = tmp_path / "junk.engine"
        junk.write_bytes(b"not an engine")

        with pytest.raises(ModelLoadError, match="did not deserialise"):
            DETECTORS.build("yolo26", backend=TENSORRT, path=junk)


def _run(script: str):
    import subprocess

    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
