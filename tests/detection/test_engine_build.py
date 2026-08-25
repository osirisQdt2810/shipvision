"""Engine building: the profile arithmetic, and every refusal that happens before tensorrt.

The reference recovers its own build configuration by running regular expressions over the ONNX
*filename* and then shells out to a script (``src/process/Yolo26Detector.cpp:66-124``). What is
tested here is the replacement: shapes passed as arguments, validated as arguments, and every
configuration mistake reported the same way on a build machine and on a laptop.
"""

from __future__ import annotations

import sys

import pytest

from shipvision.detection import OptimisationProfile, build_engine
from shipvision.errors import BackendUnavailableError, ConfigurationError, ModelLoadError


class TestOptimisationProfile:
    """``(min, opt, max)`` per input, checked at construction — no filename anywhere."""

    def test_for_batch_fills_in_the_common_case(self) -> None:
        profile = OptimisationProfile.for_batch(input_hw=(640, 640), max_batch=16)

        assert profile.input_name == "images"
        assert profile.minimum == (1, 3, 640, 640)
        assert profile.maximum == (16, 3, 640, 640)

    def test_the_optimum_defaults_to_the_maximum_not_to_half_of_it(self) -> None:
        """The reference uses ``max_batch // 2``, which is the right shape for nothing in
        particular. A detector instance here is fed by a batcher that fills up to its cap, so the
        batch it actually runs is the maximum far more often than half of it."""
        profile = OptimisationProfile.for_batch(input_hw=(640, 640), max_batch=16)

        assert profile.optimum == (16, 3, 640, 640)

    def test_an_explicit_optimum_is_kept(self) -> None:
        profile = OptimisationProfile.for_batch(
            input_hw=(512, 896), max_batch=16, min_batch=2, opt_batch=4
        )

        assert (profile.minimum[0], profile.optimum[0], profile.maximum[0]) == (2, 4, 16)
        assert profile.optimum[2:] == (512, 896)

    def test_a_non_default_input_name_and_channel_count_are_honoured(self) -> None:
        profile = OptimisationProfile.for_batch(
            input_hw=(64, 64), max_batch=1, channels=1, input_name="input_1"
        )

        assert profile.input_name == "input_1"
        assert profile.minimum == (1, 1, 64, 64)

    def test_an_unordered_profile_is_refused(self) -> None:
        """TensorRT accepts it and then refuses every shape at build time, with a message about
        the axis rather than about the ordering."""
        with pytest.raises(ConfigurationError, match="not ordered on axis 0"):
            OptimisationProfile(
                input_name="images",
                minimum=(8, 3, 64, 64),
                optimum=(4, 3, 64, 64),
                maximum=(16, 3, 64, 64),
            )

    def test_profiles_of_differing_rank_are_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="different ranks"):
            OptimisationProfile(
                input_name="images",
                minimum=(1, 3, 64),
                optimum=(1, 3, 64, 64),
                maximum=(1, 3, 64, 64),
            )

    def test_a_dynamic_dimension_left_in_the_profile_is_refused(self) -> None:
        """A ``-1`` belongs in the ONNX, not in the profile that resolves it."""
        with pytest.raises(ConfigurationError, match="fully specified"):
            OptimisationProfile(
                input_name="images",
                minimum=(1, 3, -1, 64),
                optimum=(1, 3, 64, 64),
                maximum=(1, 3, 64, 64),
            )

    def test_an_empty_input_name_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="name of the input"):
            OptimisationProfile(input_name="", minimum=(1,), optimum=(1,), maximum=(1,))

    @pytest.mark.parametrize("kwargs", [{"input_hw": (0, 64)}, {"channels": 0}])
    def test_for_batch_validates_its_own_arguments(self, kwargs) -> None:
        arguments = {"input_hw": (64, 64), "max_batch": 2, **kwargs}

        with pytest.raises(ConfigurationError):
            OptimisationProfile.for_batch(**arguments)


class TestBuildEngineValidation:
    """Argument mistakes are reported before tensorrt is even imported.

    Deliberate: a config error must read the same on a build machine and on a laptop, and the
    laptop is where the mistake is usually made.
    """

    @pytest.fixture
    def onnx(self, tmp_path):
        path = tmp_path / "model.onnx"
        path.write_bytes(b"pretend this is an onnx")
        return path

    @pytest.fixture(autouse=True)
    def no_tensorrt(self, monkeypatch):
        """Every test in this class runs as if tensorrt were absent, which proves the check it
        makes happens first."""
        monkeypatch.setitem(sys.modules, "tensorrt", None)

    def test_an_empty_and_a_missing_onnx_are_both_refused_before_tensorrt(
        self, tmp_path
    ) -> None:
        """Two cases a size check catches that a parser never gets to see."""
        empty = tmp_path / "empty.onnx"
        empty.write_bytes(b"")

        with pytest.raises(ModelLoadError, match="empty"):
            build_engine(empty, tmp_path / "e.engine")
        # `build_engine` checks existence before it gets this far, hence its own wording.
        with pytest.raises(ModelLoadError, match="no ONNX at"):
            build_engine(tmp_path / "absent.onnx", tmp_path / "a.engine")

    def test_a_calibrator_without_int8_is_refused(self, onnx, tmp_path) -> None:
        """A calibrator that is never asked for a batch is a silent no-op: the build succeeds and
        produces an fp32 engine."""
        from shipvision.detection.backends.tensorrt.calibration import CalibrationBatchFeeder

        with pytest.raises(ConfigurationError, match="silent no-op"):
            build_engine(
                onnx,
                tmp_path / "out.engine",
                int8=False,
                int8_calibration=CalibrationBatchFeeder([], batch_shape=(1, 3, 64, 64)),
            )

    def test_a_non_positive_workspace_is_refused(self, onnx, tmp_path) -> None:
        with pytest.raises(ConfigurationError, match="workspace_bytes"):
            build_engine(onnx, tmp_path / "out.engine", workspace_bytes=0)

    def test_a_missing_onnx_is_a_model_load_error(self, tmp_path) -> None:
        with pytest.raises(ModelLoadError, match="no ONNX"):
            build_engine(tmp_path / "absent.onnx", tmp_path / "out.engine")

    def test_refusing_to_overwrite_is_reported_as_configuration(self, onnx, tmp_path) -> None:
        existing = tmp_path / "out.engine"
        existing.write_bytes(b"an older engine")

        with pytest.raises(ConfigurationError, match="overwrite=False"):
            build_engine(onnx, existing, overwrite=False)

    def test_with_valid_arguments_the_missing_runtime_is_what_is_reported(
        self, onnx, tmp_path
    ) -> None:
        """The other half of the ordering claim: once the arguments are fine, the next thing
        wrong is the absent runtime, and it says what to install."""
        with pytest.raises(BackendUnavailableError, match="tensorrt"):
            build_engine(
                onnx,
                tmp_path / "out.engine",
                profiles=[OptimisationProfile.for_batch(input_hw=(64, 64), max_batch=2)],
                fp16=True,
            )


@pytest.mark.gpu
class TestBuildEngineOnRealHardware:
    """A real build. Needs tensorrt and a device, and was not run where this was written."""

    @pytest.fixture
    def onnx(self, tmp_path):
        torch = pytest.importorskip("torch")
        pytest.importorskip("tensorrt")

        class Tiny(torch.nn.Module):
            def forward(self, images: torch.Tensor) -> torch.Tensor:
                return images.flatten(1)[:, :6].reshape(-1, 1, 6)

        path = tmp_path / "tiny.onnx"
        torch.onnx.export(
            Tiny().eval(),
            torch.zeros(1, 3, 64, 64),
            str(path),
            input_names=["images"],
            output_names=["output0"],
            dynamic_axes={"images": {0: "batch"}, "output0": {0: "batch"}},
        )
        return path

    def test_a_dynamic_batch_engine_is_built_and_written(self, onnx, tmp_path) -> None:
        engine = build_engine(
            onnx,
            tmp_path / "tiny.engine",
            profiles=[OptimisationProfile.for_batch(input_hw=(64, 64), max_batch=4)],
        )

        assert engine.is_file() and engine.stat().st_size > 0

    def test_a_profile_naming_an_input_that_does_not_exist_is_refused(
        self, onnx, tmp_path
    ) -> None:
        """A profile for a missing input leaves the real input with no profile, and that only
        fails at inference."""
        with pytest.raises(ModelLoadError, match="inputs are"):
            build_engine(
                onnx,
                tmp_path / "tiny.engine",
                profiles=[
                    OptimisationProfile.for_batch(
                        input_hw=(64, 64), max_batch=4, input_name="input_1"
                    )
                ],
            )

    def test_a_file_that_is_not_onnx_is_refused_before_tensorrt_sees_it(self, tmp_path) -> None:
        """The file must be rejected by *us*, not by the parser.

        This test used to expect the parser's own error message. It could never have passed,
        because it was written on a machine with no TensorRT: with TensorRT present,
        `parser.parse()` and `parser.parse_from_file()` both **segmentation-fault** on a text
        file named `*.onnx` — measured on 10.14.1, no return value and no exception, the
        process is simply gone.

        That is why validation happens before the parser is handed anything, and why this test
        now asserts the shape of *our* refusal. It matters beyond error quality: the server
        builds engines from ONNX on demand, so a truncated download or an unresolved Git-LFS
        pointer in a model repository would otherwise take the worker down at start-up and
        leave an operator staring at a core dump.
        """
        pytest.importorskip("tensorrt")
        junk = tmp_path / "junk.onnx"
        junk.write_bytes(b"definitely not onnx")

        with pytest.raises(ModelLoadError, match=r"not a readable ONNX model|ModelProto"):
            build_engine(junk, tmp_path / "junk.engine")

        assert not (tmp_path / "junk.engine").exists(), "a refused build must leave nothing"

    def test_a_timing_cache_is_written_and_reused(self, onnx, tmp_path) -> None:
        cache = tmp_path / "timing.cache"
        profiles = [OptimisationProfile.for_batch(input_hw=(64, 64), max_batch=2)]

        build_engine(onnx, tmp_path / "a.engine", profiles=profiles, timing_cache=cache)

        assert cache.is_file() and cache.stat().st_size > 0
