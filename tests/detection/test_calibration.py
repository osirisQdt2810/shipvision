"""INT8 calibration: the batch feeder and the cache, both without tensorrt.

Subclassing ``trt.IInt8EntropyCalibrator2`` needs tensorrt at class-definition time, so the
subclass is built inside a function and everything with a decision in it lives in the two plain
classes that are tested here. That is not a workaround — it is what makes the trap in the module
docstring testable at all.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from shipvision.detection.backends.tensorrt.calibration import (
    CalibrationBatchFeeder,
    CalibrationCache,
    build_int8_calibrator,
)
from shipvision.errors import (
    BackendUnavailableError,
    ConfigurationError,
    DimensionMismatchError,
)

SHAPE = (4, 3, 8, 8)


def batch(rows, *, value=0.5, shape=SHAPE[1:]):
    """``rows`` preprocessed images whose values identify their row index."""
    data = np.empty((rows, *shape), dtype=np.float32)
    for row in range(rows):
        data[row] = value * (row + 1)
    return data


class TestCalibrationCache:
    """Persisted scales, so changing an unrelated builder flag does not recalibrate."""

    def test_an_absent_file_is_a_miss(self, tmp_path) -> None:
        cache = CalibrationCache(tmp_path / "scales.cache")

        assert cache.read() is None
        assert cache.hits == 0

    def test_a_written_blob_reads_back_and_counts_as_a_hit(self, tmp_path) -> None:
        """``hits`` is how a caller finds out whether calibration actually ran, which is
        otherwise invisible and is the first thing to suspect when an INT8 engine's accuracy
        changes without its ONNX changing."""
        cache = CalibrationCache(tmp_path / "scales.cache")

        cache.write(b"per-tensor-scales")

        assert cache.read() == b"per-tensor-scales"
        assert (cache.hits, cache.writes) == (1, 1)

    def test_the_parent_directory_is_created(self, tmp_path) -> None:
        cache = CalibrationCache(tmp_path / "nested" / "deeper" / "scales.cache")

        cache.write(b"blob")

        assert cache.read() == b"blob"

    def test_an_empty_file_is_a_miss_and_not_a_hit(self, tmp_path) -> None:
        """An empty file is a build interrupted while writing. Reading it as a hit would produce
        an engine with no scales at all."""
        path = tmp_path / "scales.cache"
        path.write_bytes(b"")
        cache = CalibrationCache(path)

        assert cache.read() is None
        assert cache.hits == 0

    def test_writing_nothing_does_not_create_a_file(self, tmp_path) -> None:
        path = tmp_path / "scales.cache"

        CalibrationCache(path).write(b"")

        assert not path.exists()

    def test_invalidate_removes_it_and_is_safe_when_absent(self, tmp_path) -> None:
        """The scales belong to a network, so a changed ONNX must invalidate them."""
        cache = CalibrationCache(tmp_path / "scales.cache")
        cache.write(b"blob")

        cache.invalidate()
        cache.invalidate()

        assert cache.read() is None

    def test_a_none_path_disables_caching_entirely(self) -> None:
        cache = CalibrationCache(None)

        cache.write(b"blob")

        assert cache.read() is None
        assert cache.writes == 0


class TestCalibrationBatchFeeder:
    """The state machine TensorRT drives, and every way calibration data can be wrong."""

    def test_batches_come_back_at_the_calibration_shape(self) -> None:
        feeder = CalibrationBatchFeeder([batch(4), batch(4)], batch_shape=SHAPE)

        assert feeder.next_batch().shape == SHAPE
        assert feeder.next_batch().shape == SHAPE
        assert feeder.next_batch() is None
        assert feeder.served == 2

    def test_a_short_final_batch_is_padded_by_repeating_its_own_rows(self) -> None:
        """Not with zeros. TensorRT calibrates at a fixed batch size and a partial batch has to
        be filled; a batch of zeros is not a plausible image and drags the activation histograms
        towards zero exactly where the entropy criterion picks its clipping threshold."""
        feeder = CalibrationBatchFeeder([batch(3, value=1.0)], batch_shape=SHAPE)

        padded = feeder.next_batch()

        assert padded.shape == SHAPE
        assert [float(padded[row].flat[0]) for row in range(4)] == [1.0, 2.0, 3.0, 1.0]

    def test_an_oversize_batch_is_truncated_to_the_calibration_shape(self) -> None:
        feeder = CalibrationBatchFeeder([batch(7, value=1.0)], batch_shape=SHAPE)

        trimmed = feeder.next_batch()

        assert [float(trimmed[row].flat[0]) for row in range(4)] == [1.0, 2.0, 3.0, 4.0]

    def test_a_limit_stops_the_calibration_early(self) -> None:
        """Accuracy plateaus after a few hundred images and build time does not."""
        feeder = CalibrationBatchFeeder([batch(4)] * 10, batch_shape=SHAPE, limit=3)

        served = 0
        while feeder.next_batch() is not None:
            served += 1

        assert served == 3

    def test_a_generator_source_is_consumed_lazily(self) -> None:
        """A calibration set larger than memory has to be a generator over files."""
        produced = []

        def source():
            for index in range(4):
                produced.append(index)
                yield batch(4)

        feeder = CalibrationBatchFeeder(source(), batch_shape=SHAPE)
        feeder.next_batch()

        assert produced == [0]

    def test_uint8_data_is_refused_because_it_is_the_trap(self) -> None:
        """Raw pixels against a model fed ``[0, 1]`` produce scales 255x too wide: the engine
        builds, runs at full speed, and is quietly wrong."""
        raw = (np.zeros((4, 3, 8, 8)) + 200).astype(np.uint8)
        feeder = CalibrationBatchFeeder([raw], batch_shape=SHAPE)

        with pytest.raises(ConfigurationError, match="quietly wrong"):
            feeder.next_batch()

    def test_a_spatial_shape_that_does_not_match_the_engine_is_refused(self) -> None:
        feeder = CalibrationBatchFeeder([batch(4, shape=(3, 16, 16))], batch_shape=SHAPE)

        with pytest.raises(DimensionMismatchError, match="same letterbox as inference"):
            feeder.next_batch()

    def test_an_empty_batch_is_refused_rather_than_read_as_the_end(self) -> None:
        feeder = CalibrationBatchFeeder([batch(0)], batch_shape=SHAPE)

        with pytest.raises(DimensionMismatchError, match="calibrates nothing"):
            feeder.next_batch()

    def test_data_outside_the_declared_value_range_is_refused(self) -> None:
        """The cheapest available defence against the preprocessing mismatch: state the range
        the preprocessing produces, and a mismatch becomes loud instead of expensive."""
        unnormalised = np.full((4, 3, 8, 8), 200.0, dtype=np.float32)
        feeder = CalibrationBatchFeeder(
            [unnormalised], batch_shape=SHAPE, value_range=(0.0, 1.0)
        )

        with pytest.raises(ConfigurationError, match="does not match inference"):
            feeder.next_batch()

    def test_data_inside_the_declared_value_range_passes(self) -> None:
        normalised = np.full((4, 3, 8, 8), 0.44, dtype=np.float32)
        feeder = CalibrationBatchFeeder(
            [normalised], batch_shape=SHAPE, value_range=(0.0, 1.0)
        )

        assert feeder.next_batch().shape == SHAPE

    def test_the_batch_size_is_what_the_engine_will_be_told(self) -> None:
        assert CalibrationBatchFeeder([], batch_shape=(16, 3, 64, 64)).batch_size == 16

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"batch_shape": (4, 3, 8)},
            {"batch_shape": (0, 3, 8, 8)},
            {"batch_shape": (4, 3, 8, -1)},
            {"value_range": (1.0, 0.0)},
            {"value_range": (1.0, 1.0)},
            {"limit": 0},
        ],
    )
    def test_an_impossible_argument_is_refused_at_construction(self, kwargs) -> None:
        arguments = {"batch_shape": SHAPE, **kwargs}

        with pytest.raises(ConfigurationError):
            CalibrationBatchFeeder([], **arguments)

    def test_the_repr_says_how_far_calibration_got(self) -> None:
        feeder = CalibrationBatchFeeder([batch(4)], batch_shape=SHAPE, limit=5)
        feeder.next_batch()

        assert "served=1" in repr(feeder) and "limit=5" in repr(feeder)


class FakeTrtWithCalibratorBase:
    """Enough of the tensorrt module for :func:`build_int8_calibrator` to subclass."""

    class IInt8EntropyCalibrator2:
        def __init__(self) -> None:
            pass


class TestInt8CalibratorConstruction:
    """The one part that needs a device — and the one refusal that does not."""

    def test_no_torch_means_nowhere_to_stage_batches(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "torch", None)
        feeder = CalibrationBatchFeeder([batch(4)], batch_shape=SHAPE)

        with pytest.raises(BackendUnavailableError, match="torch"):
            build_int8_calibrator(FakeTrtWithCalibratorBase, feeder)

    @pytest.mark.gpu
    def test_the_calibrator_hands_out_pointers_and_then_stops(self) -> None:
        """Not run in the offline tier: the buffer is real device memory."""
        trt = pytest.importorskip("tensorrt")
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")

        feeder = CalibrationBatchFeeder([batch(4), batch(4)], batch_shape=SHAPE)
        calibrator = build_int8_calibrator(trt, feeder, cache=CalibrationCache(None))

        assert calibrator.get_batch_size() == 4
        assert calibrator.get_batch(["images"]) is not None
        assert calibrator.get_batch(["images"]) is not None
        assert calibrator.get_batch(["images"]) is None
