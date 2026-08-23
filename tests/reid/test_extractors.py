"""The extractor family: the mock everything else depends on, and the two artefact loaders.

The mock's tests are the load-bearing ones. Every gallery, tracking and MTMC test that runs
without a model runs on it, so if it stops being deterministic or stops being sensitive to
crop content, a whole tier of tests starts passing for the wrong reason.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import types

import numpy as np
import pytest

from shipvision.errors import (
    BackendUnavailableError,
    ConfigurationError,
    DimensionMismatchError,
    ModelLoadError,
)
from shipvision.registry import PYTHON, TENSORRT, TORCH
from shipvision.reid import EXTRACTORS, cosine_similarity, is_normalized

CROP_H, CROP_W = 64, 32


def textured_crop(
    seed: int, *, jitter: float = 0.0, h: int = CROP_H, w: int = CROP_W
) -> np.ndarray:
    """One ``(3, h, w)`` crop with structure, not noise.

    Structure matters here for the same reason it matters in `conftest`: this extractor
    reduces a crop to a block-mean thumbnail, and independent uniform noise block-averages
    to very nearly the same thumbnail every time. Two noise images therefore *are* alike by
    the only measure the mock has, and using them as "two different objects" would be
    testing the fixture rather than the extractor.
    """
    rng = np.random.default_rng(4000 + seed)
    y, x = np.mgrid[0:h, 0:w] / max(h, w)
    crop = np.empty((3, h, w), dtype=np.float32)
    for channel in range(3):
        crop[channel] = 0.5 + 0.4 * np.sin(
            6.0 * (rng.uniform(0.5, 3.0) * y + rng.uniform(0.5, 3.0) * x)
            + rng.uniform(0.0, 6.0)
        )
    if jitter:
        crop = crop + jitter * np.random.default_rng(9000 + seed).standard_normal(crop.shape)
    return np.clip(crop, 0.0, 1.0).astype(np.float32)


def batch_of(seeds, **kwargs) -> np.ndarray:
    return np.stack([textured_crop(s, **kwargs) for s in seeds])


# ------------------------------------------------------------------------- the registry


def test_the_family_lists_the_mock_and_the_artefact_backends() -> None:
    """`mock` is its own name rather than a backend of `generic` on purpose: a missing
    engine must be an error, and a `generic` that could resolve to a mock would let a
    production deployment silently embed nothing."""
    assert EXTRACTORS.names() == ["generic", "mock"]
    assert EXTRACTORS.backends("mock") == [PYTHON]
    assert EXTRACTORS.backends("generic") == [TENSORRT, TORCH]


def test_the_mock_is_reachable_by_name_and_by_alias() -> None:
    assert type(EXTRACTORS.build("mock", dim=8)).__name__ == "MockExtractor"
    assert type(EXTRACTORS.build("fake", dim=8)).__name__ == "MockExtractor"


# ---------------------------------------------------------------------------- every one


def test_an_empty_batch_returns_zero_by_dim_not_a_flat_empty() -> None:
    """A frame with no detections is ordinary input. `(0,)` breaks every downstream
    `[:, k]` with an IndexError where `(0, dim)` yields an empty result."""
    extractor = EXTRACTORS.build("mock", dim=48)

    out = extractor.extract(np.zeros((0, 3, CROP_H, CROP_W), dtype=np.float32))

    assert out.shape == (0, 48)
    assert out.dtype == np.float32


def test_a_channels_last_batch_is_refused_rather_than_run() -> None:
    """The trap this check exists for: an `(n, h, w, 3)` batch — what OpenCV hands back —
    has the right rank and the right element count, so a model runs on it and returns
    confident nonsense. Only the channel axis says otherwise."""
    extractor = EXTRACTORS.build("mock", dim=16)
    hwc = np.zeros((2, CROP_H, CROP_W, 3), dtype=np.float32)

    with pytest.raises(DimensionMismatchError, match="channels-first"):
        extractor.extract(hwc)


def test_a_single_crop_without_a_batch_axis_is_told_what_to_call() -> None:
    extractor = EXTRACTORS.build("mock", dim=16)

    with pytest.raises(ConfigurationError, match="extract_one"):
        extractor.extract(textured_crop(0))


def test_extract_one_returns_a_single_vector() -> None:
    extractor = EXTRACTORS.build("mock", dim=16)

    out = extractor.extract_one(textured_crop(0))

    assert out.shape == (16,)
    assert is_normalized(out)


# ---------------------------------------------------------------------- the mock itself


def test_the_mock_output_is_normalised_and_dim_matches_what_it_returns() -> None:
    """`dim` is what a gallery is allocated against, so it disagreeing with the vectors is
    the one failure that cannot be detected downstream."""
    extractor = EXTRACTORS.build("mock", dim=96)

    out = extractor.extract(batch_of(range(5)))

    assert out.shape == (5, extractor.dim) == (5, 96)
    assert is_normalized(out)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_the_same_crop_always_gives_the_same_vector() -> None:
    crops = batch_of(range(4))
    first = EXTRACTORS.build("mock", dim=32).extract(crops)
    second = EXTRACTORS.build("mock", dim=32).extract(crops)

    assert np.array_equal(first, second), "two instances of one seed are one model"
    assert np.array_equal(first, EXTRACTORS.build("mock", dim=32).extract(crops))


def test_a_batch_of_one_and_a_batch_of_many_agree() -> None:
    """Otherwise an embedding would depend on how many objects happened to be in the frame
    beside it, and a gallery built offline would not match one built live."""
    extractor = EXTRACTORS.build("mock", dim=64)
    crops = batch_of(range(6))

    many = extractor.extract(crops)
    one_at_a_time = np.stack([extractor.extract(crops[i : i + 1])[0] for i in range(6)])

    assert np.allclose(many, one_at_a_time, atol=1e-6)


def test_different_crops_give_different_vectors() -> None:
    extractor = EXTRACTORS.build("mock", dim=64)

    out = extractor.extract(batch_of(range(8)))

    similarity = cosine_similarity(out, out)
    off_diagonal = similarity[~np.eye(8, dtype=bool)]
    assert np.max(np.abs(off_diagonal)) < 0.5, "eight unrelated crops must not collide"


def test_crops_that_look_alike_embed_alike_and_others_do_not() -> None:
    """The property that makes this a useful stand-in rather than a stub.

    A stub returning zeros makes every similarity identical, so a broken ranking passes. A
    stub returning noise makes every pair equally unlike, so a broken *tracker* passes —
    in high dimensions two random unit vectors are nearly orthogonal whatever produced
    them. What the rest of the library needs is the structure real embeddings have.
    """
    extractor = EXTRACTORS.build("mock", dim=256)
    same_object = extractor.extract(batch_of([1, 1, 1], jitter=0.05))
    other_objects = extractor.extract(batch_of([2, 3, 4, 5]))

    within = cosine_similarity(same_object, same_object)[0, 1:]
    across = cosine_similarity(same_object[:1], other_objects)[0]

    assert within.min() > 0.9, "three jittered views of one object must stay together"
    assert across.max() < 0.5
    assert within.min() > across.max() + 0.3


def test_uniform_crops_of_different_brightness_do_not_collide() -> None:
    """A degenerate crop is still a crop. A construction that carried only the *pattern*
    would send every flat crop to the same vector, and synthetic test images are flat."""
    extractor = EXTRACTORS.build("mock", dim=128)
    flat = np.stack(
        [np.full((3, CROP_H, CROP_W), level, dtype=np.float32) for level in (0.2, 0.5, 0.8)]
    )

    out = extractor.extract(flat)

    similarity = cosine_similarity(out, out)
    assert similarity[0, 1] < 0.6
    assert similarity[0, 2] < 0.6


def test_two_seeds_are_two_different_models() -> None:
    """How a test builds the two-models-one-gallery failure on purpose."""
    crops = batch_of(range(3))

    a = EXTRACTORS.build("mock", dim=32, seed=0).extract(crops)
    b = EXTRACTORS.build("mock", dim=32, seed=1).extract(crops)

    assert not np.allclose(a, b)


def test_the_bandwidth_controls_how_quickly_similarity_falls_off() -> None:
    """It is the knob a caller reaches for when crops are not scaled to roughly [0, 1]."""
    crops = batch_of([1, 2])

    wide = EXTRACTORS.build("mock", dim=256, bandwidth=1.0).extract(crops)
    narrow = EXTRACTORS.build("mock", dim=256, bandwidth=0.05).extract(crops)

    assert (
        cosine_similarity(wide[:1], wide[1:])[0, 0]
        > cosine_similarity(narrow[:1], narrow[1:])[0, 0]
    )


def test_a_single_channel_stream_can_be_mocked_too() -> None:
    extractor = EXTRACTORS.build("mock", dim=16, channels=1)

    out = extractor.extract(np.zeros((2, 1, CROP_H, CROP_W), dtype=np.float32))

    assert out.shape == (2, 16)
    with pytest.raises(DimensionMismatchError):
        extractor.extract(np.zeros((2, 3, CROP_H, CROP_W), dtype=np.float32))


def test_crops_of_different_sizes_still_produce_comparable_vectors() -> None:
    """A person crop is tall and a ship crop is wide. The thumbnail is what makes them
    comparable at all; without it the projection would not even accept both."""
    extractor = EXTRACTORS.build("mock", dim=64)

    tall = extractor.extract(batch_of([1], h=96, w=32))
    wide = extractor.extract(batch_of([1], h=32, w=96))

    assert tall.shape == wide.shape == (1, 64)
    assert np.isfinite(cosine_similarity(tall, wide)).all()


def test_a_crop_smaller_than_the_thumbnail_grid_is_upsampled_not_rejected() -> None:
    """A distant ship is a handful of pixels, and dropping it would lose the detection the
    gallery most needs help with."""
    extractor = EXTRACTORS.build("mock", dim=32, grid=8)

    out = extractor.extract(np.full((1, 3, 2, 3), 0.4, dtype=np.float32))

    assert out.shape == (1, 32)
    assert is_normalized(out)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"dim": 0}, "dim must be positive"),
        ({"grid": 0}, "grid must be positive"),
        ({"channels": 0}, "channels must be positive"),
        ({"bandwidth": 0.0}, "bandwidth must be positive"),
    ],
)
def test_nonsense_configuration_fails_at_construction(kwargs: dict, match: str) -> None:
    with pytest.raises(ConfigurationError, match=match):
        EXTRACTORS.build("mock", **kwargs)


@pytest.mark.slow
def test_the_mock_is_deterministic_across_processes() -> None:
    """The claim the docstring makes, tested rather than asserted.

    `hash()` is salted per process for `str` and `bytes`, so an implementation that hashed
    crop bytes would give different embeddings on every run — and a test whose expected
    answer depends on PYTHONHASHSEED is worse than no test. Two subprocesses with
    deliberately different seeds must agree exactly.
    """
    program = textwrap.dedent("""
        import numpy as np
        from shipvision.reid import EXTRACTORS
        crop = np.linspace(0.0, 1.0, 3 * 8 * 6, dtype=np.float32).reshape(1, 3, 8, 6)
        out = EXTRACTORS.build("mock", dim=8, grid=2).extract(crop)
        print(" ".join(f"{v:.9f}" for v in out[0]))
        """)
    runs = [
        subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout
        for seed in ("1", "2")
    ]

    assert runs[0] == runs[1] != ""


# ------------------------------------------------------------------- the torch extractor


@pytest.fixture(scope="module")
def torch_module():
    return pytest.importorskip("torch")


@pytest.fixture(scope="module")
def scripted_pooled(torch_module, tmp_path_factory):
    """A scripted trunk whose pooled output is ``(n, dim, 1, 1)`` — a CNN's usual shape."""
    torch = torch_module

    class Pooled(torch.nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
            self.project = torch.nn.Conv2d(3, dim, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.project(self.pool(x))

    path = tmp_path_factory.mktemp("artefacts") / "pooled.ts"
    torch.jit.script(Pooled(24)).save(str(path))
    return path


@pytest.fixture(scope="module")
def scripted_two_head(torch_module, tmp_path_factory):
    """A module returning its two heads separately — CLIP-ReID's 768 + 512 shape."""
    torch = torch_module

    class TwoHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
            self.bottleneck = torch.nn.Linear(3, 16)
            self.projection = torch.nn.Linear(3, 8)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            flat = self.pool(x).flatten(1)
            return self.bottleneck(flat), self.projection(flat)

    path = tmp_path_factory.mktemp("artefacts") / "two_head.ts"
    torch.jit.script(TwoHead()).save(str(path))
    return path


@pytest.fixture(scope="module")
def scripted_spatial(torch_module, tmp_path_factory):
    """A module that never pooled, so it returns a feature map rather than an embedding."""
    torch = torch_module

    class Spatial(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x)

    path = tmp_path_factory.mktemp("artefacts") / "spatial.ts"
    torch.jit.script(Spatial()).save(str(path))
    return path


@pytest.mark.slow
def test_the_torch_extractor_discovers_dim_from_the_artefact(scripted_pooled) -> None:
    """Discovered, not configured. A configured width that disagrees with the artefact has
    no symptom: the gallery is allocated one way and the model writes the other."""
    extractor = EXTRACTORS.build(
        "generic", backend=TORCH, path=scripted_pooled, input_size=(CROP_H, CROP_W)
    )

    assert extractor.dim == 24
    assert extractor.backend == TORCH


@pytest.mark.slow
def test_the_torch_extractor_normalises_and_batches_consistently(scripted_pooled) -> None:
    extractor = EXTRACTORS.build(
        "generic",
        backend=TORCH,
        path=scripted_pooled,
        input_size=(CROP_H, CROP_W),
        batch_size=2,
    )
    crops = batch_of(range(5))

    out = extractor.extract(crops)

    assert out.shape == (5, 24)
    assert out.dtype == np.float32
    assert is_normalized(out)
    # batch_size=2 means this batch was chunked 2+2+1; the chunking must not be visible.
    assert np.allclose(out[3], extractor.extract(crops[3:4])[0], atol=1e-5)


@pytest.mark.slow
def test_the_torch_extractor_returns_zero_by_dim_for_an_empty_batch(scripted_pooled) -> None:
    extractor = EXTRACTORS.build(
        "generic", backend=TORCH, path=scripted_pooled, input_size=(CROP_H, CROP_W)
    )

    assert extractor.extract(np.zeros((0, 3, CROP_H, CROP_W), np.float32)).shape == (0, 24)


@pytest.mark.slow
def test_two_separate_heads_are_concatenated_the_way_clip_reid_does(scripted_two_head) -> None:
    """CLIP-ReID's ViT-B-16 embedding is its 768-wide bottleneck output and its 512-wide
    projection output joined into one 1280-wide vector. A module scripted from a checkpoint
    that stops one step short returns the parts, and dropping either half would quietly
    halve the model."""
    extractor = EXTRACTORS.build(
        "generic", backend=TORCH, path=scripted_two_head, input_size=(CROP_H, CROP_W)
    )

    assert extractor.dim == 16 + 8
    assert extractor.extract(batch_of(range(2))).shape == (2, 24)


@pytest.mark.slow
def test_a_feature_map_is_refused_rather_than_flattened(scripted_spatial) -> None:
    """`reshape(n, -1)` would succeed and hand the gallery a 16 000-wide "embedding" whose
    only symptom is an out-of-memory much later."""
    with pytest.raises(ModelLoadError, match="not an embedding"):
        EXTRACTORS.build(
            "generic", backend=TORCH, path=scripted_spatial, input_size=(CROP_H, CROP_W)
        )


@pytest.mark.slow
def test_crops_of_the_wrong_size_are_refused(scripted_pooled) -> None:
    """The artefact here pools adaptively, so it would accept them and return an embedding
    of the same width and a different meaning — the failure with no symptom."""
    extractor = EXTRACTORS.build(
        "generic", backend=TORCH, path=scripted_pooled, input_size=(CROP_H, CROP_W)
    )

    with pytest.raises(DimensionMismatchError, match="imgproc boundary"):
        extractor.extract(batch_of([1], h=128, w=64))


@pytest.mark.slow
def test_a_missing_or_unreadable_artefact_is_a_model_load_error(torch_module, tmp_path) -> None:
    """Distinct from BackendUnavailableError on purpose: "there is no torch here" is a
    deployment problem and "this file is not a model" is an artefact problem, and an
    operator fixes them in different places."""
    with pytest.raises(ModelLoadError, match="no TorchScript artefact"):
        EXTRACTORS.build(
            "generic", backend=TORCH, path=tmp_path / "absent.ts", input_size=(8, 8)
        )

    junk = tmp_path / "junk.ts"
    junk.write_text("this is not a model")
    with pytest.raises(ModelLoadError, match="not a loadable TorchScript module"):
        EXTRACTORS.build("generic", backend=TORCH, path=junk, input_size=(8, 8))


@pytest.mark.slow
def test_a_bad_input_size_fails_at_load_not_on_frame_forty_thousand(
    torch_module, tmp_path
) -> None:
    torch = torch_module

    class FixedSize(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = torch.nn.Linear(3 * 8 * 8, 12)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x.flatten(1))

    path = tmp_path / "fixed.ts"
    torch.jit.script(FixedSize()).save(str(path))

    with pytest.raises(ModelLoadError, match="probe"):
        EXTRACTORS.build("generic", backend=TORCH, path=path, input_size=(16, 16))


@pytest.mark.parametrize(
    "kwargs", [{"batch_size": 0}, {"input_size": (0, 8)}, {"input_size": (8,)}]
)
def test_the_torch_extractor_validates_its_arguments_before_touching_the_disk(
    kwargs: dict, tmp_path
) -> None:
    from shipvision.reid.extractors.torch_extractor import TorchExtractor

    arguments = {"path": tmp_path / "absent.ts", "input_size": (8, 8), **kwargs}
    with pytest.raises(ConfigurationError):
        TorchExtractor(**arguments)


# ---------------------------------------------------- the runtimes that may not be there


def test_the_tensorrt_module_imports_with_no_tensorrt_installed() -> None:
    """It must, or the registry could not list a backend the machine cannot run — and an
    operator would learn that tensorrt is missing from an ImportError inside a lookup."""
    import importlib

    module = importlib.import_module("shipvision.reid.extractors.tensorrt_extractor")

    assert module.TensorRTExtractor.__name__ == "TensorRTExtractor"


@pytest.mark.parametrize(
    ("backend", "absent", "extra"),
    [
        (TENSORRT, "tensorrt", {}),
        (TORCH, "torch", {"input_size": (8, 8)}),
    ],
)
def test_a_missing_runtime_is_a_typed_error_not_an_import_error(
    backend: str, absent: str, extra: dict, monkeypatch, tmp_path
) -> None:
    """`None` in `sys.modules` is what an unimportable module looks like to `import`, so
    this exercises the real path on a machine that happens to have the runtime.

    BackendUnavailableError rather than ImportError because the two say different things to
    whoever is on call: one is "install this", the other is a stack trace they have to read
    the source to interpret.
    """
    monkeypatch.setitem(sys.modules, absent, None)

    with pytest.raises(BackendUnavailableError, match=absent):
        EXTRACTORS.build("generic", backend=backend, path=tmp_path / "absent.engine", **extra)


# ------------------------------------------------- the TensorRT engine's own batch size


class StubEngine:
    """A serialised engine's shape metadata, with nothing behind it.

    Enough of the TensorRT surface for the constructor's shape arithmetic to run on a
    machine with no GPU. `batch` is the plan's baked row count when `dynamic` is False —
    the case that matters, because a fixed-batch plan reads and writes exactly that many
    rows however few the caller asked for.
    """

    num_io_tensors = 2

    def __init__(self, *, batch: int, dynamic: bool, dim: int = 512) -> None:
        self.batch = batch
        self.dynamic = dynamic
        self.dim = dim
        self.set_input_shape_calls: list[tuple] = []

    def get_tensor_name(self, i: int) -> str:
        return ["images", "embeddings"][i]

    def get_tensor_mode(self, name: str):
        import tensorrt as trt

        return trt.TensorIOMode.INPUT if name == "images" else trt.TensorIOMode.OUTPUT

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        rows = -1 if self.dynamic else self.batch
        return (rows, 3, 256, 128) if name == "images" else (rows, self.dim)

    def get_tensor_profile_shape(self, name: str, index: int):
        one = (1, 3, 256, 128)
        return one, one, (self.batch, 3, 256, 128)

    def get_tensor_dtype(self, name: str):
        import tensorrt as trt

        return trt.DataType.FLOAT

    def create_execution_context(self):
        return StubContext(self)


class StubContext:
    def __init__(self, engine: StubEngine) -> None:
        self._engine = engine

    def set_input_shape(self, name: str, shape: tuple) -> bool:
        self._engine.set_input_shape_calls.append(shape)
        return True

    def set_tensor_address(self, name: str, pointer: int) -> None:
        return None

    def execute_async_v3(self, stream: int) -> bool:
        return True


def _stub_tensorrt(engine: StubEngine):
    module = types.ModuleType("tensorrt")
    module.TensorIOMode = types.SimpleNamespace(INPUT="input", OUTPUT="output")
    module.DataType = types.SimpleNamespace(FLOAT="float32", HALF="float16")
    module.Logger = type("Logger", (), {"WARNING": 1, "__init__": lambda self, *a: None})
    module.init_libnvinfer_plugins = lambda *a: None
    module.Runtime = type(
        "Runtime",
        (),
        {
            "__init__": lambda self, *a: None,
            "deserialize_cuda_engine": lambda self, blob: engine,
        },
    )
    return module


def _stub_torch(allocations: list[tuple[int, ...]]):
    """Just enough torch to record what the constructor would allocate on the device."""

    class Tensor:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = tuple(shape)

        def __getitem__(self, key):
            return self

        def copy_(self, *a, **k):
            return self

        def data_ptr(self) -> int:
            return 0xDEAD

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.zeros((self.shape[0], self.shape[1]), dtype=np.float32)

    class Nothing:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def empty(shape, dtype=None, device=None):
        allocations.append(tuple(shape))
        return Tensor(shape)

    module = types.ModuleType("torch")
    module.float32 = "float32"
    module.float16 = "float16"
    module.empty = empty
    module.from_numpy = lambda array: Tensor(array.shape)
    module.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device=lambda ordinal: Nothing(),
        stream=lambda stream: Nothing(),
        Stream=type("Stream", (), {"cuda_stream": 1, "synchronize": lambda self: None}),
    )
    return module


class TestTensorRTBatchAgreesWithTheEngine:
    """`max_batch` is a promise about how many rows the plan will run, not a slice size.

    A fixed-batch engine reads and writes its baked row count whatever the caller asked
    for, so buffers sized to a smaller `max_batch` are read and written out of bounds —
    a silent corruption of whatever the caching allocator handed out next, or a sticky
    illegal access that poisons the context for the life of the process. There is nothing
    to clamp: the number is the engine's, and a caller who needs a different one has a
    configuration problem.
    """

    def _build(self, monkeypatch, *, batch: int, dynamic: bool, max_batch, tmp_path):
        from shipvision.reid.extractors.tensorrt_extractor import TensorRTExtractor

        engine = StubEngine(batch=batch, dynamic=dynamic)
        allocations: list[tuple[int, ...]] = []
        monkeypatch.setitem(sys.modules, "tensorrt", _stub_tensorrt(engine))
        monkeypatch.setitem(sys.modules, "torch", _stub_torch(allocations))
        path = tmp_path / "stub.engine"
        path.write_bytes(b"not really an engine")
        extractor = TensorRTExtractor(path=path, device=0, max_batch=max_batch)
        return extractor, engine, allocations

    def test_a_smaller_max_batch_on_a_fixed_engine_is_refused(self, monkeypatch, tmp_path):
        with pytest.raises(ConfigurationError, match="32") as raised:
            self._build(monkeypatch, batch=32, dynamic=False, max_batch=8, tmp_path=tmp_path)

        assert "8" in str(raised.value), "the message must name both numbers"

    def test_a_larger_max_batch_on_a_fixed_engine_is_refused_too(self, monkeypatch, tmp_path):
        """Not honourable in the other direction either: the plan still runs 32."""
        with pytest.raises(ConfigurationError, match="fixed"):
            self._build(monkeypatch, batch=32, dynamic=False, max_batch=64, tmp_path=tmp_path)

    def test_a_fixed_engine_allocates_exactly_its_baked_row_count(self, monkeypatch, tmp_path):
        extractor, engine, allocations = self._build(
            monkeypatch, batch=32, dynamic=False, max_batch=None, tmp_path=tmp_path
        )

        assert extractor.max_batch == 32
        assert [shape[0] for shape in allocations] == [32, 32]
        extractor.extract(np.zeros((8, 3, 256, 128), np.float32))
        assert engine.set_input_shape_calls == [], "a fixed plan cannot be told a row count"

    def test_a_matching_max_batch_on_a_fixed_engine_is_accepted(self, monkeypatch, tmp_path):
        extractor, _, allocations = self._build(
            monkeypatch, batch=32, dynamic=False, max_batch=32, tmp_path=tmp_path
        )

        assert extractor.max_batch == 32
        assert [shape[0] for shape in allocations] == [32, 32]

    def test_a_dynamic_engine_still_clamps_because_it_can_honour_it(self, monkeypatch, tmp_path):
        extractor, engine, allocations = self._build(
            monkeypatch, batch=32, dynamic=True, max_batch=8, tmp_path=tmp_path
        )

        assert extractor.max_batch == 8
        assert [shape[0] for shape in allocations] == [8, 8]
        extractor.extract(np.zeros((8, 3, 256, 128), np.float32))
        assert engine.set_input_shape_calls == [(8, 3, 256, 128)], "told the real count"

    def test_a_dynamic_engine_cannot_be_pushed_past_its_profile(self, monkeypatch, tmp_path):
        extractor, _, allocations = self._build(
            monkeypatch, batch=32, dynamic=True, max_batch=64, tmp_path=tmp_path
        )

        assert extractor.max_batch == 32, "the profile maximum is still the ceiling"
        assert [shape[0] for shape in allocations] == [32, 32]
