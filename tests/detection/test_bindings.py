"""Engine introspection, tested on a machine with no TensorRT and no GPU.

This is the point of :mod:`shipvision.detection.backends.tensorrt.bindings` importing nothing:
every correctness decision in the TensorRT path — which axis is the batch, whether an extent is
dynamic, what happens when a configured input size disagrees with the binding — is exercised
here against a stub that answers like an engine. The alternative is a module whose only tests
are the ones nobody can run on the machine they are writing it on.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipvision.detection.backends.tensorrt.bindings import Binding, EngineBindings
from shipvision.detection.heads import resolve_head
from shipvision.errors import ModelLoadError


class FakeIoMode:
    INPUT = "input"
    OUTPUT = "output"


class FakeTrt:
    """Just the two pieces of the tensorrt module that :meth:`EngineBindings.read` touches."""

    TensorIOMode = FakeIoMode

    @staticmethod
    def nptype(dtype):
        return dtype


class Spec:
    """One IO tensor as a test declares it."""

    def __init__(self, name, shape, *, is_input=False, dtype=np.float32, profile=None):
        self.name = name
        self.shape = tuple(shape)
        self.is_input = is_input
        self.dtype = dtype
        self.profile = profile


class NamedEngine:
    """A TensorRT 8.5+ engine: IO tensors addressed by name."""

    def __init__(self, specs):
        self._specs = list(specs)

    @property
    def num_io_tensors(self):
        return len(self._specs)

    def _by_name(self, name):
        return next(spec for spec in self._specs if spec.name == name)

    def get_tensor_name(self, index):
        return self._specs[index].name

    def get_tensor_mode(self, name):
        return FakeIoMode.INPUT if self._by_name(name).is_input else FakeIoMode.OUTPUT

    def get_tensor_shape(self, name):
        return self._by_name(name).shape

    def get_tensor_dtype(self, name):
        return self._by_name(name).dtype

    def get_tensor_profile_shape(self, name, profile):
        assert profile == 0
        declared = self._by_name(name).profile
        if declared is None:
            raise RuntimeError("no profile for this tensor")
        return declared


class IndexedEngine:
    """A TensorRT 9-and-earlier engine: bindings addressed by index.

    Deliberately without ``num_io_tensors``, because that attribute is how the reader decides
    which API it is talking to.
    """

    def __init__(self, specs):
        self._specs = list(specs)

    @property
    def num_bindings(self):
        return len(self._specs)

    def get_binding_name(self, index):
        return self._specs[index].name

    def binding_is_input(self, index):
        return self._specs[index].is_input

    def get_binding_shape(self, index):
        return self._specs[index].shape

    def get_binding_dtype(self, index):
        return self._specs[index].dtype

    def get_profile_shape(self, profile, index):
        assert profile == 0
        declared = self._specs[index].profile
        if declared is None:
            raise RuntimeError("no profile for this binding")
        return declared


def static_detector(hw=(640, 640), batch=8):
    return [
        Spec("images", (batch, 3, *hw), is_input=True),
        Spec("output0", (batch, 300, 6)),
    ]


def dynamic_detector(*, batch=(1, 4, 16), hw=(640, 640)):
    low, opt, high = batch
    return [
        Spec(
            "images",
            (-1, 3, *hw),
            is_input=True,
            profile=((low, 3, *hw), (opt, 3, *hw), (high, 3, *hw)),
        ),
        Spec("output0", (-1, 300, 6)),
    ]


def read(specs, engine_class=NamedEngine, **kwargs):
    return EngineBindings.read(engine_class(specs), FakeTrt, **kwargs)


def _profileless(shape):
    """A dynamic input binding with no profile — which `read` cannot produce, but a future
    TensorRT whose accessor returns nothing useful could."""
    return Binding("images", 0, True, shape, np.dtype(np.float32))


class TestEngineBindingReading:
    """Both TensorRT IO APIs, reconciled into one description."""

    def test_the_named_api_is_read(self) -> None:
        bindings = read(static_detector())

        assert bindings.named_api is True
        assert [b.name for b in bindings.inputs] == ["images"]
        assert [b.name for b in bindings.outputs] == ["output0"]
        assert bindings.inputs[0].shape == (8, 3, 640, 640)
        assert bindings.inputs[0].dtype == np.dtype(np.float32)

    def test_the_indexed_api_gives_the_identical_description(self) -> None:
        """The parity claim the two code paths exist to satisfy. An engine built for TensorRT 8
        and one built for 10 must look the same to everything above this module."""
        specs = static_detector()

        named = read(specs, NamedEngine)
        indexed = read(specs, IndexedEngine)

        assert indexed.named_api is False
        assert named.inputs == indexed.inputs
        assert named.outputs == indexed.outputs

    def test_output_order_is_preserved_because_it_is_the_execution_order(self) -> None:
        specs = [
            Spec("images", (1, 3, 640, 640), is_input=True),
            Spec("proto", (1, 32, 160, 160)),
            Spec("dets", (1, 300, 38)),
        ]

        bindings = read(specs)

        assert [b.name for b in bindings.outputs] == ["proto", "dets"]
        assert [b.index for b in bindings.outputs] == [1, 2]

    def test_a_half_precision_binding_is_reported_as_float16(self) -> None:
        specs = [
            Spec("images", (1, 3, 640, 640), is_input=True, dtype=np.float16),
            Spec("output0", (1, 300, 6), dtype=np.float16),
        ]

        assert read(specs).inputs[0].dtype == np.dtype(np.float16)

    @pytest.mark.parametrize(
        "specs",
        [
            [Spec("images", (1, 3, 8, 8), is_input=True)],
            [Spec("output0", (1, 300, 6))],
        ],
    )
    def test_an_engine_missing_a_direction_is_refused(self, specs) -> None:
        with pytest.raises(ModelLoadError, match="at least one of each"):
            read(specs)


class TestImageInput:
    """Which input the frames go into, decided by name rather than by graph traversal order."""

    def test_a_single_input_is_unambiguous_whatever_it_is_called(self) -> None:
        specs = [Spec("input_1", (1, 3, 640, 640), is_input=True), Spec("out", (1, 300, 6))]

        assert read(specs).image_input.name == "input_1"

    def test_with_several_inputs_the_one_called_images_wins(self) -> None:
        specs = [
            Spec("scale_factor", (1, 2), is_input=True),
            Spec("images", (1, 3, 640, 640), is_input=True),
            Spec("out", (1, 300, 6)),
        ]

        assert read(specs).image_input.name == "images"

    def test_several_inputs_and_no_images_is_refused_rather_than_guessed(self) -> None:
        specs = [
            Spec("a", (1, 3, 640, 640), is_input=True),
            Spec("b", (1, 2), is_input=True),
            Spec("out", (1, 300, 6)),
        ]

        with pytest.raises(ModelLoadError, match="picking the first would be guessing"):
            read(specs).image_input


class TestInputHwDiscovery:
    """``input_hw`` comes from the artefact, and a caller who disagrees is refused.

    A configured 640x640 against a 512x512 engine does not crash — with a dynamic profile it
    runs and returns boxes scaled by 640/512 on every frame forever.
    """

    def test_a_static_extent_is_read_from_the_binding(self) -> None:
        assert read(static_detector(hw=(512, 896))).resolve_input_hw() == (512, 896)

    def test_a_request_that_agrees_is_accepted(self) -> None:
        assert read(static_detector(hw=(512, 512))).resolve_input_hw((512, 512)) == (512, 512)

    def test_a_request_that_disagrees_is_refused(self) -> None:
        bindings = read(static_detector(hw=(512, 512)))

        with pytest.raises(ModelLoadError, match="The engine is the artefact and wins"):
            bindings.resolve_input_hw((640, 640), artefact="yolo26n.engine")

    def test_a_dynamic_extent_defaults_to_the_profile_optimum(self) -> None:
        """The shape the engine's kernels were selected for; running anything else is a
        measurable loss even when it is legal."""
        specs = [
            Spec(
                "images",
                (-1, 3, -1, -1),
                is_input=True,
                profile=((1, 3, 320, 320), (4, 3, 640, 640), (8, 3, 1280, 1280)),
            ),
            Spec("output0", (-1, 300, 6)),
        ]

        assert read(specs).resolve_input_hw() == (640, 640)

    def test_a_dynamic_extent_accepts_a_request_inside_the_profile(self) -> None:
        specs = [
            Spec(
                "images",
                (-1, 3, -1, -1),
                is_input=True,
                profile=((1, 3, 320, 320), (4, 3, 640, 640), (8, 3, 1280, 1280)),
            ),
            Spec("output0", (-1, 300, 6)),
        ]

        assert read(specs).resolve_input_hw((960, 512)) == (960, 512)

    def test_a_dynamic_extent_refuses_a_request_outside_the_profile(self) -> None:
        specs = [
            Spec(
                "images",
                (-1, 3, -1, -1),
                is_input=True,
                profile=((1, 3, 320, 320), (4, 3, 640, 640), (8, 3, 1280, 1280)),
            ),
            Spec("output0", (-1, 300, 6)),
        ]

        with pytest.raises(ModelLoadError, match="outside its profile"):
            read(specs).resolve_input_hw((1600, 1600))

    def test_a_dynamic_extent_with_no_readable_profile_is_refused_at_read(self) -> None:
        """The engine is asked for its profile the moment a dynamic dimension is seen, so this
        fails at load rather than at the first letterbox."""
        specs = [
            Spec("images", (1, 3, -1, -1), is_input=True),
            Spec("output0", (1, 300, 6)),
        ]

        with pytest.raises(ModelLoadError, match="cannot run one"):
            read(specs)

    def test_a_hand_built_dynamic_binding_with_no_profile_is_refused_too(self) -> None:
        """The defence for a TensorRT whose profile accessor succeeds and returns nothing
        useful. Unreachable through `read` today, which is why it is exercised directly."""
        bindings = EngineBindings(
            inputs=(_profileless((-1, 3, -1, -1)),),
            outputs=(Binding("out", 1, False, (-1, 300, 6), np.dtype(np.float32)),),
            named_api=True,
        )

        with pytest.raises(ModelLoadError, match="nothing to letterbox to"):
            bindings.resolve_input_hw()

    def test_an_input_that_is_not_an_image_batch_is_refused(self) -> None:
        specs = [Spec("images", (1, 3, 640), is_input=True), Spec("out", (1, 300, 6))]

        with pytest.raises(ModelLoadError, match=r"\(n, c, h, w\)"):
            read(specs).resolve_input_hw()


class TestBatchBounds:
    """How many frames one execution may carry, and where that number comes from."""

    def test_a_static_batch_is_read_from_the_binding(self) -> None:
        bindings = read(static_detector(batch=12))

        assert bindings.max_batch == 12
        assert bindings.dynamic_batch is False

    def test_a_dynamic_batch_takes_the_profile_maximum(self) -> None:
        bindings = read(dynamic_detector(batch=(1, 4, 16)))

        assert bindings.max_batch == 16
        assert bindings.dynamic_batch is True

    def test_a_dynamic_batch_with_no_readable_profile_is_refused_at_read(self) -> None:
        specs = [Spec("images", (-1, 3, 8, 8), is_input=True), Spec("out", (-1, 300, 6))]

        with pytest.raises(ModelLoadError, match="cannot run one"):
            read(specs)

    def test_a_hand_built_dynamic_batch_with_no_profile_has_no_bound(self) -> None:
        bindings = EngineBindings(
            inputs=(_profileless((-1, 3, 8, 8)),),
            outputs=(Binding("out", 1, False, (-1, 300, 6), np.dtype(np.float32)),),
            named_api=True,
        )

        with pytest.raises(ModelLoadError, match="no optimisation profile"):
            bindings.max_batch

    def test_a_profile_of_differing_ranks_is_refused(self) -> None:
        specs = [
            Spec(
                "images",
                (-1, 3, 8, 8),
                is_input=True,
                profile=((1, 3, 8, 8), (4, 3, 8), (8, 3, 8, 8)),
            ),
            Spec("out", (-1, 300, 6)),
        ]

        with pytest.raises(ModelLoadError, match="differing rank"):
            read(specs)


class TestBufferSizing:
    """Every dynamic axis resolved, or a refusal — never a guess."""

    def test_the_batch_axis_takes_the_requested_batch(self) -> None:
        bindings = read(dynamic_detector(batch=(1, 4, 16)))

        assert bindings.output_shapes(4) == [(4, 300, 6)]

    def test_a_non_batch_dynamic_axis_takes_the_profile_maximum(self) -> None:
        binding = Binding(
            name="images",
            index=0,
            is_input=True,
            shape=(-1, 3, -1, -1),
            dtype=np.dtype(np.float32),
            profile=((1, 3, 320, 320), (4, 3, 640, 640), (8, 3, 1280, 1280)),
        )

        assert binding.sized(2) == (2, 3, 1280, 1280)

    def test_a_non_batch_dynamic_axis_with_no_profile_is_refused(self) -> None:
        binding = Binding(
            name="output0",
            index=1,
            is_input=False,
            shape=(-1, -1, 6),
            dtype=np.dtype(np.float32),
        )

        with pytest.raises(ModelLoadError, match="buffer overrun"):
            binding.sized(4)

    def test_a_fully_static_binding_ignores_the_batch_argument(self) -> None:
        binding = Binding(
            name="proto",
            index=1,
            is_input=False,
            shape=(1, 32, 160, 160),
            dtype=np.dtype(np.float32),
        )

        assert binding.sized(9) == (1, 32, 160, 160)


class TestHeadDiscoveryFromAnEngine:
    """The whole point of reading the outputs: the head follows from the artefact."""

    def test_a_one_output_engine_resolves_to_the_detection_head(self) -> None:
        bindings = read(static_detector())

        assert resolve_head(bindings.output_shapes(bindings.max_batch)).name == "yolo26"

    def test_a_two_output_engine_resolves_to_the_segmentation_head(self) -> None:
        specs = [
            Spec("images", (1, 3, 640, 640), is_input=True),
            Spec("output0", (1, 300, 38)),
            Spec("output1", (1, 32, 160, 160)),
        ]
        bindings = read(specs)

        assert resolve_head(bindings.output_shapes(1)).name == "yolo26_seg"

    def test_the_repr_names_the_bindings_for_a_log_line(self) -> None:
        text = repr(read(static_detector()))

        assert "images" in text and "output0" in text and "named" in text
