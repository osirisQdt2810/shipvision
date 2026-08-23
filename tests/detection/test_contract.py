"""The contract itself: the ABC, the typed failure, and the public surface.

Small claims, but they are the ones every other lane codes against.
"""

from __future__ import annotations

import numpy as np
import pytest

import shipvision.detection as detection
from shipvision.detection import (
    DETECTORS,
    DetectionError,
    Detector,
    empty_detections,
    frame_hw,
    frame_image,
)
from shipvision.errors import ConfigurationError, InferenceError, ShipVisionError
from shipvision.types import Detections, Frame, FrameTag


class TestDetectorContract:
    """Both abstract members are abstract, and the convenience wrapper is not the interface."""

    def test_the_abstract_base_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            Detector()  # type: ignore[abstract]

    def test_a_subclass_must_supply_both_members(self) -> None:
        class OnlyDetects(Detector):
            def detect(self, frames):
                return []

        with pytest.raises(TypeError, match="input_hw"):
            OnlyDetects()  # type: ignore[abstract]

    def test_detect_one_delegates_to_detect(self) -> None:
        class Counting(Detector):
            calls = 0

            @property
            def input_hw(self):
                return (8, 8)

            def detect(self, frames):
                type(self).calls += 1
                return [Detections(tag=f.tag) for f in frames]

        detector = Counting()
        result = detector.detect_one(
            Frame(FrameTag("cam-01", 3), image=None, height=8, width=8)
        )

        assert result.tag == FrameTag("cam-01", 3)
        assert Counting.calls == 1

    def test_the_repr_names_the_entry_and_the_extent(self) -> None:
        detector = DETECTORS.build("mock", input_hw=(320, 576))

        assert repr(detector) == (
            "<MockDetector name='mock' backend='python' input_hw=(320, 576)>"
        )


class TestDetectionError:
    """The tag survives the error path, and existing handlers still catch it."""

    def test_it_is_an_inference_error_and_a_shipvision_error(self) -> None:
        assert issubclass(DetectionError, InferenceError)
        assert issubclass(DetectionError, ShipVisionError)

    def test_it_carries_the_tag_as_an_attribute_not_only_in_the_message(self) -> None:
        """A server that must attribute a gap to a camera cannot parse a message to do it."""
        tag = FrameTag("cam-22", 91)

        failure = DetectionError("the engine returned an error", tag=tag)

        assert failure.tag is tag
        assert "cam-22#91" in str(failure)

    def test_an_unattributable_failure_has_no_tag_and_says_nothing_about_one(self) -> None:
        """An engine-wide fault is not one frame's fault, and inventing a tag for it would put a
        real-looking failure on a camera that was fine."""
        failure = DetectionError("the device fell off the bus")

        assert failure.tag is None
        assert str(failure) == "the device fell off the bus"


class TestFrameAccessors:
    """Where a frame's coordinate space comes from, and what happens when it is missing."""

    def test_the_declared_extent_wins_when_there_are_no_pixels(self) -> None:
        assert frame_hw(Frame(FrameTag("c", 0), image=None, height=720, width=1280)) == (
            720,
            1280,
        )

    def test_the_pixels_are_used_when_nothing_is_declared(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        assert frame_hw(Frame(FrameTag("c", 0), image=image)) == (480, 640)

    def test_agreement_between_the_two_is_accepted(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(FrameTag("c", 0), image=image, height=480, width=640)

        assert frame_hw(frame) == (480, 640)

    def test_a_non_array_image_is_refused_with_the_frame_named(self) -> None:
        with pytest.raises(ConfigurationError, match="cam-05#2"):
            frame_image(Frame(FrameTag("cam-05", 2), image="a device handle"))

    def test_an_array_image_is_handed_through_untouched(self) -> None:
        """Validation belongs to imgproc, which owns the dtype and layout rule."""
        image = np.zeros((4, 4, 3), dtype=np.uint8)

        assert frame_image(Frame(FrameTag("c", 0), image=image)) is image


class TestEmptyDetections:
    """The answer for a quiet frame: tagged, sized, and shaped for a downstream slice."""

    def test_it_carries_the_tag_and_the_extent(self) -> None:
        tag = FrameTag("cam-01", 8)

        result = empty_detections(tag, 1080, 1920)

        assert result.tag is tag
        assert (result.height, result.width) == (1080, 1920)

    def test_its_boxes_are_shaped_for_a_downstream_slice(self) -> None:
        """``(0,)`` breaks every downstream ``[:, 2]`` with an IndexError instead of yielding an
        empty result, and an empty frame is ordinary input."""
        result = empty_detections(FrameTag("cam-01", 0), 4, 4)

        assert len(result) == 0
        assert result.boxes.shape == (0, 4)


class TestPublicSurface:
    """Everything ``__all__`` promises resolves, including the lazily-imported backends."""

    def test_every_exported_name_resolves(self) -> None:
        missing = [name for name in detection.__all__ if not hasattr(detection, name)]

        assert missing == []

    def test_the_lazy_backends_are_part_of_the_promised_surface(self) -> None:
        """They are resolved through ``__getattr__``, so a plain ``hasattr`` sweep is not enough
        to show they are reachable by name."""
        assert detection.TensorRTDetector.__name__ == "TensorRTDetector"
        assert detection.TorchDetector.__name__ == "TorchDetector"

    def test_an_unknown_attribute_is_an_attribute_error(self) -> None:
        with pytest.raises(AttributeError, match="Yolo42Detector"):
            detection.Yolo42Detector  # noqa: B018

    def test_the_registry_lists_exactly_the_families_this_package_owns(self) -> None:
        assert DETECTORS.names() == ["mock", "yolo26"]
        assert DETECTORS.backends("yolo26") == ["tensorrt", "torch"]

    def test_there_is_no_fallback_builder_that_could_resolve_to_the_mock(self) -> None:
        """A missing engine quietly becoming a mock is a deployment that reports a successful
        start-up and detects nothing real."""
        assert not hasattr(detection, "build_detector")
