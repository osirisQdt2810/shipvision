"""The mock detector, which the tracking, MTMC and pipeline lanes all stand on.

If it stops being deterministic, or stops producing smooth motion, a whole tier of tests
elsewhere starts passing for the wrong reason — so its own tests are load-bearing.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from shipvision.detection import DETECTORS, DetectionError, MockDetector, frame_hw
from shipvision.errors import ConfigurationError, InferenceError
from shipvision.types import Frame, FrameTag, iou_matrix

from .conftest import LANDSCAPE, frame, grey_frame


class TestMockRegistration:
    """It is a name of its own, never a backend of ``yolo26``."""

    def test_it_is_registered_under_mock_with_aliases(self) -> None:
        assert DETECTORS.get("mock") is MockDetector
        assert DETECTORS.get("fake") is MockDetector
        assert DETECTORS.backends("mock") == ["python"]

    def test_it_is_not_a_backend_of_the_artefact_detector(self) -> None:
        """If it were, ``build("yolo26")`` on a machine with no engine would resolve to it and a
        deployment would report a successful start-up while detecting nothing real."""
        assert "python" not in DETECTORS.backends("yolo26")

    def test_it_reports_the_configured_input_extent(self) -> None:
        """The one detector whose ``input_hw`` is configured, because it has no artefact to read
        it from — and nothing about its output depends on it."""
        detector = MockDetector(input_hw=(512, 896))

        assert detector.input_hw == (512, 896)
        assert "512" in repr(detector)


class TestMockDeterminism:
    """The same ``(camera_id, frame_id)`` gives the same detections, always and everywhere."""

    def test_two_calls_agree_exactly(self) -> None:
        detector = MockDetector(objects=4, jitter=2.0)

        first = detector.detect_one(frame("cam-03", 17))
        second = detector.detect_one(frame("cam-03", 17))

        assert np.array_equal(first.boxes, second.boxes)
        assert first.scores.tolist() == second.scores.tolist()

    def test_two_separately_built_detectors_agree(self) -> None:
        """No hidden state: the scene is a function of the seed and the tag, not of call order."""
        one = MockDetector(objects=4, jitter=2.0)
        other = MockDetector(objects=4, jitter=2.0)

        other.detect_one(frame("cam-99", 3))
        assert np.array_equal(
            one.detect_one(frame("cam-03", 17)).boxes,
            other.detect_one(frame("cam-03", 17)).boxes,
        )

    def test_different_cameras_get_different_scenes(self) -> None:
        detector = MockDetector(objects=3)

        left = detector.detect_one(frame("cam-01", 0)).boxes
        right = detector.detect_one(frame("cam-02", 0)).boxes

        assert not np.allclose(left, right)

    def test_a_different_seed_is_a_different_world(self) -> None:
        assert not np.allclose(
            MockDetector(objects=3, seed=0).detect_one(frame()).boxes,
            MockDetector(objects=3, seed=1).detect_one(frame()).boxes,
        )

    @pytest.mark.slow
    def test_the_scene_does_not_depend_on_pythonhashseed(self) -> None:
        """``hash`` is salted per process for `str`, so a scene keyed on a camera id with
        ``hash`` would differ between runs — and a test whose expected answer depends on
        ``PYTHONHASHSEED`` is worse than no test."""
        script = textwrap.dedent("""
            from shipvision.detection import MockDetector
            from shipvision.types import Frame, FrameTag
            detector = MockDetector(objects=3, jitter=1.0)
            result = detector.detect_one(
                Frame(FrameTag("cam-42", 11), image=None, height=1080, width=1920)
            )
            print(";".join(f"{v:.6f}" for box in result.boxes for v in box))
            """)
        outputs = []
        for seed in ("0", "12345"):
            environment = {**os.environ, "PYTHONHASHSEED": seed}
            outputs.append(
                subprocess.run(
                    [sys.executable, "-c", script],
                    check=True,
                    capture_output=True,
                    text=True,
                    env=environment,
                ).stdout.strip()
            )

        assert outputs[0] == outputs[1]
        assert outputs[0]


class TestMockMotion:
    """Objects persist, move smoothly and stay in frame — which is what a tracker is tested on."""

    def test_an_object_moves_by_a_constant_step_between_frames(self) -> None:
        detector = MockDetector(objects=1, speed=0.01, jitter=0.0)

        boxes = [detector.detect_one(frame("cam-01", i)).boxes[0] for i in range(6)]
        steps = [np.linalg.norm(boxes[i + 1][:2] - boxes[i][:2]) for i in range(5)]

        expected = 0.01 * min(LANDSCAPE)
        assert steps == pytest.approx([expected] * 5, abs=1e-3)

    def test_consecutive_frames_overlap_enough_to_associate(self) -> None:
        """The property that makes a tracking test meaningful: a nearest-IoU association has a
        correct answer to find."""
        detector = MockDetector(objects=4, speed=0.01, jitter=1.0)

        previous = detector.detect_one(frame("cam-01", 40))
        current = detector.detect_one(frame("cam-01", 41))

        overlap = iou_matrix(previous.boxes, current.boxes)
        assert np.diag(overlap).min() > 0.7
        # And the diagonal must be the best match, or the scene is too crowded to be a test.
        assert (np.argmax(overlap, axis=1) == np.arange(len(previous))).all()

    def test_boxes_stay_inside_the_frame_for_a_long_sequence(self) -> None:
        """Reflected rather than wrapped: a wrapped object teleports across the scene, which no
        tracker should follow and every tracker would be judged on."""
        detector = MockDetector(objects=5, speed=0.03, jitter=0.0)

        for frame_id in range(0, 4000, 37):
            boxes = detector.detect_one(frame("cam-08", frame_id)).boxes
            assert (boxes[:, 0] >= 0.0).all() and (boxes[:, 2] <= LANDSCAPE[1]).all()
            assert (boxes[:, 1] >= 0.0).all() and (boxes[:, 3] <= LANDSCAPE[0]).all()

    def test_no_teleport_anywhere_in_a_long_sequence(self) -> None:
        detector = MockDetector(objects=3, speed=0.02, jitter=0.0)
        step = 0.02 * min(LANDSCAPE)

        centres = [
            np.stack([d.centre for d in detector.detect_one(frame("cam-08", i))])
            for i in range(600)
        ]
        jumps = [np.abs(centres[i + 1] - centres[i]).max() for i in range(len(centres) - 1)]

        assert max(jumps) <= step + 1e-3

    def test_jitter_perturbs_the_trajectory_without_replacing_it(self) -> None:
        smooth = MockDetector(objects=2, jitter=0.0)
        noisy = MockDetector(objects=2, jitter=3.0)

        clean = smooth.detect_one(frame("cam-01", 5)).boxes
        rough = noisy.detect_one(frame("cam-01", 5)).boxes

        assert not np.allclose(clean, rough)
        assert np.abs(clean - rough).max() < 40.0

    def test_zero_jitter_is_exactly_reproducible_frame_to_frame(self) -> None:
        """With ``jitter=0`` the motion is linear, so a two-frame extrapolation predicts the
        third exactly — which is what lets a Kalman test separate its filter from its data."""
        detector = MockDetector(objects=1, jitter=0.0)

        boxes = [detector.detect_one(frame("cam-01", i)).boxes[0] for i in (10, 11, 12)]

        assert (2 * boxes[1] - boxes[0]).tolist() == pytest.approx(boxes[2].tolist(), abs=1e-3)


class TestMockScene:
    """Object counts, class mixes and geometry, all with stateable answers."""

    def test_a_fixed_count_is_exact_on_every_camera(self) -> None:
        detector = MockDetector(objects=7)

        for camera in ("cam-01", "cam-02", "cam-03"):
            assert len(detector.detect_one(frame(camera, 0))) == 7

    def test_a_range_varies_between_cameras_and_stays_fixed_within_one(self) -> None:
        """Uneven cameras are the point: the failure ShipInfer exists to fix is a crowded camera
        starving a quiet one, and a load-balancing test needs a skewed scene to show it."""
        detector = MockDetector(objects=(1, 6))

        counts = {
            camera: {len(detector.detect_one(frame(camera, i))) for i in range(5)}
            for camera in (f"cam-{i:02d}" for i in range(12))
        }

        assert all(len(seen) == 1 for seen in counts.values())
        assert len({next(iter(seen)) for seen in counts.values()}) > 1
        assert all(1 <= next(iter(seen)) <= 6 for seen in counts.values())

    def test_zero_objects_gives_an_empty_detections_that_still_carries_its_tag(self) -> None:
        result = MockDetector(objects=0).detect_one(frame("cam-04", 9))

        assert len(result) == 0
        assert result.boxes.shape == (0, 4)
        assert result.tag == FrameTag("cam-04", 9)

    def test_an_unweighted_class_mix_is_round_robin(self) -> None:
        result = MockDetector(objects=6, class_mix=(0, 3, 7)).detect_one(frame())

        assert sorted(d.class_id for d in result) == [0, 0, 3, 3, 7, 7]

    def test_a_weighted_class_mix_is_exact_rather_than_sampled(self) -> None:
        """Laid out along the cumulative weight, not drawn from it, so ten objects at 90/10 are
        exactly nine and one — an answer a test can state."""
        result = MockDetector(objects=10, class_mix={0: 9.0, 5: 1.0}).detect_one(frame())

        assert sorted(d.class_id for d in result) == [0] * 9 + [5]

    def test_results_come_back_in_descending_score_order(self) -> None:
        """As every real head returns them, and stable across a sequence because the scores are
        per object rather than per frame."""
        detector = MockDetector(objects=6)

        for frame_id in (0, 5, 50):
            scores = detector.detect_one(frame("cam-01", frame_id)).scores
            assert scores.tolist() == sorted(scores.tolist(), reverse=True)

    def test_scores_stay_inside_the_configured_range(self) -> None:
        result = MockDetector(objects=8, score_range=(0.3, 0.35)).detect_one(frame())

        assert all(0.3 <= d.score <= 0.35 for d in result)

    def test_box_heights_follow_the_size_range_and_the_aspect(self) -> None:
        detector = MockDetector(objects=8, size_range=(0.1, 0.2), aspect=0.5)

        for detection in detector.detect_one(frame()):
            assert 0.1 * LANDSCAPE[0] - 1 <= detection.height <= 0.2 * LANDSCAPE[0] + 1
            assert detection.width == pytest.approx(detection.height * 0.5, rel=1e-3)

    def test_a_batch_comes_back_aligned_with_its_frames(self) -> None:
        detector = MockDetector(objects=2)
        frames = [frame("cam-01", 1), frame("cam-02", 4), frame("cam-01", 2)]

        results = detector.detect(frames)

        assert [r.tag for r in results] == [f.tag for f in frames]

    def test_an_empty_batch_gives_an_empty_list(self) -> None:
        assert MockDetector().detect([]) == []


class TestMockFailure:
    """A detector that throws is a path the server must handle, and the tag must survive it."""

    def test_the_failing_frames_are_the_ones_named_by_fail_every(self) -> None:
        detector = MockDetector(objects=1, fail_every=3)
        failed = []

        for frame_id in range(9):
            try:
                detector.detect_one(frame("cam-01", frame_id))
            except DetectionError:
                failed.append(frame_id)

        assert failed == [0, 3, 6]

    def test_which_frames_fail_does_not_depend_on_how_the_batches_were_cut(self) -> None:
        """Keyed on the frame id, not on a call counter, so a test can name the failing frames
        regardless of the batcher."""
        detector = MockDetector(objects=1, fail_every=4)

        with pytest.raises(DetectionError) as one_at_a_time:
            detector.detect_one(frame("cam-01", 8))
        with pytest.raises(DetectionError) as in_a_batch:
            detector.detect([frame("cam-01", 5), frame("cam-01", 8), frame("cam-01", 9)])

        assert one_at_a_time.value.tag == in_a_batch.value.tag == FrameTag("cam-01", 8)

    def test_the_failure_carries_the_frame_it_happened_on(self) -> None:
        """A server that must attribute a gap to a camera cannot parse a message to do it."""
        detector = MockDetector(objects=1, fail_every=5)

        with pytest.raises(DetectionError) as raised:
            detector.detect_one(frame("cam-77", 10))

        assert raised.value.tag == FrameTag("cam-77", 10)
        assert raised.value.tag.camera_id == "cam-77"
        assert "cam-77#10" in str(raised.value)

    def test_it_is_an_inference_error_so_existing_handlers_still_catch_it(self) -> None:
        detector = MockDetector(objects=1, fail_every=1)

        with pytest.raises(InferenceError):
            detector.detect_one(frame("cam-01", 0))

    def test_fail_every_none_never_fails(self) -> None:
        detector = MockDetector(objects=1, fail_every=None)

        assert len(detector.detect([frame("cam-01", i) for i in range(20)])) == 20


class TestMockFrameExtent:
    """Boxes are in the source frame's pixels, and where that extent comes from is stated."""

    def test_a_frame_with_no_pixels_works_from_its_declared_extent(self) -> None:
        """What lets the tracking and MTMC lanes drive this with no images at all."""
        result = MockDetector(objects=3).detect_one(
            Frame(FrameTag("cam-01", 0), image=None, height=720, width=1280)
        )

        assert (result.height, result.width) == (720, 1280)
        assert (result.boxes[:, 2] <= 1280).all()

    def test_a_frame_with_only_pixels_works_from_their_shape(self) -> None:
        result = MockDetector(objects=2).detect_one(grey_frame("cam-01", 0, (480, 640)))

        assert (result.height, result.width) == (480, 640)

    def test_a_declared_extent_that_contradicts_the_pixels_is_refused(self) -> None:
        """Something resized the pixels without updating the frame, and every box derived from
        it is scaled wrongly with no other symptom."""
        image = np.zeros((540, 960, 3), dtype=np.uint8)
        mismatched = Frame(FrameTag("cam-01", 0), image=image, height=1080, width=1920)

        with pytest.raises(ConfigurationError, match="without updating the frame"):
            MockDetector().detect_one(mismatched)

    def test_a_frame_with_neither_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="no coordinate space"):
            frame_hw(Frame(FrameTag("cam-01", 0), image=None))

    def test_the_extent_scales_the_scene_rather_than_clipping_it(self) -> None:
        detector = MockDetector(objects=4, size_range=(0.2, 0.2))

        small = detector.detect_one(frame("cam-01", 0, (270, 480)))
        large = detector.detect_one(frame("cam-01", 0, (1080, 1920)))

        assert all(d.height == pytest.approx(0.2 * 270, abs=1) for d in small)
        assert all(d.height == pytest.approx(0.2 * 1080, abs=1) for d in large)


class TestMockValidation:
    """A bad knob stops the process at construction, not on frame 40 000."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"input_hw": (0, 640)},
            {"input_hw": (640,)},
            {"objects": -1},
            {"objects": (5, 2)},
            {"class_mix": ()},
            {"class_mix": {}},
            {"class_mix": (-1,)},
            {"class_mix": {0: 0.0}},
            {"jitter": -1.0},
            {"speed": -0.1},
            {"aspect": 0.0},
            {"score_range": (0.9, 0.1)},
            {"score_range": (0.0, 1.5)},
            {"size_range": (-0.1, 0.2)},
            {"fail_every": 0},
            {"fail_every": -3},
        ],
    )
    def test_an_impossible_knob_is_refused(self, kwargs) -> None:
        with pytest.raises(ConfigurationError):
            MockDetector(**kwargs)

    def test_it_can_be_built_through_the_registry_with_every_knob(self) -> None:
        detector = DETECTORS.build(
            "mock",
            objects=(2, 5),
            class_mix={0: 3.0, 1: 1.0},
            jitter=1.5,
            speed=0.02,
            score_range=(0.4, 0.8),
            size_range=(0.05, 0.3),
            aspect=0.75,
            fail_every=None,
            seed=7,
        )

        assert 2 <= len(detector.detect_one(frame())) <= 5
