"""What a metric is handed, and what it refuses.

The refusals matter more than the accessors. A file whose ids do not mean what a metric
assumes produces a plausible number, and the only place that can be caught is at construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipvision.errors import ConfigurationError
from shipvision.eval.sequence import EvaluationCase, ObjectFrame, TrackSequence
from shipvision.types import FrameTag, Track

from .conftest import box, detections, frame, sequence


class TestObjectFrame:
    def test_it_refuses_the_same_identity_twice_in_one_frame(self) -> None:
        """Not a degraded measurement — a file whose ids do not mean one object per frame. The
        metrics would count the repeat as two objects that happen to be the same person, and
        HOTA's fancy-indexed accumulation would silently drop one of the two updates."""
        with pytest.raises(ConfigurationError, match="repeats an identity"):
            ObjectFrame(frame_id=1, ids=np.array([7, 7]), boxes=np.stack([box(0.0, 0.0)] * 2))

    def test_it_refuses_a_mismatch_between_ids_and_boxes(self) -> None:
        with pytest.raises(ConfigurationError, match="2 ids and 1 boxes"):
            ObjectFrame(frame_id=1, ids=np.array([7, 8]), boxes=np.stack([box(0.0, 0.0)]))

    def test_an_empty_frame_is_legal_and_has_length_zero(self) -> None:
        empty = ObjectFrame(frame_id=1, ids=np.empty(0), boxes=np.zeros((0, 4)))

        assert len(empty) == 0
        assert empty.boxes.shape == (0, 4)

    def test_ids_are_int64_and_boxes_float32_whatever_came_in(self) -> None:
        built = ObjectFrame(frame_id=1, ids=[7.0], boxes=[[0, 0, 30, 60]])

        assert built.ids.dtype == np.int64
        assert built.boxes.dtype == np.float32


class TestTrackSequence:
    def test_frames_are_held_sorted_however_they_arrived(self) -> None:
        out_of_order = TrackSequence(
            name="s", frames=(frame(3, [(1, 0.0)]), frame(1, [(1, 0.0)]), frame(2, [(1, 0.0)]))
        )

        assert out_of_order.frame_ids == (1, 2, 3)

    def test_two_entries_for_one_frame_are_refused(self) -> None:
        """A sequence is a map from frame to objects, not a list of appends. Two entries for one
        frame would be silently concatenated by every metric here."""
        with pytest.raises(ConfigurationError, match="two entries for one frame"):
            TrackSequence(name="s", frames=(frame(1, [(1, 0.0)]), frame(1, [(2, 0.0)])))

    def test_length_is_the_camera_length_not_the_annotated_one(self) -> None:
        """A sequence with detections on 40 of its 600 frames still has 600 frames, and every
        per-frame rate in a report divides by that."""
        built = TrackSequence(name="s", frames=(frame(1, [(1, 0.0)]),), length=600)

        assert built.length == 600
        assert len(built) == 1

    def test_length_cannot_be_smaller_than_the_frames_present(self) -> None:
        built = TrackSequence(
            name="s", frames=tuple(frame(t, [(1, 0.0)]) for t in range(1, 6)), length=2
        )

        assert built.length == 5

    def test_num_ids_counts_distinct_identities_across_the_whole_sequence(self) -> None:
        built = sequence(
            "s", [frame(1, [(1, 0.0), (2, 100.0)]), frame(2, [(2, 100.0), (3, 200.0)])]
        )

        assert built.num_ids == 3
        assert built.num_detections == 4

    def test_an_empty_sequence_reports_zero_rather_than_raising(self) -> None:
        empty = TrackSequence.empty("nothing", length=10)

        assert (len(empty), empty.num_ids, empty.num_detections) == (0, 0, 0)
        assert empty.length == 10


class TestFromTracks:
    def test_a_flat_stream_is_grouped_by_the_tag_frame_id(self) -> None:
        """The frame number comes from the track's own tag, which is the whole reason the tag
        travels with a result."""
        tracks = [
            Track(track_id=7, box=box(0.0, 0.0), tag=FrameTag(camera_id="c", frame_id=1)),
            Track(track_id=8, box=box(100.0, 0.0), tag=FrameTag(camera_id="c", frame_id=1)),
            Track(track_id=7, box=box(5.0, 0.0), tag=FrameTag(camera_id="c", frame_id=2)),
        ]

        built = TrackSequence.from_tracks("s", tracks, length=2)

        assert built.frame_ids == (1, 2)
        assert built.num_detections == 3
        assert sorted(built.frames[0].ids.tolist()) == [7, 8]

    def test_reordered_publication_still_produces_the_right_sequence(self) -> None:
        tracks = [
            Track(track_id=7, box=box(5.0, 0.0), tag=FrameTag(camera_id="c", frame_id=2)),
            Track(track_id=7, box=box(0.0, 0.0), tag=FrameTag(camera_id="c", frame_id=1)),
        ]

        assert TrackSequence.from_tracks("s", tracks).frame_ids == (1, 2)


class TestEvaluationCase:
    def test_it_refuses_a_case_that_mixes_two_cameras(self) -> None:
        """One tracker serves one camera, so a case that spans two measures a configuration
        nobody deploys — and would make the tracker's own tag guard fire mid-run."""
        first = detections(1, [0.0])
        second = detections(2, [0.0])
        object.__setattr__(second, "tag", FrameTag(camera_id="other", frame_id=2))

        with pytest.raises(ConfigurationError, match="mixes cameras"):
            EvaluationCase(
                name="mixed",
                detections=(first, second),
                ground_truth=sequence("gt", [frame(1, [(1, 0.0)])]),
            )

    def test_truncation_shortens_the_ground_truth_too(self, simple_case) -> None:
        """Truncating the input alone leaves the ground truth counting objects the tracker was
        never shown, which reads as a collapse in recall rather than as a shorter run."""
        short = simple_case.truncated(2)

        assert short.num_frames == 2
        assert short.ground_truth.frame_ids == (1, 2)
        assert short.ground_truth.num_detections == 4

    def test_truncation_to_zero_frames_is_refused(self, simple_case) -> None:
        with pytest.raises(ConfigurationError, match="frames must be positive"):
            simple_case.truncated(0)

    def test_the_input_detection_count_is_reported(self, simple_case) -> None:
        assert simple_case.num_input_detections == 6
        assert simple_case.num_frames == 3
