"""The MOTChallenge loader, on synthetic files small enough to reason about.

Every assertion here is about a decision that changes a published number: which rows count as
ground truth, which coordinate convention the file uses, and whether the empty frames survive.
The real dataset is exercised separately, in ``test_real_data.py``, and only under ``slow``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shipvision.errors import ConfigurationError
from shipvision.eval.datasets import (
    DISTRACTOR_CLASSES,
    discover_sequences,
    load_case,
    load_cases,
    load_detections,
    load_ground_truth,
    read_seqinfo,
    write_mot_file,
)
from shipvision.types import FrameTag, Track

SEQINFO = """[Sequence]
name={name}
imDir=img1
frameRate=30
seqLength={length}
imWidth=1920
imHeight=1080
imExt=.jpg
"""


def write_sequence(
    root: Path,
    *,
    name: str = "MOT17-99-FRCNN",
    length: int = 4,
    gt: str = "",
    det: str = "",
) -> Path:
    directory = root / name
    (directory / "gt").mkdir(parents=True)
    (directory / "det").mkdir(parents=True)
    (directory / "seqinfo.ini").write_text(SEQINFO.format(name=name, length=length))
    (directory / "gt" / "gt.txt").write_text(gt)
    (directory / "det" / "det.txt").write_text(det)
    return directory


class TestSeqInfo:
    def test_it_reads_the_length_the_camera_produced(self, tmp_path) -> None:
        directory = write_sequence(tmp_path, length=837)

        info = read_seqinfo(directory / "seqinfo.ini")

        assert info.length == 837
        assert (info.width, info.height) == (1920, 1080)
        assert info.frame_rate == 30.0

    def test_a_missing_file_raises_and_says_why_the_length_matters(self, tmp_path) -> None:
        with pytest.raises(ConfigurationError, match="inflates every per-frame rate"):
            read_seqinfo(tmp_path / "nope.ini")

    def test_a_file_without_seqlength_raises(self, tmp_path) -> None:
        path = tmp_path / "seqinfo.ini"
        path.write_text("[Sequence]\nname=x\n")

        with pytest.raises(ConfigurationError, match="no seqLength"):
            read_seqinfo(path)

    def test_a_zero_length_sequence_is_refused(self, tmp_path) -> None:
        path = tmp_path / "seqinfo.ini"
        path.write_text("[Sequence]\nname=x\nseqLength=0\n")

        with pytest.raises(ConfigurationError, match="division by zero"):
            read_seqinfo(path)


class TestTheGroundTruthClassFilter:
    """Only ``class == 1`` with ``conf == 1`` is scored. On MOT17-09 the file holds 10 411 rows
    and 5 325 real pedestrians; counting the rest inflates the crowd by nearly a factor of two
    and every metric computed over it is wrong in a direction that looks like a detector
    failure."""

    GT = "\n".join(
        [
            "1,1,10,20,30,60,1,1,1.0",  # pedestrian, scored
            "1,2,100,20,30,60,1,7,1.0",  # static person, a distractor
            "1,3,200,20,30,60,1,8,1.0",  # explicit distractor
            "1,4,300,20,30,60,1,12,1.0",  # reflection
            "1,5,400,20,30,60,1,2,1.0",  # person on a vehicle
            "1,6,500,20,30,60,0,9,1.0",  # occluder: neither scored nor forgiving
            "1,7,600,20,30,60,0,3,1.0",  # a car
            "2,1,12,20,30,60,1,1,1.0",
        ]
    )

    def test_only_the_pedestrians_are_scored(self, tmp_path) -> None:
        path = tmp_path / "gt.txt"
        path.write_text(self.GT)

        scored, _, _ = load_ground_truth(path, name="s", length=4)

        assert scored.num_detections == 2
        assert scored.num_ids == 1

    def test_the_four_distractor_classes_absorb_and_the_others_do_not(self, tmp_path) -> None:
        path = tmp_path / "gt.txt"
        path.write_text(self.GT)

        _, ignored, unscored = load_ground_truth(path, name="s", length=4)

        assert DISTRACTOR_CLASSES == (2, 7, 8, 12)
        assert sum(len(f) for f in ignored) == 4
        assert sum(len(f) for f in unscored) == 2

    def test_visibility_filtering_is_off_by_default(self, tmp_path) -> None:
        """A fully-occluded person is still a person, and a tracker that keeps its identity
        through the occlusion is doing the thing this library exists to do. Raising the floor
        makes every score go up and measures an easier benchmark, so it must be typed out."""
        path = tmp_path / "gt.txt"
        path.write_text("1,1,10,20,30,60,1,1,0.05\n1,2,100,20,30,60,1,1,0.9")

        default, _, _ = load_ground_truth(path, name="s", length=1)
        filtered, _, _ = load_ground_truth(path, name="s", length=1, min_visibility=0.5)

        assert default.num_detections == 2
        assert filtered.num_detections == 1

    def test_a_file_without_a_class_column_is_refused(self, tmp_path) -> None:
        """Silently treating every row as a pedestrian would overstate the crowd by a third and
        the only symptom would be a lower recall."""
        path = tmp_path / "gt.txt"
        path.write_text("1,1,10,20,30,60,1")

        with pytest.raises(ConfigurationError, match="class filter"):
            load_ground_truth(path, name="s", length=1)


class TestTheCoordinateConversion:
    """``x, y, w, h`` top-left becomes ``xyxy`` here and nowhere else."""

    def test_a_box_is_converted_exactly_once(self, tmp_path) -> None:
        path = tmp_path / "gt.txt"
        path.write_text("1,1,10,20,30,60,1,1,1.0")

        scored, _, _ = load_ground_truth(path, name="s", length=1)

        assert np.allclose(scored.frames[0].boxes[0], [10.0, 20.0, 40.0, 80.0])

    def test_a_detection_box_is_converted_the_same_way(self, tmp_path) -> None:
        path = tmp_path / "det.txt"
        path.write_text("1,-1,10,20,30,60,0.9")

        frames = load_detections(path, camera_id="c", length=1)

        assert np.allclose(frames[0].boxes[0], [10.0, 20.0, 40.0, 80.0])

    def test_writing_converts_back_so_the_round_trip_is_one_function_pair(
        self, tmp_path
    ) -> None:
        tracks = [
            Track(
                track_id=7,
                box=np.array([10.0, 20.0, 40.0, 80.0]),
                tag=FrameTag(camera_id="c", frame_id=1),
            )
        ]

        written = write_mot_file(tmp_path / "out.txt", tracks)

        assert written == 1
        assert (tmp_path / "out.txt").read_text().startswith("1,7,10.00,20.00,30.00,60.00")


class TestPublicDetections:
    def test_every_frame_is_emitted_including_the_empty_ones(self, tmp_path) -> None:
        """A tracker ages its tracks on an empty frame and eventually forgets them. Skipping the
        frames with no detections measures a tracker that never has to forget anything."""
        path = tmp_path / "det.txt"
        path.write_text("1,-1,10,20,30,60,0.9\n4,-1,10,20,30,60,0.9")

        frames = load_detections(path, camera_id="c", length=4)

        assert len(frames) == 4
        assert [len(f) for f in frames] == [1, 0, 0, 1]

    def test_the_tag_carries_the_sequence_name_as_the_camera(self, tmp_path) -> None:
        """So that feeding one tracker two sequences fails loudly instead of merging their
        identities — the same guard that stops one instance serving two real cameras."""
        path = tmp_path / "det.txt"
        path.write_text("1,-1,10,20,30,60,0.9")

        frames = load_detections(path, camera_id="MOT17-09-FRCNN", length=1, frame_rate=30.0)

        assert frames[0].tag.camera_id == "MOT17-09-FRCNN"
        assert frames[0].tag.frame_id == 1
        assert frames[0].tag.timestamp == pytest.approx(1 / 30)

    def test_the_frame_size_is_passed_through(self, tmp_path) -> None:
        """Trackers that decline to recover a track against the frame border need it, and a zero
        disables that policy silently."""
        path = tmp_path / "det.txt"
        path.write_text("1,-1,10,20,30,60,0.9")

        frames = load_detections(path, camera_id="c", length=1, height=1080, width=1920)

        assert (frames[0].height, frames[0].width) == (1080, 1920)

    def test_an_unnormalised_confidence_file_is_refused(self, tmp_path) -> None:
        """MOTChallenge public detections are normalised. An unnormalised file would make every
        score threshold in every tracker meaningless while the run still succeeded."""
        path = tmp_path / "det.txt"
        path.write_text("1,-1,10,20,30,60,42.0")

        with pytest.raises(ConfigurationError, match=r"outside \[0, 1\]"):
            load_detections(path, camera_id="c", length=1)

    def test_a_score_floor_is_applied_when_asked_and_not_otherwise(self, tmp_path) -> None:
        """ByteTrack's whole contribution is what it does with the low-scoring boxes, so
        pre-filtering the input by default would hide it."""
        path = tmp_path / "det.txt"
        path.write_text("1,-1,10,20,30,60,0.9\n1,-1,100,20,30,60,0.1")

        assert len(load_detections(path, camera_id="c", length=1)[0]) == 2
        assert len(load_detections(path, camera_id="c", length=1, min_score=0.5)[0]) == 1

    def test_an_empty_file_is_legal(self, tmp_path) -> None:
        path = tmp_path / "det.txt"
        path.write_text("")

        frames = load_detections(path, camera_id="c", length=3)

        assert [len(f) for f in frames] == [0, 0, 0]


class TestLoadCase:
    GT = "1,1,10,20,30,60,1,1,1.0\n2,1,12,20,30,60,1,1,1.0"
    DET = "1,-1,10,20,30,60,0.9\n2,-1,12,20,30,60,0.9"

    def test_it_assembles_a_case_from_a_sequence_directory(self, tmp_path) -> None:
        directory = write_sequence(tmp_path, gt=self.GT, det=self.DET, length=4)

        case = load_case(directory)

        assert case.name == "MOT17-99-FRCNN"
        assert case.num_frames == 4
        assert case.ground_truth.num_detections == 2
        assert case.metadata["people_per_frame"] == pytest.approx(0.5)

    def test_a_missing_ground_truth_raises_rather_than_scoring_zero(self, tmp_path) -> None:
        """The MOT17 test split ships without one. A loader that returned an empty ground truth
        would report MOTA 0 and look like a broken tracker."""
        directory = write_sequence(tmp_path, gt=self.GT, det=self.DET)
        (directory / "gt" / "gt.txt").unlink()

        with pytest.raises(ConfigurationError, match="test split"):
            load_case(directory)

    def test_a_missing_detection_file_raises(self, tmp_path) -> None:
        directory = write_sequence(tmp_path, gt=self.GT, det=self.DET)
        (directory / "det" / "det.txt").unlink()

        with pytest.raises(ConfigurationError, match="no detector installed"):
            load_case(directory)

    def test_truncation_is_available_for_a_smoke_run(self, tmp_path) -> None:
        directory = write_sequence(tmp_path, gt=self.GT, det=self.DET, length=4)

        case = load_case(directory, frames=1)

        assert case.num_frames == 1
        assert case.ground_truth.num_detections == 1

    def test_discovery_is_sorted_so_a_report_is_diffable(self, tmp_path) -> None:
        for name in ("MOT17-13-FRCNN", "MOT17-02-FRCNN", "MOT17-09-FRCNN"):
            write_sequence(tmp_path, name=name, gt=self.GT, det=self.DET)

        found = discover_sequences(tmp_path)

        assert [f.name for f in found] == [
            "MOT17-02-FRCNN",
            "MOT17-09-FRCNN",
            "MOT17-13-FRCNN",
        ]

    def test_a_directory_with_no_sequences_raises(self, tmp_path) -> None:
        with pytest.raises(ConfigurationError, match="no sequences under"):
            discover_sequences(tmp_path)

    def test_a_named_sequence_that_does_not_exist_raises(self, tmp_path) -> None:
        """A typo in a sequence list that silently evaluates six sequences instead of seven
        produces a number nobody can reproduce."""
        write_sequence(tmp_path, gt=self.GT, det=self.DET)

        with pytest.raises(ConfigurationError, match="no such sequence"):
            load_cases(tmp_path, sequences=["MOT17-04-FRCNN"])

    def test_the_named_subset_is_returned_in_the_order_asked_for(self, tmp_path) -> None:
        for name in ("MOT17-02-FRCNN", "MOT17-09-FRCNN"):
            write_sequence(tmp_path, name=name, gt=self.GT, det=self.DET)

        cases = load_cases(tmp_path, sequences=["MOT17-09-FRCNN", "MOT17-02-FRCNN"])

        assert [c.name for c in cases] == ["MOT17-09-FRCNN", "MOT17-02-FRCNN"]
