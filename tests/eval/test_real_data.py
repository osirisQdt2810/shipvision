"""The real MOT17 train split. Marked ``slow`` and skipped when the data is absent.

These are the tests that would catch a loader that reads a synthetic file correctly and a real
one wrongly — a class column in a different position, a sequence whose annotations stop before
its last frame, a detection file with an unnormalised confidence. The offline tier cannot catch
any of that, and it must not have to.

The measured densities are asserted, not assumed. They come from ``scripts/fetch_datasets.py
--report`` and they are how the loader's class filter is checked against something outside
itself: get the filter wrong and MOT17-09 reports 19.8 people per frame instead of 10.1,
because the file holds 10 411 rows of which 5 325 are pedestrians.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shipvision.eval import evaluate, format_table, load_case
from shipvision.eval.datasets import discover_sequences
from shipvision.tracking import TRACKERS

pytestmark = pytest.mark.slow

#: ``sequence -> (frames, ground-truth detections, people per frame)``, measured. The middle
#: column is the one a wrong class filter breaks.
EXPECTED = {
    "MOT17-02-FRCNN": (600, 18581, 31.0),
    "MOT17-04-FRCNN": (1050, 47557, 45.3),
    "MOT17-05-FRCNN": (837, 6917, 8.3),
    "MOT17-09-FRCNN": (525, 5325, 10.1),
    "MOT17-10-FRCNN": (654, 12839, 19.6),
    "MOT17-11-FRCNN": (900, 9436, 10.5),
    "MOT17-13-FRCNN": (750, 11642, 15.5),
}

#: The sequences inside this library's operating point of 10-20 people per frame. MOT17-02 and
#: MOT17-04 are deliberately excluded from the on-target set and reported separately: at 31 and
#: 45 people per frame they are a different problem, and averaging them in hides which regime a
#: number came from.
ON_TARGET = (
    "MOT17-05-FRCNN",
    "MOT17-09-FRCNN",
    "MOT17-10-FRCNN",
    "MOT17-11-FRCNN",
    "MOT17-13-FRCNN",
)


class TestTheLoaderAgreesWithTheMeasuredDensities:
    def test_all_seven_sequences_are_found(self, mot17_root: Path) -> None:
        found = discover_sequences(mot17_root)

        assert [f.name for f in found] == sorted(EXPECTED)

    @pytest.mark.parametrize("name", sorted(EXPECTED))
    def test_the_frame_count_and_the_crowd_size_match(
        self, mot17_root: Path, name: str
    ) -> None:
        frames, detections_, density = EXPECTED[name]

        case = load_case(mot17_root / name)

        assert case.num_frames == frames
        assert case.ground_truth.num_detections == detections_
        assert case.metadata["people_per_frame"] == pytest.approx(density, abs=0.05)

    @pytest.mark.parametrize("name", sorted(EXPECTED))
    def test_the_distractor_rows_are_separated_rather_than_counted(
        self, mot17_root: Path, name: str
    ) -> None:
        """MOT17-09's file holds 10 411 rows against 5 325 pedestrians. Counting the rest would
        report a crowd nearly twice the real one, and every metric over it would look like a
        detector failure."""
        case = load_case(mot17_root / name)
        extra = sum(len(f) for f in case.ignored) + sum(len(f) for f in case.unscored)

        assert extra > 0
        assert case.ground_truth.num_detections == EXPECTED[name][1]

    @pytest.mark.parametrize("name", sorted(EXPECTED))
    def test_every_frame_is_present_including_the_ones_with_no_detections(
        self, mot17_root: Path, name: str
    ) -> None:
        case = load_case(mot17_root / name)

        assert len(case.detections) == EXPECTED[name][0]
        assert [d.tag.frame_id for d in case.detections[:3]] == [1, 2, 3]
        # The frame size comes from seqinfo.ini and is not the same for all seven: MOT17-05
        # is 640x480 while the rest are 1920x1080. A tracker that declines to recover a track
        # against the frame border needs the real numbers, and a hard-coded 1080 would silently
        # disable that policy on the one sequence where the border is closest.
        assert case.detections[0].height > 0
        assert case.detections[0].width > 0

    def test_the_public_detections_are_normalised_and_below_the_ground_truth_count(
        self, mot17_root: Path
    ) -> None:
        """Public detections are the input, and they have worse recall than the ground truth by
        design — which is why MOTA on this benchmark is dominated by the detector and HOTA is
        the better tuning objective."""
        case = load_case(mot17_root / "MOT17-09-FRCNN")
        scores = np.concatenate([d.scores for d in case.detections if len(d)])

        assert 0.0 <= scores.min() and scores.max() <= 1.0
        assert case.num_input_detections < case.ground_truth.num_detections


class TestATrackerScoresPlausiblyOnRealFootage:
    """Not a quality claim about the trackers — a claim that the whole path works end to end on
    real data and lands in the range the published MOT17 leaderboard occupies. A metric bug
    large enough to matter puts these outside the bracket."""

    @pytest.mark.parametrize("name", ["sort", "bytetrack", "ocsort", "botsort", "deepsortv2"])
    def test_every_tracker_lands_in_the_published_range_on_mot17_09(
        self, mot17_root: Path, name: str
    ) -> None:
        case = load_case(mot17_root / "MOT17-09-FRCNN")

        result = evaluate(TRACKERS.build(name), case)

        assert 0.30 < result.hota.hota < 0.70, f"{name} HOTA {result.hota.hota:.4f}"
        assert 0.30 < result.clear.mota < 0.75, f"{name} MOTA {result.clear.mota:.4f}"
        assert 0.40 < result.identity.idf1 < 0.85
        assert result.clear.motp > 0.80
        assert result.clear.id_switches < result.num_gt_ids * 20

    def test_the_on_target_sequences_produce_a_table(self, mot17_root: Path) -> None:
        """One row per sequence and a summed aggregate — the shape a PR is supposed to paste."""
        results = [
            evaluate(TRACKERS.build("sort"), load_case(mot17_root / name, frames=120))
            for name in ON_TARGET
        ]

        text = format_table(results, title="sort / on-target / first 120 frames")

        assert all(name in text for name in ON_TARGET)
        assert "COMBINED" in text

    def test_the_ms_per_frame_is_measured_and_under_the_budget(self, mot17_root: Path) -> None:
        """1000 frames per second across fifty cameras is the target. A single-camera tracker
        that costs more than a millisecond a frame cannot be part of it, so the number is
        asserted rather than merely printed — loosely, because this runs on shared CI."""
        case = load_case(mot17_root / "MOT17-09-FRCNN")

        result = evaluate(TRACKERS.build("sort"), case)

        assert 0.0 < result.ms_per_frame < 10.0
