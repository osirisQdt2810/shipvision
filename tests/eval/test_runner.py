"""Driving a tracker over a case, and the two hazards in doing so.

Both hazards are aliasing hazards, and both produce a plausible number rather than an error:
a tracker's published tracks are live objects it will mutate, and a tracker's state is per
camera so reusing an instance carries one sequence's tracks into the next.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipvision.errors import ConfigurationError, TrackingError
from shipvision.eval.runner import evaluate, evaluate_all, run, score
from shipvision.eval.sequence import EvaluationCase, TrackSequence
from shipvision.tracking import TRACKERS
from shipvision.types import Detections, FrameTag, Track, TrackState

from .conftest import box, detections, frame, sequence


class MutatingTracker:
    """A tracker that publishes one live :class:`Track` and mutates it every frame.

    This is not a strawman: it is what
    :class:`~shipvision.tracking.pool.TrackPool` does. ``output()`` returns references to the
    pool's own objects, so a caller that buffers a whole run and reads the ids afterwards gets
    the last frame's state on every entry.
    """

    def __init__(self) -> None:
        self.track = Track(
            track_id=1, box=box(0.0, 0.0), tag=FrameTag(camera_id="c", frame_id=1)
        )

    def reset(self) -> None:
        return None

    def update(self, dets: Detections, *, image: np.ndarray | None = None) -> list[Track]:
        self.track.track_id = 100 + dets.tag.frame_id
        self.track.tag = dets.tag
        self.track.box = box(float(dets.tag.frame_id), 0.0)
        self.track.state = TrackState.CONFIRMED
        return [self.track]


class TestTheLiveTrackHazard:
    """The regression that found this: a run over MOT17-09 produced a frame in which one
    identity appeared twenty-six times, because every buffered reference was the same object."""

    def test_the_run_snapshots_each_frame_instead_of_buffering_references(self) -> None:
        case = EvaluationCase(
            name="alias",
            detections=tuple(detections(t, [float(t)]) for t in (1, 2, 3)),
            ground_truth=sequence(
                "gt", [frame(t, [(1, float(t))]) for t in (1, 2, 3)], length=3
            ),
        )

        predictions, _ = run(MutatingTracker(), case)

        assert predictions.frame_ids == (1, 2, 3)
        assert [f.ids.tolist() for f in predictions] == [[101], [102], [103]]

    def test_buffering_the_references_would_have_collapsed_the_run(self) -> None:
        """The wrong version, written out: three frames become one frame of three identical
        entries, which :class:`ObjectFrame` then rejects for repeating an identity. A metric
        that did not validate would have silently scored the last frame three times."""
        tracker = MutatingTracker()
        buffered: list[Track] = []
        for t in (1, 2, 3):
            buffered.extend(tracker.update(detections(t, [float(t)])))

        assert {track.track_id for track in buffered} == {103}
        with pytest.raises(ConfigurationError, match="repeats an identity"):
            TrackSequence.from_tracks("aliased", buffered)


class TestRunningARealTracker:
    def test_sort_on_its_own_ground_truth_scores_perfectly_after_the_warm_up(
        self, simple_case
    ) -> None:
        """SORT's default ``min_hits`` is 3, so on a three-frame sequence only the last frame is
        published: 2 of 6 ground-truth boxes found, no false positives. That is the tracker's
        documented lifecycle rather than a metric problem, and asserting it here is what makes
        the runner's frame accounting checkable."""
        result = evaluate(TRACKERS.build("sort"), simple_case)

        assert result.clear.true_positives == 2
        assert result.clear.false_positives == 0
        assert result.clear.false_negatives == 4
        assert result.num_frames == 3

    def test_min_hits_of_one_finds_every_object_with_no_error(self, simple_case) -> None:
        result = evaluate(TRACKERS.build("sort", min_hits=1), simple_case)

        assert result.clear.mota == 1.0
        assert (result.clear.false_positives, result.clear.false_negatives) == (0, 0)
        assert result.identity.idf1 == 1.0

    def test_hota_still_sees_the_filter_smoothing_that_clear_at_half_cannot(
        self, simple_case
    ) -> None:
        """The published box is the Kalman estimate, not the detection, so it sits a pixel or
        two off. CLEAR at IoU 0.5 cannot see that at all — MOTA is a flat 1.0 — while HOTA's
        sweep charges it at the 0.90 and 0.95 thresholds and comes out at 0.968.

        This is the clearest small demonstration of why both are reported: a change that made
        the filter smoother would move HOTA and leave MOTA exactly where it was."""
        result = evaluate(TRACKERS.build("sort", min_hits=1), simple_case)

        assert result.clear.mota == 1.0
        assert result.clear.motp < 1.0
        assert 0.9 < result.hota.hota < 1.0
        assert np.all(result.hota.hota_curve[:17] == 1.0)
        assert result.hota.hota_curve[-1] < 0.7
        assert result.hota.loc_a < 1.0

    def test_the_timing_is_recorded_and_positive(self, simple_case) -> None:
        """Cost belongs next to quality: a tracker that wins HOTA by a point at 4 ms a frame
        cannot run fifty cameras, and separating the two measurements is how that is missed."""
        result = evaluate(TRACKERS.build("sort", min_hits=1), simple_case)

        assert result.seconds > 0.0
        assert result.ms_per_frame == pytest.approx(1000.0 * result.seconds / 3)

    @pytest.mark.parametrize("name", ["sort", "bytetrack", "ocsort", "botsort", "deepsortv2"])
    def test_every_registered_tracker_runs_and_produces_a_finite_score(
        self, name, simple_case
    ) -> None:
        """Not a quality claim — a claim that the evaluation path works for all five, including
        the two that would use appearance if any were offered and must degrade to geometry when
        none is."""
        result = evaluate(TRACKERS.build(name, min_hits=1), simple_case)

        assert np.isfinite(result.hota.hota)
        assert np.isfinite(result.clear.mota)
        assert result.num_pred_dets > 0


class TestStateIsPerCamera:
    def test_evaluate_all_builds_a_fresh_tracker_per_case(self, simple_case) -> None:
        """A tracker is stateful and single-camera by construction. The factory argument makes
        the correct thing the only expressible thing."""
        other = EvaluationCase(
            name="other",
            detections=tuple(
                Detections(
                    tag=FrameTag(camera_id="other-camera", frame_id=t),
                    items=list(detections(t, [0.0]).items),
                )
                for t in (1, 2, 3)
            ),
            ground_truth=sequence("gt", [frame(t, [(1, 0.0)]) for t in (1, 2, 3)], length=3),
        )

        results = evaluate_all(lambda: TRACKERS.build("sort", min_hits=1), [simple_case, other])

        assert [r.name for r in results] == ["two-objects", "other"]

    def test_reusing_one_instance_across_two_cameras_raises(self, simple_case) -> None:
        """Which is the whole reason ``evaluate_all`` takes a factory. Asserted so the guard
        cannot quietly disappear."""
        tracker = TRACKERS.build("sort", min_hits=1)
        run(tracker, simple_case)

        other = EvaluationCase(
            name="other",
            detections=(
                Detections(
                    tag=FrameTag(camera_id="other-camera", frame_id=1),
                    items=list(detections(1, [0.0]).items),
                ),
            ),
            ground_truth=sequence("gt", [frame(1, [(1, 0.0)])], length=1),
        )

        with pytest.raises(TrackingError, match="one camera"):
            run(tracker, other, reset=False)

    def test_evaluating_nothing_raises_rather_than_reporting_zero(self) -> None:
        with pytest.raises(ConfigurationError, match="not a score of 0"):
            evaluate_all(lambda: TRACKERS.build("sort"), [])


class TestScoreIsIndependentOfHowThePredictionWasProduced:
    def test_it_grades_two_hand_built_sequences(self, two_objects, perfect) -> None:
        """The scoring path must work for a submission file read off disk, not only for a
        tracker built here — which is what makes it usable as a cross-check."""
        result = score(two_objects, perfect, seconds=0.5, name="hand-built")

        assert result.name == "hand-built"
        assert result.clear.mota == 1.0
        assert result.seconds == 0.5

    def test_every_score_the_report_and_the_tuner_use_is_present(
        self, two_objects, perfect
    ) -> None:
        scores = score(two_objects, perfect).scores()

        for key in (
            "HOTA",
            "DetA",
            "AssA",
            "LocA",
            "IDF1",
            "MOTA",
            "MOTP",
            "IDSW",
            "ms_per_frame",
        ):
            assert key in scores
