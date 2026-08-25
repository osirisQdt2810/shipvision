"""Aggregation across sequences: sum the counts, divide once.

This is the test the operating-point argument rests on. MOT17-05 is 837 frames at 8.3 people
per frame and MOT17-04 is 1050 frames at 45.3; a mean of per-sequence scores weights a fifth
of the objects like four fifths of them, and the resulting number describes a benchmark nobody
ran. Every case below is built so the two aggregations give visibly different answers, and the
summed one is asserted.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipvision.errors import ConfigurationError
from shipvision.eval.metrics import COMBINED, combine
from shipvision.eval.runner import score
from shipvision.eval.sequence import TrackSequence

from .conftest import frame, sequence


@pytest.fixture
def one_short_one_long() -> tuple:
    """One perfect frame against ten missed ones.

    Sequence ``tiny``: 1 frame, 1 ground-truth box, found. MOTA = 1.0, HOTA = 1.0.
    Sequence ``long``: 10 frames, 1 box each, nothing published. MOTA = 0.0, HOTA = 0.0.

    A mean of the two scores is 0.5. Summing the counts gives GT = 11, FN = 10, so
    MOTA = 1 - 10/11 = 1/11 = 0.0909 — five times smaller, because ten of the eleven objects
    were missed and that is what the aggregate is supposed to say.
    """
    tiny_gt = sequence("tiny", [frame(1, [(1, 0.0)])], length=1)
    tiny_pred = sequence("tiny-p", [frame(1, [(71, 0.0)])], length=1)
    long_gt = sequence("long", [frame(t, [(1, 0.0)]) for t in range(1, 11)], length=10)

    tiny = score(tiny_gt, tiny_pred, name="tiny")
    long = score(long_gt, TrackSequence.empty("long-p", length=10), name="long")
    return tiny, long


class TestSummingBeatsAveraging:
    def test_mota_is_the_ratio_of_summed_counts(self, one_short_one_long) -> None:
        tiny, long = one_short_one_long
        total = combine([tiny, long])

        assert (tiny.clear.mota, long.clear.mota) == (1.0, 0.0)
        assert total.clear.num_gt_dets == 11
        assert total.clear.false_negatives == 10
        assert total.clear.mota == pytest.approx(1 / 11)
        assert total.clear.mota != pytest.approx((tiny.clear.mota + long.clear.mota) / 2)

    def test_hota_is_recomputed_from_the_summed_counts(self, one_short_one_long) -> None:
        """TP = 1, FN = 10, FP = 0 at every threshold, so DetA = 1/11 and AssA = 1 (the one
        true positive is a whole trajectory on both sides). HOTA = sqrt(1/11) = 0.30151, not
        the 0.5 an average of 1.0 and 0.0 would give."""
        tiny, long = one_short_one_long
        total = combine([tiny, long])

        assert total.hota.det_a == pytest.approx(1 / 11)
        assert total.hota.ass_a == pytest.approx(1.0)
        assert total.hota.hota == pytest.approx(float(np.sqrt(1 / 11)))
        assert total.hota.hota != pytest.approx(0.5)

    def test_idf1_is_the_ratio_of_summed_counts(self, one_short_one_long) -> None:
        """IDTP = 1, IDFN = 10, IDFP = 0 → IDF1 = 2/(2 + 10) = 1/6."""
        total = combine(list(one_short_one_long))

        assert total.identity.true_positives == 1
        assert total.identity.false_negatives == 10
        assert total.identity.idf1 == pytest.approx(1 / 6)

    def test_the_frame_count_and_the_time_add_up(self, one_short_one_long) -> None:
        total = combine(list(one_short_one_long))

        assert total.num_frames == 11
        assert total.num_gt_dets == 11


class TestTheAggregateIsLabelled:
    def test_it_is_named_combined_and_not_after_a_sequence(self, one_short_one_long) -> None:
        """A row labelled with a sequence name that is really an aggregate is how a table gets
        quoted as if it were a single-sequence result."""
        total = combine(list(one_short_one_long))

        assert total.name == COMBINED

    def test_adding_two_results_also_relabels(self, one_short_one_long) -> None:
        tiny, long = one_short_one_long

        assert (tiny + long).name == COMBINED

    def test_a_caller_may_name_it_something_else(self, one_short_one_long) -> None:
        assert combine(list(one_short_one_long), name="on-target").name == "on-target"


class TestRefusals:
    def test_combining_nothing_raises_rather_than_reporting_zero(self) -> None:
        """An empty run is not a score of zero. A study that scored a crashed configuration as
        0.0 would rank it against real ones."""
        with pytest.raises(ConfigurationError, match="not a score of zero"):
            combine([])

    def test_an_unknown_metric_name_raises_on_lookup(self, one_short_one_long) -> None:
        """A typo'd ``--metric`` must stop the study, not silently optimise something else."""
        tiny, _ = one_short_one_long

        with pytest.raises(ConfigurationError, match="unknown metric"):
            tiny.score("HOTA1")


class TestMillisecondsPerFrame:
    def test_it_divides_the_summed_time_by_the_summed_frames(self) -> None:
        """Cost aggregates the same way quality does. Averaging two per-frame rates would weight
        a one-frame sequence like a thousand-frame one."""
        gt = sequence("a", [frame(1, [(1, 0.0)])], length=1)
        long_gt = sequence("b", [frame(t, [(1, 0.0)]) for t in range(1, 10)], length=9)
        fast = score(gt, gt, seconds=0.001, name="a")
        slow = score(long_gt, long_gt, seconds=0.009, name="b")

        total = combine([fast, slow])

        assert fast.ms_per_frame == pytest.approx(1.0)
        assert slow.ms_per_frame == pytest.approx(1.0)
        assert total.ms_per_frame == pytest.approx(1.0)
        assert total.num_frames == 10

    def test_an_untimed_run_reports_zero_rather_than_dividing_by_nothing(self) -> None:
        gt = sequence("a", [frame(1, [(1, 0.0)])], length=1)

        assert score(gt, gt, name="a").ms_per_frame == 0.0
