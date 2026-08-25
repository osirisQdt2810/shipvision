"""The table. Its job is to make the wrong comparison hard to draw by eye.

Three properties are asserted: the aggregate row is computed from summed counts and labelled
so it cannot be read as a sequence, cost sits in the same table as quality, and a comparison
over different sequence sets is refused rather than printed.
"""

from __future__ import annotations

import pytest

from shipvision.errors import ConfigurationError
from shipvision.eval.metrics import COMBINED
from shipvision.eval.report import format_comparison, format_table, rows
from shipvision.eval.runner import score
from shipvision.eval.sequence import TrackSequence

from .conftest import frame, sequence


@pytest.fixture
def two_results() -> list:
    """A perfect one-frame sequence and a missed ten-frame one — the aggregation case again,
    here to check that the printed aggregate is the summed one and not the mean."""
    tiny = sequence("tiny", [frame(1, [(1, 0.0)])], length=1)
    long = sequence("long", [frame(t, [(1, 0.0)]) for t in range(1, 11)], length=10)
    return [
        score(
            tiny,
            sequence("tiny-p", [frame(1, [(71, 0.0)])], length=1),
            name="tiny",
            seconds=0.001,
        ),
        score(long, TrackSequence.empty("long-p", length=10), name="long", seconds=0.010),
    ]


class TestTheTable:
    def test_there_is_one_row_per_sequence_plus_a_combined_row(self, two_results) -> None:
        table = rows(two_results)

        assert [row["sequence"] for row in table] == ["tiny", "long", COMBINED]

    def test_the_combined_row_is_the_summed_score_not_the_mean(self, two_results) -> None:
        """MOTA 100.00 and 0.00 average to 50.00. Summing the counts gives 1 - 10/11 = 9.09."""
        table = rows(two_results)

        assert table[0]["MOTA"] == "100.00"
        assert table[1]["MOTA"] == "0.00"
        assert table[2]["MOTA"] == "9.09"

    def test_scores_print_as_percentages_and_counts_as_integers(self, two_results) -> None:
        """A leaderboard entry can be compared with a row here without arithmetic, and an IDSW
        of '1200.00%' — which has happened in a real report — cannot be produced."""
        table = rows(two_results)

        assert table[1]["FN"] == "10"
        assert table[1]["IDSW"] == "0"
        assert "." in table[0]["HOTA"]

    def test_cost_is_a_column_and_not_an_appendix(self, two_results) -> None:
        table = rows(two_results)

        assert table[0]["ms_per_frame"] == "1.000"
        assert table[2]["ms_per_frame"] == "1.000"

    def test_a_single_sequence_gets_no_aggregate_row(self, two_results) -> None:
        """A COMBINED row over one sequence is noise that invites being quoted as a second
        result."""
        assert [row["sequence"] for row in rows(two_results[:1])] == ["tiny"]

    def test_the_aggregate_can_be_suppressed(self, two_results) -> None:
        assert len(rows(two_results, aggregate=False)) == 2

    def test_an_unknown_column_raises(self, two_results) -> None:
        with pytest.raises(ConfigurationError, match="unknown metric"):
            rows(two_results, columns=("HOTA", "MOTA2"))

    def test_reporting_nothing_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="not a table of zeros"):
            rows([])


class TestRendering:
    def test_the_header_and_every_row_are_present(self, two_results) -> None:
        text = format_table(two_results, title="on-target")

        lines = text.splitlines()
        assert lines[0] == "on-target"
        assert lines[1].startswith("sequence")
        assert any(line.startswith("tiny") for line in lines)
        assert any(line.startswith(COMBINED) for line in lines)

    def test_the_columns_line_up(self, two_results) -> None:
        """A table nobody can read is a table nobody checks."""
        lines = format_table(two_results).splitlines()
        widths = {len(line) for line in lines if line.strip() and not line.startswith("-")}

        assert len(widths) == 1


class TestComparison:
    def test_it_groups_by_sequence_so_the_eye_compares_like_with_like(
        self, two_results
    ) -> None:
        """Grouping by tracker produces a table where the eye compares one tracker's 45-people
        sequence against another's 8-people one, and those are different problems."""
        text = format_comparison({"sort": two_results, "bytetrack": two_results})

        lines = [line for line in text.splitlines() if line.strip()]
        assert lines[2].startswith("tiny")
        assert "sort" in lines[2]
        assert "bytetrack" in lines[3]

    def test_a_comparison_over_different_sequences_is_refused(self, two_results) -> None:
        """The usual way it happens is one tracker crashing on one sequence and its row quietly
        going missing, which turns a comparison of algorithms into a comparison of handicaps."""
        with pytest.raises(ConfigurationError, match="different sequences"):
            format_comparison({"sort": two_results, "bytetrack": two_results[:1]})

    def test_comparing_nothing_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="nothing to compare"):
            format_comparison({})
