"""Turning results into a table somebody will actually read.

Three rules, and each one exists because its opposite has produced a misleading report:

**Per sequence, never one number.** MOT17-05 has 8.3 people per frame and MOT17-04 has 45.3.
This library's operating point is 10-20, so a single averaged score is dominated by two
sequences that are outside it. Every table here has one row per sequence, and the aggregate is
an extra row rather than the only row.

**The aggregate row sums counts and divides once — it is not the mean of the rows above it.**
See :mod:`shipvision.eval.metrics.base`. The two differ by several points on an uneven
sequence set, and the mean-of-rows version silently weights a 525-frame sequence like a
1050-frame one.

**Cost sits in the same table as quality.** ``ms/frame`` is a column, not an appendix. A
tracker that wins HOTA by a point and costs 4 ms a frame cannot run fifty cameras on this
box, and a report that puts speed in a second table lets that trade pass unnoticed.

Scores are printed as percentages, which is how MOTChallenge publishes them — so a number
here can be compared with a leaderboard entry without arithmetic. Counts print as integers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from shipvision.errors import ConfigurationError
from shipvision.eval.metrics.base import COMBINED, SequenceResult, combine

__all__ = ["COUNT_COLUMNS", "DEFAULT_COLUMNS", "format_comparison", "format_table", "rows"]

#: What a tracking report needs to show at once. HOTA first because it is the summary score,
#: then the two halves it is the geometric mean of — a change that trades one for the other
#: leaves HOTA flat and is invisible without them.
DEFAULT_COLUMNS: tuple[str, ...] = (
    "HOTA",
    "DetA",
    "AssA",
    "LocA",
    "IDF1",
    "MOTA",
    "MOTP",
    "IDSW",
    "FP",
    "FN",
    "ms_per_frame",
)

#: Columns that are tallies rather than scores, so they print as integers and are not
#: multiplied by a hundred. An IDSW of "1200.00%" has happened in a real report.
COUNT_COLUMNS = frozenset({"IDSW", "FP", "FN", "MT", "ML", "Frag"})


def _cell(column: str, value: float) -> str:
    if column in COUNT_COLUMNS:
        return f"{round(value)}"
    if column == "ms_per_frame":
        return f"{value:.3f}"
    return f"{100.0 * value:.2f}"


def rows(
    results: Sequence[SequenceResult],
    *,
    columns: Sequence[str] = DEFAULT_COLUMNS,
    aggregate: bool = True,
) -> list[dict[str, str]]:
    """The table as data, so a caller can emit CSV, JSON or Markdown without re-deriving it.

    The ``sequence`` key is added first and holds the row label; every other key is one of
    ``columns``. Insertion order is the print order.
    """
    if not results:
        raise ConfigurationError("nothing to report; an empty run is not a table of zeros")
    ordered = list(results)
    if aggregate and len(ordered) > 1:
        ordered.append(combine(ordered))
    table: list[dict[str, str]] = []
    for result in ordered:
        scores = result.scores()
        missing = [column for column in columns if column not in scores]
        if missing:
            raise ConfigurationError(
                f"unknown metric(s) {missing}; available: {sorted(scores)}"
            )
        row = {"sequence": result.name}
        row.update({column: _cell(column, scores[column]) for column in columns})
        table.append(row)
    return table


def _render(header: Sequence[str], body: Sequence[Sequence[str]], *, title: str = "") -> str:
    widths = [
        max(len(str(header[index])), *(len(str(row[index])) for row in body))
        for index in range(len(header))
    ]
    lines: list[str] = []
    if title:
        lines.append(title)

    def line(cells: Sequence[str]) -> str:
        first = str(cells[0]).ljust(widths[0])
        rest = "  ".join(
            str(cell).rjust(widths[index + 1]) for index, cell in enumerate(cells[1:])
        )
        return f"{first}  {rest}".rstrip()

    lines.append(line(header))
    lines.append("-" * len(lines[-1]))
    lines.extend(line(row) for row in body)
    return "\n".join(lines)


def format_table(
    results: Sequence[SequenceResult],
    *,
    columns: Sequence[str] = DEFAULT_COLUMNS,
    title: str = "",
    aggregate: bool = True,
) -> str:
    """One tracker over several sequences. Scores as percentages, counts as integers.

    The last row is :data:`~shipvision.eval.metrics.base.COMBINED` when there is more than one
    sequence — computed from summed counts, and labelled so it cannot be mistaken for a
    sequence with an unfortunate name.
    """
    table = rows(results, columns=columns, aggregate=aggregate)
    header = ["sequence", *columns]
    body = [[row["sequence"], *(row[column] for column in columns)] for row in table]
    return _render(header, body, title=title)


def format_comparison(
    results: Mapping[str, Sequence[SequenceResult]],
    *,
    columns: Sequence[str] = DEFAULT_COLUMNS,
    title: str = "",
) -> str:
    """Several trackers over the same sequences: one block per sequence, one row per tracker.

    Grouped by sequence rather than by tracker on purpose. Grouping by tracker produces a
    table where the eye compares a tracker's MOT17-04 row against another's MOT17-05 row, and
    those are different problems — 45 people per frame against 8. Grouping by sequence makes
    the only comparison the table invites the correct one.

    Raises:
        ConfigurationError: the trackers were not run over the same sequences. A comparison
            over different inputs is not a comparison, and the usual way it happens is one
            tracker crashing on one sequence and its row quietly going missing.
    """
    if not results:
        raise ConfigurationError("nothing to compare")
    names = {tracker: [r.name for r in runs] for tracker, runs in results.items()}
    first = next(iter(names.values()))
    disagreeing = {tracker: seen for tracker, seen in names.items() if seen != first}
    if disagreeing:
        raise ConfigurationError(
            f"the trackers were run over different sequences: expected {first}, got "
            f"{disagreeing}. A table over different inputs compares handicaps, not algorithms"
        )

    header = ["sequence / tracker", *columns]
    body: list[list[str]] = []
    for index, sequence in enumerate([*first, COMBINED]):
        for tracker, runs in results.items():
            result = runs[index] if index < len(first) else combine(list(runs))
            scores = result.scores()
            body.append([f"{sequence:<14} {tracker}", *(_cell(c, scores[c]) for c in columns)])
        body.append(["" for _ in header])
    return _render(header, body[:-1], title=title)
