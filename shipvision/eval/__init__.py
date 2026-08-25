"""Measurement. HOTA, IDF1 and CLEAR for MOT, and the loader that feeds them real footage.

This package is why the library/server split exists. An algorithm is judged by a number on
recorded footage, and if that number takes a GPU, a model repository and an engine build to
produce, it will not be produced — so the tracker that shipped first wins by default rather
than by evidence. Everything here runs on boxes alone, in seconds, with no device::

    from shipvision.eval import evaluate, format_table, load_case
    from shipvision.mot import TRACKERS

    case = load_case("data/mot17/train/MOT17-09-FRCNN")
    print(format_table([evaluate(TRACKERS.build("bytetrack"), case)]))

The three metrics are three different matchings, and the differences *are* the metrics — see
:mod:`shipvision.eval.association`. All three are re-implemented here from their papers rather
than vendored, and cross-checked against TrackEval's numbers on real sequences; the checks
that a reader can verify by hand live in ``tests/eval/``, where each one writes its arithmetic
out in the docstring.

Two conventions worth knowing before reading a number this package produced:

**Aggregation sums counts, never averages scores.** A mean of per-sequence HOTA weights
MOT17-09's 525 frames like MOT17-04's 1050 and describes a benchmark nobody ran.

**Report per sequence and say which regime it came from.** This library's operating point is
10-20 people per frame. MOT17-02 and MOT17-04, at 31 and 45, are a different problem, and one
averaged number over all seven hides which one it measured.

Re-identification has its own ranking metrics — CMC and mAP, with the camera-exclusion
protocol that stops a query matching itself — and they live in
:mod:`shipvision.reid.metrics` rather than here, because they answer a question about an
embedding rather than about a sequence of frames.
"""

from shipvision.eval.association import (
    AlignedSequence,
    align,
    drop_predictions_matching,
    iou_similarity,
    match_preferring,
    solve_maximum,
)
from shipvision.eval.datasets import (
    DISTRACTOR_CLASSES,
    PEDESTRIAN_CLASS,
    MotSequenceFiles,
    SeqInfo,
    discover_sequences,
    load_case,
    load_cases,
    load_detections,
    load_ground_truth,
    read_seqinfo,
    write_mot_file,
)
from shipvision.eval.metrics import (
    ALPHAS,
    COMBINED,
    ClearCounts,
    HotaCounts,
    IdentityCounts,
    SequenceResult,
    clear_counts,
    combine,
    hota_counts,
    identity_counts,
)
from shipvision.eval.report import DEFAULT_COLUMNS, format_comparison, format_table, rows
from shipvision.eval.runner import evaluate, evaluate_all, run, score
from shipvision.eval.sequence import EvaluationCase, ObjectFrame, TrackSequence

__all__ = [
    "ALPHAS",
    "COMBINED",
    "DEFAULT_COLUMNS",
    "DISTRACTOR_CLASSES",
    "PEDESTRIAN_CLASS",
    "AlignedSequence",
    "ClearCounts",
    "EvaluationCase",
    "HotaCounts",
    "IdentityCounts",
    "MotSequenceFiles",
    "ObjectFrame",
    "SeqInfo",
    "SequenceResult",
    "TrackSequence",
    "align",
    "clear_counts",
    "combine",
    "discover_sequences",
    "drop_predictions_matching",
    "evaluate",
    "evaluate_all",
    "format_comparison",
    "format_table",
    "hota_counts",
    "identity_counts",
    "iou_similarity",
    "load_case",
    "load_cases",
    "load_detections",
    "load_ground_truth",
    "match_preferring",
    "read_seqinfo",
    "rows",
    "run",
    "score",
    "solve_maximum",
    "write_mot_file",
]
