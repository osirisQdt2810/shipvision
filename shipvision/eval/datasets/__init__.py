"""Dataset loaders. One per on-disk layout, and MOTChallenge is the only one so far.

A loader's job is to end at :class:`~shipvision.eval.sequence.EvaluationCase` and nothing
else, so that everything above it — the metrics, the report, the tuning objective — works
unchanged on a synthetic three-frame case built in a fixture. That is what keeps the offline
test tier free of a dataset dependency: a test that needs a scenario builds one, and only the
handful marked ``slow`` reach for the real footage.
"""

from shipvision.eval.datasets.mot import (
    DISTRACTOR_CLASSES,
    PEDESTRIAN_CLASS,
    MotSequenceFiles,
    discover_sequences,
    load_case,
    load_cases,
    load_detections,
    load_ground_truth,
    read_mot_file,
    write_mot_file,
)
from shipvision.eval.datasets.seqinfo import SeqInfo, read_seqinfo

__all__ = [
    "DISTRACTOR_CLASSES",
    "PEDESTRIAN_CLASS",
    "MotSequenceFiles",
    "SeqInfo",
    "discover_sequences",
    "load_case",
    "load_cases",
    "load_detections",
    "load_ground_truth",
    "read_mot_file",
    "read_seqinfo",
    "write_mot_file",
]
