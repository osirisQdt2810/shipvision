"""The metrics here, against TrackEval's own implementation, on real footage.

A metric nobody can compare against is a metric nobody should trust — the same argument this
library makes about fused kernels, applied to the thing that decides whether a kernel helped.
The three metrics in :mod:`shipvision.eval.metrics` are written from their papers, and this is
the test that says so out loud: it runs every registered tracker over a real MOT17 sequence,
hands the same output to TrackEval, and compares seventeen fields.

Skipped unless a TrackEval checkout is reachable. It is MIT-licensed and lives outside this
repository on purpose — vendoring it would make the reimplementation pointless and would put
a second copy of the arithmetic under maintenance here.

**The harness writes boxes at nine significant digits, not the two decimals of a real
submission file.** Two decimals move an IoU by about 1e-4, which is enough to push a pair
across one of HOTA's nineteen thresholds; the first version of this comparison spent an hour
looking like a metric disagreement and was a rounding artefact of the file it wrote. The
residue that is left — a few parts in a billion on MOTP and LocA — is this library computing
IoU in float32 against TrackEval's float64, and it is the only difference there is.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from shipvision.eval import load_case, run, score
from shipvision.eval.datasets import read_seqinfo
from shipvision.mot import TRACKERS

pytestmark = pytest.mark.slow

#: Where to find a TrackEval checkout. MIT-licensed; the default is where the internal
#: reference repositories put one.
TRACKEVAL_ENV = "SHIPVISION_TRACKEVAL"
TRACKEVAL_DEFAULT = Path(
    "/home/dungha15/workspaces/phucnp/shipinfer/references/"
    "gitea-multi-object-tracking-pyservice/app/pysrc/evaluation/mot/TrackEval"
)

#: ``our name -> (TrackEval metric group, TrackEval field, is it a per-threshold array)``.
#:
#: The array flag matters: TrackEval keeps HOTA's nineteen thresholds as an array and reports
#: the mean of it, so a comparison that read element 0 would be comparing HOTA at IoU 0.05
#: against HOTA averaged over the sweep.
FIELDS: tuple[tuple[str, str, str, bool], ...] = (
    ("HOTA", "HOTA", "HOTA", True),
    ("DetA", "HOTA", "DetA", True),
    ("AssA", "HOTA", "AssA", True),
    ("AssRe", "HOTA", "AssRe", True),
    ("AssPr", "HOTA", "AssPr", True),
    ("LocA", "HOTA", "LocA", True),
    ("IDF1", "Identity", "IDF1", False),
    ("IDP", "Identity", "IDP", False),
    ("IDR", "Identity", "IDR", False),
    ("MOTA", "CLEAR", "MOTA", False),
    ("MOTP", "CLEAR", "MOTP", False),
    ("IDSW", "CLEAR", "IDSW", False),
    ("FP", "CLEAR", "CLR_FP", False),
    ("FN", "CLEAR", "CLR_FN", False),
    ("MT", "CLEAR", "MT", False),
    ("ML", "CLEAR", "ML", False),
    ("Frag", "CLEAR", "Frag", False),
)

#: Every count must agree exactly; the two overlap sums may differ by float32 noise.
TOLERANCE = 1e-6

SEQUENCE = "MOT17-09-FRCNN"


@pytest.fixture(scope="module")
def trackeval():
    """The TrackEval package, or a skip."""
    root = Path(os.environ.get(TRACKEVAL_ENV, TRACKEVAL_DEFAULT))
    if not (root / "trackeval").is_dir():
        pytest.skip(f"no TrackEval checkout at {root}; set {TRACKEVAL_ENV} to point at one")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return pytest.importorskip("trackeval", reason="TrackEval could not be imported")


def write_submission(path: Path, predictions) -> None:
    """MOTChallenge format at nine significant digits. See the module docstring."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            f"{f.frame_id},{i},{b[0]:.9g},{b[1]:.9g},"
            f"{b[2] - b[0]:.9g},{b[3] - b[1]:.9g},1,-1,-1,-1\n"
            for f in predictions
            for i, b in zip(f.ids.tolist(), f.boxes, strict=True)
        )
    )


@pytest.fixture(scope="module")
def comparison(trackeval, mot17_root, tmp_path_factory):
    """``{tracker: (ours, theirs)}`` for every registered tracker on one sequence.

    Module-scoped because the expensive half is TrackEval's own evaluation, and running it once
    for all five trackers is the difference between a test that gets run and one that gets
    marked skip and forgotten.
    """
    names = sorted(TRACKERS.names())
    work = tmp_path_factory.mktemp("trackeval")
    gt_root, tracker_root = work / "gt", work / "trackers"
    source = mot17_root / SEQUENCE
    (gt_root / SEQUENCE / "gt").mkdir(parents=True)
    (gt_root / SEQUENCE / "gt" / "gt.txt").write_bytes((source / "gt" / "gt.txt").read_bytes())
    length = read_seqinfo(source / "seqinfo.ini").length

    case = load_case(source)
    ours = {}
    for name in names:
        predictions, seconds = run(TRACKERS.build(name), case)
        ours[name] = score(
            case.ground_truth,
            predictions,
            ignored=case.ignored,
            unscored=case.unscored,
            seconds=seconds,
            name=SEQUENCE,
        )
        write_submission(tracker_root / name / "data" / f"{SEQUENCE}.txt", predictions)

    evaluation = trackeval.Evaluator.get_default_eval_config()
    evaluation.update(
        {
            "USE_PARALLEL": False,
            "PRINT_RESULTS": False,
            "PRINT_CONFIG": False,
            "OUTPUT_SUMMARY": False,
            "OUTPUT_DETAILED": False,
            "PLOT_CURVES": False,
            "TIME_PROGRESS": False,
        }
    )
    dataset = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset.update(
        {
            "GT_FOLDER": str(gt_root),
            "TRACKERS_FOLDER": str(tracker_root),
            "SKIP_SPLIT_FOL": True,
            "SEQ_INFO": {SEQUENCE: length},
            "TRACKERS_TO_EVAL": names,
            "PRINT_CONFIG": False,
            "OUTPUT_FOLDER": str(work / "out"),
        }
    )
    metrics_config = {
        "METRICS": ["HOTA", "CLEAR", "Identity"],
        "THRESHOLD": 0.5,
        "PRINT_CONFIG": False,
    }
    output, _ = trackeval.Evaluator(evaluation).evaluate(
        [trackeval.datasets.MotChallenge2DBox(dataset)],
        [
            trackeval.metrics.HOTA(metrics_config),
            trackeval.metrics.CLEAR(metrics_config),
            trackeval.metrics.Identity(metrics_config),
        ],
    )
    theirs = {name: output["MotChallenge2DBox"][name][SEQUENCE]["pedestrian"] for name in names}
    return {name: (ours[name], theirs[name]) for name in names}


class TestParityWithTrackEval:
    @pytest.mark.parametrize("tracker", sorted(TRACKERS.names()))
    def test_every_reported_field_agrees(self, comparison, tracker: str) -> None:
        mine, reference = comparison[tracker]
        scores = mine.scores()
        disagreements: list[str] = []
        for label, group, key, is_array in FIELDS:
            expected = (
                float(np.mean(reference[group][key]))
                if is_array
                else float(reference[group][key])
            )
            delta = abs(scores[label] - expected)
            if delta > TOLERANCE:
                disagreements.append(
                    f"{label}: ours {scores[label]:.9f} theirs {expected:.9f} "
                    f"(delta {delta:.2e})"
                )

        assert not disagreements, f"{tracker} on {SEQUENCE}:\n  " + "\n  ".join(disagreements)

    @pytest.mark.parametrize("tracker", sorted(TRACKERS.names()))
    def test_every_count_agrees_exactly_and_not_only_within_a_tolerance(
        self, comparison, tracker: str
    ) -> None:
        """The tallies are integers on both sides. A tolerance would hide an off-by-one in the
        ignore-region preprocessing, which is exactly the bug this comparison first found."""
        mine, reference = comparison[tracker]
        scores = mine.scores()

        assert scores["IDSW"] == float(reference["CLEAR"]["IDSW"])
        assert scores["FP"] == float(reference["CLEAR"]["CLR_FP"])
        assert scores["FN"] == float(reference["CLEAR"]["CLR_FN"])
        assert scores["MT"] == float(reference["CLEAR"]["MT"])
        assert scores["ML"] == float(reference["CLEAR"]["ML"])
        assert scores["Frag"] == float(reference["CLEAR"]["Frag"])
