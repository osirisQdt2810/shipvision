# Runs roboflow/trackers (Apache-2.0), src/trackers/core/mcbyte/mask_association.py and
# src/trackers/core/masks/base.py, commit ced34f04886da91dc6bec3dfe02f0a0427231ce8. Changed:
# no line of it is copied — the two modules are imported by path and their answers dumped.
"""Generate golden traces for shipvision's McByte port, from the REFERENCE implementation.

Runs roboflow/trackers' own `mask_association` (Apache-2.0) over hand-built cases and dumps
inputs + outputs to JSON. The port's tests convert to cost space on the way in:
cost = 1 - similarity; max_cost = 1 - minimum_similarity; boosts subtract where the
reference adds. Run once, BEFORE the port exists; never regenerate from the port.

Point ``ROBOFLOW_TRACKERS_SRC`` at the checkout's ``src/`` to re-derive the oracle elsewhere.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

#: The reference checkout's ``src/``, overridable so the oracle can be re-derived by somebody
#: whose reference tree is not where this one's was. The default is the path it was first
#: dumped from, kept because it also documents which tree the committed JSON came out of.
DEFAULT_REF_SRC = "/home/dungha15/workspaces/shipinfer/references/roboflow-trackers/src"
REF_SRC = Path(os.environ.get("ROBOFLOW_TRACKERS_SRC", DEFAULT_REF_SRC))

# Load the two needed modules by path: the package __init__ drags rich/cv2/supervision,
# but masks/base.py and mcbyte/mask_association.py need only numpy.
import importlib.util  # noqa: E402
import types  # noqa: E402

import numpy as np  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


for _pkg in ("trackers", "trackers.core", "trackers.core.masks", "trackers.core.mcbyte"):
    sys.modules.setdefault(_pkg, types.ModuleType(_pkg))
_masks_base = _load("trackers.core.masks.base", REF_SRC / "trackers/core/masks/base.py")
_assoc = _load(
    "trackers.core.mcbyte.mask_association",
    REF_SRC / "trackers/core/mcbyte/mask_association.py",
)
MaskOutput = _masks_base.MaskOutput
_get_ambiguous_candidate_matrix = _assoc._get_ambiguous_candidate_matrix
_get_clear_matches = _assoc._get_clear_matches
_get_isolated_candidate_matrix = _assoc._get_isolated_candidate_matrix
_get_remaining_indices = _assoc._get_remaining_indices
condition_similarity_with_masks = _assoc.condition_similarity_with_masks


def _mask_output(spec):
    if spec is None:
        return None
    masks = np.zeros(tuple(spec["shape"]), dtype=bool)
    for row, (r0, r1, c0, c1) in zip(range(masks.shape[0]), spec["true_rects"], strict=True):
        masks[row, r0:r1, c0:c1] = True
    return MaskOutput(
        masks=masks,
        tracklet_mask_dict={int(k): v for k, v in spec["tracklet_mask"].items()},
        mask_avg_prob_dict={int(k): v for k, v in spec["avg_prob"].items()},
    )


CASES = {
    # name: (similarity, raw_iou, tracklet_ids, detection_boxes, mask_spec, min_sim, kwargs)
    "doctest_ambiguous_row_boosted": (
        [[0.7, 0.6]],
        [[0.7, 0.6]],
        [10],
        [[0, 0, 5, 5], [5, 5, 10, 10]],
        {
            "shape": [1, 10, 10],
            "true_rects": [[0, 5, 0, 5]],
            "tracklet_mask": {"10": 0},
            "avg_prob": {"10": 0.9},
        },
        0.5,
        {},
    ),
    "one_by_one_clear_lock": (
        [[0.8]],
        [[0.8]],
        [1],
        [[0, 0, 4, 4]],
        None,
        0.5,
        {},
    ),
    "all_gated_locks_nothing": (
        [[0.2, 0.1], [0.05, 0.3]],
        [[0.2, 0.1], [0.05, 0.3]],
        [1, 2],
        [[0, 0, 4, 4], [4, 4, 8, 8]],
        None,
        0.5,
        {},
    ),
    "ambiguous_column_two_tracks_one_det": (
        [[0.7], [0.6]],
        [[0.7], [0.6]],
        [1, 2],
        [[0, 0, 4, 4]],
        None,
        0.5,
        {},
    ),
    "middle_lock_reduces_3x3": (
        [[0.9, 0.6, 0.0], [0.0, 0.0, 0.8], [0.55, 0.0, 0.3]],
        [[0.9, 0.6, 0.0], [0.0, 0.0, 0.8], [0.55, 0.0, 0.3]],
        [1, 2, 3],
        [[0, 0, 4, 4], [4, 0, 8, 4], [0, 4, 4, 8]],
        None,
        0.5,
        {},
    ),
    "isolated_below_threshold_flag_off": (
        [[0.3]],
        [[0.3]],
        [10],
        [[0, 0, 5, 5], [90, 90, 99, 99]][:1],
        {
            "shape": [1, 10, 10],
            "true_rects": [[0, 5, 0, 5]],
            "tracklet_mask": {"10": 0},
            "avg_prob": {"10": 0.9},
        },
        0.5,
        {"enable_isolated_mask_matching": False},
    ),
    "isolated_below_threshold_flag_on": (
        [[0.3]],
        [[0.3]],
        [10],
        [[0, 0, 5, 5]],
        {
            "shape": [1, 10, 10],
            "true_rects": [[0, 5, 0, 5]],
            "tracklet_mask": {"10": 0},
            "avg_prob": {"10": 0.9},
        },
        0.5,
        {"enable_isolated_mask_matching": True},
    ),
    "confidence_below_floor_changes_nothing": (
        [[0.7, 0.6]],
        [[0.7, 0.6]],
        [10],
        [[0, 0, 5, 5], [5, 5, 10, 10]],
        {
            "shape": [1, 10, 10],
            "true_rects": [[0, 5, 0, 5]],
            "tracklet_mask": {"10": 0},
            "avg_prob": {"10": 0.5},
        },
        0.5,
        {},
    ),
    "coverage_below_floor_changes_nothing": (
        # mask true on the LEFT half; box on the RIGHT half -> coverage ~0
        [[0.7, 0.6]],
        [[0.7, 0.6]],
        [10],
        [[5, 0, 10, 10], [0, 0, 5, 5]],
        {
            "shape": [1, 10, 10],
            "true_rects": [[0, 10, 0, 5]],
            "tracklet_mask": {"10": 0},
            "avg_prob": {"10": 0.9},
        },
        0.5,
        {},
    ),
    "fill_ratio_below_floor_changes_nothing": (
        # 2x2 mask inside a 10x10 box -> fill 0.04 < 0.05
        [[0.7, 0.6]],
        [[0.7, 0.6]],
        [10],
        [[0, 0, 10, 10], [10, 10, 20, 20]],
        {
            "shape": [1, 20, 20],
            "true_rects": [[0, 2, 0, 2]],
            "tracklet_mask": {"10": 0},
            "avg_prob": {"10": 0.9},
        },
        0.5,
        {},
    ),
    "no_mask_output_is_identity_on_scores": (
        [[0.7, 0.6], [0.55, 0.8]],
        [[0.7, 0.6], [0.55, 0.8]],
        [1, 2],
        [[0, 0, 4, 4], [4, 4, 8, 8]],
        None,
        0.5,
        {},
    ),
}


def main() -> None:
    ref_sha = subprocess.run(
        ["git", "-C", str(REF_SRC.parent), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    out = {
        "provenance": {
            "upstream": "roboflow/trackers (Apache-2.0)",
            "path": "src/trackers/core/mcbyte/mask_association.py",
            "commit": ref_sha,
            "generated": str(date.today()),
            "generator": "shipinfer scratchpad gen_mcbyte_golden.py (committed alongside)",
            "conversion": "cost = 1 - similarity; max_cost = 1 - minimum_similarity; "
            "reference adds fill_ratio where the port subtracts it",
        },
        "cases": {},
    }
    for name, (sim, raw, ids, boxes, mask_spec, min_sim, kwargs) in CASES.items():
        similarity = np.asarray(sim, dtype=np.float32)
        raw_iou = np.asarray(raw, dtype=np.float32)
        det_boxes = np.asarray(boxes, dtype=np.float32)
        result = condition_similarity_with_masks(
            similarity=similarity,
            raw_iou_similarity=raw_iou,
            tracklet_ids=list(ids),
            detection_boxes=det_boxes,
            mask_output=_mask_output(mask_spec),
            minimum_similarity=min_sim,
            **kwargs,
        )
        out["cases"][name] = {
            "inputs": {
                "similarity": similarity.tolist(),
                "raw_iou_similarity": raw_iou.tolist(),
                "tracklet_ids": list(ids),
                "detection_boxes": det_boxes.tolist(),
                "mask_spec": mask_spec,
                "minimum_similarity": min_sim,
                "kwargs": kwargs,
            },
            "helpers": {
                "clear_matches": _get_clear_matches(similarity, min_sim),
                "ambiguous": _get_ambiguous_candidate_matrix(similarity, min_sim).tolist(),
                "isolated": _get_isolated_candidate_matrix(raw_iou, min_sim).tolist(),
                "remaining": _get_remaining_indices(
                    similarity.shape[0],
                    similarity.shape[1],
                    _get_clear_matches(similarity, min_sim),
                ),
            },
            "result": {
                "conditioned_similarity": result.conditioned_similarity.tolist(),
                "locked_matches": [list(p) for p in result.locked_matches],
                "remaining_track_indices": list(result.remaining_track_indices),
                "remaining_detection_indices": list(result.remaining_detection_indices),
            },
        }
    path = Path(__file__).with_name("mcbyte_association_golden.json")
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {path} ({len(out['cases'])} cases, reference {ref_sha[:9]})")


if __name__ == "__main__":
    main()
