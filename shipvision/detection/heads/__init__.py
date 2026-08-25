"""Output-tensor decoding, one head per model family.

A head is the only place that knows what the numbers in a model's output mean, which is why
it is a registry rather than a branch: adding YOLOv8's ``(B, 84, 8400)`` anchor grid, or an
RT-DETR head, is a new file and a decorator, and nothing in the runtimes changes. It is also
the part of a detector that can be tested exhaustively with no hardware at all — a synthesised
output tensor whose correct answer you chose is a complete test of a decode.

Read :mod:`shipvision.detection.heads.base` for the three rules every head shares: the
inclusive confidence boundary, the half-away-from-zero class rounding, and the descending-score
output order.

    from shipvision.detection.heads import HEADS

    head = HEADS.build("yolo26", conf_threshold=0.3)
    results = head.decode([output], geometries, tags)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from shipvision.detection.heads.base import (
    HEADS,
    Candidates,
    DetectionHead,
    build_detections,
    round_class_ids,
)
from shipvision.detection.heads.masks import (
    bilinear_resize,
    box_crop_bounds,
    fuse_mask_logits,
    unpad_mask,
)
from shipvision.detection.heads.yolo26 import Yolo26Head
from shipvision.detection.heads.yolo26_seg import Yolo26SegHead
from shipvision.errors import ModelLoadError

__all__ = [
    "HEADS",
    "Candidates",
    "DetectionHead",
    "Yolo26Head",
    "Yolo26SegHead",
    "bilinear_resize",
    "box_crop_bounds",
    "build_detections",
    "fuse_mask_logits",
    "resolve_head",
    "round_class_ids",
    "unpad_mask",
]


def resolve_head(
    output_shapes: Sequence[Sequence[int]],
    *,
    name: str | None = None,
    artefact: str = "artefact",
    **kwargs: Any,
) -> DetectionHead:
    """Pick the head an artefact's outputs can actually feed, and build it.

    The head is **discovered from the artefact** for the same reason ``input_hw`` is: a
    detection engine has one output and a segmentation engine has two, the artefact knows
    which it is, and a config file that says otherwise is a disagreement with no symptom —
    boxes come out either way, and only the masks quietly stop existing.

    Args:
        output_shapes: the artefact's output shapes, in its own order. Only the ranks are read,
            because a binding's order and name are whatever the exporter chose while the rank
            of a prototype tensor is not.
        name: pin a head. It is then *checked* against the artefact rather than trusted, so
            naming ``yolo26`` for a two-output engine fails at load.
        artefact: what to call the artefact in an error message — a path, usually.
        **kwargs: forwarded to the head's constructor.

    Raises:
        ModelLoadError: the outputs match no head, or contradict the head that was named.
    """
    shapes = [tuple(int(v) for v in shape) for shape in output_shapes]
    if name is None:
        name = _head_for(shapes, artefact)

    head_class = HEADS.get(name)
    expected = head_class.expected_outputs
    if expected != len(shapes):
        raise ModelLoadError(
            f"head {name!r} decodes {expected} output tensor(s) but {artefact} has "
            f"{len(shapes)}: {shapes}. One of the two is wrong, and continuing would mean "
            f"choosing which — a segmentation engine decoded as a detector returns believable "
            f"boxes and no masks at all"
        )
    return HEADS.build(name, **kwargs)


def _head_for(shapes: Sequence[tuple[int, ...]], artefact: str) -> str:
    ranks = sorted(len(shape) for shape in shapes)
    if ranks == [3]:
        return "yolo26"
    if ranks == [3, 4]:
        return "yolo26_seg"
    raise ModelLoadError(
        f"cannot tell which head decodes {artefact}: its outputs have ranks {ranks} and "
        f"shapes {list(shapes)}. A YOLO26 detector is one (B, N, D) tensor and a YOLO26-seg "
        f"is that plus a rank-4 (B, C, ph, pw) prototype basis. Name the head explicitly if "
        f"the export is something else"
    )
