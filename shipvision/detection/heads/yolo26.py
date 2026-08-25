"""The YOLO26 detection head: ``(B, N, 6)`` in network space to boxes in image pixels.

YOLO26 is **end-to-end / NMS-free** (``readme.MD:4-5``). Its exported graph already performs
top-k selection and duplicate suppression, so its output is not a grid of anchor logits but a
fixed-length list of final detections: ``[x1, y1, x2, y2, conf, cls]`` per row, in **network**
coordinates, padded with low-confidence rows to a constant ``N``.

That makes this the shortest decode in the family and moves all the risk elsewhere. There is
no anchor arithmetic, no stride table, no ``cxcywh`` conversion and no sigmoid — the three
things that can still be wrong are the confidence boundary, the class-id rounding and the
letterbox inverse, and all three are decided once in
:mod:`shipvision.detection.heads.base`.

Suppression is still available and is off by default. Running greedy NMS over an end-to-end
head's output merges two genuinely distinct overlapping objects, which is why the reference
ships with ``nms-method: 0`` (``config/yolo26_param.yaml``). It is *available* because a
non-end-to-end export of the same architecture exists, and because at a low confidence
threshold duplicate removal is wanted even from an end-to-end head.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from shipvision.detection.base import validate_alignment
from shipvision.detection.heads.base import HEADS, DetectionHead, build_detections
from shipvision.errors import DimensionMismatchError
from shipvision.imgproc.geometry import LetterboxGeometry
from shipvision.registry import PYTHON
from shipvision.types import Detections, FrameTag

__all__ = ["Yolo26Head"]


@HEADS.register("yolo26", backend=PYTHON, aliases=("yolo26-det", "end2end"))
class Yolo26Head(DetectionHead):
    """One output tensor, ``(B, N, 6)`` or wider. See :class:`DetectionHead` for the arguments.

    A wider ``D`` is accepted rather than refused: the segmentation export is ``(B, N, 38)``
    and its first six columns mean exactly the same thing, so a caller who deliberately wants
    boxes only from a segmentation engine gets them by pairing this head with the detection
    output alone. Columns past the sixth are ignored *here* and consumed by
    :class:`~shipvision.detection.heads.yolo26_seg.Yolo26SegHead`.
    """

    expected_outputs = 1

    def decode(
        self,
        outputs: Sequence[np.ndarray],
        geometries: Sequence[LetterboxGeometry],
        tags: Sequence[FrameTag],
    ) -> list[Detections]:
        """See :meth:`DetectionHead.decode`."""
        self._require_outputs(outputs)
        batch = as_detection_batch(outputs[0])
        validate_alignment(batch.shape[0], len(geometries), len(tags), what="yolo26 decode")

        return [
            build_detections(tag, geometry, self._candidates(batch[index]))
            for index, (geometry, tag) in enumerate(zip(geometries, tags, strict=True))
        ]


def as_detection_batch(output: np.ndarray) -> np.ndarray:
    """``(B, N, D)`` float32 with ``D >= 6``, accepting a single image's ``(N, D)``.

    Shared with the segmentation head. A ``(N, D)`` array is promoted to a batch of one
    because that is what a recorded fixture and a single-frame call look like, and because
    the alternative — letting it through — would treat ``N`` as the batch axis and decode
    every proposal as its own frame.
    """
    array = np.asarray(output, dtype=np.float32)
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3 or array.shape[2] < 6:
        raise DimensionMismatchError(
            f"a YOLO26 detection output is (B, N, D) with D >= 6 = "
            f"[x1, y1, x2, y2, conf, cls], got {array.shape}. YOLO26 is end-to-end, so this "
            f"is a list of final detections and not an anchor grid — a (B, 84, 8400) tensor "
            f"is a YOLOv8 head and needs its own decode"
        )
    return array
