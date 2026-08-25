"""Shared scaffolding for the detection tests.

Two things live here. The first is :func:`to_network_space`, the *forward* letterbox map for
boxes — deliberately written out longhand as ``x * scale + pad`` rather than by calling
anything in the library, because it is the oracle the library's inverse is checked against. If
both directions came from the same code, a sign error in the pad would round-trip perfectly.

The second is :func:`detection_output`, which synthesises the tensor a YOLO26 engine would
emit. Every decode test starts by choosing the boxes it wants back and building the tensor that
should produce them, which is the only way to test a decode: "it produced some detections"
proves nothing about whether they are in the right place.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipvision.imgproc.geometry import LetterboxGeometry
from shipvision.types import Frame, FrameTag

#: A 1080p frame into a 640x640 network input. The total vertical pad is 280 — even.
LANDSCAPE = (1080, 1920)

#: 1083 rows instead of 1080, which makes the resized height 361 and the total vertical pad
#: 279 — **odd**, so the image sits one pixel above centre and ``pad_top != pad_bottom``. This
#: is the case that catches a half-pixel error in the inverse, and it is the reason the shape
#: looks arbitrary.
ODD_PAD = (1083, 1920)

NETWORK = (640, 640)


def geometry(source_hw=LANDSCAPE, target_hw=NETWORK) -> LetterboxGeometry:
    return LetterboxGeometry.plan(source_hw, target_hw)


def to_network_space(boxes, geom: LetterboxGeometry) -> np.ndarray:
    """``(n, 4)`` xyxy in source pixels to the same boxes in network pixels.

    The forward letterbox map, written independently of
    :meth:`~shipvision.imgproc.geometry.LetterboxGeometry.invert_boxes` on purpose — see the
    module docstring. float32 throughout because that is what the whole pipeline is, and a
    float64 oracle would report the library's rounding as an error.
    """
    source = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scale = np.float32(geom.scale)
    out = np.empty_like(source)
    out[:, 0::2] = source[:, 0::2] * scale + np.float32(geom.pad_left)
    out[:, 1::2] = source[:, 1::2] * scale + np.float32(geom.pad_top)
    return out


def detection_output(boxes, scores, class_ids, *, extra=None, rows=None) -> np.ndarray:
    """A ``(1, N, D)`` YOLO26 detection tensor holding exactly these proposals.

    Args:
        boxes: ``(n, 4)`` xyxy in **network** space.
        scores: ``(n,)`` confidences.
        class_ids: ``(n,)`` class ids, as floats — the channel is float in a real output, which
            is the whole reason there is a rounding rule to pin.
        extra: ``(n, k)`` mask coefficients appended after the sixth column.
        rows: pad out to this many proposals with zero-confidence rows, as a real end-to-end
            export does. `None` uses exactly ``n``.
    """
    box_array = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    count = box_array.shape[0]
    width = 6 if extra is None else 6 + np.asarray(extra).shape[1]
    total = count if rows is None else rows
    assert total >= count

    output = np.zeros((1, total, width), dtype=np.float32)
    output[0, :count, 0:4] = box_array
    output[0, :count, 4] = np.asarray(scores, dtype=np.float32)
    output[0, :count, 5] = np.asarray(class_ids, dtype=np.float32)
    if extra is not None:
        output[0, :count, 6:] = np.asarray(extra, dtype=np.float32)
    return output


def frame(camera_id="cam-01", frame_id=0, source_hw=LANDSCAPE, image=None) -> Frame:
    """A frame with a declared extent and no pixels, which is all the mock detector reads."""
    return Frame(
        tag=FrameTag(camera_id, frame_id),
        image=image,
        height=source_hw[0],
        width=source_hw[1],
    )


def grey_frame(camera_id="cam-01", frame_id=0, source_hw=LANDSCAPE) -> Frame:
    """A frame that really does carry uint8 BGR pixels, for the letterboxing backends."""
    height, width = source_hw
    image = np.full((height, width, 3), 96, dtype=np.uint8)
    return Frame(tag=FrameTag(camera_id, frame_id), image=image)


@pytest.fixture
def tag() -> FrameTag:
    return FrameTag("cam-07", 42, timestamp=1.5)
