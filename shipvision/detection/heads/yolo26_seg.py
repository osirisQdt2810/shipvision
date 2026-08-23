"""The YOLO26 segmentation head: ``(B, N, 38)`` plus ``(1, 32, h/4, w/4)`` to masks.

Two outputs (``readme.MD:5``). The detection tensor's first six columns are exactly the
detection head's — ``[x1, y1, x2, y2, conf, cls]`` in network space — and the remaining
``D - 6`` are mask coefficients, sliced from index 6
(``Yolo26PostProcessor.cpp:63``). The second output is the prototype basis: 32 planes at a
quarter of the network resolution, shared by every detection in the frame.

**The coefficient count is discovered, not configured.** ``D - 6`` from the detection output
must equal the prototype tensor's channel count, and the two disagreeing means the engine's
two outputs did not come from the same export — so it is a load-time-shaped failure that this
head raises rather than a warning it logs. The reference logs
(``Yolo26SegPostProcessor.cpp:96-98``) and carries on with whichever number is smaller, which
produces masks built from a truncated basis: plausible, wrong, and silent.

**Masks come back in the box's frame of reference**, matching
:class:`shipvision.types.Detection`'s contract, and are cropped there straight out of the
resample rather than by materialising a full-frame plane first — see
:func:`~shipvision.detection.heads.masks.bilinear_resize`'s ``window``. A 1080x1920 float
plane per detection is 8 MB, and twenty ships a frame would spend the whole latency budget in
``malloc``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from shipvision.detection.base import validate_alignment
from shipvision.detection.heads.base import HEADS, Candidates, build_detections
from shipvision.detection.heads.masks import (
    box_crop_bounds,
    fuse_mask_logits,
    unpad_mask,
)
from shipvision.detection.heads.yolo26 import Yolo26Head, as_detection_batch
from shipvision.errors import ConfigurationError, DimensionMismatchError
from shipvision.imgproc.geometry import LetterboxGeometry
from shipvision.registry import PYTHON
from shipvision.types import Detections, FrameTag

__all__ = ["Yolo26SegHead"]


@HEADS.register("yolo26_seg", backend=PYTHON, aliases=("yolo26-seg", "yolo26seg"))
class Yolo26SegHead(Yolo26Head):
    """Boxes exactly as :class:`~shipvision.detection.heads.yolo26.Yolo26Head`, plus masks.

    Args:
        mask_threshold: a mask probability at or above this is inside the instance.
            Inclusive, and 0.5 by default — ``Yolo26SegDetectorParams::maskConfThres``.
        binarise: return a boolean mask. `False` returns the float32 probabilities in the same
            window instead, which is what a quality score or a soft IoU wants; the reference
            has no such option and thresholds unconditionally.
        **kwargs: everything :class:`~shipvision.detection.heads.base.DetectionHead` takes.
    """

    expected_outputs = 2
    produces_masks = True

    def __init__(
        self, *, mask_threshold: float = 0.5, binarise: bool = True, **kwargs: object
    ) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        if not 0.0 <= mask_threshold <= 1.0:
            raise ConfigurationError(
                f"mask_threshold is a probability and must be in [0, 1], got {mask_threshold}"
            )
        self.mask_threshold = float(mask_threshold)
        self.binarise = bool(binarise)

    def decode(
        self,
        outputs: Sequence[np.ndarray],
        geometries: Sequence[LetterboxGeometry],
        tags: Sequence[FrameTag],
    ) -> list[Detections]:
        """See :meth:`~shipvision.detection.heads.base.DetectionHead.decode`."""
        self._require_outputs(outputs)
        detections, prototypes = _split_outputs(outputs)
        validate_alignment(
            detections.shape[0], len(geometries), len(tags), what="yolo26-seg decode"
        )
        _require_matching_widths(detections.shape[2], prototypes.shape[1])
        _require_prototype_batch(prototypes.shape[0], detections.shape[0])

        results: list[Detections] = []
        for index, (geometry, tag) in enumerate(zip(geometries, tags, strict=True)):
            plane = detections[index]
            candidates = self._candidates(plane)
            basis = prototypes[index if prototypes.shape[0] > 1 else 0]
            masks = self._masks(plane, candidates, basis, geometry)
            results.append(build_detections(tag, geometry, candidates, masks))
        return results

    def _masks(
        self,
        plane: np.ndarray,
        candidates: Candidates,
        prototypes: np.ndarray,
        geometry: LetterboxGeometry,
    ) -> list[np.ndarray | None]:
        """One mask per survivor, in that survivor's box frame of reference.

        The boxes are inverted a second time here rather than being threaded down from
        :func:`~shipvision.detection.heads.base.build_detections`, because the mask window has
        to be in source pixels and the alternative is either passing the inverted boxes into
        the mask code (coupling the two) or inverting in the caller and handing both around.
        Inversion is two subtractions and a divide over a handful of boxes; the coupling would
        outlive the saving.
        """
        if len(candidates) == 0:
            return []

        coefficients = plane[candidates.rows, 6:]
        logits = fuse_mask_logits(coefficients, prototypes)
        boxes = geometry.invert_boxes(candidates.boxes)

        masks: list[np.ndarray | None] = []
        for index in range(len(candidates)):
            bounds = box_crop_bounds(
                boxes[index], geometry.source_height, geometry.source_width
            )
            y0, y1, x0, x1 = bounds
            probability = unpad_mask(logits[index], geometry, window=(y0, y1, x0, x1))
            masks.append(
                probability >= np.float32(self.mask_threshold)
                if self.binarise
                else probability
            )
        return masks


def _split_outputs(outputs: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """``(detections, prototypes)`` identified by rank, not by binding order.

    An engine's output order is whatever the exporter emitted, and the reference deals with
    that by configuring two indices (``detOutputIndex``, ``protoOutputIndex``) that a YAML
    file can get the wrong way round — it even warns when they are equal, which is a check
    for a mistake that only a configured index makes possible. The shapes are unambiguous
    instead: the detections are rank 3 and the prototype basis is rank 4.
    """
    ranks = [np.asarray(output).ndim for output in outputs]
    if sorted(ranks) != [3, 4]:
        raise DimensionMismatchError(
            f"a YOLO26-seg engine returns one (B, N, D) detection tensor and one "
            f"(B, C, ph, pw) prototype tensor; got outputs with ranks {ranks}"
        )
    detection_index = ranks.index(3)
    proto_index = ranks.index(4)
    prototypes = np.asarray(outputs[proto_index], dtype=np.float32)
    if prototypes.shape[1] <= 0:
        raise DimensionMismatchError(
            f"the prototype tensor has {prototypes.shape[1]} channels; a mask basis needs at "
            f"least one plane"
        )
    return as_detection_batch(outputs[detection_index]), prototypes


def _require_prototype_batch(prototype_batch: int, frames: int) -> None:
    """The prototype tensor must cover every frame, or be the one plane set they all share.

    A batch of 1 against several frames is normal and is what ``(1, 32, h/4, w/4)`` in the
    readme means — the export writes one basis per call. Anything between the two is an
    engine whose batch axis was partly collapsed, and the reference's response is to clamp
    the frame index into range (``Yolo26SegPostProcessor.cpp:128``), which silently gives the
    last frame's basis to every frame after it.
    """
    if prototype_batch not in (1, frames):
        raise DimensionMismatchError(
            f"the prototype tensor has a batch of {prototype_batch} for {frames} frames; "
            f"expected {frames}, or 1 for a basis shared by the whole batch. Clamping the "
            f"index into range would give one frame's masks to another"
        )


def _require_matching_widths(detection_width: int, prototype_channels: int) -> None:
    if detection_width - 6 != prototype_channels:
        raise DimensionMismatchError(
            f"the detection output is {detection_width} wide, so it carries "
            f"{detection_width - 6} mask coefficients, but the prototype tensor has "
            f"{prototype_channels} planes. These must match: the reference takes the smaller "
            f"of the two and builds every mask from a truncated basis, which looks like a "
            f"model that segments badly rather than like a wiring mistake"
        )
