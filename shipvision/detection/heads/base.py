"""The head contract: an output tensor becomes tagged detections in image pixels.

A *head* is the only part of a detector that knows what the numbers in an output tensor mean.
Keeping it separate from the runtime is what lets one decode serve TensorRT, TorchScript and
a recorded ``.npy`` fixture — which is also what makes the decode testable with no GPU, and
the decode is where the subtle errors live. There is no anchor arithmetic to get wrong in a
YOLO26 head, and there is still a half-pixel letterbox inverse, a rounding rule and a
threshold's inclusivity to get wrong.

Three decisions are fixed here, once, for every head:

**Confidence admission is ``score >= conf_threshold`` — inclusive.** A box exactly at the
threshold is kept. That matches ``Yolo26PostProcessor.cpp:51`` (``if (conf < confThres)
continue``) and matches :func:`shipvision.imgproc.nms.candidates.prepare`, so the head and the
suppression it calls agree on the boundary rather than differing by one box at exactly the
threshold — which is a difference nobody notices and nobody can reproduce.

**Class ids round half away from zero.** See :func:`round_class_ids`. The reference uses C++
``std::round``, and Python's built-in :func:`round` and :func:`numpy.round` both round half to
*even*, so ``1.5`` becomes ``2`` in C++ and ``2`` in numpy, while ``2.5`` becomes ``3`` in C++
and ``2`` in numpy. Two of every four half-integers disagree. We match the reference.

**Output order is descending score, ties by ascending row index.** The reference groups by
class and emits class-by-class, so its output order depends on ``std::set<int>`` iteration
order over the labels present in the frame — meaning the same frame with one extra low-score
object can reorder every detection before it. Sorting once at the end makes the order a
function of the detections alone, which is what a downstream tracker's tie-breaking needs.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from shipvision.errors import ConfigurationError, DimensionMismatchError
from shipvision.imgproc.geometry import LetterboxGeometry
from shipvision.imgproc.nms import METHODS, NONE, suppress
from shipvision.registry import PYTHON, Registry
from shipvision.types import Detection, Detections, FrameTag

__all__ = [
    "HEADS",
    "Candidates",
    "DetectionHead",
    "build_detections",
    "round_class_ids",
]


def round_class_ids(values: np.ndarray) -> np.ndarray:
    """Float class ids to ``int32``, rounding **half away from zero**.

    ``std::round`` in ``Yolo26PostProcessor.cpp:50`` rounds half away from zero;
    :func:`numpy.round` and Python's :func:`round` round half to even. On a well-behaved
    export the class channel holds exact integers and the rule never matters — but an fp16
    engine emits ``2.5`` for class 2 or 3 often enough, and then the two rules disagree on
    half of all half-integers. Matching the reference is the choice that keeps a re-export
    comparable against the C++ deployment it replaces.

    Implemented as ``sign(x) * floor(|x| + 0.5)`` rather than ``floor(x + 0.5)`` so that the
    rule is genuinely symmetric; a negative class id is nonsense, but rounding it towards a
    *different* nonsense depending on sign would make the validation below inconsistent.
    """
    array = np.asarray(values, dtype=np.float32)
    rounded = np.sign(array) * np.floor(np.abs(array) + np.float32(0.5))
    return rounded.astype(np.int32)


@dataclass(slots=True, frozen=True)
class Candidates:
    """The survivors of one image's decode, still in network space.

    A small value object rather than four parallel lists, because the four arrays must stay
    aligned through a confidence filter, a per-class suppression and a final sort, and
    ``rows`` — the index back into the raw ``(N, D)`` output — is what a segmentation head
    needs to find each survivor's mask coefficients afterwards.

    Attributes:
        rows: ``(k,)`` int64 indices into the model's ``N`` proposals.
        boxes: ``(k, 4)`` xyxy float32, **network space**. Un-mapped by the caller.
        scores: ``(k,)`` float32, decayed if a soft suppression method was used.
        class_ids: ``(k,)`` int32.
    """

    rows: np.ndarray
    boxes: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray

    def __len__(self) -> int:
        return int(self.rows.shape[0])


class DetectionHead(abc.ABC):
    """Turns a model's raw outputs into :class:`~shipvision.types.Detections`.

    Args:
        conf_threshold: keep a proposal when ``score >= conf_threshold``. Inclusive.
        iou_threshold: passed to suppression as the overlap above which a box is punished.
        nms_method: one of :data:`shipvision.imgproc.nms.METHODS`. The default is ``"none"``
            because YOLO26 is an **end-to-end, NMS-free** detector: its exported graph has
            already suppressed duplicates, and running greedy NMS over its output again
            merges genuinely distinct overlapping objects. Suppression is still supported —
            a non-end-to-end export exists, and at a low confidence threshold duplicate
            removal is wanted even from an end-to-end head.
        class_agnostic: suppress across all classes together instead of per class. Per class
            is the default and is what the reference does: a person standing in front of a
            ship overlaps it almost completely, and class-agnostic suppression deletes one of
            them.
        sigma: the gaussian's width, read by ``nms_method="gauss"`` only.
        max_detections: keep at most this many per frame, highest score first. `None` for no
            cap. A cap is not tidiness — a broken engine can emit thousands of high-score
            proposals, and a bounded output is what stops one bad frame from stalling every
            stage behind it.
        num_classes: optional. When set, a decoded class id outside ``[0, num_classes)``
            raises instead of flowing downstream. This is the one thing about an end-to-end
            output that *cannot* be discovered from the artefact — a ``(B, N, 6)`` tensor
            says nothing about how many classes produced it — so it is configuration, and
            checking it catches the "engine has 80 classes, config has 2" mismatch on the
            first frame rather than in a report a week later.
    """

    name: str = "head"
    backend: str = PYTHON

    expected_outputs: int = 1
    """How many output tensors this head consumes. Used by
    :func:`shipvision.detection.heads.resolve_head` to pick a head from an artefact's output
    arity, and to refuse a head the caller named that the artefact cannot feed."""

    produces_masks: bool = False

    def __init__(
        self,
        *,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        nms_method: str = NONE,
        class_agnostic: bool = False,
        sigma: float = 0.5,
        max_detections: int | None = 300,
        num_classes: int | None = None,
    ) -> None:
        if not 0.0 <= conf_threshold <= 1.0:
            raise ConfigurationError(f"conf_threshold must be in [0, 1], got {conf_threshold}")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ConfigurationError(f"iou_threshold must be in [0, 1], got {iou_threshold}")
        if nms_method not in METHODS:
            raise ConfigurationError(
                f"unknown nms method {nms_method!r}; expected one of {METHODS}"
            )
        if max_detections is not None and max_detections <= 0:
            raise ConfigurationError(
                f"max_detections must be positive or None, got {max_detections}"
            )
        if num_classes is not None and num_classes <= 0:
            raise ConfigurationError(f"num_classes must be positive, got {num_classes}")

        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.nms_method = nms_method
        self.class_agnostic = bool(class_agnostic)
        self.sigma = float(sigma)
        self.max_detections = None if max_detections is None else int(max_detections)
        self.num_classes = None if num_classes is None else int(num_classes)

    # -- the contract -----------------------------------------------------------------

    @abc.abstractmethod
    def decode(
        self,
        outputs: Sequence[np.ndarray],
        geometries: Sequence[LetterboxGeometry],
        tags: Sequence[FrameTag],
    ) -> list[Detections]:
        """Raw model outputs to one :class:`~shipvision.types.Detections` per frame.

        Args:
            outputs: the model's output tensors, in the artefact's own order. Batched: the
                leading axis of a per-image tensor is the frame axis.
            geometries: the letterbox that mapped each frame into the network input, in frame
                order. Used to invert the boxes — never recomputed from the two shapes.
            tags: one tag per frame, in the same order.

        Returns:
            ``len(tags)`` results, boxes in original image pixels.

        Raises:
            DimensionMismatchError: the outputs, geometries and tags do not describe the same
                batch, or an output has a layout this head does not decode.
        """

    # -- shared machinery -------------------------------------------------------------

    def _require_outputs(self, outputs: Sequence[np.ndarray]) -> None:
        """Refuse an output arity this head does not consume, naming the head that would.

        Deliberately strict rather than "use the first tensor and ignore the rest". A
        segmentation engine wired to a detection head produces perfectly plausible boxes and
        no masks, and nothing downstream can tell that from a frame where nothing was
        segmentable — which is precisely the class of failure this library refuses to make
        silent.
        """
        if len(outputs) != self.expected_outputs:
            raise DimensionMismatchError(
                f"{type(self).__name__} decodes {self.expected_outputs} output tensor(s) but "
                f"was given {len(outputs)}. A detection engine has one output and a "
                f"segmentation engine has two (detections plus the mask prototypes); pick "
                f"the head that matches the artefact rather than ignoring a tensor"
            )

    def _candidates(self, plane: np.ndarray) -> Candidates:
        """One image's ``(N, D)`` proposals, filtered, suppressed and sorted.

        The whole per-image decode except the letterbox inverse, which needs the geometry and
        so belongs to the caller. Kept here rather than in the concrete head because a
        segmentation head differs from a detection head only in what it does *after* this.
        """
        pred = np.asarray(plane, dtype=np.float32)
        if pred.ndim != 2 or pred.shape[1] < 6:
            raise DimensionMismatchError(
                f"a YOLO26 image plane is (N, D) with D >= 6 = [x1, y1, x2, y2, conf, cls], "
                f"got {pred.shape}. A (B, N, D) tensor is a batch — index it first"
            )

        # `>=`, and NaN fails every comparison, so a NaN confidence is dropped rather than
        # being admitted and then sorting unpredictably.
        rows = np.flatnonzero(pred[:, 4] >= np.float32(self.conf_threshold))
        if rows.size == 0:
            return _no_candidates()

        boxes = self._sane_boxes(pred[rows, 0:4])
        scores = self._validated_scores(pred[rows, 4])
        class_ids = self._validated_class_ids(pred[rows, 5])
        return self._suppress(rows.astype(np.int64), boxes, scores, class_ids)

    @staticmethod
    def _sane_boxes(boxes: np.ndarray) -> np.ndarray:
        """``(k, 4)`` xyxy with ``x2 >= x1`` and ``y2 >= y1``, by widening not by dropping.

        ``Detection`` refuses an inside-out box, and an engine occasionally emits one — a
        zero-area proposal at the frame edge whose corners round the wrong way in fp16. The
        reference clamps the *extent* to zero (``max(0.0f, x2 - x1)``) and so keeps the box
        with no area; doing the same keeps that behaviour rather than silently changing the
        detection count between the C++ deployment and this one.
        """
        out = np.array(boxes, dtype=np.float32, copy=True)
        np.maximum(out[:, 2], out[:, 0], out=out[:, 2])
        np.maximum(out[:, 3], out[:, 1], out=out[:, 3])
        return out

    @staticmethod
    def _validated_scores(values: np.ndarray) -> np.ndarray:
        """``(k,)`` float32 confidences, clipped into ``[0, 1]`` — but only just.

        ``Detection`` refuses a score outside ``[0, 1]``, and a sigmoid in fp16 does return
        ``1.0000001``, so a hair of overshoot is clipped rather than raised on. A score of
        15.0 is a different thing entirely: the most likely cause is a ``(B, N, 6)`` tensor
        whose columns are not ``[x1, y1, x2, y2, conf, cls]`` — a raw ``xywh`` layout puts a
        pixel width in the confidence slot — and clipping that to 1.0 would turn a wiring
        mistake into a frame full of maximally-confident detections.
        """
        scores = np.asarray(values, dtype=np.float32)
        if scores.size and float(scores.max()) > 1.0 + 1e-3:
            raise DimensionMismatchError(
                f"decoded confidences reach {float(scores.max()):.4g}, which is not a "
                f"probability. The output columns are probably not "
                f"[x1, y1, x2, y2, conf, cls] — an xywh layout puts a pixel width here"
            )
        return np.clip(scores, 0.0, 1.0)

    def _validated_class_ids(self, values: np.ndarray) -> np.ndarray:
        class_ids = round_class_ids(values)
        if class_ids.size == 0:
            return class_ids
        low = int(class_ids.min())
        high = int(class_ids.max())
        limit = self.num_classes
        if low < 0 or (limit is not None and high >= limit):
            raise DimensionMismatchError(
                f"decoded class ids span [{low}, {high}], outside "
                f"[0, {'inf' if limit is None else limit}). Either the output layout is not "
                f"[x1, y1, x2, y2, conf, cls] — a (B, N, 6) tensor with the score and the "
                f"class swapped decodes without complaint — or num_classes does not match "
                f"the engine"
            )
        return class_ids

    def _suppress(
        self,
        rows: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
    ) -> Candidates:
        """Per-class (or class-agnostic) suppression, then one deterministic sort.

        Suppression itself is :func:`shipvision.imgproc.nms.suppress` — the one implementation
        of the five methods and of the admission, overlap and departure rules.
        :meth:`~shipvision.imgproc.base.ImageOps.nms_with_scores` is concrete on the base
        class and delegates to exactly this function, so calling it directly costs nothing and
        saves the head from having to hold an image-ops backend for what is a scalar loop over
        a few dozen survivors.

        ``score_threshold=conf_threshold`` is passed on purpose: for a soft method the whole
        output is a re-weighted score, and without a floor every box comes back merely
        re-ranked. That is the mistake the ``nms_with_scores`` docstring warns about.
        """
        groups = (
            [np.arange(rows.size, dtype=np.int64)]
            if self.class_agnostic
            else [np.flatnonzero(class_ids == cls) for cls in np.unique(class_ids)]
        )

        kept: list[np.ndarray] = []
        kept_scores: list[np.ndarray] = []
        for group in groups:
            local, decayed = suppress(
                boxes[group],
                scores[group],
                iou_threshold=self.iou_threshold,
                method=self.nms_method,
                sigma=self.sigma,
                score_threshold=self.conf_threshold,
            )
            kept.append(group[local])
            kept_scores.append(decayed)

        selected = np.concatenate(kept) if kept else np.zeros(0, dtype=np.int64)
        final_scores = (
            np.concatenate(kept_scores) if kept_scores else np.zeros(0, dtype=np.float32)
        )
        if selected.size == 0:
            return _no_candidates()

        # Descending score, ties by ascending row index. `lexsort` is stable and reads its
        # keys last-first, so the primary key goes last.
        order = np.lexsort((rows[selected], -final_scores))
        if self.max_detections is not None:
            order = order[: self.max_detections]
        selected = selected[order]
        return Candidates(
            rows=rows[selected],
            boxes=boxes[selected],
            scores=final_scores[order].astype(np.float32, copy=False),
            class_ids=class_ids[selected],
        )

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} conf={self.conf_threshold} iou={self.iou_threshold} "
            f"nms={self.nms_method!r}>"
        )


def _no_candidates() -> Candidates:
    """The empty selection, with every array at its documented empty shape."""
    return Candidates(
        rows=np.zeros(0, dtype=np.int64),
        boxes=np.zeros((0, 4), dtype=np.float32),
        scores=np.zeros(0, dtype=np.float32),
        class_ids=np.zeros(0, dtype=np.int32),
    )


def build_detections(
    tag: FrameTag,
    geometry: LetterboxGeometry,
    candidates: Candidates,
    masks: Sequence[np.ndarray | None] | None = None,
) -> Detections:
    """Candidates in network space plus a geometry to one frame's ``Detections``.

    This is the only place in the package where a box leaves network space, and it does it
    with :meth:`~shipvision.imgproc.geometry.LetterboxGeometry.invert_boxes` — the algebraic
    inverse of the forward map, using the numbers that were actually used. Nothing here
    re-derives ``gain``, ``padH`` or ``padW`` from the two shapes the way
    ``Yolo26PostProcessor.cpp:118-120`` does, because that arithmetic exists correctly in one
    place and a second copy is a second rounding rule.
    """
    boxes = geometry.invert_boxes(candidates.boxes)
    items = [
        Detection(
            box=boxes[index],
            score=float(candidates.scores[index]),
            class_id=int(candidates.class_ids[index]),
            mask=None if masks is None else masks[index],
        )
        for index in range(len(candidates))
    ]
    return Detections(
        tag=tag,
        items=items,
        height=geometry.source_height,
        width=geometry.source_width,
    )


#: The head family: one entry per output layout a detector can produce. A new model family is
#: a new file and a decorator here, never a branch inside an existing decode.
HEADS: Registry[DetectionHead] = Registry("detection head")
