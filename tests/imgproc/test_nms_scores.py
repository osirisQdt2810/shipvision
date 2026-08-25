"""``nms_with_scores`` must be the backend's ``nms`` plus the scores, not a second algorithm.

The two methods are documented as interchangeable — "the indices it returns always agree with
:meth:`nms`" — and they did agree. What differed was the *route*: ``nms_with_scores`` called
the shared numpy loop for every method, including ``classic``, which has a CUDA bitmask kernel
and a ``torchvision.ops.nms`` path. Measured on the native backend at 25 000 proposals and
IoU 0.5 that was 77 ms against 11 586 ms, a factor of 150, and the ``nms`` docstring pointed
callers at the slow one. Against a ~1 ms/frame budget a detector lane that followed the
documentation lost device NMS entirely, with no error and no wrong number to notice.

Only ``linear`` and ``gauss`` have to share the sequential loop, and for a stated reason: soft
decay is order-dependent, so box j's score depends on which boxes were picked before it and
there is no bitmask formulation. For ``classic``, ``neighborhood`` and ``none`` the kept
scores *are* the original scores, so the score-carrying variant is one gather away from the
indices.

These tests pin the route as well as the answer, with a spy rather than a timing threshold: a
wall-clock assertion on a shared GPU box is a flake, and "did it call the backend" is the
property that actually matters.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipvision.imgproc import IMGPROC
from shipvision.imgproc.nms import SOFT_METHODS, suppress
from tests.imgproc.conftest import backend_params

DEVICE_METHODS = ["classic", "neighborhood", "none"]
"""The methods whose kept scores are the originals, so they can go through ``nms``."""

SHARED_METHODS = sorted(SOFT_METHODS)
"""The methods that must not: their scores only exist inside the sequential loop."""


@pytest.fixture(params=backend_params())
def candidate(request):
    """One image-ops backend that this machine can actually build."""
    return IMGPROC.build("default", backend=request.param)


def proposals(*, seed: int = 3, count: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """Overlapping boxes with distinct scores, the shape a raw detector output has."""
    rng = np.random.default_rng(seed)
    top_left = rng.uniform(0.0, 400.0, size=(count, 2)).astype(np.float32)
    extent = rng.uniform(10.0, 120.0, size=(count, 2)).astype(np.float32)
    boxes = np.concatenate([top_left, top_left + extent], axis=1).astype(np.float32)
    scores = (np.arange(count, dtype=np.float32) + 1.0) / (count + 1)
    rng.shuffle(scores)
    return boxes, scores


class TestTheFastMethodsGoThroughTheBackend:
    """``classic``, ``neighborhood`` and ``none`` reach ``self.nms``, once, per call.

    This is the finding, expressed as a route rather than as a duration. On the native backend
    ``self.nms`` is the CUDA bitmask; on the torch backend it is ``torchvision.ops.nms``; on
    numpy it is the same loop either way — and the assertion is identical for all three, which
    is the point of a seam.
    """

    @pytest.mark.parametrize("method", DEVICE_METHODS)
    def test_it_calls_nms_exactly_once(self, candidate, monkeypatch, method: str) -> None:
        boxes, scores = proposals()
        calls: list[str] = []
        original = candidate.nms

        def spy(*args, **kwargs):
            calls.append(kwargs.get("method", "classic"))
            return original(*args, **kwargs)

        monkeypatch.setattr(candidate, "nms", spy)

        candidate.nms_with_scores(boxes, scores, iou_threshold=0.5, method=method)

        assert calls == [method], "nms_with_scores bypassed the backend's own nms"

    @pytest.mark.parametrize("method", SHARED_METHODS)
    def test_a_soft_method_does_not_call_nms(self, candidate, monkeypatch, method: str) -> None:
        """It must not: ``nms`` returns indices, and a soft method's answer is the scores.

        Routing ``linear`` through ``nms`` would either recompute the decay twice or return
        undecayed scores, and the second failure is silent.
        """
        boxes, scores = proposals()
        calls: list[str] = []
        original = candidate.nms

        def spy(*args, **kwargs):
            calls.append(kwargs.get("method", "classic"))
            return original(*args, **kwargs)

        monkeypatch.setattr(candidate, "nms", spy)

        candidate.nms_with_scores(boxes, scores, iou_threshold=0.5, method=method)

        assert calls == []


class TestTheIndicesStillAgree:
    """Changing the route may not change the answer — for any method, on any backend."""

    @pytest.mark.parametrize("method", [*DEVICE_METHODS, *SHARED_METHODS])
    @pytest.mark.parametrize("score_threshold", [0.0, 0.25])
    def test_the_pair_matches_nms(self, candidate, method: str, score_threshold: float) -> None:
        boxes, scores = proposals(seed=11)

        indices = candidate.nms(
            boxes,
            scores,
            iou_threshold=0.5,
            method=method,
            score_threshold=score_threshold,
        )
        paired, _ = candidate.nms_with_scores(
            boxes,
            scores,
            iou_threshold=0.5,
            method=method,
            score_threshold=score_threshold,
        )

        assert paired.tolist() == indices.tolist()
        assert paired.dtype == np.int64

    @pytest.mark.parametrize("method", [*DEVICE_METHODS, *SHARED_METHODS])
    def test_the_pair_matches_the_shared_reference(self, candidate, method: str) -> None:
        """And against ``suppress`` itself, which is what it used to call unconditionally.

        The scores as well as the indices: the whole risk of taking the fast route for
        ``classic`` is handing back the wrong score alongside a right index, and a score is
        not something a downstream tracker can sanity-check.
        """
        boxes, scores = proposals(seed=23)
        expected_indices, expected_scores = suppress(
            boxes, scores, iou_threshold=0.45, method=method, score_threshold=0.1
        )

        indices, kept_scores = candidate.nms_with_scores(
            boxes, scores, iou_threshold=0.45, method=method, score_threshold=0.1
        )

        assert indices.tolist() == expected_indices.tolist()
        assert kept_scores.tolist() == pytest.approx(expected_scores.tolist())
        assert kept_scores.dtype == np.float32

    def test_an_empty_input_gives_two_empty_arrays(self, candidate) -> None:
        indices, kept_scores = candidate.nms_with_scores(
            np.zeros((0, 4), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            iou_threshold=0.5,
        )

        assert indices.shape == (0,)
        assert kept_scores.shape == (0,)
        assert (indices.dtype, kept_scores.dtype) == (np.int64, np.float32)

    def test_a_classic_survivor_keeps_its_own_score(self, candidate) -> None:
        """Hand-checked, so a gather that used the wrong index array cannot pass.

        Three boxes: the first two overlap at IoU 1/3, the third is far away. At
        ``iou_threshold=0.2`` the middle one is suppressed and the survivors are the highest
        score and the distant box, each carrying the score it came in with.
        """
        boxes = np.array(
            [[0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 15.0, 10.0], [500.0, 500.0, 510.0, 510.0]],
            dtype=np.float32,
        )
        scores = np.array([0.9, 0.6, 0.7], dtype=np.float32)

        indices, kept_scores = candidate.nms_with_scores(boxes, scores, iou_threshold=0.2)

        assert indices.tolist() == [0, 2]
        assert kept_scores.tolist() == pytest.approx([0.9, 0.7])


class TestSoftMethodsStillDecay:
    """The half of the contract the fast route must not break.

    ``linear`` and ``gauss`` exist to *lower* scores, so their kept scores have to differ from
    the input. Asserted explicitly, because "returns the original scores" is exactly what a
    careless routing of every method through ``nms`` would produce, and it looks fine.
    """

    @pytest.mark.parametrize("method", SHARED_METHODS)
    def test_an_overlapped_box_comes_back_with_a_lower_score(
        self, candidate, method: str
    ) -> None:
        boxes = np.array([[0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 15.0, 10.0]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)

        indices, kept_scores = candidate.nms_with_scores(
            boxes, scores, iou_threshold=0.2, method=method
        )

        assert indices.tolist() == [0, 1]
        assert kept_scores[0] == pytest.approx(0.9)
        assert kept_scores[1] < 0.8

    @pytest.mark.parametrize("method", DEVICE_METHODS)
    def test_a_non_soft_method_returns_the_scores_it_was_given(
        self, candidate, method: str
    ) -> None:
        boxes, scores = proposals(seed=31, count=150)

        indices, kept_scores = candidate.nms_with_scores(
            boxes, scores, iou_threshold=0.6, method=method
        )

        assert kept_scores.tolist() == pytest.approx(scores[indices].tolist())
