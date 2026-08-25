"""The decode: a synthesised output tensor, and the boxes it must produce.

This is where the invisible bugs live. A detector that returns plausible boxes in the wrong
place looks exactly like one that works — every camera is shifted by the same amount, so
nothing looks anomalous — which is why every test here starts from boxes chosen in source
pixels and asserts they come back.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipvision.detection.heads import HEADS, Yolo26Head, resolve_head, round_class_ids
from shipvision.errors import (
    BackendUnavailableError,
    ConfigurationError,
    DimensionMismatchError,
    ModelLoadError,
)
from shipvision.imgproc import IMGPROC
from shipvision.imgproc.geometry import LetterboxGeometry
from shipvision.imgproc.nms import CLASSIC, METHODS, suppress
from shipvision.types import FrameTag

from .conftest import LANDSCAPE, NETWORK, ODD_PAD, detection_output, geometry, to_network_space

SOURCE_BOXES = np.array(
    [
        [100.0, 200.0, 400.0, 700.0],
        [12.0, 33.0, 900.0, 1000.0],
        [1500.0, 50.0, 1900.0, 480.0],
    ],
    dtype=np.float32,
)


class TestLetterboxInversion:
    """A box decoded from network space must land back where it started, in image pixels.

    The single most valuable claim in this package. An error of half a pixel here is invisible
    in any smoke test and shifts every detection on every camera in production, permanently.
    """

    @pytest.mark.parametrize(
        ("source_hw", "label"), [(LANDSCAPE, "even pad"), (ODD_PAD, "odd pad")]
    )
    def test_boxes_round_trip_through_the_letterbox(self, source_hw, label, tag) -> None:
        geom = geometry(source_hw)
        output = detection_output(
            to_network_space(SOURCE_BOXES, geom), [0.9, 0.8, 0.7], [0.0, 1.0, 2.0]
        )

        result = HEADS.build("yolo26", conf_threshold=0.1).decode([output], [geom], [tag])[0]

        assert len(result) == 3
        # A thousandth of a pixel. The bound is tight on purpose: the only error left after an
        # exact algebraic inverse is float32 rounding, so anything larger is a real mistake and
        # not a tolerance to be widened.
        assert (
            np.abs(np.sort(result.boxes, axis=0) - np.sort(SOURCE_BOXES, axis=0)).max() < 1e-3
        )

    def test_the_odd_pad_case_really_is_odd(self) -> None:
        """Guard on the fixture: the test above only tests what it claims if this holds."""
        geom = geometry(ODD_PAD)

        assert (geom.target_height - geom.resized_height) % 2 == 1
        assert geom.pad_top != geom.pad_bottom

    def test_the_references_recomputed_pad_shifts_every_box(self, tag) -> None:
        """Why the geometry is carried rather than re-derived, in numbers.

        ``Yolo26PostProcessor.cpp:118-125`` recomputes the pad as
        ``(target_h - gain * raw_h) * 0.5`` in float. For a 1083-row source that is 139.5 where
        the letterbox actually used 139, so its inverse puts every box 1.5 source pixels too
        high — on this camera only, which makes it the hardest kind of bug to attribute.
        """
        geom = geometry(ODD_PAD)
        network = to_network_space(SOURCE_BOXES, geom)
        output = detection_output(network, [0.9, 0.8, 0.7], [0.0, 0.0, 0.0])

        ours = HEADS.build("yolo26", conf_threshold=0.1).decode([output], [geom], [tag])[0]

        gain = np.float32(geom.scale)
        reference_pad = (np.float32(geom.target_height) - gain * np.float32(ODD_PAD[0])) * 0.5
        reference_y = (network[:, 1] - reference_pad) / gain

        assert float(reference_pad) == pytest.approx(139.5, abs=1e-3)
        assert np.abs(
            np.sort(reference_y) - np.sort(SOURCE_BOXES[:, 1])
        ).max() == pytest.approx(1.5, abs=0.01)
        assert np.abs(np.sort(ours.boxes[:, 1]) - np.sort(SOURCE_BOXES[:, 1])).max() < 1e-3

    def test_the_carried_geometry_is_used_and_not_recomputed(self, tag) -> None:
        """Hand in a geometry that ``plan`` would never produce, and watch the decode obey it.

        This is the property the whole design rests on: post-processing inverts with the numbers
        that were actually used, so a backend free to letterbox differently — a native kernel, a
        cached resize — cannot silently disagree with the decode.
        """
        fabricated = LetterboxGeometry(
            scale=0.25,
            pad_left=7,
            pad_top=13,
            source_height=800,
            source_width=1000,
            target_height=NETWORK[0],
            target_width=NETWORK[1],
        )
        output = detection_output([[7.0, 13.0, 107.0, 63.0]], [0.9], [0.0])

        result = HEADS.build("yolo26").decode([output], [fabricated], [tag])[0]

        assert result.boxes[0].tolist() == pytest.approx([0.0, 0.0, 400.0, 200.0])
        assert (result.height, result.width) == (800, 1000)

    def test_a_box_beyond_the_frame_is_clipped_to_the_frame_edge(self, tag) -> None:
        """Clipped to the continuous extent, not to ``extent - 1``: a detection that genuinely
        touches the right edge has ``x2 == width``."""
        geom = geometry(LANDSCAPE)
        output = detection_output([[-40.0, -40.0, 900.0, 900.0]], [0.9], [0.0])

        result = HEADS.build("yolo26").decode([output], [geom], [tag])[0]

        assert result.boxes[0].tolist() == [0.0, 0.0, 1920.0, 1080.0]


class TestConfidenceFiltering:
    """Admission is ``score >= conf_threshold``. Inclusive, and both sides are tested."""

    def test_a_score_exactly_at_the_threshold_is_kept(self, tag) -> None:
        threshold = np.float32(0.25)
        output = detection_output([[0.0, 0.0, 10.0, 10.0]], [threshold], [0.0])

        result = HEADS.build("yolo26", conf_threshold=float(threshold)).decode(
            [output], [geometry()], [tag]
        )[0]

        assert len(result) == 1

    def test_a_score_one_float_below_the_threshold_is_dropped(self, tag) -> None:
        threshold = np.float32(0.25)
        just_below = np.nextafter(threshold, np.float32(0.0))
        output = detection_output([[0.0, 0.0, 10.0, 10.0]], [just_below], [0.0])

        result = HEADS.build("yolo26", conf_threshold=float(threshold)).decode(
            [output], [geometry()], [tag]
        )[0]

        assert just_below < threshold
        assert len(result) == 0

    def test_the_padding_rows_of_an_end_to_end_output_are_dropped(self, tag) -> None:
        """A real ``(B, 300, 6)`` export is mostly zero-confidence padding, and a head that
        admitted it would return three hundred boxes at the origin on every frame."""
        output = detection_output(
            to_network_space(SOURCE_BOXES[:1], geometry()), [0.9], [0.0], rows=300
        )

        result = HEADS.build("yolo26", conf_threshold=0.05).decode(
            [output], [geometry()], [tag]
        )[0]

        assert len(result) == 1

    def test_a_zero_threshold_still_drops_a_nan_confidence(self, tag) -> None:
        """NaN fails every comparison, which is the behaviour wanted: a NaN admitted at
        ``conf_threshold=0.0`` would then sort unpredictably against real scores."""
        output = detection_output(
            [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]], [np.nan, 0.0], [0.0, 0.0]
        )

        result = HEADS.build("yolo26", conf_threshold=0.0).decode(
            [output], [geometry()], [tag]
        )[0]

        assert len(result) == 1
        assert result[0].score == 0.0


class TestClassIdRounding:
    """Half away from zero, as C++ ``std::round`` does — not numpy's half-to-even."""

    @pytest.mark.parametrize(
        ("value", "expected"), [(0.5, 1), (1.5, 2), (2.5, 3), (3.49, 3), (7.0, 7)]
    )
    def test_a_half_integer_rounds_away_from_zero(self, value, expected) -> None:
        assert int(round_class_ids(np.array([value], dtype=np.float32))[0]) == expected

    def test_this_disagrees_with_numpy_and_with_python(self) -> None:
        """The whole reason the rule is written down: two of every four half-integers differ."""
        halves = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)

        ours = round_class_ids(halves).tolist()

        assert ours == [1, 2, 3, 4]
        assert np.round(halves).astype(int).tolist() == [0, 2, 2, 4]
        assert [round(float(v)) for v in halves] == [0, 2, 2, 4]

    def test_the_decoded_class_id_is_a_python_int(self, tag) -> None:
        """A ``np.int32`` compares equal to an int and serialises as ``{"class_id": 2}`` in
        some JSON encoders and blows up in others, so the boundary is crossed here."""
        output = detection_output([[0.0, 0.0, 10.0, 10.0]], [0.9], [2.5])

        detection = HEADS.build("yolo26").decode([output], [geometry()], [tag])[0][0]

        assert detection.class_id == 3
        assert type(detection.class_id) is int

    def test_a_negative_class_id_is_refused_rather_than_passed_on(self, tag) -> None:
        output = detection_output([[0.0, 0.0, 10.0, 10.0]], [0.9], [-1.0])

        with pytest.raises(DimensionMismatchError, match="class ids"):
            HEADS.build("yolo26").decode([output], [geometry()], [tag])

    def test_a_class_id_beyond_a_declared_num_classes_is_refused(self, tag) -> None:
        """Catches "the engine has 80 classes and the config says 2" on the first frame."""
        output = detection_output([[0.0, 0.0, 10.0, 10.0]], [0.9], [17.0])

        with pytest.raises(DimensionMismatchError, match="num_classes"):
            HEADS.build("yolo26", num_classes=2).decode([output], [geometry()], [tag])


class TestEmptyOutput:
    """A frame with nothing in it is ordinary input, and it still carries its tag."""

    @pytest.mark.parametrize("rows", [0, 1, 300])
    def test_no_admitted_proposals_gives_an_empty_detections_not_none(self, rows, tag) -> None:
        geom = geometry()
        output = np.zeros((1, rows, 6), dtype=np.float32)

        result = HEADS.build("yolo26", conf_threshold=0.25).decode([output], [geom], [tag])[0]

        assert result is not None
        assert len(result) == 0
        assert result.boxes.shape == (0, 4)
        assert result.scores.shape == (0,)
        assert result.class_ids.shape == (0,)
        assert result.tag == tag
        assert (result.height, result.width) == (geom.source_height, geom.source_width)

    def test_every_frame_in_a_batch_keeps_its_own_tag(self) -> None:
        """The mis-tagging failure: a result list that lost an element re-attributes every
        detection after it to the previous camera, and all of them look real."""
        geom = geometry()
        tags = [FrameTag("cam-01", 5), FrameTag("cam-02", 9), FrameTag("cam-03", 11)]
        output = np.zeros((3, 4, 6), dtype=np.float32)
        output[1, 0] = [10.0, 10.0, 20.0, 20.0, 0.9, 0.0]

        results = HEADS.build("yolo26").decode([output], [geom] * 3, tags)

        assert [r.tag for r in results] == tags
        assert [len(r) for r in results] == [0, 1, 0]

    def test_a_batch_whose_tags_do_not_match_the_output_is_refused(self, tag) -> None:
        output = np.zeros((3, 4, 6), dtype=np.float32)

        with pytest.raises(DimensionMismatchError, match="same length"):
            HEADS.build("yolo26").decode([output], [geometry()] * 3, [tag])


class TestSuppression:
    """Off by default, because YOLO26 is end-to-end — and available, because exports vary."""

    DUPLICATES = np.array(
        [[100.0, 100.0, 200.0, 200.0], [104.0, 104.0, 204.0, 204.0]], dtype=np.float32
    )

    def test_the_default_keeps_both_of_two_near_identical_boxes(self, tag) -> None:
        """``nms-method: 0`` in the reference config. An end-to-end head has already suppressed
        duplicates, and suppressing again merges genuinely distinct overlapping objects."""
        output = detection_output(self.DUPLICATES, [0.9, 0.8], [0.0, 0.0])

        result = HEADS.build("yolo26").decode([output], [geometry()], [tag])[0]

        assert len(result) == 2

    def test_classic_nms_collapses_them(self, tag) -> None:
        output = detection_output(self.DUPLICATES, [0.9, 0.8], [0.0, 0.0])

        result = HEADS.build("yolo26", nms_method="classic", iou_threshold=0.5).decode(
            [output], [geometry()], [tag]
        )[0]

        assert len(result) == 1
        assert result[0].score == pytest.approx(0.9)

    def test_suppression_is_per_class_so_a_person_in_front_of_a_ship_survives(
        self, tag
    ) -> None:
        output = detection_output(self.DUPLICATES, [0.9, 0.8], [0.0, 1.0])

        per_class = HEADS.build("yolo26", nms_method="classic", iou_threshold=0.5)
        agnostic = HEADS.build(
            "yolo26", nms_method="classic", iou_threshold=0.5, class_agnostic=True
        )

        assert len(per_class.decode([output], [geometry()], [tag])[0]) == 2
        assert len(agnostic.decode([output], [geometry()], [tag])[0]) == 1

    def test_a_soft_method_returns_the_decayed_score_and_drops_what_falls_below(
        self, tag
    ) -> None:
        """Soft NMS suppresses by lowering scores, so the confidence threshold is what actually
        removes a box — passing it as the floor is the difference between suppression and a
        re-ranking."""
        output = detection_output(self.DUPLICATES, [0.9, 0.5], [0.0, 0.0])

        lenient = HEADS.build("yolo26", nms_method="linear", conf_threshold=0.05)
        strict = HEADS.build("yolo26", nms_method="linear", conf_threshold=0.4)

        decayed = lenient.decode([output], [geometry()], [tag])[0]
        assert len(decayed) == 2
        assert decayed[1].score < 0.5

        assert len(strict.decode([output], [geometry()], [tag])[0]) == 1

    def test_max_detections_bounds_a_frame_and_keeps_the_best(self, tag) -> None:
        boxes = np.stack(
            [np.array([x, 0.0, x + 5.0, 5.0], dtype=np.float32) for x in range(0, 100, 10)]
        )
        scores = np.linspace(0.1, 0.9, boxes.shape[0], dtype=np.float32)
        output = detection_output(boxes, scores, np.zeros(boxes.shape[0]))

        result = HEADS.build("yolo26", conf_threshold=0.05, max_detections=3).decode(
            [output], [geometry()], [tag]
        )[0]

        assert len(result) == 3
        assert result.scores.tolist() == pytest.approx(sorted(scores.tolist())[-3:][::-1])


class TestOutputOrdering:
    """Descending score, ties by ascending proposal index — a function of the detections alone.

    The reference emits class-by-class in ``std::set<int>`` order, so adding one low-score
    object of a new class reorders every detection before it.
    """

    def test_results_are_sorted_by_descending_score_across_classes(self, tag) -> None:
        boxes = np.array(
            [[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0], [100.0, 100.0, 110.0, 110.0]],
            dtype=np.float32,
        )
        output = detection_output(boxes, [0.4, 0.95, 0.7], [3.0, 1.0, 2.0])

        result = HEADS.build("yolo26", conf_threshold=0.1).decode(
            [output], [geometry()], [tag]
        )[0]

        assert result.scores.tolist() == pytest.approx([0.95, 0.7, 0.4])
        assert result.class_ids.tolist() == [1, 2, 3]

    def test_a_tie_breaks_towards_the_lower_proposal_index(self, tag) -> None:
        boxes = np.array([[200.0, 0.0, 210.0, 10.0], [0.0, 0.0, 10.0, 10.0]], dtype=np.float32)
        output = detection_output(boxes, [0.5, 0.5], [0.0, 0.0])

        result = HEADS.build("yolo26").decode([output], [geometry()], [tag])[0]

        assert result.boxes[0][0] > result.boxes[1][0]


class TestOutputLayoutValidation:
    """Refusals, not coercions: every one of these has a plausible-looking failure mode."""

    def test_a_single_image_output_is_promoted_to_a_batch_of_one(self, tag) -> None:
        plane = detection_output([[0.0, 0.0, 10.0, 10.0]], [0.9], [0.0])[0]

        result = HEADS.build("yolo26").decode([plane], [geometry()], [tag])

        assert len(result) == 1 and len(result[0]) == 1

    def test_a_narrow_output_is_refused(self, tag) -> None:
        with pytest.raises(DimensionMismatchError, match="D >= 6"):
            HEADS.build("yolo26").decode(
                [np.zeros((1, 4, 5), dtype=np.float32)], [geometry()], [tag]
            )

    def test_a_confidence_column_holding_pixels_is_refused(self, tag) -> None:
        """What a raw ``cxcywh`` head looks like when it is decoded as an end-to-end one: the
        fifth column is a width in pixels, and clipping it to 1.0 would produce a frame full of
        maximally-confident detections."""
        output = detection_output([[0.0, 0.0, 10.0, 10.0]], [64.0], [0.0])

        with pytest.raises(DimensionMismatchError, match="probability"):
            HEADS.build("yolo26").decode([output], [geometry()], [tag])

    def test_a_slightly_over_unit_confidence_is_clipped_rather_than_refused(self, tag) -> None:
        """An fp16 sigmoid really does return 1.0000001, and ``Detection`` refuses > 1."""
        output = detection_output([[0.0, 0.0, 10.0, 10.0]], [1.0000001], [0.0])

        result = HEADS.build("yolo26").decode([output], [geometry()], [tag])[0]

        assert result[0].score == 1.0

    def test_an_inside_out_box_is_widened_rather_than_dropped(self, tag) -> None:
        """``Detection`` refuses ``x2 < x1``; the reference clamps the extent to zero and keeps
        the detection, so matching it keeps the detection count comparable."""
        output = detection_output([[300.0, 300.0, 290.0, 280.0]], [0.9], [0.0])

        result = HEADS.build("yolo26").decode([output], [geometry()], [tag])[0]

        assert len(result) == 1
        assert result[0].area == 0.0

    def test_a_detection_head_refuses_a_segmentation_engines_two_outputs(self, tag) -> None:
        """Strict rather than "use the first and ignore the rest": boxes with no masks are
        indistinguishable from a frame where nothing was segmentable."""
        outputs = [np.zeros((1, 4, 38), dtype=np.float32), np.zeros((1, 32, 8, 8), np.float32)]

        with pytest.raises(DimensionMismatchError, match="segmentation engine has two"):
            HEADS.build("yolo26").decode(outputs, [geometry()], [tag])


class TestHeadConfiguration:
    """A bad threshold must stop the process at start-up, not on frame 40 000."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"conf_threshold": -0.1},
            {"conf_threshold": 1.5},
            {"iou_threshold": 2.0},
            {"nms_method": "softmax"},
            {"max_detections": 0},
            {"num_classes": 0},
        ],
    )
    def test_an_impossible_argument_is_refused_at_construction(self, kwargs) -> None:
        with pytest.raises(ConfigurationError):
            Yolo26Head(**kwargs)

    def test_the_head_reports_what_it_was_registered_as(self) -> None:
        head = HEADS.build("yolo26")

        assert (head.name, head.backend) == ("yolo26", "python")
        assert HEADS.get("end2end") is Yolo26Head


class TestHeadResolution:
    """Which head runs is read off the artefact, and a caller who disagrees is refused."""

    def test_one_output_resolves_to_the_detection_head(self) -> None:
        assert resolve_head([(1, 300, 6)]).name == "yolo26"

    def test_two_outputs_with_a_rank_four_prototype_resolve_to_segmentation(self) -> None:
        assert resolve_head([(1, 300, 38), (1, 32, 160, 160)]).name == "yolo26_seg"

    def test_the_prototype_may_come_first(self) -> None:
        """Binding order is whatever the exporter emitted; rank is not."""
        assert resolve_head([(1, 32, 160, 160), (1, 300, 38)]).name == "yolo26_seg"

    def test_a_named_head_that_the_artefact_cannot_feed_is_refused(self) -> None:
        with pytest.raises(ModelLoadError, match="decodes 1 output"):
            resolve_head([(1, 300, 38), (1, 32, 160, 160)], name="yolo26")

    def test_an_unrecognisable_output_arity_is_refused(self) -> None:
        with pytest.raises(ModelLoadError, match="cannot tell which head"):
            resolve_head([(1, 84, 8400), (1, 32, 160, 160), (1, 5)])

    def test_head_options_reach_the_constructor(self) -> None:
        head = resolve_head([(1, 300, 6)], conf_threshold=0.4, nms_method="classic")

        assert (head.conf_threshold, head.nms_method) == (0.4, "classic")


class TestSuppressionRoutesThroughTheBackend:
    """A head given an `image_ops` must use it, and must agree with the numpy path.

    Suppression used to always call `shipvision.imgproc.nms.suppress` directly, on the
    argument that `nms_with_scores` was the same implementation so a backend bought nothing.
    That was true when written. Once `nms_with_scores` learned to keep the CUDA bitmask kernel
    and `torchvision.ops.nms` for the hard methods, this head became the only remaining caller
    of the slow path in the library — measured at roughly 150x the device one over 25 000
    proposals. These tests keep the fast route wired and keep the two answers identical.
    """

    def _overlapping(self, count: int = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(11)
        origins = rng.uniform(0, 120, size=(count, 2)).astype(np.float32)
        boxes = np.concatenate([origins, origins + 60.0], axis=1).astype(np.float32)
        scores = rng.uniform(0.3, 0.99, size=count).astype(np.float32)
        classes = (rng.random(count) > 0.5).astype(np.int32)
        return boxes, scores, classes

    def test_the_backend_is_actually_called(self) -> None:
        calls: list[str] = []

        class Spy:
            def nms_with_scores(self, boxes, scores, **kwargs):
                calls.append(kwargs["method"])
                return suppress(boxes, scores, **kwargs)

        head = Yolo26Head(nms_method=CLASSIC, image_ops=Spy())
        boxes, scores, classes = self._overlapping()

        head._suppress(np.arange(boxes.shape[0]), boxes, scores, classes)

        assert calls, "the head kept the numpy path despite being given a backend"
        assert set(calls) == {CLASSIC}

    def test_no_backend_still_works(self) -> None:
        """A head must be constructible with nothing installed — that is what keeps the
        offline tier and the numpy oracle honest."""
        head = Yolo26Head(nms_method=CLASSIC)

        assert head.image_ops is None
        boxes, scores, classes = self._overlapping()
        kept = head._suppress(np.arange(boxes.shape[0]), boxes, scores, classes)
        assert len(kept.boxes) > 0

    @pytest.mark.parametrize("method", sorted(METHODS))
    def test_every_backend_agrees_with_the_numpy_path(self, method: str) -> None:
        boxes, scores, classes = self._overlapping()
        rows = np.arange(boxes.shape[0])

        reference = Yolo26Head(nms_method=method)._suppress(rows, boxes, scores, classes)

        # Every backend this machine can actually build. Asking the registry rather than
        # hard-coding a list means a machine with no torch and no compiled extension runs the
        # numpy-vs-numpy case and still asserts something, instead of skipping.
        for backend in IMGPROC.backends("default"):
            try:
                ops = IMGPROC.build("default", backend=backend)
            except BackendUnavailableError:
                continue
            through = Yolo26Head(nms_method=method, image_ops=ops)._suppress(
                rows, boxes, scores, classes
            )
            assert np.array_equal(
                through.boxes, reference.boxes
            ), f"{backend} disagreed with numpy on {method}"
            assert np.allclose(through.scores, reference.scores, atol=1e-5)
