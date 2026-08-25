"""The contract. Every stage and both repos speak these types, so their edges are pinned
here rather than rediscovered per algorithm."""

from __future__ import annotations

import numpy as np
import pytest

from shipvision import (
    ConfigurationError,
    Detection,
    Detections,
    Embedding,
    FrameTag,
    GlobalTrack,
    Track,
    TrackState,
    cxcyah_to_xyxy,
    cxcywh_to_xyxy,
    iou_matrix,
    xyxy_to_cxcyah,
    xyxy_to_cxcywh,
)

TAG = FrameTag(camera_id="cam-01", frame_id=7, timestamp=1_700_000_000.5)

# --------------------------------------------------------------------------- FrameTag

# -------------------------------------------------------------------------- Detection

# ------------------------------------------------------------------------- Detections

# -------------------------------------------------------------------------- Embedding

# ------------------------------------------------------------------------------ Track

# ------------------------------------------------------------------------ GlobalTrack

# ------------------------------------------------------------------------ conversions

# --------------------------------------------------------------------------- iou


class TestFrameTag:
    """Where and when a frame came from. It travels with everything, so it cannot change."""

    def test_a_tag_is_immutable_and_prints_readably(self) -> None:
        """It travels through every stage; a stage that could mutate it could silently
        re-attribute a result to another camera."""
        assert str(TAG) == "cam-01#7"
        with pytest.raises(AttributeError):
            TAG.camera_id = "cam-02"  # type: ignore[misc]

    def test_a_negative_frame_id_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="non-negative"):
            FrameTag(camera_id="cam-01", frame_id=-1)


class TestDetection:
    """One detected object, and the box conventions it enforces at construction."""

    def test_a_detection_normalises_its_box_to_float32_xyxy(self) -> None:
        d = Detection(box=[10, 20, 110, 220])

        assert d.box.dtype == np.float32
        assert d.box.shape == (4,)
        assert (d.width, d.height, d.area) == (100.0, 200.0, 20_000.0)
        assert d.centre == (60.0, 120.0)

    def test_an_inside_out_box_is_refused_and_says_why(self) -> None:
        """The real bug it catches: a converter that wrote xywh into an xyxy field produces
        exactly this — (x, y, w, h) = (10, 20, 5, 5) reads as x2 < x1 — and every downstream
        IoU silently becomes zero."""
        with pytest.raises(ConfigurationError, match="xywh"):
            Detection(box=[10, 20, 5, 5])

    def test_a_box_of_the_wrong_length_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="4 values"):
            Detection(box=[1, 2, 3])

    def test_a_score_outside_zero_one_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match=r"\[0, 1\]"):
            Detection(box=[0, 0, 1, 1], score=1.5)

    def test_a_zero_area_box_is_allowed(self) -> None:
        """A detector legitimately emits these at a frame edge; refusing them would make the
        library reject valid input."""
        d = Detection(box=[10, 10, 10, 10])

        assert d.area == 0.0


class TestDetections:
    """A frame's worth of detections, and the batched views the algorithms consume."""

    def test_an_empty_frame_gives_correctly_shaped_empty_arrays(self) -> None:
        """`(0,)` instead of `(0, 4)` breaks every downstream `[:, 2]` with an IndexError
        instead of yielding an empty result — and an empty frame is normal input, not an edge
        case."""
        empty = Detections(tag=TAG)

        assert len(empty) == 0
        assert empty.boxes.shape == (0, 4)
        assert empty.boxes.dtype == np.float32
        assert empty.scores.shape == (0,)
        assert empty.class_ids.shape == (0,)
        assert empty.embeddings is None

    def test_the_batched_views_line_up_with_the_items(self) -> None:
        dets = Detections(
            tag=TAG,
            items=[
                Detection(box=[0, 0, 10, 10], score=0.9, class_id=1),
                Detection(box=[5, 5, 20, 20], score=0.4, class_id=2),
            ],
        )

        assert dets.boxes.shape == (2, 4)
        assert dets.scores.tolist() == pytest.approx([0.9, 0.4])
        assert dets.class_ids.tolist() == [1, 2]
        assert dets[0].score == pytest.approx(0.9)

    def test_embeddings_are_all_or_nothing(self) -> None:
        """A half-embedded batch becomes a cost matrix where some rows are appearance-based and
        some are not, which is not a matrix anyone can reason about. The caller must decide, so
        it is told rather than handed a guess."""
        dets = Detections(
            tag=TAG,
            items=[
                Detection(box=[0, 0, 1, 1], embedding=np.ones(8, np.float32)),
                Detection(box=[1, 1, 2, 2]),
            ],
        )

        assert dets.embeddings is None

        dets.items[1].embedding = np.zeros(8, np.float32)
        assert dets.embeddings is not None
        assert dets.embeddings.shape == (2, 8)

    def test_filtering_keeps_the_tag(self) -> None:
        """Losing it here is how a result ends up under the wrong camera's name."""
        dets = Detections(
            tag=TAG,
            items=[
                Detection(box=[0, 0, 1, 1], score=0.9, class_id=1),
                Detection(box=[1, 1, 2, 2], score=0.2, class_id=1),
                Detection(box=[2, 2, 3, 3], score=0.8, class_id=2),
            ],
            height=1080,
            width=1920,
        )

        kept = dets.filter(min_score=0.5, class_ids=[1])

        assert len(kept) == 1
        assert kept.tag is dets.tag
        assert (kept.height, kept.width) == (1080, 1920)
        assert len(dets) == 3, "filter returns a new object rather than mutating"


class TestNonFiniteInput:
    """A NaN or inf must be refused at the boundary, never carried.

    This is not defensive tidiness. The adversarial review of the re-identification package
    measured what one poisoned row does: a single NaN gallery vector out of sixty made every
    entry of a re-ranked score matrix NaN, and the evaluation then reported **mAP 0.2236
    against the true 0.1196** — better than reality — because `argsort` on an all-NaN row
    falls back to array order. A failure that flatters the measurement is the worst kind to
    let through, so it is stopped where the value enters.
    """

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_a_non_finite_box_is_refused(self, bad: float) -> None:
        with pytest.raises(ConfigurationError, match="non-finite"):
            Detection(box=[0.0, 0.0, bad, 1.0])

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_a_non_finite_detection_embedding_is_refused(self, bad: float) -> None:
        with pytest.raises(ConfigurationError, match="non-finite"):
            Detection(box=[0, 0, 1, 1], embedding=np.array([1.0, bad, 3.0], np.float32))

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_a_non_finite_embedding_is_refused(self, bad: float) -> None:
        with pytest.raises(ConfigurationError, match="non-finite"):
            Embedding(vector=np.array([1.0, 2.0, bad], np.float32))

    def test_the_error_says_how_many_and_where(self) -> None:
        """One bad crop in tens of thousands is the usual cause, so knowing it was one value
        of 512 rather than the whole batch is the difference between a bug hunt and a
        dropped frame."""
        vector = np.ones(512, np.float32)
        vector[7] = np.nan
        vector[300] = np.inf

        with pytest.raises(ConfigurationError, match=r"2 non-finite value\(s\).*index 7"):
            Embedding(vector=vector)

    def test_a_non_finite_track_box_is_refused(self) -> None:
        with pytest.raises(ConfigurationError):
            Track(track_id=1, box=[0.0, 0.0, np.nan, 1.0], tag=TAG)

    def test_a_finite_box_is_untouched(self) -> None:
        """The guard must not cost the ordinary path anything or change a value.

        The box only — this used to assert the same of the embedding, which is what pinned the
        bug: the module's stated contract is that embeddings are stored L2-normalised, and
        `[0, 1, 2, 3]` coming back unchanged is that contract being false. Direction is what
        must survive normalisation, and `TestEmbedding` asserts it does.
        """
        detection = Detection(
            box=[1.5, 2.5, 3.5, 4.5], embedding=np.arange(4, dtype=np.float32)
        )

        assert detection.box.tolist() == [1.5, 2.5, 3.5, 4.5]


class TestEmbedding:
    """One appearance vector plus the context needed to judge it."""

    def test_an_embedding_flattens_and_records_its_width(self) -> None:
        e = Embedding(
            vector=np.ones((1, 512), np.float64), identity="ship-3", camera_id="cam-01"
        )

        assert e.vector.shape == (512,)
        assert e.vector.dtype == np.float32
        assert e.dim == 512

    def test_an_empty_or_badly_scored_embedding_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="cannot be empty"):
            Embedding(vector=np.zeros(0, np.float32))
        with pytest.raises(ConfigurationError, match=r"\[0, 1\]"):
            Embedding(vector=np.ones(4, np.float32), quality=2.0)


class TestTrack:
    """One identity within one camera, and the question of when it may be published."""

    def test_a_track_is_publishable_only_when_confirmed_and_current(self) -> None:
        """A LOST track's box is a Kalman prediction no detector saw. Emitting it as an
        observation is how a phantom object drifts across a scene."""
        confirmed = Track(track_id=1, box=[0, 0, 10, 10], tag=TAG)
        assert confirmed.is_publishable

        stale = Track(track_id=1, box=[0, 0, 10, 10], tag=TAG, time_since_update=1)
        assert not stale.is_publishable

        tentative = Track(track_id=1, box=[0, 0, 10, 10], tag=TAG, state=TrackState.TENTATIVE)
        assert not tentative.is_publishable

        lost = Track(track_id=1, box=[0, 0, 10, 10], tag=TAG, state=TrackState.LOST)
        assert not lost.is_publishable

    def test_an_unknown_state_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="unknown state"):
            Track(track_id=1, box=[0, 0, 1, 1], tag=TAG, state="zombie")

    def test_track_states_survive_serialisation_as_plain_strings(self) -> None:
        """They cross a process boundary on the way to MTMC. An enum that serialises as
        "TrackState.CONFIRMED" on one side and must be parsed on the other is a bug waiting for
        a version skew."""
        import json

        assert json.loads(json.dumps(TrackState.CONFIRMED)) == TrackState.CONFIRMED
        assert TrackState.CONFIRMED == "confirmed"

    def test_a_track_exposes_its_camera_without_the_caller_unpacking_the_tag(self) -> None:
        assert Track(track_id=1, box=[0, 0, 1, 1], tag=TAG).camera_id == "cam-01"


class TestGlobalTrack:
    """One identity across cameras, and why an unassigned id is None rather than -1."""

    def test_an_unassigned_global_track_is_none_not_minus_one(self) -> None:
        """-1 is the references' convention and it leaks: it compares, sorts and serialises as
        a perfectly ordinary id, so an unassigned track flows downstream looking assigned."""
        track = Track(track_id=5, box=[0, 0, 1, 1], tag=TAG)

        unassigned = GlobalTrack(global_id=None, track=track)
        assert not unassigned.is_assigned
        assert unassigned.global_id is None
        assert not isinstance(unassigned.global_id, int), "the sentinel must not be a number"

        assigned = GlobalTrack(
            global_id=17, track=track, members=(("cam-01", 5), ("cam-02", 9))
        )
        assert assigned.is_assigned
        assert len(assigned.members) == 2


class TestBoxConversions:
    """Round-trips between xyxy and the two centre forms. A transposition here is silent."""

    @pytest.mark.parametrize(
        "boxes",
        [
            [[10.0, 20.0, 110.0, 220.0]],
            [[0.0, 0.0, 1.0, 1.0], [100.0, 200.0, 300.0, 500.0]],
        ],
    )
    def test_cxcyah_round_trips(self, boxes: list[list[float]]) -> None:
        array = np.array(boxes, dtype=np.float32)

        assert np.allclose(cxcyah_to_xyxy(xyxy_to_cxcyah(array)), array, atol=1e-4)

    def test_cxcyah_is_aspect_and_height_not_width(self) -> None:
        """The transposition this pins down tracks square objects perfectly and falls apart on
        a ship, which is why it is worth a test of its own."""
        state = xyxy_to_cxcyah(np.array([[0.0, 0.0, 100.0, 200.0]], dtype=np.float32))[0]

        assert state[2] == pytest.approx(0.5), "aspect = width / height = 100 / 200"
        assert state[3] == pytest.approx(200.0), "the fourth component is HEIGHT"

    def test_a_zero_height_box_does_not_divide_by_zero(self) -> None:
        state = xyxy_to_cxcyah(np.array([[5.0, 5.0, 15.0, 5.0]], dtype=np.float32))

        assert np.all(np.isfinite(state))

    def test_cxcywh_round_trips(self) -> None:
        array = np.array([[10.0, 20.0, 110.0, 220.0]], dtype=np.float32)

        assert np.allclose(cxcywh_to_xyxy(xyxy_to_cxcywh(array)), array, atol=1e-4)


class TestIouMatrix:
    """Pairwise overlap, including the degenerate cases that are normal input."""

    def test_iou_of_known_overlap(self) -> None:
        a = np.array([[0, 0, 10, 10]], np.float32)
        b = np.array([[5, 5, 15, 15]], np.float32)

        # intersection 5x5 = 25; union 100 + 100 - 25 = 175.
        assert float(iou_matrix(a, b)[0, 0]) == pytest.approx(25 / 175, abs=1e-6)

    def test_iou_of_identical_boxes_is_one_and_of_disjoint_is_zero(self) -> None:
        a = np.array([[0, 0, 10, 10]], np.float32)
        far = np.array([[100, 100, 110, 110]], np.float32)

        assert float(iou_matrix(a, a)[0, 0]) == pytest.approx(1.0)
        assert float(iou_matrix(a, far)[0, 0]) == 0.0

    def test_iou_touching_edges_is_zero_not_negative(self) -> None:
        """Without the clip, the negative overlap multiplies into a positive area and two boxes
        that merely touch appear to overlap."""
        a = np.array([[0, 0, 10, 10]], np.float32)
        b = np.array([[10, 0, 20, 10]], np.float32)

        assert float(iou_matrix(a, b)[0, 0]) == 0.0

    def test_iou_with_an_empty_side_has_the_right_shape(self) -> None:
        a = np.zeros((3, 4), np.float32)
        empty = np.zeros((0, 4), np.float32)

        assert iou_matrix(a, empty).shape == (3, 0)
        assert iou_matrix(empty, a).shape == (0, 3)

    def test_iou_matrix_shape_and_symmetry(self) -> None:
        rng = np.random.default_rng(3)
        a = np.sort(rng.random((5, 4)).astype(np.float32) * 100, axis=1)[:, [0, 1, 2, 3]]
        a = np.stack([a[:, 0], a[:, 1], a[:, 0] + 10, a[:, 1] + 10], axis=1).astype(np.float32)

        m = iou_matrix(a, a)

        assert m.shape == (5, 5)
        assert np.allclose(m, m.T)
        assert np.allclose(np.diag(m), 1.0)


class TestEmbeddingsAreStoredNormalised:
    """The contract in the module docstring, made true rather than merely written down.

    A gallery that believes it computes cosine similarity as a plain dot product — which is the
    entire stated reason for normalising here rather than inside every distance function — gets
    silently wrong numbers against un-normalised rows: two identical `np.ones(512)` vectors
    score 512, every `sim > 0.5` gate admits everything, and a quality-weighted aggregator ends
    up weighted by whichever crop had the largest activations rather than by `quality`.

    Asserted on all three carriers, because a rule that holds on one of them is worse than no
    rule: it makes the exception the thing nobody checks for.
    """

    RAW = np.array([3.0, 4.0], dtype=np.float32)

    def test_an_embedding_is_normalised_on_the_way_in(self) -> None:
        assert np.linalg.norm(Embedding(vector=self.RAW).vector) == pytest.approx(1.0)

    def test_a_detection_normalises_the_one_it_carries(self) -> None:
        detection = Detection(box=[0.0, 0.0, 1.0, 1.0], embedding=self.RAW)

        assert np.linalg.norm(detection.embedding) == pytest.approx(1.0)

    def test_a_track_normalises_the_one_it_carries(self) -> None:
        """The carrier that reaches furthest: a track is what MTMC clusters on, and it was
        checked nowhere at all before."""
        track = Track(track_id=1, tag=TAG, box=[0.0, 0.0, 1.0, 1.0], embedding=self.RAW)

        assert np.linalg.norm(track.embedding) == pytest.approx(1.0)

    def test_the_direction_survives(self) -> None:
        """Normalising must not be a different vector, only a shorter one."""
        stored = Embedding(vector=self.RAW).vector

        assert stored.tolist() == pytest.approx([0.6, 0.8])

    def test_an_already_normalised_vector_is_a_fixed_point(self) -> None:
        """Otherwise normalising twice — once in an extractor, once here — would drift."""
        once = Embedding(vector=self.RAW).vector
        twice = Embedding(vector=once).vector

        assert twice.tolist() == pytest.approx(once.tolist())

    def test_an_all_zero_vector_is_refused_rather_than_turned_into_nan(self) -> None:
        """It has no direction. Dividing gives NaN; leaving it gives a row at cosine 0 from
        everything, which is a plausible answer to every query and so the worse failure."""
        with pytest.raises(ConfigurationError, match="no direction"):
            Embedding(vector=np.zeros(4, dtype=np.float32))

    def test_a_zero_vector_on_a_detection_is_refused_too(self) -> None:
        with pytest.raises(ConfigurationError, match="no direction"):
            Detection(box=[0.0, 0.0, 1.0, 1.0], embedding=np.zeros(4, dtype=np.float32))
