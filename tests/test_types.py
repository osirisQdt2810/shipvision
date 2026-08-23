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


def test_a_tag_is_immutable_and_prints_readably() -> None:
    """It travels through every stage; a stage that could mutate it could silently
    re-attribute a result to another camera."""
    assert str(TAG) == "cam-01#7"
    with pytest.raises(AttributeError):
        TAG.camera_id = "cam-02"  # type: ignore[misc]


def test_a_negative_frame_id_is_refused() -> None:
    with pytest.raises(ConfigurationError, match="non-negative"):
        FrameTag(camera_id="cam-01", frame_id=-1)


# -------------------------------------------------------------------------- Detection


def test_a_detection_normalises_its_box_to_float32_xyxy() -> None:
    d = Detection(box=[10, 20, 110, 220])

    assert d.box.dtype == np.float32
    assert d.box.shape == (4,)
    assert (d.width, d.height, d.area) == (100.0, 200.0, 20_000.0)
    assert d.centre == (60.0, 120.0)


def test_an_inside_out_box_is_refused_and_says_why() -> None:
    """The real bug it catches: a converter that wrote xywh into an xyxy field produces
    exactly this — (x, y, w, h) = (10, 20, 5, 5) reads as x2 < x1 — and every downstream
    IoU silently becomes zero."""
    with pytest.raises(ConfigurationError, match="xywh"):
        Detection(box=[10, 20, 5, 5])


def test_a_box_of_the_wrong_length_is_refused() -> None:
    with pytest.raises(ConfigurationError, match="4 values"):
        Detection(box=[1, 2, 3])


def test_a_score_outside_zero_one_is_refused() -> None:
    with pytest.raises(ConfigurationError, match=r"\[0, 1\]"):
        Detection(box=[0, 0, 1, 1], score=1.5)


def test_a_zero_area_box_is_allowed() -> None:
    """A detector legitimately emits these at a frame edge; refusing them would make the
    library reject valid input."""
    d = Detection(box=[10, 10, 10, 10])

    assert d.area == 0.0


# ------------------------------------------------------------------------- Detections


def test_an_empty_frame_gives_correctly_shaped_empty_arrays() -> None:
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


def test_the_batched_views_line_up_with_the_items() -> None:
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


def test_embeddings_are_all_or_nothing() -> None:
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


def test_filtering_keeps_the_tag() -> None:
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


# -------------------------------------------------------------------------- Embedding


def test_an_embedding_flattens_and_records_its_width() -> None:
    e = Embedding(vector=np.ones((1, 512), np.float64), identity="ship-3", camera_id="cam-01")

    assert e.vector.shape == (512,)
    assert e.vector.dtype == np.float32
    assert e.dim == 512


def test_an_empty_or_badly_scored_embedding_is_refused() -> None:
    with pytest.raises(ConfigurationError, match="cannot be empty"):
        Embedding(vector=np.zeros(0, np.float32))
    with pytest.raises(ConfigurationError, match=r"\[0, 1\]"):
        Embedding(vector=np.ones(4, np.float32), quality=2.0)


# ------------------------------------------------------------------------------ Track


def test_a_track_is_publishable_only_when_confirmed_and_current() -> None:
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


def test_an_unknown_state_is_refused() -> None:
    with pytest.raises(ConfigurationError, match="unknown state"):
        Track(track_id=1, box=[0, 0, 1, 1], tag=TAG, state="zombie")


def test_track_states_survive_serialisation_as_plain_strings() -> None:
    """They cross a process boundary on the way to MTMC. An enum that serialises as
    "TrackState.CONFIRMED" on one side and must be parsed on the other is a bug waiting for
    a version skew."""
    import json

    assert json.loads(json.dumps(TrackState.CONFIRMED)) == TrackState.CONFIRMED
    assert TrackState.CONFIRMED == "confirmed"


def test_a_track_exposes_its_camera_without_the_caller_unpacking_the_tag() -> None:
    assert Track(track_id=1, box=[0, 0, 1, 1], tag=TAG).camera_id == "cam-01"


# ------------------------------------------------------------------------ GlobalTrack


def test_an_unassigned_global_track_is_none_not_minus_one() -> None:
    """-1 is the references' convention and it leaks: it compares, sorts and serialises as
    a perfectly ordinary id, so an unassigned track flows downstream looking assigned."""
    track = Track(track_id=5, box=[0, 0, 1, 1], tag=TAG)

    unassigned = GlobalTrack(global_id=None, track=track)
    assert not unassigned.is_assigned
    assert unassigned.global_id is None
    assert not isinstance(unassigned.global_id, int), "the sentinel must not be a number"

    assigned = GlobalTrack(global_id=17, track=track, members=(("cam-01", 5), ("cam-02", 9)))
    assert assigned.is_assigned
    assert len(assigned.members) == 2


# ------------------------------------------------------------------------ conversions


@pytest.mark.parametrize(
    "boxes",
    [
        [[10.0, 20.0, 110.0, 220.0]],
        [[0.0, 0.0, 1.0, 1.0], [100.0, 200.0, 300.0, 500.0]],
    ],
)
def test_cxcyah_round_trips(boxes: list[list[float]]) -> None:
    array = np.array(boxes, dtype=np.float32)

    assert np.allclose(cxcyah_to_xyxy(xyxy_to_cxcyah(array)), array, atol=1e-4)


def test_cxcyah_is_aspect_and_height_not_width() -> None:
    """The transposition this pins down tracks square objects perfectly and falls apart on
    a ship, which is why it is worth a test of its own."""
    state = xyxy_to_cxcyah(np.array([[0.0, 0.0, 100.0, 200.0]], dtype=np.float32))[0]

    assert state[2] == pytest.approx(0.5), "aspect = width / height = 100 / 200"
    assert state[3] == pytest.approx(200.0), "the fourth component is HEIGHT"


def test_a_zero_height_box_does_not_divide_by_zero() -> None:
    state = xyxy_to_cxcyah(np.array([[5.0, 5.0, 15.0, 5.0]], dtype=np.float32))

    assert np.all(np.isfinite(state))


def test_cxcywh_round_trips() -> None:
    array = np.array([[10.0, 20.0, 110.0, 220.0]], dtype=np.float32)

    assert np.allclose(cxcywh_to_xyxy(xyxy_to_cxcywh(array)), array, atol=1e-4)


# --------------------------------------------------------------------------- iou


def test_iou_of_known_overlap() -> None:
    a = np.array([[0, 0, 10, 10]], np.float32)
    b = np.array([[5, 5, 15, 15]], np.float32)

    # intersection 5x5 = 25; union 100 + 100 - 25 = 175.
    assert float(iou_matrix(a, b)[0, 0]) == pytest.approx(25 / 175, abs=1e-6)


def test_iou_of_identical_boxes_is_one_and_of_disjoint_is_zero() -> None:
    a = np.array([[0, 0, 10, 10]], np.float32)
    far = np.array([[100, 100, 110, 110]], np.float32)

    assert float(iou_matrix(a, a)[0, 0]) == pytest.approx(1.0)
    assert float(iou_matrix(a, far)[0, 0]) == 0.0


def test_iou_touching_edges_is_zero_not_negative() -> None:
    """Without the clip, the negative overlap multiplies into a positive area and two boxes
    that merely touch appear to overlap."""
    a = np.array([[0, 0, 10, 10]], np.float32)
    b = np.array([[10, 0, 20, 10]], np.float32)

    assert float(iou_matrix(a, b)[0, 0]) == 0.0


def test_iou_with_an_empty_side_has_the_right_shape() -> None:
    a = np.zeros((3, 4), np.float32)
    empty = np.zeros((0, 4), np.float32)

    assert iou_matrix(a, empty).shape == (3, 0)
    assert iou_matrix(empty, a).shape == (0, 3)


def test_iou_matrix_shape_and_symmetry() -> None:
    rng = np.random.default_rng(3)
    a = np.sort(rng.random((5, 4)).astype(np.float32) * 100, axis=1)[:, [0, 1, 2, 3]]
    a = np.stack([a[:, 0], a[:, 1], a[:, 0] + 10, a[:, 1] + 10], axis=1).astype(np.float32)

    m = iou_matrix(a, a)

    assert m.shape == (5, 5)
    assert np.allclose(m, m.T)
    assert np.allclose(np.diag(m), 1.0)
