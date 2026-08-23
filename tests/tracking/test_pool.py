"""The shared lifecycle, tested directly rather than through five trackers.

:class:`~shipvision.tracking.pool.TrackPool` is where a bug would be invisible: every tracker
would still run, still publish, and be wrong in the same way, so a scenario test on any one of
them would not narrow it down. The dense-array alignment invariant in particular — ``tracks[i]``
describes ``means[i]`` — has no observable symptom other than identities attaching to the wrong
boxes several frames later.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipvision.errors import ConfigurationError
from shipvision.tracking import TRACKERS, TrackPool
from shipvision.types import Detection, Detections, FrameTag, Track, TrackState

BOX_W, BOX_H = 220.0, 130.0
LANE_Y = 500.0


def _det(cx: float, score: float = 0.95) -> Detection:
    return Detection(
        box=np.array(
            [cx - BOX_W / 2, LANE_Y - BOX_H / 2, cx + BOX_W / 2, LANE_Y + BOX_H / 2],
            np.float32,
        ),
        score=score,
    )


def _tag(frame_id: int) -> FrameTag:
    return FrameTag("approach-channel", frame_id)


def _walk(pool: TrackPool, positions: list[float], *, start: int = 0) -> int:
    """Observe one object at each position on consecutive frames. Returns the next frame id."""
    for offset, cx in enumerate(positions):
        frame_id = start + offset
        pool.predict(_tag(frame_id))
        if frame_id == start and len(pool) == 0:
            pool.spawn([_det(cx)], [0])
        else:
            pool.apply_matches([(0, 0)], [_det(cx)])
    return start + len(positions)


# ---------------------------------------------------------------- the alignment invariant


class TestArrayAlignment:
    """``tracks[i]`` must always describe ``means[i]``. Breaking it has no symptom of its own —
    only identities attaching to the wrong boxes a second later.
    """

    def test_sweeping_the_middle_track_keeps_every_array_aligned(self) -> None:
        """Three tracks, the middle one dies. The survivors must keep their own filter states.

        This is the failure with no symptom of its own: if ``sweep`` rebuilt the track list but
        not the covariance array, tracks 1 and 3 would carry each other's motion from here on and
        the only evidence would be identities drifting onto the wrong objects a second later.
        """
        pool = TrackPool(max_age=1, min_hits=1)
        pool.predict(_tag(0))
        pool.spawn([_det(200), _det(900), _det(1600)], [0, 1, 2])
        ids = [t.track_id for t in pool.tracks]

        # Two misses: with max_age=1 the middle track goes LOST on the first and REMOVED on the
        # second, which is the transition `sweep` acts on.
        for frame_id, (left, right) in ((1, (210.0, 1610.0)), (2, (220.0, 1620.0))):
            pool.predict(_tag(frame_id))
            pool.apply_matches([(0, 0), (2, 1)], [_det(left), _det(right)])
            pool.mark_missed([1])
        pool.sweep()

        assert [t.track_id for t in pool.tracks] == [ids[0], ids[2]]
        assert pool.means.shape == (2, 8)
        assert pool.covariances.shape == (2, 8, 8)
        # Row 0 must still be the one near x=220 and row 1 the one near x=1620, in the predicted
        # boxes *and* in the last-observation array that observation-centric recovery reads.
        np.testing.assert_allclose(pool.means[:, 0], [220.0, 1620.0], atol=5.0)
        np.testing.assert_allclose(pool.observed_boxes()[:, 0], [110.0, 1510.0], atol=1.0)

    def test_a_track_may_not_match_two_detections_in_one_frame(self) -> None:
        """A multi-stage tracker that lets one track match twice writes its filter update twice
        and silently keeps the second. Refusing it is how a staging bug fails loudly."""
        pool = TrackPool(max_age=5, min_hits=1)
        pool.predict(_tag(0))
        pool.spawn([_det(200)], [0])
        pool.predict(_tag(1))
        with pytest.raises(ConfigurationError, match="at most one detection"):
            pool.apply_matches([(0, 0), (0, 1)], [_det(205), _det(210)])

    def test_spawn_before_predict_is_refused(self) -> None:
        """A track spawned without a frame open would have no tag, and an untagged track is the
        one failure mode worse than a dropped frame."""
        pool = TrackPool(max_age=5, min_hits=1)
        with pytest.raises(ConfigurationError, match="predict"):
            pool.spawn([_det(200)], [0])

    def test_the_tag_of_the_open_frame_is_stamped_on_every_live_track(self) -> None:
        pool = TrackPool(max_age=5, min_hits=1)
        _walk(pool, [200.0, 220.0, 240.0])
        pool.predict(_tag(9))
        assert all(t.tag == _tag(9) for t in pool.tracks)


class TestLifecycle:
    """Promotion, ageing and death, in one place so five trackers cannot disagree about them."""

    def test_a_tentative_track_dies_on_its_first_miss(self) -> None:
        """An unconfirmed track that misses even once was probably a false positive. Keeping it
        alive costs an identity slot and invites a wrong association."""
        pool = TrackPool(max_age=30, min_hits=3)
        pool.predict(_tag(0))
        pool.spawn([_det(200)], [0])
        assert pool.tracks[0].state == TrackState.TENTATIVE

        pool.predict(_tag(1))
        pool.mark_missed([0])
        pool.sweep()
        assert len(pool) == 0

    def test_a_confirmed_track_goes_lost_before_it_is_removed(self) -> None:
        pool = TrackPool(max_age=2, min_hits=1)
        _walk(pool, [200.0, 220.0])
        assert pool.tracks[0].state == TrackState.CONFIRMED

        for frame_id in (2, 3):
            pool.predict(_tag(frame_id))
            pool.mark_missed([0])
            pool.sweep()
            assert pool.tracks[0].state == TrackState.LOST

        pool.predict(_tag(4))
        pool.mark_missed([0])
        pool.sweep()
        assert len(pool) == 0

    def test_a_min_hits_of_one_publishes_immediately(self) -> None:
        """A caller who asked for immediate publication and got silence would reasonably call
        that a bug rather than a policy, so promotion is checked on spawn as well as on match.
        """
        pool = TrackPool(max_age=5, min_hits=1)
        pool.predict(_tag(0))
        pool.spawn([_det(200)], [0])
        assert len(pool.output()) == 1


class TestObservationCentricReUpdate:
    """OC-SORT's ORU, stated as an equality rather than as an improvement."""

    def test_re_update_leaves_the_filter_where_an_unbroken_observation_would_have(self) -> None:
        """OC-SORT's re-update, stated as an equality rather than as an improvement.

        Three pools see the same object. It travels at 30 px/frame for ten frames, then:

        * pool **A** loses it for five frames and finds it again 60 px on, with ``re_update=True``;
        * pool **B** the same, with ``re_update=False`` — one distant measurement into a filter
          whose covariance has been inflating for the whole gap;
        * pool **C** never loses it: the detector actually reports the five boxes on the straight
          line between the two real observations.

        ORU's claim is that A ends up where C is, and that is asserted exactly — mean and
        covariance, to within float32 noise. B does not, and its position variance is 41% higher,
        because it absorbed one measurement where the others absorbed six.

        Worth recording, because it decides how much this mechanism is worth on this codebase:
        the *magnitude* of the A-versus-B difference is small — 1.3 px of position and 0.17
        px/frame of velocity here. That is a property of this library's Kalman filter, whose
        velocity process noise is height/160 and which therefore holds its velocity estimate
        tightly; the measured gain on velocity from a position innovation is about 0.09, so a
        single post-gap measurement already corrects most of what it is going to correct. ORU's
        value scales with how loosely the filter constrains velocity, and the honest test of it is
        this equality rather than a scenario tuned until the two land on opposite sides of an IoU
        threshold.
        """
        lead = [200.0 + 30.0 * i for i in range(10)]
        last_seen = lead[-1]
        gap = 5
        found_at = last_seen + 60.0
        slope = (found_at - last_seen) / (gap + 1)

        def gapped(*, re_update: bool) -> TrackPool:
            pool = TrackPool(max_age=40, min_hits=1, re_update=re_update)
            frame_id = _walk(pool, lead)
            for _ in range(gap):
                pool.predict(_tag(frame_id))
                pool.mark_missed([0])
                frame_id += 1
            pool.predict(_tag(frame_id))
            pool.apply_matches([(0, 0)], [_det(found_at)])
            return pool

        def uninterrupted() -> TrackPool:
            pool = TrackPool(max_age=40, min_hits=1)
            frame_id = _walk(pool, lead)
            for step in range(1, gap + 2):
                pool.predict(_tag(frame_id))
                pool.apply_matches([(0, 0)], [_det(last_seen + slope * step)])
                frame_id += 1
            return pool

        re_updated = gapped(re_update=True)
        plain = gapped(re_update=False)
        observed = uninterrupted()

        np.testing.assert_allclose(re_updated.means, observed.means, rtol=1e-5, atol=1e-3)
        np.testing.assert_allclose(
            re_updated.covariances, observed.covariances, rtol=1e-5, atol=1e-3
        )

        # And it is genuinely a different answer from the plain update, in the right direction:
        # six absorbed measurements leave less uncertainty than one.
        assert not np.allclose(plain.means, observed.means, atol=1e-3)
        assert re_updated.covariances[0, 0, 0] < plain.covariances[0, 0, 0]
        assert plain.covariances[0, 0, 0] / re_updated.covariances[0, 0, 0] == pytest.approx(
            1.41, abs=0.05
        )

    def test_re_update_is_a_no_op_for_a_track_that_never_missed_a_frame(self) -> None:
        """Otherwise every frame would pay for a mechanism that only applies after a gap."""
        with_oru = TrackPool(max_age=40, min_hits=1, re_update=True)
        without = TrackPool(max_age=40, min_hits=1, re_update=False)
        positions = [200.0 + 30.0 * i for i in range(12)]
        _walk(with_oru, positions)
        _walk(without, positions)
        np.testing.assert_allclose(with_oru.means, without.means, rtol=0, atol=0)


class TestObservationHistory:
    """The bounded ring the momentum term reads, and what it means for it to be empty."""

    def test_a_heading_needs_two_observations_and_is_unknown_before_that(self) -> None:
        """``(0, 0)`` means "no information", and every consumer must treat it that way rather
        than as "not moving" — a brand-new track is not stationary, it is unmeasured."""
        pool = TrackPool(max_age=40, min_hits=1, observation_history=4)
        pool.predict(_tag(0))
        pool.spawn([_det(200)], [0])
        np.testing.assert_allclose(pool.directions(3), [[0.0, 0.0]])

        _walk(pool, [230.0, 260.0, 290.0, 320.0], start=1)
        np.testing.assert_allclose(pool.directions(3), [[1.0, 0.0]], atol=1e-5)

    def test_the_observation_history_is_bounded(self) -> None:
        """A process here runs for weeks. An unbounded per-track ring is a slow leak with no
        symptom until a camera has been up for a month."""
        pool = TrackPool(max_age=200, min_hits=1, observation_history=4)
        _walk(pool, [200.0 + 10.0 * i for i in range(300)])
        assert len(pool._observations[0]) == 4

    def test_without_a_history_there_is_no_heading_to_read(self) -> None:
        """A tracker that does not ask for observations must not silently get a one-frame
        heading, which is mostly detector jitter."""
        pool = TrackPool(max_age=40, min_hits=1)
        _walk(pool, [200.0 + 30.0 * i for i in range(6)])
        np.testing.assert_allclose(pool.directions(3), [[0.0, 0.0]])


class TestCameraCompensation:
    """Warping every predicted state into this frame's coordinates, which is why the state is
    parameterised as ``(cx, cy, aspect, height)``.
    """

    def test_a_camera_pan_moves_every_prediction_by_the_pan(self) -> None:
        pool = TrackPool(max_age=40, min_hits=1)
        _walk(pool, [500.0, 500.0, 500.0])
        pool.predict(_tag(3))
        before = pool.boxes().copy()

        pool.apply_camera_motion(np.array([[1.0, 0.0, -45.0], [0.0, 1.0, 12.0]], np.float32))
        after = pool.boxes()
        np.testing.assert_allclose(after[:, [0, 2]], before[:, [0, 2]] - 45.0, atol=1e-3)
        np.testing.assert_allclose(after[:, [1, 3]], before[:, [1, 3]] + 12.0, atol=1e-3)

    def test_a_camera_zoom_scales_the_height_and_leaves_the_aspect_alone(self) -> None:
        """Why the filter state is ``(cx, cy, aspect, height)``: under a similarity transform the
        aspect ratio is invariant, so a zoom is one multiplication rather than a re-derivation.
        """
        pool = TrackPool(max_age=40, min_hits=1)
        _walk(pool, [500.0, 500.0, 500.0])
        pool.predict(_tag(3))
        before = pool.boxes()[0]
        aspect_before = (before[2] - before[0]) / (before[3] - before[1])

        pool.apply_camera_motion(np.array([[1.5, 0.0, 0.0], [0.0, 1.5, 0.0]], np.float32))
        after = pool.boxes()[0]
        aspect_after = (after[2] - after[0]) / (after[3] - after[1])

        assert aspect_after == pytest.approx(aspect_before, rel=1e-4)
        assert (after[3] - after[1]) == pytest.approx(1.5 * (before[3] - before[1]), rel=1e-4)

    def test_it_also_warps_what_the_pool_remembers_seeing(self) -> None:
        """The last-observation array and the observation ring are image coordinates too.

        Leaving them in the previous frame's system would give the observation-centric stages
        a stale frame of reference: recovery would score this frame's detection against a box
        measured before the camera moved, and a heading measured across a pan would be mostly
        the pan. Unreachable today, since only BoT-SORT compensates and it has no recovery
        stage — which is why it is asserted here rather than left for whoever combines them.
        """
        pool = TrackPool(max_age=40, min_hits=1, observation_history=4)
        _walk(pool, [500.0, 520.0, 540.0, 560.0])
        observed_before = pool.observed_boxes().copy()
        heading_before = pool.directions(3).copy()

        pool.predict(_tag(4))
        pool.apply_camera_motion(np.array([[1.0, 0.0, -45.0], [0.0, 1.0, 0.0]], np.float32))

        observed_after = pool.observed_boxes()
        np.testing.assert_allclose(
            observed_after[:, [0, 2]], observed_before[:, [0, 2]] - 45.0, atol=1e-3
        )
        # The heading is a difference of two warped observations, so a pure translation
        # cancels out of it entirely — the object's direction of travel did not change
        # because the camera moved.
        np.testing.assert_allclose(pool.directions(3), heading_before, atol=1e-5)

    def test_a_misshaped_affine_is_refused(self) -> None:
        pool = TrackPool(max_age=40, min_hits=1)
        with pytest.raises(ConfigurationError, match=r"\(2, 3\)"):
            pool.apply_camera_motion(np.eye(3, dtype=np.float32))


class TestAppearanceBlending:
    """A track's appearance follows the object slowly, at a rate the caller may set per
    detection.
    """

    def test_appearance_is_blended_not_replaced(self) -> None:
        """One badly-cropped frame must not become the reference every future match is scored
        against, or the identity walks away from itself."""
        pool = TrackPool(max_age=40, min_hits=1, embedding_momentum=0.9)
        first = np.array([1.0, 0.0, 0.0], np.float32)
        second = np.array([0.0, 1.0, 0.0], np.float32)

        pool.predict(_tag(0))
        pool.spawn([Detection(box=_det(200).box, score=0.9, embedding=first)], [0])
        np.testing.assert_allclose(pool.tracks[0].embedding, first)

        pool.predict(_tag(1))
        pool.apply_matches(
            [(0, 0)], [Detection(box=_det(205).box, score=0.9, embedding=second)]
        )
        blended = pool.tracks[0].embedding
        assert float(np.linalg.norm(blended)) == pytest.approx(1.0, abs=1e-5)
        assert (
            blended[0] > blended[1] > 0.0
        ), "the track should have moved a little, not swapped"

    def test_a_per_detection_momentum_overrides_the_default(self) -> None:
        """DeepSORTv2 derives the rate from confidence and crowding, so a clean isolated crop
        moves a track's appearance further than a half-occluded one does."""
        first = np.array([1.0, 0.0], np.float32)
        second = np.array([0.0, 1.0], np.float32)
        moved = []
        for momentum in (0.99, 0.5):
            pool = TrackPool(max_age=40, min_hits=1, embedding_momentum=0.9)
            pool.predict(_tag(0))
            pool.spawn([Detection(box=_det(200).box, score=0.9, embedding=first)], [0])
            pool.predict(_tag(1))
            pool.apply_matches(
                [(0, 0)],
                [Detection(box=_det(205).box, score=0.9, embedding=second)],
                embedding_momentum=np.array([momentum], np.float32),
            )
            moved.append(float(pool.tracks[0].embedding[1]))
        assert moved[1] > moved[0], "a lower retention must move the track's appearance further"

    def test_embeddings_are_all_or_nothing_across_the_pool(self) -> None:
        """A cost matrix whose rows are half appearance-based and half not is not a matrix anyone
        can threshold, so the pool refuses to hand out a partial one."""
        pool = TrackPool(max_age=40, min_hits=1)
        pool.predict(_tag(0))
        pool.spawn(
            [
                Detection(
                    box=_det(200).box, score=0.9, embedding=np.array([1.0, 0.0], np.float32)
                ),
                Detection(box=_det(900).box, score=0.9),
            ],
            [0, 1],
        )
        assert pool.embeddings() is None


# -------------------------------------------------------------------------- the lifecycle


# ------------------------------------------------------------------------- ORU, exactly


# --------------------------------------------------------------- observations and headings


# --------------------------------------------------------------------- camera compensation


# ------------------------------------------------------------------------------ appearance


class TestOutputIsASnapshot:
    """A returned track must not change under its reader.

    The pool mutates its `Track` objects in place every frame. While `output()` returned
    references, a caller that consumed each frame immediately saw nothing wrong, and a caller
    that buffered a run and read the ids afterwards got the *final* frame's state on every
    entry. The evaluation harness hit it: one identity appeared 26 times in a single frame of
    MOT17-09, which reads as a tracker defect and was a lifetime-of-the-object defect.
    """

    def test_a_buffered_run_keeps_each_frame_as_it_was(self) -> None:
        tracker = TRACKERS.build("sort", min_hits=1, max_age=5)
        buffered: list[list[Track]] = []
        for frame in range(8):
            detections = Detections(
                tag=FrameTag(camera_id="cam-a", frame_id=frame),
                items=[Detection(box=[10 + frame * 20, 30, 60 + frame * 20, 130], score=0.9)],
            )
            buffered.append(list(tracker.update(detections)))

        published = [frames for frames in buffered if frames]
        assert len(published) >= 4, "the scenario must actually publish something"

        boxes = [frames[0].box[0] for frames in published]
        assert len(set(boxes)) == len(boxes), (
            f"every buffered frame reports x1={boxes[0]} — the pool's live objects were "
            f"handed out instead of copies, so the whole run collapsed onto the last frame"
        )
        frame_ids = [frames[0].tag.frame_id for frames in published]
        assert frame_ids == sorted(frame_ids)
        assert len(set(frame_ids)) == len(frame_ids)

    def test_mutating_a_returned_track_cannot_corrupt_the_pool(self) -> None:
        """The other direction: a consumer that edits what it was given must not reach back
        into the tracker's state."""
        tracker = TRACKERS.build("sort", min_hits=1, max_age=5)
        first = tracker.update(
            Detections(
                tag=FrameTag(camera_id="cam-a", frame_id=0),
                items=[Detection(box=[10, 30, 60, 130], score=0.9)],
            )
        )
        assert first, "the scenario must publish on the first frame"
        stolen_id = first[0].track_id
        first[0].track_id = -999
        first[0].box[:] = 0.0

        second = tracker.update(
            Detections(
                tag=FrameTag(camera_id="cam-a", frame_id=1),
                items=[Detection(box=[12, 30, 62, 130], score=0.9)],
            )
        )

        assert second[0].track_id == stolen_id
        assert second[0].box[0] > 0.0
