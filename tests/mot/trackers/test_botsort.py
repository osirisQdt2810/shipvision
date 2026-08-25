"""BoT-SORT's two changes to ByteTrack, each measured against ByteTrack failing.

Because BoT-SORT *is* ByteTrack plus two things, every test here has a third arm: the same
BoT-SORT with the thing under test switched off. That is what separates "BoT-SORT is better"
(a claim nobody can act on) from "camera-motion compensation is what saved this" (a claim that
tells an operator whether to turn it on).
"""

from __future__ import annotations

import numpy as np
import pytest

from shipvision.errors import TrackingError
from shipvision.mot import TRACKERS
from shipvision.reid import EXTRACTORS
from shipvision.types import Detection, Detections, FrameTag

CAMERA = "ptz-north"
PERSON_W, PERSON_H = 60.0, 140.0


def _person(cx: float, cy: float, embedding: np.ndarray | None = None) -> Detection:
    return Detection(
        box=np.array(
            [
                cx - PERSON_W / 2,
                cy - PERSON_H / 2,
                cx + PERSON_W / 2,
                cy + PERSON_H / 2,
            ],
            np.float32,
        ),
        score=0.95,
        embedding=embedding,
    )


# ============================================================ camera-motion compensation

# Three people standing still on a quay. The camera is on a PTZ head and, ten frames in,
# starts panning at 45 px/frame — a third of a person's width every frame, which after two
# frames is more overlap than ByteTrack's 0.2 IoU threshold allows.
WORLD = ((400.0, 500.0), (900.0, 620.0), (1400.0, 400.0))
STILL_FRAMES, PANNING_FRAMES, PAN = 10, 15, 45.0


def _pan_scenario(
    name: str,
    *,
    feed_telemetry: bool = False,
    images: list[np.ndarray] | None = None,
    **options: object,
) -> tuple[set[int], set[int]]:
    """Run the pan and return (identities before it started, identities on the last frame)."""
    tracker = TRACKERS.build(name, min_hits=2, max_age=30, **options)
    published: list[set[int]] = []
    for frame_id in range(STILL_FRAMES + PANNING_FRAMES):
        shift = 0.0 if frame_id < STILL_FRAMES else -PAN * (frame_id - STILL_FRAMES + 1)
        items = [_person(x + shift, y) for x, y in WORLD]
        if feed_telemetry:
            moved = 0.0 if frame_id < STILL_FRAMES else -PAN
            tracker.camera_motion.push(
                np.array([[1.0, 0.0, moved], [0.0, 1.0, 0.0]], np.float32)
            )
        tracks = tracker.update(
            Detections(tag=FrameTag(CAMERA, frame_id), items=items, height=1080, width=1920),
            image=None if images is None else images[frame_id],
        )
        published.append({t.track_id for t in tracks})
    return set().union(*published[:STILL_FRAMES]), published[-1]


class TestCameraMotionFromTelemetry:
    """The affine supplied by a PTZ head, which is the best estimate available where the
    hardware reports its own pan.
    """

    def test_camera_motion_compensation_survives_a_pan_that_bytetrack_does_not(self) -> None:
        """The failure CMC exists for, and it is total rather than gradual.

        A pan moves *every* track's prediction wrong by the same amount on the same frame, so the
        whole association fails at once. ByteTrack's tracks all go LOST, the detections all spawn
        new tentative tracks, and on the next frame the pan has moved on again — so the new tracks
        never confirm either. The result is not degraded tracking, it is a scene that re-births
        every frame and publishes nothing.

        Warping the predictions by the frame-to-frame affine first puts them in this frame's
        coordinates, and the association is then exactly as easy as it was before the camera
        moved.

        Three arms, and the middle one is the important one: BoT-SORT with ``cmc="none"`` must
        fail *identically* to ByteTrack. Without that, this test would only show that BoT-SORT
        differs from ByteTrack somehow, and BoT-SORT also changes the cost function.
        """
        before, after = _pan_scenario("botsort", cmc="external", feed_telemetry=True)
        assert before and after == before, "BoT-SORT lost identities across a compensated pan"

        for label, name, options in (
            ("bytetrack", "bytetrack", {}),
            ("botsort with cmc off", "botsort", {"cmc": "none"}),
        ):
            baseline_before, baseline_after = _pan_scenario(name, **options)
            assert baseline_before, f"{label} never established the tracks"
            assert baseline_after != baseline_before, (
                f"{label} kept the identities through the pan; if the baseline survives, this "
                f"test says nothing about camera-motion compensation"
            )

    def test_an_affine_that_is_never_pushed_is_not_quietly_reused(self) -> None:
        """A stale affine applied to a later frame moves every prediction by a motion that did
        not happen, and over-compensation loses identities exactly as under-compensation does. So
        each pushed affine is consumed by one frame and the next frame gets identity unless it is
        told otherwise — which, on this scenario, means failing like ByteTrack.
        """
        before, after = _pan_scenario("botsort", cmc="external", feed_telemetry=False)
        assert before and after != before

    def test_strict_telemetry_refuses_to_guess_when_a_packet_is_dropped(self) -> None:
        """On an installation where telemetry is supposed to be on every frame, silence is an
        incident. Treating it as "the camera did not move" is a slow degradation nobody notices.
        """
        tracker = TRACKERS.build(
            "botsort", min_hits=1, cmc="external", cmc_options={"strict": True}
        )
        with pytest.raises(TrackingError, match="no camera motion was pushed"):
            tracker.update(Detections(tag=FrameTag(CAMERA, 0), items=[_person(400, 500)]))


class TestCameraMotionFromPixels:
    """The same compensation with no telemetry, recovered from the frames by sparse optical
    flow.
    """

    def test_the_optical_flow_estimator_recovers_a_known_pan(self) -> None:
        """Before trusting it inside a tracker, check it against a translation we chose.

        Sub-pixel accuracy is not a nice-to-have here: the affine is applied to every prediction,
        so an error in it is an error on every track at once, and a wrong compensation is worse
        than none.
        """
        pytest.importorskip("cv2")
        from shipvision.mot import CAMERA_MOTION

        estimator = CAMERA_MOTION.build("sparse_flow")
        views = _panning_views()

        # Every frame in order, because the estimator is stateful: skipping one would compare
        # against a two-frame-old reference and "recover" twice the pan.
        for frame_id, view in enumerate(views):
            affine = estimator.estimate(view)
            expected = 0.0 if frame_id < STILL_FRAMES else -PAN
            assert affine[0, 2] == pytest.approx(expected, abs=0.5), f"frame {frame_id}"
            assert affine[1, 2] == pytest.approx(0.0, abs=0.5)
            np.testing.assert_allclose(affine[:, :2], np.eye(2), atol=1e-3)

    def test_compensation_from_pixels_alone_also_survives_the_pan(self) -> None:
        """The deployment case with no telemetry: the affine comes from the frames themselves.

        Same assertion as the telemetry version, and the same ByteTrack failure underneath it —
        which is the point of running both. If only the telemetry arm passed, the estimator would
        be untested end to end; if only this one passed, an estimator bug would be hiding behind
        a lucky threshold.
        """
        pytest.importorskip("cv2")
        views = _panning_views()
        before, after = _pan_scenario("botsort", cmc="sparse_flow", images=views)
        assert before and after == before

    def test_the_flow_estimator_refuses_to_invent_an_answer_without_pixels(self) -> None:
        """Returning identity would be indistinguishable from a correct answer and wrong on
        exactly the frames that matter."""
        pytest.importorskip("cv2")
        tracker = TRACKERS.build("botsort", min_hits=1, cmc="sparse_flow")
        with pytest.raises(TrackingError, match="needs the frame"):
            tracker.update(Detections(tag=FrameTag(CAMERA, 0), items=[_person(400, 500)]))


class TestMinimumAppearanceFusion:
    """Fusing IoU and appearance by minimum rather than by weighted sum, so either signal on
    its own can settle a pair.
    """

    def test_appearance_breaks_a_tie_that_geometry_gets_backwards(self) -> None:
        """Minimum fusion against the same tracker with no embeddings to fuse.

        Two people stand shoulder to shoulder and change places. Every pair of (prediction,
        detection) overlaps, the motion gate objects to none of them, and the *wrong* reading has
        the better IoU — so geometry alone swaps the identities, and both baselines here do.

        Fusing by minimum, the correct pairing scores ``0.5 x 0.0002`` on appearance and the
        wrong one is pushed to 1.0 by the appearance gate, so the correct reading wins by three
        orders of magnitude rather than by a margin.

        Both baselines are asserted to fail: ByteTrack, which has no appearance term at all, and
        BoT-SORT itself when the detections carry no embedding — which also pins down that the
        win comes from the appearance and not from anything else BoT-SORT does.
        """
        assert _pair_run(with_appearance=True), "BoT-SORT swapped two visibly different people"
        assert not _pair_run(with_appearance=False), (
            "BoT-SORT with no embeddings was expected to swap them; if it does not, the scenario "
            "is not geometrically ambiguous and this test measures nothing"
        )
        assert not _pair_run(
            with_appearance=True, name="bytetrack"
        ), "ByteTrack was expected to swap them — it has no appearance term to consult"

    def test_missing_embeddings_fall_back_to_geometry_rather_than_claiming_a_match(
        self,
    ) -> None:
        """A frame where the re-ID stage did not run must not be read as "everything looks
        identical", which is the strongest claim available and made on no evidence.

        Asserted by construction rather than by outcome: an all-zero appearance cost would make
        every pair free, so the tracker would match across the whole frame. It does not.
        """
        tracker = TRACKERS.build("botsort", min_hits=1, max_age=5)
        published = []
        for frame_id in range(6):
            items = [_person(300.0, 500.0), _person(1500.0, 500.0)]
            published.append(
                tracker.update(
                    Detections(
                        tag=FrameTag(CAMERA, frame_id), items=items, height=1080, width=1920
                    )
                )
            )
        assert len(published[-1]) == 2
        centres = sorted(float(t.box[0] + t.box[2]) / 2.0 for t in published[-1])
        assert centres == pytest.approx([300.0, 1500.0], abs=5.0)


class TestOperability:
    """What an operator can see and what a reconnect has to clear."""

    def test_describe_names_the_estimator_in_use(self) -> None:
        """An operator reading a log line needs to know whether compensation was on."""
        assert "none" in TRACKERS.build("botsort").describe()
        assert "external" in TRACKERS.build("botsort", cmc="external").describe()

    def test_reset_clears_the_estimator_as_well_as_the_pool(self) -> None:
        """A reconnected camera has no continuity with the frame before it, so a reference frame
        kept across the break would produce a large spurious motion on the first frame back."""
        tracker = TRACKERS.build("botsort", min_hits=1, cmc="external")
        tracker.camera_motion.push(np.array([[1.0, 0.0, -30.0], [0.0, 1.0, 0.0]], np.float32))
        tracker.reset()
        affine = tracker.camera_motion.estimate(None)
        np.testing.assert_allclose(affine, [[1, 0, 0], [0, 1, 0]])


# ------------------------------------------------------------- and the same from pixels only


def _panning_views() -> list[np.ndarray]:
    """A textured quay wall sliding left at 45 px/frame, as the camera would see it.

    Structure at two scales on purpose: ``goodFeaturesToTrack`` needs corners, and uniform
    noise upsampled in blocks gives plenty of them while staying deterministic.
    """
    rng = np.random.default_rng(7)
    coarse = (rng.random((150, 400)) * 255).astype(np.uint8)
    backdrop = np.kron(coarse, np.ones((8, 8), np.uint8))
    views = []
    for frame_id in range(STILL_FRAMES + PANNING_FRAMES):
        shift = int(0.0 if frame_id < STILL_FRAMES else PAN * (frame_id - STILL_FRAMES + 1))
        views.append(backdrop[:1080, shift : shift + 1920].copy())
    return views


# =============================================================== minimum appearance fusion


def _crop(kind: str, jitter: float = 0.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """A ``(3, h, w)`` crop with real structure.

    Structure rather than noise, because the mock extractor block-averages: two independent
    uniform-noise crops converge on the same flat thumbnail and score ~0.9 against each
    other, which is the right answer to the question it was asked and not the question this
    test means to ask.
    """
    height, width = 64, 32
    rows = np.linspace(0.0, 1.0, height)[:, None]
    cols = np.linspace(0.0, 1.0, width)[None, :]
    if kind == "striped":
        plane = ((np.sin(rows * 20.0) > 0).astype(np.float32) * 0.8 + 0.1) * np.ones_like(cols)
    else:
        plane = (cols * 0.8 + 0.1) * np.ones_like(rows)
    crop = np.repeat(plane[None, :, :], 3, axis=0).astype(np.float32)
    if jitter and rng is not None:
        crop = crop + (rng.random(crop.shape) - 0.5) * jitter
    return np.clip(crop, 0.0, 1.0)


# Two people standing shoulder to shoulder, 14 px apart with 60 px-wide boxes, so every
# (track, detection) pair overlaps heavily. On the disputed frame they change places by 20 px
# each — inside the motion gate, so the filter has no objection to either reading — and the
# *wrong* pairing has the better overlap: 0.905 IoU against 0.714. Geometry alone therefore
# swaps them, definitively rather than by a coin toss.
_PAIR_LEAD = 10
_PAIR_START = (400.0, 414.0)
_PAIR_END = (420.0, 394.0)


def _pair_run(*, with_appearance: bool, name: str = "botsort") -> bool:
    """Run the shoulder-to-shoulder swap. Returns True if the identities were preserved."""
    extractor = EXTRACTORS.build("mock", dim=64, seed=3)
    rng = np.random.default_rng(11)
    looks = {
        "striped": lambda: extractor.extract_one(_crop("striped", 0.05, rng)),
        "graded": lambda: extractor.extract_one(_crop("graded", 0.05, rng)),
    }

    tracker = TRACKERS.build(name, min_hits=2, max_age=30)
    identity_of: dict[str, int] = {}
    for frame_id in range(_PAIR_LEAD + 1):
        swapped = frame_id == _PAIR_LEAD
        xs = _PAIR_END if swapped else _PAIR_START
        items = [
            _person(xs[0], 500.0, looks["striped"]() if with_appearance else None),
            _person(xs[1], 500.0, looks["graded"]() if with_appearance else None),
        ]
        tracks = tracker.update(
            Detections(tag=FrameTag(CAMERA, frame_id), items=items, height=1080, width=1920)
        )
        if len(tracks) != 2:
            continue
        # The striped person is at xs[0]; find the track sitting on that box.
        by_x = sorted(tracks, key=lambda t: float(t.box[0]))
        nearest = by_x[0] if xs[0] < xs[1] else by_x[1]
        if not swapped:
            identity_of["striped"] = nearest.track_id
        else:
            return bool(identity_of) and nearest.track_id == identity_of["striped"]
    return False
