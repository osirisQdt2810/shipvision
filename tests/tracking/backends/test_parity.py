"""The compiled trackers against the readable ones, on sequences with a right answer.

A compiled tracker nobody can compare against is a compiled tracker nobody can trust, and for a
tracker the comparison has to be over a *sequence* rather than over one call: the whole of what
a tracker does is carry state, and a per-frame check would pass for an implementation whose
filter diverges by frame forty.

**What is compared, and what is not.** Identities are compared exactly and boxes with a
tolerance — see :func:`~tests.tracking.backends.conftest.assert_same_tracking`. Track ids are
compared *structurally*, because the counter that issues them is process-wide by design.

**Ties are excluded by construction, not by tolerance.** The C++ assignment solver and
``scipy.optimize.linear_sum_assignment`` optimise the same total, so they agree on the cost —
but two assignments of equal total cost are both optimal and nothing says the two must pick the
same one. Every scenario here is therefore built so the right answer is unambiguous: objects
are far apart relative to their motion, and no two candidate pairings score the same. A tracker
whose output depends on which of two equal-cost matches it gets is not reproducible anyway.

The whole file skips when there is no build, exactly like the imgproc parity tests. That is
not a hole in the offline tier: with no build there is nothing on the machine to disagree.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipvision.errors import ConfigurationError
from shipvision.registry import NATIVE, PYTHON
from shipvision.tracking import TRACKERS
from shipvision.types import Detection, Detections, FrameTag
from tests.tracking.backends.conftest import NATIVE_BUILT, NO_BUILD, assert_same_tracking
from tests.tracking.conftest import CAMERA, box, det, drive, frame

pytestmark = [pytest.mark.native, pytest.mark.skipif(not NATIVE_BUILT, reason=NO_BUILD)]


def both(algorithm: str, **options: object) -> tuple[object, object]:
    """The readable tracker and the compiled one, configured identically."""
    return (
        TRACKERS.build(algorithm, backend=PYTHON, **options),
        TRACKERS.build(algorithm, backend=NATIVE, **options),
    )


class TestTheCompiledTrackerTracksTheSameObjects:
    """One identity per object, in the same frames, with the same boxes."""

    def test_two_objects_moving_apart_are_tracked_identically(self, algorithm: str) -> None:
        frames = [[det(100 + 6 * i, 200), det(900 - 6 * i, 600)] for i in range(20)]
        reference, candidate = both(algorithm, min_hits=2, max_age=10)

        worst = assert_same_tracking(drive(reference, frames), drive(candidate, frames))

        assert worst < 1e-3, f"boxes drifted by {worst} px over twenty frames"

    def test_a_track_withheld_until_confirmation_is_withheld_by_both(
        self, algorithm: str
    ) -> None:
        """The frames before the third hit are the ones a divergent lifecycle shows up in, and
        they are the frames a per-frame spot check would call "both empty, fine"."""
        frames = [[det(100 + 4 * i, 200)] for i in range(6)]
        reference, candidate = both(algorithm, min_hits=3)

        published_reference = drive(reference, frames)
        published_candidate = drive(candidate, frames)

        assert [len(step) for step in published_reference] == [0, 0, 1, 1, 1, 1]
        assert_same_tracking(published_reference, published_candidate)

    def test_a_detection_gap_is_survived_or_dropped_by_both(self, algorithm: str) -> None:
        """A person walks behind a pillar for three frames. Whether the identity survives is
        the algorithm's business; that the two backends *agree* is this test's."""
        frames = [[] if 8 <= i < 11 else [det(100 + 8 * i, 300)] for i in range(20)]
        reference, candidate = both(algorithm, min_hits=2, max_age=5)

        assert_same_tracking(drive(reference, frames), drive(candidate, frames))

    def test_a_track_that_ages_out_is_dropped_from_both_pools(self, algorithm: str) -> None:
        """Publication stopping is not the same as memory being freed, and only one of them is
        visible in the output. A process here runs for weeks."""
        frames = [[det(100, 200)] for _ in range(4)] + [[] for _ in range(8)]
        reference, candidate = both(algorithm, min_hits=2, max_age=3)

        drive(reference, frames)
        drive(candidate, frames)

        assert reference.pool_size == 0
        assert candidate.pool_size == 0

    def test_an_empty_frame_ages_both_pools_the_same_way(self, algorithm: str) -> None:
        """An empty frame is information, not a no-op: a tracker that skips the update keeps
        dead objects alive forever."""
        frames = [[det(100, 200)] for _ in range(4)] + [[], []] + [[det(100, 200)]]
        reference, candidate = both(algorithm, min_hits=2, max_age=5)

        assert_same_tracking(drive(reference, frames), drive(candidate, frames))
        assert reference.pool_size == candidate.pool_size

    def test_a_one_frame_false_positive_becomes_an_identity_in_neither(
        self, algorithm: str
    ) -> None:
        frames = [[det(100 + 3 * i, 200)] for i in range(10)]
        frames[4] = [*frames[4], det(1500, 900)]
        reference, candidate = both(algorithm, min_hits=3, max_age=5)

        published = drive(candidate, frames)

        assert_same_tracking(drive(reference, frames), published)
        assert len({t.track_id for step in published for t in step}) == 1

    def test_reset_forgets_the_same_amount_on_both_sides(self, algorithm: str) -> None:
        frames = [[det(100 + 5 * i, 200)] for i in range(6)]
        reference, candidate = both(algorithm, min_hits=2)
        drive(reference, frames)
        drive(candidate, frames)

        reference.reset()
        candidate.reset()

        assert reference.pool_size == 0
        assert candidate.pool_size == 0
        # And the sequence may start again from frame zero, which `begin` refuses without it.
        assert_same_tracking(drive(reference, frames), drive(candidate, frames))


class TestTheHardSequence:
    """Everything at once: jitter, a gap, a crossing, a low-confidence spell and noise.

    The scenarios above each isolate one behaviour, which is what makes a failure readable.
    This one exists because the interesting divergences are *cumulative* — a filter that is
    slightly wrong tracks perfectly for thirty frames and then associates differently once.
    """

    def test_sixty_frames_of_realistic_input_produce_the_same_identities(
        self, algorithm: str
    ) -> None:
        rng = np.random.default_rng(20260825)
        frames = []
        for index in range(60):
            items = []
            if index not in (20, 21, 22):  # object A is occluded for three frames
                items.append(
                    det(100 + 6 * index + rng.normal(0, 1.5), 200 + rng.normal(0, 1.5))
                )
            # Object B crosses the frame the other way and dips into the low-score tier, which
            # is the one detection ByteTrack keeps and SORT discards.
            items.append(
                det(
                    900 - 6 * index + rng.normal(0, 1.5),
                    205 + rng.normal(0, 1.5),
                    score=0.3 if 30 <= index < 36 else 0.8,
                )
            )
            if index % 7 == 0:  # detector noise, well below any threshold
                items.append(det(rng.uniform(0, 1900), rng.uniform(0, 1000), score=0.15))
            frames.append(items)

        reference, candidate = both(algorithm, min_hits=2, max_age=10)

        worst = assert_same_tracking(drive(reference, frames), drive(candidate, frames))

        assert worst < 1e-3, f"boxes drifted by {worst} px over sixty frames"


class TestAppearanceSurvivesTheBoundary:
    """ByteTrack's own association is geometric, but a track still *carries* an appearance
    vector — the cross-camera tier downstream has no access to the crops. The C++ pool never
    sees it, so the averaging happens on the Python side, and this is what says the two
    backends produce the same vector rather than merely both producing one."""

    def test_the_published_embedding_matches_the_readable_backend(self) -> None:
        rng = np.random.default_rng(11)
        base = rng.normal(size=32).astype(np.float32)
        base /= np.linalg.norm(base)
        frames = [
            [
                det(
                    100 + 5 * i,
                    200,
                    embedding=(lambda v: v / np.linalg.norm(v))(
                        base + 0.05 * rng.normal(size=32).astype(np.float32)
                    ),
                )
            ]
            for i in range(12)
        ]
        reference, candidate = both("bytetrack", min_hits=2, embedding_momentum=0.9)

        published_reference = drive(reference, frames)
        published_candidate = drive(candidate, frames)

        assert_same_tracking(published_reference, published_candidate)
        expected = published_reference[-1][0].embedding
        actual = published_candidate[-1][0].embedding
        assert actual is not None
        assert np.abs(expected - actual).max() < 1e-5

    def test_a_dead_track_stops_costing_memory_on_the_native_side(self) -> None:
        """The id and embedding maps live in Python, keyed on a pool-local id. If they did not
        follow the C++ pool's own eviction they would grow by one entry per object ever seen,
        which on a camera that runs for a week is the leak nothing else would notice."""
        candidate = TRACKERS.build("bytetrack", backend=NATIVE, min_hits=2, max_age=2)
        embedding = np.eye(1, 8, dtype=np.float32)[0]
        drive(candidate, [[det(100, 200, embedding=embedding)] for _ in range(4)])

        drive(candidate, [[] for _ in range(6)], start=4)

        assert candidate.pool_size == 0
        assert candidate._native_pool._embeddings == {}
        assert candidate._native_pool._ids == {}


class TestTheCompiledTrackerRefusesWhatTheReadableOneRefuses:
    """A bad config must fail at start-up with the same typed error on either backend, because
    the ``except`` clause that handles it was written once."""

    @pytest.mark.parametrize(
        ("algorithm", "options"),
        [
            ("sort", {"det_threshold": 1.5}),
            ("sort", {"iou_threshold": 0.0}),
            ("sort", {"max_age": 0}),
            ("sort", {"min_hits": 0}),
            ("bytetrack", {"track_threshold": 0.1, "low_threshold": 0.5}),
            ("bytetrack", {"max_age": 0}),
            ("bytetrack", {"embedding_momentum": 1.0}),
            ("ocsort", {"delta_t": 0}),
            ("ocsort", {"momentum_weight": 1.5}),
            ("ocsort", {"recovery_iou_threshold": 0.0}),
            ("botsort", {"appearance_gate": 0.0}),
            ("botsort", {"appearance_weight": 1.5}),
            ("botsort", {"track_threshold": 0.1, "low_threshold": 0.5}),
            ("deepsortv2", {"cascade_stride": 0}),
            ("deepsortv2", {"border_fraction": 0.6}),
            ("deepsortv2", {"appearance_momentum": (0.95, 0.9)}),
        ],
    )
    def test_both_backends_raise_a_configuration_error(
        self, algorithm: str, options: dict
    ) -> None:
        with pytest.raises(ConfigurationError):
            TRACKERS.build(algorithm, backend=PYTHON, **options)
        with pytest.raises(ConfigurationError):
            TRACKERS.build(algorithm, backend=NATIVE, **options)

    def test_a_typo_in_a_config_key_is_not_silently_dropped(self, algorithm: str) -> None:
        """A dropped keyword means the tracker runs with a default nobody chose."""
        with pytest.raises(TypeError):
            TRACKERS.build(algorithm, backend=NATIVE, min_hitz=2)

    def test_a_second_camera_on_one_instance_is_refused(self, algorithm: str) -> None:
        """The tag discipline lives in Python and applies to both backends, which is the whole
        reason the tag never crosses the boundary."""
        from shipvision.errors import TrackingError

        candidate = TRACKERS.build(algorithm, backend=NATIVE, min_hits=1)
        candidate.update(
            Detections(tag=FrameTag(camera_id=CAMERA, frame_id=0), items=[det(100, 200)])
        )

        with pytest.raises(TrackingError, match="one camera"):
            candidate.update(
                Detections(tag=FrameTag(camera_id="other", frame_id=1), items=[det(100, 200)])
            )


class TestTheTwoBackendsAgreeOnWhatTheyAreCarrying:
    """``tracks`` is the live set, published or not, and it is what a caller reads to answer
    "is this identity still alive". It comes from a different C++ entry point than ``update``
    does, so it gets its own check."""

    def test_the_live_set_matches_after_a_partial_occlusion(self, algorithm: str) -> None:
        frames = [[det(100 + 5 * i, 200)] for i in range(5)] + [[]]
        reference, candidate = both(algorithm, min_hits=2, max_age=5)
        drive(reference, frames)
        drive(candidate, frames)

        expected = sorted((t.state, t.hits, t.time_since_update) for t in reference.tracks)
        actual = sorted((t.state, t.hits, t.time_since_update) for t in candidate.tracks)

        assert actual == expected

    def test_every_published_track_carries_the_input_tag(self, algorithm: str) -> None:
        """The tag is the thing a tracker must never lose: a mis-tagged result is a
        real-looking detection on a camera where nothing happened."""
        frames = [[det(100 + 5 * i, 200)] for i in range(5)]
        candidate = TRACKERS.build(algorithm, backend=NATIVE, min_hits=1)

        published = drive(candidate, frames, start=7)

        for offset, step in enumerate(published):
            for track in step:
                assert track.tag == FrameTag(camera_id=CAMERA, frame_id=7 + offset)

    def test_the_box_of_a_stationary_object_stays_where_the_readable_one_puts_it(
        self, algorithm: str
    ) -> None:
        """Twelve frames of a filter correcting itself against the same measurement is where a
        transposed matrix or a wrong noise term shows up as a slow drift rather than a jump."""
        frames = [[Detection(box=box(400, 500), score=0.9)] for _ in range(12)]
        reference, candidate = both(algorithm, min_hits=2)

        worst = assert_same_tracking(drive(reference, frames), drive(candidate, frames))

        assert worst < 1e-4


def _unit(vector: np.ndarray) -> np.ndarray:
    """L2-normalised, which is how this library stores an embedding at every boundary."""
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def _wardrobe(count: int, *, seed: int) -> list[np.ndarray]:
    """One near-orthogonal appearance vector per object: people in different clothing.

    Near-orthogonal rather than merely different, so a same-object cosine distance sits far
    below DeepSORTv2's 0.15 gate and a cross-object one far above it. A scenario whose
    appearance distances land near a gate would be measuring float32 rounding rather than
    measuring the two implementations.
    """
    rng = np.random.default_rng(seed)
    return [_unit(rng.normal(size=16).astype(np.float32)) for _ in range(count)]


def _looks_like(base: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The same object, seen again: the vector plus the jitter a re-ID model really produces."""
    return _unit(base + 0.05 * rng.normal(size=base.shape).astype(np.float32))


class TestOcSortsThreeFixesCrossTheBoundary:
    """OC-SORT is three named corrections to SORT, and the generic parity scenarios above
    exercise none of them: they never gap a *fast* track, never hide an object that then stops,
    and never make two candidates geometrically interchangeable. Each test here is one of the
    three, plus the same run with that fix switched off — because a flag that is silently
    dropped at the binding produces two trackers that agree perfectly and are both the wrong
    algorithm."""

    #: Fast enough that the prediction is a different box after the gap, which is the whole
    #: precondition for the re-update mattering: at 4 px/frame the filter coasts back onto the
    #: object by luck and every configuration agrees.
    SPEED = 14.0

    def _gap_sequence(self) -> list[list]:
        """A fast walker the detector loses for four frames and then finds again."""
        return [
            [] if 10 <= index < 14 else [det(120 + self.SPEED * index, 300)]
            for index in range(26)
        ]

    @pytest.mark.parametrize("re_update", [True, False])
    def test_a_gapped_track_is_re_derived_the_same_way_on_both_sides(
        self, re_update: bool
    ) -> None:
        """ORU. Parametrised over the switch because that is what says the switch arrived: with
        it dropped, both runs would still agree and both would be estimation-centric SORT."""
        frames = self._gap_sequence()
        reference, candidate = both(
            "ocsort", min_hits=2, max_age=10, re_update=re_update, recover=False
        )

        published = drive(candidate, frames)
        worst = assert_same_tracking(drive(reference, frames), published)

        assert worst < 1e-3, f"boxes drifted by {worst} px across the gap"
        # The known-correct answer: one object was present throughout, so one identity.
        assert len({track.track_id for step in published for track in step}) == 1

    def test_the_re_update_actually_changes_the_track_it_is_switched_on_for(self) -> None:
        """Otherwise the test above proves only that two implementations of nothing agree.

        Measured over the whole run rather than at the end, because that is where the effect
        lives: the correction pins the *position* on the re-acquisition frame whichever way the
        filter got there, and what ORU changes is the velocity — so the two runs separate on the
        frames just after the gap and converge again as the filter re-settles. The separation is
        two orders of magnitude above the 1e-3 px the parity comparison calls float32 rounding.
        """
        frames = self._gap_sequence()
        runs = [
            drive(
                TRACKERS.build(
                    "ocsort",
                    backend=NATIVE,
                    min_hits=2,
                    max_age=10,
                    re_update=re_update,
                    recover=False,
                ),
                frames,
            )
            for re_update in (True, False)
        ]

        worst = max(
            float(np.abs(with_oru[0].box - plain[0].box).max())
            for with_oru, plain in zip(*runs, strict=True)
            if with_oru and plain
        )

        assert worst > 0.05, (
            f"the re-updated filter tracked within {worst} px of the plain one, so the "
            f"re_update flag is not reaching the C++ pool"
        )

    def test_an_object_that_stopped_while_hidden_is_recovered_by_both(self) -> None:
        """OCR. The object walks, is hidden for four frames, and reappears where it was last
        seen rather than where the filter carried it — so the prediction has walked a whole
        box-width away and only an association against the last *observation* can find it."""
        last_seen = 120 + self.SPEED * 9
        frames = [
            (
                [det(120 + self.SPEED * index, 300)]
                if index < 10
                else ([] if index < 14 else [det(last_seen, 300)])
            )
            for index in range(22)
        ]
        reference, candidate = both("ocsort", min_hits=2, max_age=10, recover=True)

        published = drive(candidate, frames)

        assert_same_tracking(drive(reference, frames), published)
        assert (
            len({track.track_id for step in published for track in step}) == 1
        ), "the stopped object was given a new identity, so the recovery stage did not run"

    @pytest.mark.parametrize("momentum_weight", [0.0, 0.2])
    def test_two_objects_passing_each_other_keep_their_own_identities(
        self, momentum_weight: float
    ) -> None:
        """OCM. Different speeds and different heights, so no two pairings score the same —
        a symmetric crossing has two optimal assignments and the two solvers are entitled to
        pick different ones."""
        frames = [
            [det(200 + 11 * index, 300), det(900 - 7 * index, 340)] for index in range(24)
        ]
        reference, candidate = both(
            "ocsort", min_hits=2, max_age=10, momentum_weight=momentum_weight
        )

        published = drive(candidate, frames)

        assert_same_tracking(drive(reference, frames), published)
        assert len({track.track_id for step in published for track in step}) == 2


class TestBotSortsTwoChangesCrossTheBoundary:
    """BoT-SORT is ByteTrack plus camera-motion compensation and minimum-fused appearance.
    Neither reaches C++ the way the rest of a tracker does — the affine is produced by a Python
    estimator and the appearance distances are built from vectors that never cross — so each is
    a place where the compiled tracker could quietly be running plain ByteTrack."""

    #: Faster than a third of a person's width per frame, which is more than ByteTrack's IoU
    #: threshold survives: uncompensated, the whole scene is re-born on the first panning frame.
    PAN = 45.0
    QUAY = ((400.0, 500.0), (900.0, 620.0), (1400.0, 400.0))

    STILL = 8

    def _panning_run(
        self, tracker: object, *, telemetry: bool = True, frames: int = 22
    ) -> list[list]:
        """Drive a fixed scene under a camera that starts panning on frame eight."""
        published = []
        for frame_id in range(frames):
            panning = frame_id >= self.STILL
            shift = -self.PAN * (frame_id - self.STILL + 1) if panning else 0.0
            if telemetry:
                moved = -self.PAN if panning else 0.0
                tracker.camera_motion.push(
                    np.array([[1.0, 0.0, moved], [0.0, 1.0, 0.0]], np.float32)
                )
            items = [det(x + shift, y, w=60.0, h=140.0) for x, y in self.QUAY]
            published.append(tracker.update(frame(items, frame_id)))
        return published

    @staticmethod
    def _survivors(published: list[list]) -> tuple[set[int], set[int]]:
        """The identities established before the pan, and the ones still published at the end.

        Counting *distinct ids over the whole run* would not see the failure: a scene that
        re-births every frame never confirms any of its new tracks, so it publishes nothing new
        and the id count stays put. What changes is which identities are still alive at the end.
        """
        established: set[int] = set()
        for step in published[: TestBotSortsTwoChangesCrossTheBoundary.STILL]:
            established |= {track.track_id for track in step}
        return established, {track.track_id for track in published[-1]}

    def test_a_panning_camera_is_compensated_identically(self) -> None:
        reference, candidate = both("botsort", cmc="external", min_hits=2, max_age=30)

        published = self._panning_run(candidate)

        assert_same_tracking(self._panning_run(reference), published)
        # The known-correct answer: three people stood still while the camera moved, so the
        # three identities that existed before the pan are the three that end it.
        established, surviving = self._survivors(published)
        assert len(established) == 3
        assert surviving == established

    def test_the_affine_actually_moves_the_compiled_predictions(self) -> None:
        """The compensated run against the same compiled tracker with the telemetry withheld.

        Without this arm, a binding that dropped the affine on the floor would pass the test
        above — the numpy tracker would be wrong in exactly the same way, and two implementations
        of plain ByteTrack agree beautifully.
        """
        uncompensated = TRACKERS.build(
            "botsort", backend=NATIVE, cmc="external", min_hits=2, max_age=30
        )

        established, surviving = self._survivors(
            self._panning_run(uncompensated, telemetry=False)
        )

        assert established, "the uncompensated tracker never established the tracks at all"
        assert surviving != established, (
            "an uncompensated tracker kept its identities through a 45 px/frame pan, so this "
            "scenario says nothing about camera-motion compensation"
        )

    def test_appearance_is_fused_the_same_way_on_both_sides(self) -> None:
        """Two people crossing, told apart by what they are wearing. The minimum fusion is
        BoT-SORT's second change, and it only exists on a frame where both signals are
        present."""
        rng = np.random.default_rng(7)
        wardrobe = _wardrobe(2, seed=3)
        frames = [
            [
                det(200 + 11 * index, 300, embedding=_looks_like(wardrobe[0], rng)),
                det(900 - 7 * index, 340, embedding=_looks_like(wardrobe[1], rng)),
            ]
            for index in range(24)
        ]
        reference, candidate = both("botsort", min_hits=2, max_age=10)

        published = drive(candidate, frames)

        assert_same_tracking(drive(reference, frames), published)
        assert len({track.track_id for step in published for track in step}) == 2

    def test_a_frame_with_no_embeddings_falls_back_to_geometry_on_both_sides(self) -> None:
        """ "No appearance evidence" is a distinct answer from "the appearance distance is
        zero", and the two backends have to reach it in the same place — the compiled one is
        handed an empty matrix, the readable one gets ``None``."""
        frames = [[det(200 + 9 * index, 300), det(900 - 6 * index, 340)] for index in range(18)]
        reference, candidate = both("botsort", min_hits=2, max_age=10)

        assert_same_tracking(drive(reference, frames), drive(candidate, frames))


class TestDeepSortV2sCascadeCrossesTheBoundary:
    """Four association stages, a per-detection appearance rate, and a border rule — none of
    which the generic scenarios reach. Stage A never runs without embeddings, stage C never
    runs without a gap, and the border rule never runs without a frame size."""

    def _walkers(self, count: int, *, seed: int, length: int = 30) -> list[list]:
        """``count`` people crossing the quay at different speeds and heights, each with a
        consistent appearance. Different speeds because two identical trajectories make two
        assignments equally optimal, and the two solvers may then legitimately disagree."""
        rng = np.random.default_rng(seed)
        wardrobe = _wardrobe(count, seed=seed + 1)
        return [
            [
                det(
                    150 + (9 + 4 * person) * index,
                    250 + 90 * person,
                    h=100.0 + 20.0 * person,
                    embedding=_looks_like(wardrobe[person], rng),
                )
                for person in range(count)
            ]
            for index in range(length)
        ]

    def test_the_cascade_groups_the_same_detections_over_a_long_sequence(self) -> None:
        frames = self._walkers(3, seed=11)
        reference, candidate = both("deepsortv2", min_hits=2, max_age=10)

        published = drive(candidate, frames)

        worst = assert_same_tracking(drive(reference, frames), published)
        assert worst < 1e-3, f"boxes drifted by {worst} px over thirty frames"
        assert len({track.track_id for step in published for track in step}) == 3

    @pytest.mark.parametrize("recover", [True, False])
    def test_stage_c_agrees_whether_it_is_on_or_off(self, recover: bool) -> None:
        """Stage C is the observation-centric recovery, and it is the only stage that can pick
        up an object that stopped moving while it was hidden. Parametrised over the switch,
        because a flag dropped at the binding leaves two trackers that agree and are both
        missing a stage."""
        rng = np.random.default_rng(5)
        wardrobe = _wardrobe(1, seed=13)
        last_seen = 400 + 13 * 9
        frames = [
            (
                []
                if 10 <= index < 14
                else [
                    det(
                        400 + 13 * index if index < 10 else last_seen,
                        400,
                        embedding=_looks_like(wardrobe[0], rng),
                    )
                ]
            )
            for index in range(24)
        ]
        reference, candidate = both("deepsortv2", min_hits=2, max_age=10, recover=recover)

        assert_same_tracking(drive(reference, frames), drive(candidate, frames))

    @pytest.mark.parametrize("skip_border_recovery", [True, False])
    def test_the_border_rule_reaches_the_compiled_tracker(
        self, skip_border_recovery: bool
    ) -> None:
        """An object leaving the frame is half out of it, so its last observation is a truncated
        box that overlaps whatever else is at that edge. The rule needs the frame size, which is
        the one piece of ``Detections`` other trackers never look at — so it is the one a
        binding is most likely to drop."""
        rng = np.random.default_rng(9)
        wardrobe = _wardrobe(2, seed=17)
        frames = []
        for index in range(24):
            items = [
                det(1880 - 4 * index, 540, embedding=_looks_like(wardrobe[0], rng)),
            ]
            if index < 10 or index >= 14:
                items.append(det(1890, 300, embedding=_looks_like(wardrobe[1], rng)))
            frames.append(items)
        reference, candidate = both(
            "deepsortv2",
            min_hits=2,
            max_age=10,
            skip_border_recovery=skip_border_recovery,
        )

        assert_same_tracking(drive(reference, frames), drive(candidate, frames))

    def test_the_dynamic_appearance_rate_produces_the_same_gallery_vector(self) -> None:
        """The rate depends on how confident and how isolated each detection was, and it is
        computed on the Python side for both backends. What this asserts is that the compiled
        tracker's *matches* line up with the same detections, so the same rate is applied to the
        same track — a vector that agreed while the associations differed would be a
        coincidence, not parity."""
        rng = np.random.default_rng(21)
        wardrobe = _wardrobe(2, seed=23)
        frames = [
            [
                det(
                    200 + 12 * index,
                    300,
                    score=0.55 if index % 3 == 0 else 0.95,
                    embedding=_looks_like(wardrobe[0], rng),
                ),
                det(1500 - 8 * index, 700, embedding=_looks_like(wardrobe[1], rng)),
            ]
            for index in range(20)
        ]
        reference, candidate = both("deepsortv2", min_hits=2, max_age=10)

        published_reference = drive(reference, frames)
        published_candidate = drive(candidate, frames)

        assert_same_tracking(published_reference, published_candidate)
        expected = sorted((float(t.box[0]), t.embedding) for t in published_reference[-1])
        actual = sorted((float(t.box[0]), t.embedding) for t in published_candidate[-1])
        assert len(actual) == 2
        for (_, want), (_, got) in zip(expected, actual, strict=True):
            assert got is not None
            assert np.abs(want - got).max() < 1e-5

    def test_a_frame_with_no_embeddings_still_agrees(self) -> None:
        """Stage A falls back to gated GIoU alone and stage B loses its veto. A deployment with
        no re-ID pass runs exactly this path, so it is not a corner case."""
        frames = [
            [det(150 + 9 * index, 250), det(1500 - 11 * index, 600)] for index in range(24)
        ]
        reference, candidate = both("deepsortv2", min_hits=2, max_age=10)

        assert_same_tracking(drive(reference, frames), drive(candidate, frames))


class TestTheHardSequenceWithAppearance:
    """The cumulative case for the two trackers that associate on appearance.

    :class:`TestTheHardSequence` covers the geometric path for all five, and it deliberately
    carries no embeddings — so for BoT-SORT and DeepSORTv2 it exercises the fallback rather
    than the algorithm. This is the same shape of sequence with a re-ID vector on every crop:
    jitter, an occlusion, a crossing, a low-confidence spell and detector noise, over sixty
    frames. The divergences worth catching here are cumulative — a cost that is slightly wrong
    tracks perfectly for thirty frames and then associates differently once.
    """

    @pytest.mark.parametrize("algorithm", ["botsort", "deepsortv2"])
    def test_sixty_frames_of_embedded_input_produce_the_same_identities(
        self, algorithm: str
    ) -> None:
        rng = np.random.default_rng(20260825)
        wardrobe = _wardrobe(2, seed=31)
        stranger = _wardrobe(1, seed=97)[0]
        frames = []
        for index in range(60):
            items = []
            if index not in (20, 21, 22):  # object A is occluded for three frames
                items.append(
                    det(
                        100 + 6 * index + rng.normal(0, 1.5),
                        200 + rng.normal(0, 1.5),
                        embedding=_looks_like(wardrobe[0], rng),
                    )
                )
            # Object B crosses the other way and dips into the low-score tier, which is the one
            # detection ByteTrack keeps and its second stage has to rescue on geometry alone.
            items.append(
                det(
                    900 - 6 * index + rng.normal(0, 1.5),
                    260 + rng.normal(0, 1.5),
                    score=0.3 if 30 <= index < 36 else 0.8,
                    embedding=_looks_like(wardrobe[1], rng),
                )
            )
            if index % 7 == 0:  # detector noise, well below any threshold, in nobody's clothing
                items.append(
                    det(
                        rng.uniform(0, 1900),
                        rng.uniform(0, 1000),
                        score=0.15,
                        embedding=_looks_like(stranger, rng),
                    )
                )
            frames.append(items)

        reference, candidate = both(algorithm, min_hits=2, max_age=10)

        worst = assert_same_tracking(drive(reference, frames), drive(candidate, frames))

        assert worst < 1e-3, f"boxes drifted by {worst} px over sixty frames"
