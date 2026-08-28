"""The compiled cross-camera *matchers* against the readable ones, and the clusters they feed.

``test_parity.py`` next door checks the individual fused passes — threshold, ground distance,
gate, veto, distance conversion — each against its numpy twin. This file checks the layer above
it: the whole matchers of :mod:`shipvision.mtmc.matchers` ported to C++ under
``csrc/shipvision/mtmc/matchers/``, the average-linkage clusterer that consumes their matrices, and
the identities that come out of the far end.

Three claims, in the order they build on each other:

* **Same matrix.** Element for element, on scenes whose right answer is known — including an
  uncalibrated site, where the geometry must fall *open* rather than refusing everyone.
* **Same clusters.** A matrix that is off by 1e-7 is still the same matrix; a cluster boundary
  that moved is a different answer. Compared as a partition, because label numbering is
  arbitrary — scipy numbers by merge order, the C++ by first appearance.
* **Same global ids.** The end of the pipeline, through the real
  :class:`~shipvision.mtmc.tracker.ClusterMTMCTracker` with one half swapped. This is the only
  claim an operator can check, and it is the one a matrix comparison does not imply: the
  identity assigner is stateful, so a single disagreed cluster on instant 3 renames things on
  instant 4 and never converges back.

The whole file skips when there is no build, like every other parity file here.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipvision.errors import ConfigurationError
from shipvision.mtmc import (
    MTMC_MATCHERS,
    NEVER_MERGE,
    AgglomerativeClusterer,
    BaseClusterer,
    BaseMatcher,
    ClusterMTMCTracker,
    GroundPlane,
    Homography,
    TrackObservation,
    foot_points,
)
from shipvision.mtmc.backends.native import native_available
from shipvision.mtmc.matchers.appearance.utils import stack_embeddings
from shipvision.registry import NATIVE, PYTHON
from shipvision.reid.distance import cosine_similarity
from tests.mtmc.conftest import (
    make_cluster,
    make_track,
    one_person_two_cameras,
    two_people_two_cameras,
)

pytestmark = [
    pytest.mark.native,
    pytest.mark.skipif(
        not native_available(),
        reason="shipvision._C is not built, or predates the cross-camera helpers",
    ),
]

NAMES = ["appearance", "gated", "spatial"]

#: An identity homography per camera: the image plane *is* the map. Enough to exercise the
#: composition, with a real projective warp used where the *projection* is what is under test.
CALIBRATED = GroundPlane({"cam-a": Homography(np.eye(3)), "cam-b": Homography(np.eye(3))})


def _c():
    """The extension, imported at call time so collection works without a build."""
    from shipvision import _C

    return _C


# -- the numpy edge the compiled matchers take ------------------------------------------------
#
# Three arrays and a camera order, never a list of objects: one Python object per track per
# instant is exactly the per-frame overhead a native backend exists to remove. Camera identity
# crosses as an integer code, assigned by first appearance — which is what
# `BaseMatcher.mergeable_mask` does, and the codes are never compared across calls, so any
# consistent numbering describes the same same-camera exclusion.


def instant_arrays(
    observations: tuple[TrackObservation, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """``(boxes, frame_sizes, camera_codes, camera order)`` for one synchronised instant."""
    codes: dict[str, int] = {}
    order: list[str] = []
    for observation in observations:
        if observation.camera_id not in codes:
            codes[observation.camera_id] = len(codes)
            order.append(observation.camera_id)
    boxes = np.zeros((len(observations), 4), dtype=np.float32)
    sizes = np.zeros((len(observations), 2), dtype=np.int32)
    for index, observation in enumerate(observations):
        boxes[index] = np.asarray(observation.box, dtype=np.float32)
        sizes[index] = (observation.frame_width, observation.frame_height)
    camera_codes = np.array(
        [codes[observation.camera_id] for observation in observations], dtype=np.int32
    )
    return boxes, sizes, camera_codes, order


def plane_arrays(
    ground_plane: GroundPlane | None, order: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The ground plane as three arrays indexed by the camera codes ``order`` defines.

    An uncalibrated camera gets an identity matrix and a *false* flag rather than being left
    out. The flag is what carries the meaning: an identity homography is a real mapping — the
    image plane as the map — so a plane that expressed "no calibration" as a matrix would put
    every one of that camera's tracks at a plausible place instead of at no place.
    """
    matrices = np.tile(np.eye(3, dtype=np.float64), (len(order), 1, 1)).reshape(
        len(order), 3, 3
    )
    sizes = np.zeros((len(order), 2), dtype=np.int32)
    calibrated = np.zeros(len(order), dtype=np.uint8)
    for index, camera_id in enumerate(order):
        homography = None if ground_plane is None else ground_plane.get(camera_id)
        if homography is None:
            continue
        matrices[index] = homography.matrix
        sizes[index] = (homography.camera_width, homography.camera_height)
        calibrated[index] = 1
    return matrices, sizes, calibrated


def gram_of(observations: tuple[TrackObservation, ...]) -> np.ndarray:
    """``(n, n)`` float32 cosine similarity — the gemm the compiled matcher does not do.

    ``features @ features.T`` is what BLAS is for, so it stays here on purpose; the compiled
    matcher owns the threshold, the veto and the distance conversion around it.
    """
    features = stack_embeddings(observations)
    if features.size == 0:
        return np.zeros((len(observations), len(observations)), dtype=np.float32)
    return np.ascontiguousarray(cosine_similarity(features, features), dtype=np.float32)


def canonical(labels: np.ndarray) -> np.ndarray:
    """Relabel by first appearance, so two partitions compare with one ``==``.

    Only equality between labels ever carries meaning. scipy numbers by merge order and the
    compiled clusterer numbers by first appearance, so a raw comparison would fail on two
    identical answers.
    """
    seen: dict[int, int] = {}
    out = np.empty(len(labels), dtype=np.int32)
    for index, label in enumerate(labels):
        out[index] = seen.setdefault(int(label), len(seen))
    return out


# -- the adapters the parity tests drive ------------------------------------------------------
#
# Thin on purpose: they translate the numpy edge above into the compiled classes and nothing
# else, so a disagreement below is the C++ disagreeing rather than an adapter deciding
# something. They live in the test rather than in `shipvision/mtmc/backends/native.py` because
# wiring these classes into `MTMC_MATCHERS` under `backend="native"` is a change to the Python
# package, and this file's job is to establish that the C++ half is worth wiring in.


class NativeCoreMatcher(BaseMatcher):
    """One of the three compiled matchers, behind the ``BaseMatcher`` contract."""

    backend = NATIVE

    def __init__(
        self,
        name: str,
        *,
        ground_plane: GroundPlane | None = None,
        appearance_threshold: float = 0.86,
        spatial_threshold: float = 280.0,
        foot_ratio: float = 1.0,
        aspect_ratio: float = 0.25,
    ) -> None:
        self.name = name
        self.ground_plane = ground_plane
        self.appearance_threshold = appearance_threshold
        self.spatial_threshold = spatial_threshold
        self.foot_ratio = foot_ratio
        self.aspect_ratio = aspect_ratio

    def _spatial(self, matrices, sizes, calibrated):
        return _c().MtmcSpatialMatcher(
            spatial_threshold=self.spatial_threshold,
            foot_ratio=self.foot_ratio,
            aspect_ratio=self.aspect_ratio,
            homographies=matrices,
            calibration_sizes=sizes,
            calibrated=calibrated,
        )

    def build(self, observations) -> np.ndarray:
        # No early return for an empty instant: every camera being quiet at once is ordinary
        # input, and letting the adapter answer it here would leave the shape the compiled side
        # produces — (0, 0), not (0,) — untested by the very case it exists for.
        observations = tuple(observations)
        boxes, frame_sizes, camera_codes, order = instant_arrays(observations)
        matrices, sizes, calibrated = plane_arrays(self.ground_plane, order)
        if self.name == "appearance":
            matcher = _c().MtmcAppearanceMatcher(appearance_threshold=self.appearance_threshold)
            return np.asarray(matcher.build(gram_of(observations), camera_codes), np.float32)
        if self.name == "spatial":
            matcher = self._spatial(matrices, sizes, calibrated)
            return np.asarray(matcher.build(boxes, frame_sizes, camera_codes), np.float32)
        matcher = _c().MtmcGatedMatcher(
            appearance_threshold=self.appearance_threshold,
            spatial_threshold=self.spatial_threshold,
            foot_ratio=self.foot_ratio,
            aspect_ratio=self.aspect_ratio,
            homographies=matrices,
            calibration_sizes=sizes,
            calibrated=calibrated,
        )
        return np.asarray(
            matcher.build(gram_of(observations), boxes, frame_sizes, camera_codes), np.float32
        )


class NativeCoreClusterer(BaseClusterer):
    """The compiled average-linkage clusterer, behind the ``BaseClusterer`` contract."""

    backend = NATIVE

    def __init__(self, *, distance_threshold: float = 0.14) -> None:
        self.distance_threshold = distance_threshold

    def fit_predict(self, distances: np.ndarray) -> np.ndarray:
        matrix = np.ascontiguousarray(distances, dtype=np.float64)
        clusterer = _c().MtmcAgglomerativeClusterer(distance_threshold=self.distance_threshold)
        return np.asarray(clusterer.fit_predict(matrix), dtype=np.int32)


def both(name: str, **options: object) -> tuple[BaseMatcher, BaseMatcher]:
    """The readable matcher and the compiled one, configured identically.

    Options a given matcher does not accept are dropped rather than forwarded, which is what
    :meth:`shipvision.mtmc.tracker.ClusterMTMCTracker._build_matcher` does and for the same
    reason: ``appearance`` has no geometry and therefore no ground plane.
    """
    import inspect

    accepted = inspect.signature(MTMC_MATCHERS.get(name, PYTHON).__init__).parameters
    kept = {key: value for key, value in options.items() if key in accepted}
    return MTMC_MATCHERS.build(name, backend=PYTHON, **kept), NativeCoreMatcher(name, **kept)


class TestTheCompiledMatcherProducesTheSameMatrix:
    """Element for element. A distance matrix is harder to eyeball than a tracker's output —
    every entry is plausible — so the comparison is exact rather than qualitative."""

    @pytest.mark.parametrize("name", NAMES)
    def test_on_two_people_seen_by_two_cameras(self, name: str) -> None:
        observations = two_people_two_cameras().observations
        reference, candidate = both(name, ground_plane=CALIBRATED)

        actual = candidate.build(observations)
        expected = reference.build(observations)

        assert actual.shape == expected.shape == (4, 4)
        assert actual.dtype == expected.dtype == np.float32
        assert np.abs(actual - expected).max() < 1e-6

    @pytest.mark.parametrize("name", NAMES)
    def test_on_an_uncalibrated_site(self, name: str) -> None:
        """No homographies at all, which is the normal state of a site on its first day. The
        geometry must fall open rather than refusing everyone — that is what lets a camera take
        part on appearance alone — and both implementations must fall open the same way."""
        observations = two_people_two_cameras().observations
        reference, candidate = both(name)

        assert (
            np.abs(candidate.build(observations) - reference.build(observations)).max() < 1e-6
        )

    @pytest.mark.parametrize("name", NAMES)
    def test_on_an_instant_with_no_tracks_at_all(self, name: str) -> None:
        """Every camera can be quiet at once. The shape is the thing: ``(0,)`` instead of
        ``(0, 0)`` turns an ordinary instant into an IndexError three frames later."""
        reference, candidate = both(name, ground_plane=CALIBRATED)

        assert candidate.build(()).shape == (0, 0) == reference.build(()).shape

    @pytest.mark.parametrize("name", NAMES)
    def test_two_tracks_in_one_camera_are_never_mergeable(self, name: str) -> None:
        """Even when they carry the *same identity's* embedding, which is the adversarial case:
        appearance says yes as loudly as it can and the answer must still be no. Lose this and
        MTMC quietly becomes a within-camera deduplicator — every count drops, every metric
        improves, and the system is worse."""
        cluster = make_cluster(
            {
                "cam-a": [
                    make_track(camera="cam-a", track_id=1, identity=0, view=0),
                    make_track(camera="cam-a", track_id=2, identity=0, view=0),
                ]
            }
        )
        reference, candidate = both(name, ground_plane=CALIBRATED)

        actual = candidate.build(cluster.observations)

        assert actual[0, 1] == pytest.approx(NEVER_MERGE)
        assert actual[1, 0] == pytest.approx(NEVER_MERGE)
        assert np.array_equal(actual, reference.build(cluster.observations))

    @pytest.mark.parametrize("name", NAMES)
    def test_the_result_is_symmetric_finite_and_zero_on_the_diagonal(self, name: str) -> None:
        """What every clusterer requires. ``squareform`` has no tolerance for asymmetry — it
        silently reads the upper triangle — and it rejects a non-finite entry outright."""
        observations = two_people_two_cameras().observations
        _, candidate = both(name, ground_plane=CALIBRATED)

        actual = candidate.build(observations)

        assert np.array_equal(actual, actual.T)
        assert np.array_equal(
            np.diagonal(actual), np.zeros(len(observations), dtype=np.float32)
        )
        assert np.isfinite(actual).all()

    @pytest.mark.parametrize("name", NAMES)
    @pytest.mark.parametrize("seed", [11, 12, 13])
    def test_it_agrees_on_a_randomised_site(self, name: str, seed: int) -> None:
        """The scripted scenes above are the ones somebody thought of. This one is three cameras,
        one of them uncalibrated, with boxes that sometimes run off the bottom of the frame — so
        it exercises the truncation branch, the mixed-calibration branch and the horizon clamp
        together, which is where a port drifts without any single case looking wrong."""
        rng = np.random.default_rng(seed)
        cameras = ["cam-a", "cam-b", "cam-c"]
        tracks: dict[str, list] = {}
        for camera_index, camera in enumerate(cameras):
            group = []
            for slot in range(int(rng.integers(1, 5))):
                left = float(rng.uniform(0.0, 1700.0))
                top = float(rng.uniform(0.0, 700.0))
                # A third of the boxes are deliberately clipped by the bottom edge, which is what
                # sends the foot point through the aspect-ratio extrapolation instead of to the
                # box's own bottom.
                bottom = (
                    1079.0 if rng.random() < 0.34 else top + float(rng.uniform(140.0, 350.0))
                )
                group.append(
                    make_track(
                        camera=camera,
                        track_id=slot + 1,
                        identity=int(rng.integers(0, 3)),
                        view=camera_index,
                        box=(left, top, left + float(rng.uniform(40.0, 120.0)), bottom),
                    )
                )
            tracks[camera] = group
        warp = np.array(
            [[1.2, 0.1, 5.0], [0.0, 1.6, -3.0], [1e-4, 3e-4, 1.0]], dtype=np.float64
        )
        # cam-c has no homography: an uncalibrated camera in a calibrated group is the state a
        # site is in for as long as it takes somebody to click the new camera's points.
        plane = GroundPlane({"cam-a": Homography(warp), "cam-b": Homography(warp * 1.05)})
        observations = make_cluster(tracks).observations
        reference, candidate = both(name, ground_plane=plane)

        assert (
            np.abs(candidate.build(observations) - reference.build(observations)).max() < 1e-5
        )

    def test_a_scene_the_appearance_threshold_rejects_is_never_merge_not_one(self) -> None:
        """Two different people, so every cross-camera pair is below the bar. This is the case
        where an implementation that wrote 1.0 for "no evidence" still looks like a distance
        matrix — and leaves a ruled-out pair merely expensive rather than forbidden."""
        cluster = make_cluster(
            {
                "cam-a": [make_track(camera="cam-a", track_id=1, identity=0)],
                "cam-b": [make_track(camera="cam-b", track_id=1, identity=5)],
            }
        )
        reference, candidate = both("appearance", appearance_threshold=0.99)

        actual = candidate.build(cluster.observations)

        assert actual[0, 1] == pytest.approx(NEVER_MERGE)
        assert np.array_equal(actual, reference.build(cluster.observations))


class TestTheSpatialHalfAgreesPassForPass:
    """The matcher-level comparison above would pass for a compiled projection that happened to
    be cancelled out by a wrong gate, so each stage of the geometry is checked on its own."""

    def scene(self, ground_plane: GroundPlane | None = None):
        observations = two_people_two_cameras().observations
        boxes, sizes, codes, order = instant_arrays(observations)
        matrices, calibration, calibrated = plane_arrays(ground_plane, order)
        native = _c().MtmcSpatialMatcher(
            spatial_threshold=280.0,
            homographies=matrices,
            calibration_sizes=calibration,
            calibrated=calibrated,
        )
        readable = MTMC_MATCHERS.build(
            "spatial", backend=PYTHON, ground_plane=ground_plane or GroundPlane()
        )
        return observations, (boxes, sizes, codes), native, readable

    def test_the_projected_ground_positions_match(self) -> None:
        observations, arrays, native, readable = self.scene(CALIBRATED)

        points, known = native.ground_positions(*arrays)
        expected_points, expected_known = readable.ground_positions(observations)

        assert np.allclose(points, expected_points, equal_nan=True)
        assert np.array_equal(known, expected_known)

    def test_an_uncalibrated_camera_is_nowhere_rather_than_at_the_origin(self) -> None:
        """The origin is a real place on the map. The reference implementation used ``(0, 0)``
        plus a side-list of invalid indices, and one forgotten check leaves every uncalibrated
        camera's tracks coincident with each other — which reads as a crowd, not as a bug."""
        observations, arrays, native, readable = self.scene(None)

        points, known = native.ground_positions(*arrays)

        assert np.isnan(points).all()
        assert not known.any()
        assert np.array_equal(known, readable.ground_positions(observations)[1])

    def test_the_ground_distances_match(self) -> None:
        observations, arrays, native, readable = self.scene(CALIBRATED)

        assert np.allclose(
            native.ground_distances(*arrays), readable.ground_distances(observations), atol=1e-6
        )

    def test_the_gate_falls_open_on_a_pair_it_cannot_judge(self) -> None:
        """The single most consequential line in the spatial half: an uncalibrated camera keeps
        taking part on appearance alone rather than quietly never merging with anyone."""
        observations, arrays, native, readable = self.scene(None)

        gate = native.gate(*arrays)

        assert gate.all()
        assert np.array_equal(gate, readable.gate(observations))

    def test_a_projective_homography_lands_in_the_same_place(self) -> None:
        """An identity homography is a warp that cannot tell a transposed matrix from a correct
        one. This one has a real perspective term, so a row-major/column-major mistake moves
        every point."""
        warp = np.array(
            [[1.4, 0.2, -30.0], [0.1, 1.9, 12.0], [2.0e-4, 5.0e-4, 1.0]], dtype=np.float64
        )
        plane = GroundPlane({"cam-a": Homography(warp), "cam-b": Homography(warp.T)})
        observations, arrays, native, readable = self.scene(plane)

        assert np.allclose(
            native.ground_positions(*arrays)[0],
            readable.ground_positions(observations)[0],
            atol=1e-4,
        )

    def test_a_homography_calibrated_at_another_resolution_is_rescaled_the_same_way(
        self,
    ) -> None:
        """A homography fitted on 1080p stills does not apply to the 720p stream the same camera
        serves at night: the coordinates differ by a factor of 1.5 and the projection lands
        somewhere else on the map, silently. The calibration size is what fixes that, and both
        implementations have to apply it identically or the two disagree only at night."""
        calibrated_at_720p = Homography(
            np.array([[1.1, 0.05, 4.0], [0.0, 1.3, -7.0], [1e-4, 2e-4, 1.0]]),
            camera_width=1280,
            camera_height=720,
        )
        plane = GroundPlane({"cam-a": calibrated_at_720p, "cam-b": calibrated_at_720p})
        observations, arrays, native, readable = self.scene(plane)

        assert np.allclose(
            native.ground_positions(*arrays)[0],
            readable.ground_positions(observations)[0],
            atol=1e-4,
        )

    def test_the_foot_point_of_a_box_the_frame_cut_off_is_extrapolated_identically(
        self,
    ) -> None:
        """A box touching the bottom edge has its feet outside the image, so its bottom-centre is
        somewhere around the waist. Skip the extrapolation and every track in the near field of
        every camera projects metres short of where it is — consistently, which reads as a
        systematic map offset rather than as a bug."""
        boxes = np.array(
            [[100.0, 300.0, 200.0, 700.0], [100.0, 800.0, 200.0, 1079.0]], dtype=np.float32
        )
        heights = np.array([1080.0, 1080.0], dtype=np.float64)

        actual = _c().mtmc_foot_points(boxes, heights, 1.0, 0.25)

        assert np.array_equal(actual, foot_points(boxes, heights))
        # The second box is the truncated one: its foot is 400px (width / 0.25) below its top,
        # not at its own bottom edge.
        assert actual[1, 1] == pytest.approx(800.0 + 400.0)


class TestTheCompiledClustererFindsTheSameClusters:
    """A matrix that differs by 1e-7 is still the same matrix; a cluster boundary that moved is a
    different answer. Compared as a partition, because label numbering is arbitrary."""

    def labels(
        self, matrix: np.ndarray, threshold: float = 0.14
    ) -> tuple[np.ndarray, np.ndarray]:
        readable = AgglomerativeClusterer(distance_threshold=threshold).fit_predict(matrix)
        compiled = NativeCoreClusterer(distance_threshold=threshold).fit_predict(matrix)
        return canonical(readable), canonical(compiled)

    def test_a_scene_with_a_known_answer_gives_two_clusters_on_both_sides(self) -> None:
        """Two identities, each seen twice: tracks 0 and 2 are one object, 1 and 3 the other.
        "It produced clusters" proves nothing — a matcher returning zeros produces clusters, and
        so does one that merges everything — so the expected partition is written out."""
        matcher = MTMC_MATCHERS.build("gated", backend=PYTHON, ground_plane=CALIBRATED)
        matrix = matcher.build(two_people_two_cameras().observations)

        readable, compiled = self.labels(matrix)

        assert np.array_equal(compiled, readable)
        assert compiled[0] == compiled[2] and compiled[1] == compiled[3]
        assert compiled[0] != compiled[1]

    def test_a_never_merge_pair_keeps_two_clusters_apart(self) -> None:
        """One forbidden pair drags a candidate merge's mean distance to ~50 000, which is the
        whole reason the sentinel is a large finite number rather than infinity."""
        matrix = np.array(
            [
                [0.0, 0.01, NEVER_MERGE],
                [0.01, 0.0, 0.01],
                [NEVER_MERGE, 0.01, 0.0],
            ],
            dtype=np.float32,
        )

        readable, compiled = self.labels(matrix)

        assert np.array_equal(compiled, readable)
        assert compiled[0] != compiled[2]

    def test_a_pair_exactly_on_the_threshold_is_grouped_by_both(self) -> None:
        """scipy's distance criterion keeps every merge at *or below* the cut. A strict
        comparison in the compiled version would split a group whose members sit exactly on the
        threshold — and a threshold copied out of a tuning run is made of exactly those."""
        matrix = np.array([[0.0, 0.14], [0.14, 0.0]], dtype=np.float64)

        readable, compiled = self.labels(matrix)

        assert np.array_equal(compiled, readable)
        assert compiled[0] == compiled[1]

    def test_an_empty_instant_and_a_single_track_are_ordinary_input(self) -> None:
        """One visible track is the normal state of a quiet site, and it is also the input scipy
        refuses outright — so it has to be answered before any linkage happens."""
        assert NativeCoreClusterer().fit_predict(np.zeros((0, 0))).shape == (0,)
        assert np.array_equal(NativeCoreClusterer().fit_predict(np.zeros((1, 1))), [0])

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_it_agrees_with_scipy_on_random_matrices(self, seed: int) -> None:
        """The scripted cases above are the ones somebody thought of. Average linkage is where a
        port drifts quietly — the size-weighted update (UPGMA) and the unweighted one (WPGMA)
        are both called "average" and differ only once a cluster has three members."""
        rng = np.random.default_rng(seed)
        for _ in range(40):
            count = int(rng.integers(2, 12))
            matrix = rng.uniform(0.0, 0.4, size=(count, count))
            matrix = 0.5 * (matrix + matrix.T)
            np.fill_diagonal(matrix, 0.0)
            for _ in range(int(rng.integers(0, count))):
                i, j = rng.integers(0, count, size=2)
                if i != j:
                    matrix[i, j] = matrix[j, i] = NEVER_MERGE
            threshold = float(rng.uniform(0.05, 0.3))

            readable, compiled = self.labels(matrix, threshold)

            assert np.array_equal(compiled, readable)


class TestTheGlobalIdsAreTheSame:
    """The end of the pipeline: the real tracker, with the compiled half swapped in.

    The claim a matrix comparison does not imply. The identity assigner is stateful, so one
    disagreed cluster on instant 3 renames things on instant 4 and never converges back — which
    is what an operator would actually see.
    """

    def trackers(
        self, ground_plane: GroundPlane
    ) -> tuple[ClusterMTMCTracker, ClusterMTMCTracker]:
        readable = ClusterMTMCTracker(
            matrix_builder=MTMC_MATCHERS.build(
                "gated", backend=PYTHON, ground_plane=ground_plane
            ),
            clusterer=AgglomerativeClusterer(distance_threshold=0.14),
        )
        compiled = ClusterMTMCTracker(
            matrix_builder=NativeCoreMatcher("gated", ground_plane=ground_plane),
            clusterer=NativeCoreClusterer(distance_threshold=0.14),
        )
        return readable, compiled

    def ids(self, tracker: ClusterMTMCTracker, scene, steps: int) -> list[list[int | None]]:
        return [
            [result.global_id for result in tracker.track(scene(frame_id))]
            for frame_id in range(steps)
        ]

    def test_two_people_over_six_instants_get_the_same_identities(self) -> None:
        readable, compiled = self.trackers(CALIBRATED)

        assert self.ids(compiled, two_people_two_cameras, 6) == self.ids(
            readable, two_people_two_cameras, 6
        )

    def test_one_person_seen_twice_is_one_identity_on_both_sides(self) -> None:
        """The simplest scene with a right answer, asserted rather than merely compared: two
        implementations that both said "two people" would agree and both be wrong."""
        readable, compiled = self.trackers(CALIBRATED)

        compiled_ids = self.ids(compiled, one_person_two_cameras, 5)

        assert compiled_ids == self.ids(readable, one_person_two_cameras, 5)
        assert len(set(compiled_ids[-1])) == 1
        assert compiled_ids[-1][0] is not None

    def test_an_uncalibrated_site_still_agrees(self) -> None:
        """Geometry off, appearance alone. This is the configuration a site runs in on its first
        day, and it is the one where a gate that fell *closed* instead of open would still look
        like a working system — with every camera's identities quietly separate."""
        readable, compiled = self.trackers(GroundPlane())

        assert self.ids(compiled, two_people_two_cameras, 6) == self.ids(
            readable, two_people_two_cameras, 6
        )


class TestTheCompiledMatcherRefusesTheSameInput:
    """A bad threshold has to stop the process at start-up, not at frame 40 000 — and the two
    implementations have to draw the line in the same place, or a config that the readable
    backend rejects boots on the compiled one."""

    def test_an_out_of_range_appearance_threshold_is_refused(self) -> None:
        with pytest.raises(ConfigurationError):
            MTMC_MATCHERS.build("appearance", backend=PYTHON, appearance_threshold=1.5)
        with pytest.raises(ValueError):
            _c().MtmcAppearanceMatcher(appearance_threshold=1.5)

    def test_a_non_positive_spatial_threshold_is_refused(self) -> None:
        empty = (np.zeros((0, 3, 3)), np.zeros((0, 2), np.int32), np.zeros(0, np.uint8))
        with pytest.raises(ConfigurationError):
            MTMC_MATCHERS.build("spatial", backend=PYTHON, spatial_threshold=0.0)
        with pytest.raises(ValueError):
            _c().MtmcSpatialMatcher(
                spatial_threshold=0.0,
                homographies=empty[0],
                calibration_sizes=empty[1],
                calibrated=empty[2],
            )

    def test_a_non_finite_distance_matrix_is_refused_by_the_clusterer(self) -> None:
        """``inf`` is the obvious way to say "never merge these" and it is wrong: average linkage
        computes ``inf - inf`` and gets NaN, which does not fail — it produces a dendrogram whose
        merges are arbitrary. That is why the sentinel is finite, and this is where a builder
        that ignored it finds out."""
        matrix = np.array([[0.0, np.inf], [np.inf, 0.0]], dtype=np.float64)

        with pytest.raises(ValueError):
            NativeCoreClusterer().fit_predict(matrix)

    def test_a_camera_code_per_track_is_required(self) -> None:
        """The same-camera exclusion is the one rule this matrix must never lose, so the codes
        are not optional and a short array is not silently padded."""
        with pytest.raises(ValueError):
            _c().MtmcAppearanceMatcher().build(
                np.zeros((3, 3), dtype=np.float32), np.zeros(2, dtype=np.int32)
            )
