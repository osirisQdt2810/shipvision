"""The distance matrices — and the one rule that makes cross-camera tracking cross-camera."""

from __future__ import annotations

import numpy as np
import pytest

from shipvision.errors import ConfigurationError, DimensionMismatchError, TrackingError
from shipvision.mtmc import (
    MATRIX_BUILDERS,
    NEVER_MERGE,
    AppearanceMatrixBuilder,
    BaseMatrixBuilder,
    GatedMatrixBuilder,
    GroundPlane,
    Homography,
    SpatialMatrixBuilder,
    foot_points,
)
from tests.mtmc.conftest import (
    FRAME_HEIGHT,
    identity_homography,
    make_cluster,
    make_track,
    view_of,
)


def flat_plane() -> GroundPlane:
    """Both test cameras mapping image pixels straight onto ground units."""
    return GroundPlane({"cam-a": identity_homography(), "cam-b": identity_homography()})


def build(name: str, **kwargs: object) -> BaseMatrixBuilder:
    """A builder by name, given a ground plane only if it has somewhere to put one."""
    if name != "appearance":
        kwargs.setdefault("ground_plane", flat_plane())
    return MATRIX_BUILDERS.build(name, **kwargs)


def observations(*specs: tuple[str, int, int]) -> tuple:
    """``(camera, track_id, identity)`` triples to a flat observation tuple."""
    by_camera: dict[str, list] = {}
    for camera, track_id, identity in specs:
        by_camera.setdefault(camera, []).append(
            make_track(camera=camera, track_id=track_id, identity=identity)
        )
    return make_cluster(by_camera).observations


def placed(*specs: tuple[str, int, int, float]) -> tuple:
    """``(camera, track_id, identity, x)`` — the same, with a box at a chosen x position.

    With the identity homography from conftest, ``x`` *is* the ground-plane x coordinate of
    the foot point, so a spatial expectation can be read off the test rather than derived.
    """
    by_camera: dict[str, list] = {}
    for camera, track_id, identity, x in specs:
        by_camera.setdefault(camera, []).append(
            make_track(
                camera=camera,
                track_id=track_id,
                identity=identity,
                box=(x - 50.0, 300.0, x + 50.0, 700.0),
            )
        )
    return make_cluster(by_camera).observations


class TestSameCameraExclusion:
    """The single most important claim in the package, asserted for every builder.

    Two tracks in one camera view are two different objects by definition — if they were the
    same object, the single-camera tracker had one job and failed at it. Merge them anyway and
    MTMC silently becomes a within-camera deduplicator: every count drops, every metric
    improves, and the system is worse. A builder that forgets the mask produces entirely
    plausible output, which is why this is checked per builder rather than once.
    """

    @pytest.mark.parametrize("name", MATRIX_BUILDERS.names())
    def test_two_tracks_in_one_camera_can_never_merge(self, name: str) -> None:
        builder = build(name)
        # One camera, identical embeddings, identical positions: nothing but the mask can
        # keep these two apart.
        same = view_of(0)
        pair = make_cluster(
            {
                "cam-a": [
                    make_track(camera="cam-a", track_id=1, identity=0, embedding=same),
                    make_track(camera="cam-a", track_id=2, identity=0, embedding=same),
                ]
            }
        ).observations

        distances = builder.build(pair)

        assert distances[0, 1] == pytest.approx(NEVER_MERGE)
        assert distances[1, 0] == pytest.approx(NEVER_MERGE)

    @pytest.mark.parametrize("name", MATRIX_BUILDERS.names())
    def test_the_identical_pair_across_two_cameras_does_merge(self, name: str) -> None:
        """The other half. Without it the test above would pass on a builder that refuses
        every pair, which is the failure mode that looks like caution."""
        builder = build(name)
        same = view_of(0)
        pair = make_cluster(
            {
                "cam-a": [make_track(camera="cam-a", track_id=1, identity=0, embedding=same)],
                "cam-b": [make_track(camera="cam-b", track_id=1, identity=0, embedding=same)],
            }
        ).observations

        assert builder.build(pair)[0, 1] < 0.01


class TestMatrixContract:
    """What every clusterer is entitled to assume about every builder's output."""

    @pytest.mark.parametrize("name", MATRIX_BUILDERS.names())
    def test_the_matrix_is_symmetric_zero_diagonal_and_finite(self, name: str) -> None:
        """scipy refuses a non-finite condensed matrix outright, and its symmetry check is
        exact."""
        builder = build(name)
        obs = observations(("cam-a", 1, 0), ("cam-b", 1, 0), ("cam-a", 2, 1), ("cam-b", 2, 1))

        distances = builder.build(obs)

        assert distances.shape == (4, 4)
        assert np.all(np.isfinite(distances))
        assert np.allclose(distances, distances.T)
        assert np.allclose(np.diag(distances), 0.0)

    @pytest.mark.parametrize("name", MATRIX_BUILDERS.names())
    def test_an_empty_instant_gives_a_zero_by_zero_matrix(self, name: str) -> None:
        """(0, 0), not (0,). An instant with no tracks is ordinary input, and the wrong shape
        turns it into an IndexError three frames later."""
        assert build(name).build(()).shape == (0, 0)


class TestAppearanceMatrix:
    """Cosine similarity, hard-thresholded, on embeddings that behave like real ones."""

    def test_two_views_of_one_object_are_close_and_two_objects_never_merge(self) -> None:
        builder = AppearanceMatrixBuilder(appearance_threshold=0.5)
        obs = observations(("cam-a", 1, 0), ("cam-b", 1, 0), ("cam-b", 2, 1))

        distances = builder.build(obs)

        assert distances[0, 1] < 0.1
        assert distances[0, 2] == pytest.approx(NEVER_MERGE)

    def test_similarity_below_the_threshold_becomes_never_merge_not_a_short_distance(
        self,
    ) -> None:
        """ "Scored 0.8, below the bar" and "same camera" both mean do-not-group. Expressing
        the first as a distance of 0.2 would let average linkage merge it once a threshold
        moved.

        Built from two vectors whose cosine is exactly 0.8 rather than from the fixtures, so
        the number the threshold is compared against is visible in the test.
        """
        first = np.zeros(64, dtype=np.float32)
        first[0] = 1.0
        second = np.zeros(64, dtype=np.float32)
        second[0], second[1] = 0.8, np.sqrt(1.0 - 0.64)
        pair = make_cluster(
            {
                "cam-a": [make_track(camera="cam-a", track_id=1, identity=0, embedding=first)],
                "cam-b": [make_track(camera="cam-b", track_id=1, identity=0, embedding=second)],
            }
        ).observations

        assert AppearanceMatrixBuilder(appearance_threshold=0.9).build(pair)[
            0, 1
        ] == pytest.approx(NEVER_MERGE)
        # And the other half: just under the cosine, it is an ordinary short distance.
        assert AppearanceMatrixBuilder(appearance_threshold=0.7).build(pair)[
            0, 1
        ] == pytest.approx(0.2, abs=1e-5)

    def test_an_un_normalised_embedding_gives_the_same_answer_as_a_normalised_one(self) -> None:
        """The contract says embeddings arrive normalised; the builder does not bet the site
        on every upstream stage having honoured it. An un-normalised vector turns cosine
        similarity into a dot product of arbitrary scale — which does not fail, it just makes
        every threshold in the package mean something different."""
        builder = AppearanceMatrixBuilder(appearance_threshold=0.5)
        unit = view_of(0)
        pair = make_cluster(
            {
                "cam-a": [make_track(camera="cam-a", track_id=1, identity=0, embedding=unit)],
                "cam-b": [
                    make_track(camera="cam-b", track_id=1, identity=0, embedding=unit * 17.0)
                ],
            }
        ).observations

        assert builder.build(pair)[0, 1] == pytest.approx(0.0, abs=1e-5)

    def test_a_track_without_an_embedding_raises_and_names_the_missing_stage(self) -> None:
        builder = AppearanceMatrixBuilder()
        obs = observations(("cam-a", 1, 0), ("cam-b", 1, 0))
        obs[1].track.embedding = None

        with pytest.raises(TrackingError, match="needs an embedding on every track"):
            builder.build(obs)

    def test_embeddings_of_two_widths_in_one_group_are_a_typed_failure(self) -> None:
        """Two cameras running different re-ID models. The only other symptom is a similarity
        matrix that cannot be formed — or worse, one a broadcast quietly produced."""
        builder = AppearanceMatrixBuilder()
        obs = observations(("cam-a", 1, 0), ("cam-b", 1, 0))
        obs[1].track.embedding = np.ones(7, dtype=np.float32)

        with pytest.raises(DimensionMismatchError, match="not running the same re-ID model"):
            builder.build(obs)

    def test_the_threshold_is_validated_at_construction(self) -> None:
        with pytest.raises(ConfigurationError, match=r"\[-1, 1\]"):
            AppearanceMatrixBuilder(appearance_threshold=1.5)


class TestFootPoint:
    """Where an object meets the ground, including when the frame cut its feet off."""

    def test_the_foot_point_of_an_untruncated_box_is_its_bottom_centre(self) -> None:
        point = foot_points(np.array([[100.0, 300.0, 200.0, 700.0]]), np.array([FRAME_HEIGHT]))

        assert point[0] == pytest.approx([150.0, 700.0])

    def test_a_box_clipped_by_the_bottom_edge_has_its_feet_extrapolated_below_it(self) -> None:
        """The common near-field case. The box stops at the frame edge but the person does
        not, so the bottom-centre is around their waist. Skip the aspect test and every
        near-field track on every camera projects metres short of where it is — consistently,
        which reads as a map offset rather than as a bug."""
        clipped = np.array([[100.0, 900.0, 200.0, 1080.0]])  # 100 wide, ends at the edge

        point = foot_points(clipped, np.array([FRAME_HEIGHT]), aspect_ratio=0.25)

        # 100px wide / 0.25 = a 400px-tall person, so the feet are 400px below the box top.
        assert point[0] == pytest.approx([150.0, 1300.0])
        assert point[0][1] > FRAME_HEIGHT

    def test_mismatched_lengths_are_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="frame heights"):
            foot_points(np.zeros((2, 4)), np.array([FRAME_HEIGHT]))


class TestSpatialMatrix:
    """Ground-plane geometry on its own."""

    def test_distance_is_measured_on_the_plane_not_in_the_image(self) -> None:
        builder = SpatialMatrixBuilder(ground_plane=flat_plane())
        obs = placed(("cam-a", 1, 0, 100.0), ("cam-b", 1, 0, 300.0))

        assert builder.ground_distances(obs)[0, 1] == pytest.approx(200.0, rel=1e-4)

    def test_an_uncalibrated_camera_yields_unknowable_rather_than_the_origin(self) -> None:
        """The reference put uncalibrated tracks at (0, 0) and kept a side-list of indices.
        The origin is a real place on the map: one forgotten check and every uncalibrated
        camera's tracks are coincident with each other."""
        builder = SpatialMatrixBuilder(
            ground_plane=GroundPlane({"cam-a": identity_homography()})
        )
        obs = placed(("cam-a", 1, 0, 100.0), ("cam-b", 1, 0, 100.0))

        points, known = builder.ground_positions(obs)

        assert known.tolist() == [True, False]
        assert np.isnan(points[1]).all()
        assert not np.isfinite(builder.ground_distances(obs)[0, 1])

    def test_clustering_on_position_alone_refuses_to_merge_what_it_cannot_judge(self) -> None:
        """The opposite of what the gate does with the same input, and both are right: with no
        other evidence in play, "I cannot tell" must not become "merge them"."""
        builder = SpatialMatrixBuilder(
            ground_plane=GroundPlane({"cam-a": identity_homography()})
        )
        obs = placed(("cam-a", 1, 0, 100.0), ("cam-b", 1, 0, 100.0))

        assert builder.build(obs)[0, 1] == pytest.approx(NEVER_MERGE)

    def test_a_resolution_change_is_rescaled_into_the_calibrated_domain(self) -> None:
        """A homography fitted on 1080p stills does not apply to the 720p night stream: the
        pixel coordinates differ by a factor of 1.5 and the projection lands somewhere else,
        silently."""
        plane = GroundPlane(
            {"cam-a": Homography(matrix=np.eye(3), camera_width=1920, camera_height=1080)}
        )
        builder = SpatialMatrixBuilder(ground_plane=plane)
        half = make_cluster(
            {
                "cam-a": [
                    make_track(
                        camera="cam-a",
                        track_id=1,
                        identity=0,
                        box=(450.0, 100.0, 510.0, 340.0),
                    )
                ]
            },
            height=540,
            width=960,
        ).observations

        points, known = builder.ground_positions(half)

        assert known.tolist() == [True]
        # Foot point (480, 340) in a 960x540 frame is (960, 680) in the calibrated 1920x1080.
        assert points[0] == pytest.approx([960.0, 680.0], rel=1e-4)

    def test_parameters_are_validated_at_construction(self) -> None:
        with pytest.raises(ConfigurationError, match="spatial_threshold must be positive"):
            SpatialMatrixBuilder(spatial_threshold=0.0)
        with pytest.raises(ConfigurationError, match="aspect_ratio"):
            SpatialMatrixBuilder(aspect_ratio=0.0)


class TestSpatialGating:
    """Appearance vetoed by geometry: the production builder, and its fallbacks."""

    def test_two_identical_looking_people_far_apart_do_not_merge(self) -> None:
        """The whole point of the gate. Two crew in identical overalls score high on
        appearance from any model; they are 200 ground units apart, so they are two people."""
        builder = GatedMatrixBuilder(
            ground_plane=flat_plane(), appearance_threshold=0.5, spatial_threshold=100.0
        )
        same = view_of(0)
        far = make_cluster(
            {
                "cam-a": [
                    make_track(
                        camera="cam-a",
                        track_id=1,
                        identity=0,
                        embedding=same,
                        box=(50.0, 300.0, 150.0, 700.0),
                    )
                ],
                "cam-b": [
                    make_track(
                        camera="cam-b",
                        track_id=1,
                        identity=0,
                        embedding=same,
                        box=(250.0, 300.0, 350.0, 700.0),
                    )
                ],
            }
        ).observations

        assert builder.build(far)[0, 1] == pytest.approx(NEVER_MERGE)

    def test_the_same_pair_standing_two_units_apart_does_merge(self) -> None:
        """The other half of the assertion above. Without it the test would pass on a gate
        that rejects everything."""
        builder = GatedMatrixBuilder(
            ground_plane=flat_plane(), appearance_threshold=0.5, spatial_threshold=100.0
        )
        same = view_of(0)
        near = make_cluster(
            {
                "cam-a": [
                    make_track(
                        camera="cam-a",
                        track_id=1,
                        identity=0,
                        embedding=same,
                        box=(50.0, 300.0, 150.0, 700.0),
                    )
                ],
                "cam-b": [
                    make_track(
                        camera="cam-b",
                        track_id=1,
                        identity=0,
                        embedding=same,
                        box=(52.0, 300.0, 152.0, 700.0),
                    )
                ],
            }
        ).observations

        assert builder.build(near)[0, 1] < 0.01

    def test_an_uncalibrated_camera_degrades_to_appearance_instead_of_excluding_itself(
        self,
    ) -> None:
        """A new camera goes live before anyone clicks its calibration points, and a PTZ
        camera invalidates its own the moment it moves. Excluding it is the quiet failure: its
        identities simply never merge with anyone and nothing in the metrics says so."""
        builder = GatedMatrixBuilder(
            ground_plane=GroundPlane({"cam-a": identity_homography()}),
            appearance_threshold=0.5,
            spatial_threshold=1.0,  # so tight that any judged pair would be rejected
        )
        obs = placed(("cam-a", 1, 0, 100.0), ("cam-b", 1, 0, 900.0))

        assert builder.build(obs)[0, 1] < 0.1

    def test_a_gate_with_no_homographies_is_exactly_the_appearance_builder(self) -> None:
        """Which is what makes "gated" a safe default: an uncalibrated site gets
        appearance-only behaviour without a config change."""
        gated = GatedMatrixBuilder(appearance_threshold=0.5)
        plain = AppearanceMatrixBuilder(appearance_threshold=0.5)
        obs = observations(("cam-a", 1, 0), ("cam-b", 1, 0), ("cam-b", 2, 1))

        assert np.allclose(gated.build(obs), plain.build(obs))
