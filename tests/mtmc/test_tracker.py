"""The four components together, on scenes with a stateable right answer."""

from __future__ import annotations

import numpy as np
import pytest

from shipvision.errors import ConfigurationError
from shipvision.mtmc import (
    MTMC,
    AppearanceMatcher,
    ClusterMTMCTracker,
    FrameTrackCluster,
    GroundPlane,
)
from shipvision.types import GlobalTrack
from tests.mtmc.conftest import (
    identity_homography,
    make_cluster,
    make_track,
    one_person_two_cameras,
    two_people_two_cameras,
    view_of,
)

pytest.importorskip("scipy", reason="the clusterer is scipy's job, not ours")


def assigned(results: list[GlobalTrack]) -> list[GlobalTrack]:
    return [result for result in results if result.is_assigned]


def identities(results: list[GlobalTrack]) -> set[int]:
    return {result.global_id for result in results if result.is_assigned}


def warm(tracker: ClusterMTMCTracker, scene, steps: int = 3) -> list[GlobalTrack]:
    """Feed ``steps`` instants of ``scene`` so tracks clear the tentative-age gate."""
    results: list[GlobalTrack] = []
    for frame_id in range(steps):
        results = tracker.track(scene(frame_id))
    return results


def tracker(**kwargs) -> ClusterMTMCTracker:
    """A tracker with the gate wide open unless a test says otherwise."""
    kwargs.setdefault("min_hits", 1)
    return ClusterMTMCTracker(**kwargs)


class TestCrossCameraIdentity:
    """The scenes the whole package exists for."""

    def test_one_person_on_two_cameras_becomes_one_global_id(self) -> None:
        results = tracker().track(one_person_two_cameras())

        assert len(results) == 2
        assert len(identities(results)) == 1
        assert all(result.is_assigned for result in results)

    def test_two_people_on_two_cameras_become_two_global_ids(self) -> None:
        results = tracker().track(two_people_two_cameras())

        assert len(results) == 4
        assert len(identities(results)) == 2
        assert all(result.global_id is not None for result in results)

    def test_two_tracks_in_one_camera_never_share_a_global_id(self) -> None:
        """End to end, with identical embeddings: the mask is the only thing standing between
        MTMC and being a within-camera deduplicator."""
        same = view_of(0)
        scene = make_cluster(
            {
                "cam-a": [
                    make_track(camera="cam-a", track_id=1, identity=0, embedding=same),
                    make_track(camera="cam-a", track_id=2, identity=0, embedding=same),
                ]
            }
        )

        results = tracker().track(scene)

        assert len(identities(results)) == 2

    def test_an_identity_keeps_its_global_id_across_instants(self) -> None:
        instance = tracker()
        first = instance.track(one_person_two_cameras(frame_id=0))
        later = instance.track(one_person_two_cameras(frame_id=1))

        assert identities(first) == identities(later)

    def test_members_name_every_camera_currently_holding_the_identity(self) -> None:
        results = tracker().track(one_person_two_cameras())

        members = {result.members for result in results}

        assert members == {(("cam-a", 1), ("cam-b", 1))}

    def test_a_camera_joining_late_adopts_the_established_id(self) -> None:
        instance = tracker()
        established = identities(instance.track(one_person_two_cameras(frame_id=0)))

        third = make_cluster(
            {
                "cam-a": [make_track(camera="cam-a", track_id=1, identity=0, view=0)],
                "cam-b": [make_track(camera="cam-b", track_id=1, identity=0, view=1)],
                "cam-c": [make_track(camera="cam-c", track_id=1, identity=0, view=2)],
            },
            frame_id=1,
        )
        results = instance.track(third)

        assert identities(results) == established
        assert len(assigned(results)) == 3


class TestOutputContract:
    """Every input track comes back, and unassigned means None."""

    def test_a_gated_out_track_comes_back_with_global_id_none(self) -> None:
        """Reported rather than dropped. A caller that has to diff two lists to discover which
        of its tracks were judged too new is a caller that will not."""
        instance = ClusterMTMCTracker(min_hits=3)

        results = instance.track(one_person_two_cameras())

        assert len(results) == 2
        assert all(result.global_id is None for result in results)
        assert all(result.metadata["gated"] for result in results)
        assert not any(result.is_assigned for result in results)

    def test_a_track_becomes_assigned_once_it_has_enough_consecutive_hits(self) -> None:
        """The other half: the gate delays, it does not discard."""
        instance = ClusterMTMCTracker(min_hits=3)

        first = instance.track(one_person_two_cameras(frame_id=0))
        second = instance.track(one_person_two_cameras(frame_id=1))
        third = instance.track(one_person_two_cameras(frame_id=2))

        assert not assigned(first)
        assert not assigned(second)
        assert len(assigned(third)) == 2

    def test_a_track_too_small_to_embed_is_never_admitted(self) -> None:
        instance = ClusterMTMCTracker(min_hits=1, min_height_fraction=0.2)
        tiny = make_cluster(
            {
                "cam-a": [
                    make_track(
                        camera="cam-a", track_id=1, identity=0, box=(0.0, 0.0, 20.0, 80.0)
                    )
                ],
                "cam-b": [make_track(camera="cam-b", track_id=1, identity=0, view=1)],
            }
        )

        results = warm(instance, lambda f: tiny, steps=5)

        assert results[0].global_id is None  # the tiny one, on cam-a
        assert results[1].is_assigned  # the ordinary one, on cam-b

    def test_results_are_returned_in_input_order(self) -> None:
        """So a caller can zip them against what it passed in."""
        scene = two_people_two_cameras()

        results = tracker().track(scene)

        assert [result.track for result in results] == [
            observation.track for observation in scene.observations
        ]

    def test_the_cluster_label_is_carried_so_the_grouping_is_visible(self) -> None:
        """The only way to see, from the output, which tracks were grouped in this call."""
        results = tracker().track(one_person_two_cameras())

        labels = {result.cluster_id for result in results}

        assert labels == {"0"}

    def test_an_empty_instant_returns_nothing_and_does_not_raise(self) -> None:
        assert tracker().track(FrameTrackCluster()) == []

    def test_a_single_visible_track_is_the_normal_case(self) -> None:
        """One track means no matrix and no clustering — scipy refuses a 1x1 condensed
        matrix, and a quiet site is not an edge case."""
        lonely = make_cluster({"cam-a": [make_track(camera="cam-a", track_id=1, identity=0)]})

        results = tracker().track(lonely)

        assert len(assigned(results)) == 1

    def test_being_handed_one_camera_at_a_time_is_refused(self) -> None:
        """Which is the mistake that turns cross-camera association into single-camera
        deduplication, and it would otherwise fail silently."""
        with pytest.raises(ConfigurationError, match="synchronised instant"):
            tracker().track([make_track(camera="cam-a", track_id=1, identity=0)])


class TestSpatialGatingEndToEnd:
    """Geometry vetoing appearance, through the whole pipeline."""

    def test_two_look_alikes_far_apart_on_the_ground_stay_two_identities(self) -> None:
        plane = GroundPlane({"cam-a": identity_homography(), "cam-b": identity_homography()})
        instance = tracker(ground_plane=plane, spatial_threshold=100.0)
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
        )

        assert len(identities(instance.track(far))) == 2

    def test_the_same_pair_standing_together_is_one_identity(self) -> None:
        plane = GroundPlane({"cam-a": identity_homography(), "cam-b": identity_homography()})
        instance = tracker(ground_plane=plane, spatial_threshold=100.0)
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
        )

        assert len(identities(instance.track(near))) == 1

    def test_an_uncalibrated_camera_still_takes_part(self) -> None:
        """It falls back to appearance rather than raising or quietly never merging."""
        instance = tracker(
            ground_plane=GroundPlane({"cam-a": identity_homography()}),
            spatial_threshold=1.0,
        )

        results = instance.track(one_person_two_cameras())

        assert len(identities(results)) == 1


class TestTtlEndToEnd:
    """Leaving and coming back, through the tracker."""

    def test_returning_inside_the_ttl_keeps_the_id(self) -> None:
        instance = tracker(max_age=5)
        before = identities(instance.track(one_person_two_cameras(frame_id=0)))

        for frame_id in range(1, 6):
            instance.track(make_cluster({}, frame_id=frame_id))
        after = identities(instance.track(one_person_two_cameras(frame_id=6)))

        assert after == before

    def test_returning_after_the_ttl_gets_a_new_id(self) -> None:
        instance = tracker(max_age=5)
        before = identities(instance.track(one_person_two_cameras(frame_id=0)))

        for frame_id in range(1, 8):
            instance.track(make_cluster({}, frame_id=frame_id))
        after = identities(instance.track(one_person_two_cameras(frame_id=8)))

        assert after != before
        assert min(after) > max(before)

    def test_reset_forgets_identities_without_reusing_their_ids(self) -> None:
        instance = tracker()
        before = identities(instance.track(one_person_two_cameras(frame_id=0)))

        instance.reset()
        after = identities(instance.track(one_person_two_cameras(frame_id=1)))

        assert not (before & after)


class TestBoundedGrowthEndToEnd:
    """A process here runs for weeks. This is the shape of evidence that claim needs."""

    @pytest.mark.slow
    def test_five_thousand_instants_of_churning_identities_stay_flat(self) -> None:
        instance = ClusterMTMCTracker(
            min_hits=2, max_age=5, capacity=64, max_tracks=128, validate_every_step=False
        )

        def churn(start: int, count: int) -> None:
            for step in range(start, start + count):
                # Each identity lives for three instants under one track id, then both the
                # identity and the track id are replaced: continuous turnover, which is what
                # a gate or a quay looks like over an hour.
                epoch = step // 3
                instance.track(
                    make_cluster(
                        {
                            "cam-a": [
                                make_track(
                                    camera="cam-a",
                                    track_id=epoch,
                                    identity=epoch % 89,
                                    view=0,
                                    frame_id=step,
                                )
                            ],
                            "cam-b": [
                                make_track(
                                    camera="cam-b",
                                    track_id=epoch,
                                    identity=epoch % 89,
                                    view=1,
                                    frame_id=step,
                                )
                            ],
                        },
                        frame_id=step,
                    )
                )

        churn(0, 500)
        early = instance.sizes()
        churn(500, 4500)
        late = instance.sizes()

        assert late == early
        assert late["owner"] <= 2 * 6  # two cameras, at most max_age + 1 instants alive
        assert late["gate_hits"] <= 2
        assert late["global_ids"] <= 64
        assert instance.assigner.issued > 1000  # it kept discovering, it did not stall
        instance.assigner.validate()


class TestConstruction:
    """A config typo stops the process at start-up, not at frame 40 000."""

    def test_the_tracker_is_selectable_by_name_and_by_alias(self) -> None:
        assert isinstance(MTMC.build("cluster"), ClusterMTMCTracker)
        assert isinstance(MTMC.build("vtx"), ClusterMTMCTracker)

    def test_a_named_builder_only_receives_the_arguments_it_accepts(self) -> None:
        """ "appearance" has no geometry and therefore no spatial threshold; forwarding one
        would make it unselectable from config."""
        instance = ClusterMTMCTracker(matrix_builder="appearance", appearance_threshold=0.5)

        assert isinstance(instance.builder, AppearanceMatcher)
        assert instance.builder.appearance_threshold == 0.5

    def test_a_prebuilt_component_is_used_as_given(self) -> None:
        builder = AppearanceMatcher(appearance_threshold=0.42)

        instance = ClusterMTMCTracker(matrix_builder=builder, appearance_threshold=0.99)

        assert instance.builder is builder

    def test_an_unknown_builder_name_fails_at_construction(self) -> None:
        with pytest.raises(ConfigurationError, match="unknown mtmc matcher"):
            ClusterMTMCTracker(matrix_builder="telepathy")

    def test_a_nonsense_builder_type_fails_at_construction(self) -> None:
        with pytest.raises(ConfigurationError, match="registered name or a BaseMatcher"):
            ClusterMTMCTracker(matrix_builder=np.eye(3))

    def test_an_out_of_range_threshold_fails_at_construction(self) -> None:
        with pytest.raises(ConfigurationError):
            ClusterMTMCTracker(min_height_fraction=1.5)
