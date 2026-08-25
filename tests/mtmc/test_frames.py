"""The input unit's contract: the key, the validation, the flattening."""

from __future__ import annotations

import numpy as np
import pytest

from shipvision.errors import ConfigurationError
from shipvision.mtmc import CameraTracks, FrameTrackCluster, TrackKey
from shipvision.types import FrameTag, Track
from tests.mtmc.conftest import FRAME_HEIGHT, FRAME_WIDTH, make_cluster, make_track


class TestTrackKey:
    """The cross-camera key is a type, not a formatted string."""

    def test_it_is_hashable_orderable_and_a_plain_tuple(self) -> None:
        """It goes into dicts, gets sorted for deterministic tie-breaks, and lands in
        GlobalTrack.members, which is declared as tuple[tuple[str, int], ...]."""
        a = TrackKey("cam-a", 1)
        b = TrackKey("cam-b", 1)

        assert {a: 1}[TrackKey("cam-a", 1)] == 1
        assert sorted([b, a]) == [a, b]
        assert a == ("cam-a", 1)
        assert isinstance(a, tuple)

    def test_the_same_track_id_on_two_cameras_is_two_different_keys(self) -> None:
        """The whole reason the key is a pair: track 7 means nothing without its camera."""
        assert TrackKey("cam-a", 7) != TrackKey("cam-b", 7)

    def test_a_camera_id_containing_the_reference_separator_still_round_trips(self) -> None:
        """The reference keyed on f"{camera}_{track_id}" and split it back apart. A camera
        called "quay_3" then parsed to camera "quay" and track "3_id"."""
        key = TrackKey("quay_3", 12)

        assert key.camera_id == "quay_3"
        assert key.track_id == 12


class TestFrameTrackClusterValidation:
    """A malformed instant fails at construction, not at frame 40 000."""

    def test_a_view_without_frame_dimensions_is_refused(self) -> None:
        """Zero dimensions would make the height gate, the truncation test and the homography
        scaling silently wrong rather than loudly absent."""
        with pytest.raises(ConfigurationError, match="positive frame dimensions"):
            CameraTracks(
                tag=FrameTag(camera_id="cam-a", frame_id=0), tracks=(), height=0, width=0
            )

    def test_a_track_tagged_with_another_camera_is_refused(self) -> None:
        """(camera_id, frame_id) survives every path — including the path where somebody
        groups tracks by hand and gets one wrong."""
        stray = make_track(camera="cam-b", track_id=1, identity=0)

        with pytest.raises(ConfigurationError, match="is tagged camera"):
            CameraTracks(
                tag=FrameTag(camera_id="cam-a", frame_id=0),
                tracks=(stray,),
                height=FRAME_HEIGHT,
                width=FRAME_WIDTH,
            )

    def test_one_camera_cannot_report_the_same_track_twice(self) -> None:
        with pytest.raises(ConfigurationError, match="twice in one frame"):
            CameraTracks(
                tag=FrameTag(camera_id="cam-a", frame_id=0),
                tracks=(
                    make_track(camera="cam-a", track_id=1, identity=0),
                    make_track(camera="cam-a", track_id=1, identity=1),
                ),
                height=FRAME_HEIGHT,
                width=FRAME_WIDTH,
            )

    def test_one_camera_cannot_appear_twice_in_one_instant(self) -> None:
        """Two frames from one camera in one instant are two instants, and merging them makes
        the same-camera exclusion mask leak."""
        view = CameraTracks(
            tag=FrameTag(camera_id="cam-a", frame_id=0),
            tracks=(),
            height=FRAME_HEIGHT,
            width=FRAME_WIDTH,
        )

        with pytest.raises(ConfigurationError, match="appears twice"):
            FrameTrackCluster.from_views([view, view])

    def test_filter_checks_its_length(self) -> None:
        cluster = make_cluster({"cam-a": [make_track(camera="cam-a", track_id=1, identity=0)]})

        with pytest.raises(ConfigurationError, match="entries for"):
            cluster.filter([True, False])


class TestFrameTrackClusterFlattening:
    """One synchronised instant becomes one ordered list of observations."""

    def test_observations_are_flattened_in_view_then_track_order(self) -> None:
        cluster = make_cluster(
            {
                "cam-a": [
                    make_track(camera="cam-a", track_id=5, identity=0),
                    make_track(camera="cam-a", track_id=9, identity=1),
                ],
                "cam-b": [make_track(camera="cam-b", track_id=2, identity=0)],
            }
        )

        assert cluster.keys == (
            TrackKey("cam-a", 5),
            TrackKey("cam-a", 9),
            TrackKey("cam-b", 2),
        )
        assert len(cluster) == 3
        assert cluster.cameras == ("cam-a", "cam-b")

    def test_an_observation_carries_the_frame_size_its_box_was_measured_in(self) -> None:
        cluster = make_cluster({"cam-a": [make_track(camera="cam-a", track_id=1, identity=0)]})
        observation = cluster.observations[0]

        assert observation.frame_height == FRAME_HEIGHT
        assert observation.frame_width == FRAME_WIDTH
        assert observation.height_fraction == pytest.approx(400.0 / FRAME_HEIGHT)

    def test_an_empty_instant_is_ordinary_input(self) -> None:
        """Every camera can be watching nothing. That is a quiet night, not an error."""
        cluster = FrameTrackCluster()

        assert len(cluster) == 0
        assert cluster.observations == ()
        assert cluster.keys == ()

    def test_the_tracks_are_not_copied(self) -> None:
        """This is built a thousand times a second; copying a few hundred embeddings each
        time would cost more than the association it feeds."""
        track = make_track(camera="cam-a", track_id=1, identity=0)
        cluster = make_cluster({"cam-a": [track]})

        assert cluster.observations[0].track is track

    def test_from_tracks_groups_by_camera(self) -> None:
        tracks = [
            make_track(camera="cam-a", track_id=1, identity=0),
            make_track(camera="cam-b", track_id=1, identity=0),
            make_track(camera="cam-a", track_id=2, identity=1),
        ]

        cluster = FrameTrackCluster.from_tracks(tracks, height=FRAME_HEIGHT, width=FRAME_WIDTH)

        assert sorted(cluster.cameras) == ["cam-a", "cam-b"]
        assert len(cluster) == 3

    def test_filter_keeps_input_order(self) -> None:
        cluster = make_cluster(
            {
                "cam-a": [
                    make_track(camera="cam-a", track_id=1, identity=0),
                    make_track(camera="cam-a", track_id=2, identity=1),
                ]
            }
        )

        kept = cluster.filter([False, True])

        assert [o.key.track_id for o in kept] == [2]

    def test_a_track_with_no_embedding_is_still_a_valid_observation(self) -> None:
        """MTMC refuses it later, with a typed error naming the missing stage. It is not the
        input type's job to insist on a re-ID model having run."""
        track = Track(
            track_id=1,
            box=np.array([0.0, 0.0, 10.0, 200.0], dtype=np.float32),
            tag=FrameTag(camera_id="cam-a", frame_id=0),
        )

        cluster = make_cluster({"cam-a": [track]})

        assert cluster.observations[0].embedding is None
