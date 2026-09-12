"""Which tracks are worth asking about, and why the order of the two gates matters."""

from __future__ import annotations

import pytest

from shipvision.errors import ConfigurationError
from shipvision.mtmc import ObservationGate, TrackKey
from shipvision.mtmc.frames import FrameTrackCluster
from tests.mtmc.conftest import make_cluster, make_track

BIG_BOX = (100.0, 300.0, 200.0, 700.0)  # 400px of 1080 -> 0.370 of frame height
SMALL_BOX = (100.0, 300.0, 120.0, 380.0)  # 80px of 1080 -> 0.074 of frame height


def instant(*boxes: tuple[str, int, tuple[float, float, float, float]]) -> FrameTrackCluster:
    """The scene kept whole, because the roster lives on the cluster and not on the tracks.

    A camera named with no boxes is an EMPTY VIEW: it reported this instant and saw nothing,
    which the observations alone cannot express.
    """
    by_camera: dict[str, list] = {}
    for camera, track_id, box in boxes:
        by_camera.setdefault(camera, [])
        if track_id:
            by_camera[camera].append(
                make_track(camera=camera, track_id=track_id, identity=0, box=box)
            )
    return make_cluster(by_camera)


def observations(*boxes: tuple[str, int, tuple[float, float, float, float]]) -> tuple:
    return instant(*boxes).observations


class TestHeightGate:
    """A crop too small to embed produces a confident embedding anyway."""

    def test_a_box_below_the_fraction_is_dropped(self) -> None:
        gate = ObservationGate(min_hits=1, min_height_fraction=0.1)

        admitted = gate.filter(observations(("cam-a", 1, SMALL_BOX), ("cam-a", 2, BIG_BOX)))

        assert [o.key.track_id for o in admitted] == [2]

    def test_the_threshold_is_a_fraction_not_a_pixel_count(self) -> None:
        """The same physical distance is a different pixel count on a 1080p and a 4K camera, so
        a threshold in pixels has to be retuned per camera model."""
        gate = ObservationGate(min_hits=1, min_height_fraction=0.1)
        tall_in_a_small_frame = make_cluster(
            {"cam-a": [make_track(camera="cam-a", track_id=1, identity=0, box=SMALL_BOX)]},
            height=200,
            width=400,
        ).observations

        assert len(gate.filter(tall_in_a_small_frame)) == 1


class TestAgeGate:
    """A track the single-camera tracker has only just noticed may not exist."""

    def test_a_track_is_admitted_only_after_enough_consecutive_observations(self) -> None:
        gate = ObservationGate(min_hits=3)
        scene = observations(("cam-a", 1, BIG_BOX))

        assert gate.filter(scene) == []
        assert gate.filter(scene) == []
        assert len(gate.filter(scene)) == 1

    def test_a_track_its_own_camera_reported_without_starts_the_count_again(self) -> None:
        """ "Consecutive" is the claim, and a track that flickers is exactly the track that
        should not be trusted with a cross-camera identity. The camera being THERE is what
        makes the absence evidence: it reported, and this track was not in what it said."""
        gate = ObservationGate(min_hits=3)
        scene = observations(("cam-a", 1, BIG_BOX))

        gate.filter(scene)
        gate.filter(scene)
        gate.filter(observations(("cam-a", 2, BIG_BOX)))  # cam-a is here; track 1 is not
        assert gate.hits(TrackKey("cam-a", 1)) == 0

        assert gate.filter(scene) == []

    def test_a_camera_that_reported_an_empty_view_breaks_the_streak(self) -> None:
        """An empty view is a REPORT, not a silence. Derived from the observations it leaves
        no trace, so a track that flickers on and off — exactly the track this gate rejects —
        would keep its streak across every instant it was missing from and be admitted on the
        third sighting, having never had two in a row. The roster is what tells them apart."""
        gate = ObservationGate(min_hits=3)
        seen = instant(("cam-a", 1, BIG_BOX))
        empty = instant(("cam-a", 0, BIG_BOX))  # cam-a reported; its view held nothing
        assert empty.cameras == ("cam-a",) and empty.observations == ()

        gate.filter(seen.observations, cameras=seen.cameras)
        gate.filter(empty.observations, cameras=empty.cameras)

        assert (
            gate.hits(TrackKey("cam-a", 1)) == 0
        ), "the camera was there and the track was not"
        assert gate.filter(seen.observations, cameras=seen.cameras) == []

    def test_without_a_roster_an_empty_view_is_indistinguishable_from_an_absence(self) -> None:
        """The documented cost of the fallback, pinned so it stays a decision. A caller that
        cannot name the cameras gets the weaker rule, and the flicker above survives it."""
        gate = ObservationGate(min_hits=3)
        seen = instant(("cam-a", 1, BIG_BOX))

        gate.filter(seen.observations)
        gate.filter(instant(("cam-a", 0, BIG_BOX)).observations)

        assert gate.hits(TrackKey("cam-a", 1)) == 1, "carried, because nothing said otherwise"

    def test_an_instant_its_camera_was_not_in_is_not_a_miss(self) -> None:
        """The other half, and the one that decides a fleet. A synchronised instant holds
        whichever cameras landed inside its window — at fifty cameras, measured, about a
        quarter of them — so treating "my camera was not in this instant" as a miss makes
        three consecutive sightings a 1.4% event and the gate admits almost nothing."""
        gate = ObservationGate(min_hits=3)
        scene = observations(("cam-a", 1, BIG_BOX))

        gate.filter(scene)
        gate.filter(scene)
        gate.filter(observations(("cam-b", 7, BIG_BOX)))  # an instant cam-a was not in
        assert gate.hits(TrackKey("cam-a", 1)) == 2, "the streak is carried, not broken"

        assert gate.filter(scene) != [], "so the third sighting admits it"

    def test_a_camera_gone_long_enough_loses_its_streaks(self) -> None:
        """The bound. An absent camera says nothing about its tracks, but one that has gone
        for good must not leave its streaks in the map for the life of the process."""
        gate = ObservationGate(min_hits=3, max_absent_instants=2)
        gate.filter(observations(("cam-a", 1, BIG_BOX)))
        elsewhere = observations(("cam-b", 7, BIG_BOX))

        gate.filter(elsewhere)
        assert gate.hits(TrackKey("cam-a", 1)) == 1, "one instant away is not gone"
        gate.filter(elsewhere)
        assert gate.hits(TrackKey("cam-a", 1)) == 1, "two is the bound, still held"
        gate.filter(elsewhere)

        assert gate.hits(TrackKey("cam-a", 1)) == 0, "past it, the streak is dropped"


class TestGateOrder:
    """Height first, then age. The order is load-bearing."""

    def test_a_track_banks_no_age_while_it_is_too_small(self) -> None:
        """Swap the two gates and a track banks three instants of age while it is unusably
        small, then enters the matrix on its first usable frame with the gate already
        satisfied — which is the frame its embedding is least trustworthy on."""
        gate = ObservationGate(min_hits=2, min_height_fraction=0.1)
        far_away = observations(("cam-a", 1, SMALL_BOX))
        close_up = observations(("cam-a", 1, BIG_BOX))

        for _ in range(5):
            gate.filter(far_away)
        assert gate.hits(TrackKey("cam-a", 1)) == 0

        assert gate.filter(close_up) == []  # first usable instant: still tentative
        assert len(gate.filter(close_up)) == 1


class TestGateIsBounded:
    """State that only ever holds the tracks currently in flight."""

    def test_the_hit_map_never_exceeds_the_tracks_of_one_instant(self) -> None:
        gate = ObservationGate(min_hits=1)

        for step in range(5000):
            gate.filter(observations(("cam-a", step, BIG_BOX), ("cam-b", step, BIG_BOX)))

        assert gate.sizes() == {"hits": 2, "absent": 0}
        assert len(gate) == 2

    def test_a_camera_that_never_comes_back_does_not_grow_the_map(self) -> None:
        """The bound the carried streaks need: `cam-a` reports once and is never heard from
        again, and its key leaves the map `max_absent_instants` instants later rather than
        staying for the life of the process."""
        gate = ObservationGate(min_hits=1, max_absent_instants=4)
        gate.filter(observations(("cam-a", 1, BIG_BOX)))

        for step in range(5000):
            gate.filter(observations(("cam-b", step, BIG_BOX)))

        assert gate.sizes() == {"hits": 1, "absent": 0}, "only cam-b's current track"

    def test_reset_forgets_everything(self) -> None:
        gate = ObservationGate(min_hits=2)
        gate.filter(observations(("cam-a", 1, BIG_BOX)))

        gate.reset()

        assert gate.hits(TrackKey("cam-a", 1)) == 0


class TestGateConstruction:
    def test_min_hits_below_one_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="min_hits must be at least 1"):
            ObservationGate(min_hits=0)

    def test_a_height_fraction_outside_zero_to_one_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="min_height_fraction"):
            ObservationGate(min_height_fraction=1.0)
