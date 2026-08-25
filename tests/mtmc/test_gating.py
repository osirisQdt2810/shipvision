"""Which tracks are worth asking about, and why the order of the two gates matters."""

from __future__ import annotations

import pytest

from shipvision.errors import ConfigurationError
from shipvision.mtmc import ObservationGate, TrackKey
from tests.mtmc.conftest import make_cluster, make_track

BIG_BOX = (100.0, 300.0, 200.0, 700.0)  # 400px of 1080 -> 0.370 of frame height
SMALL_BOX = (100.0, 300.0, 120.0, 380.0)  # 80px of 1080 -> 0.074 of frame height


def observations(*boxes: tuple[str, int, tuple[float, float, float, float]]) -> tuple:
    by_camera: dict[str, list] = {}
    for camera, track_id, box in boxes:
        by_camera.setdefault(camera, []).append(
            make_track(camera=camera, track_id=track_id, identity=0, box=box)
        )
    return make_cluster(by_camera).observations


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

    def test_a_missed_instant_starts_the_count_again(self) -> None:
        """ "Consecutive" is the claim, and a track that flickers is exactly the track that
        should not be trusted with a cross-camera identity."""
        gate = ObservationGate(min_hits=3)
        scene = observations(("cam-a", 1, BIG_BOX))

        gate.filter(scene)
        gate.filter(scene)
        gate.filter(())  # the track is gone for one instant
        assert gate.hits(TrackKey("cam-a", 1)) == 0

        assert gate.filter(scene) == []


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

        assert gate.sizes() == {"hits": 2}
        assert len(gate) == 2

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
