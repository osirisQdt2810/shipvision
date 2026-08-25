"""The compiled cross-camera matrices against the readable ones, entry point for entry point.

A distance matrix is a harder thing to eyeball than a tracker's output — every entry is
plausible — so the comparison is exact rather than qualitative: same shape, same dtype, and
element-wise equality within float32 rounding, on scenes whose right answer is known.

Two properties get their own tests because they are the ones that fail *silently*:

* **The same-camera exclusion.** Two tracks in one camera can never be the same object. Lose it
  and MTMC quietly becomes a within-camera deduplicator: every count drops, every metric
  improves, and the system is worse.
* **NEVER_MERGE is exact.** ``to_distance`` turns a zero similarity into a large finite
  sentinel, and a compiled version that produced 1.0 instead would leave a pair the evidence
  ruled out merely expensive — which average linkage buys the moment somebody loosens a
  threshold.

The whole file skips when there is no build, like every other parity file here.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipvision.mtmc import MTMC_MATCHERS, NEVER_MERGE, GroundPlane, Homography
from shipvision.mtmc.backends.native import native_available
from shipvision.registry import NATIVE, PYTHON
from tests.mtmc.conftest import make_cluster, make_track, two_people_two_cameras

pytestmark = [
    pytest.mark.native,
    pytest.mark.skipif(
        not native_available(),
        reason="shipvision._C is not built, or predates the cross-camera helpers",
    ),
]

NAMES = ["appearance", "gated", "spatial"]

#: An identity homography per camera: the image plane *is* the map. Enough to exercise the
#: geometry — the projection is numpy's on both sides, and what is being compared is the
#: O(n^2) work over the projected points.
CALIBRATED = GroundPlane({"cam-a": Homography(np.eye(3)), "cam-b": Homography(np.eye(3))})


def both(name: str, **options: object) -> tuple[object, object]:
    """The readable matcher and the compiled one, configured identically.

    Options a given matcher does not accept are dropped rather than forwarded, which is what
    :meth:`shipvision.mtmc.tracker.ClusterMTMCTracker._build_matcher` does and for the same
    reason: ``appearance`` has no geometry and therefore no ground plane, and passing it one
    would make it unselectable from a config that also configures ``gated``.
    """
    import inspect

    accepted = inspect.signature(MTMC_MATCHERS.get(name, PYTHON).__init__).parameters
    kept = {key: value for key, value in options.items() if key in accepted}
    return (
        MTMC_MATCHERS.build(name, backend=PYTHON, **kept),
        MTMC_MATCHERS.build(name, backend=NATIVE, **kept),
    )


class TestTheCompiledMatrixEqualsTheReadableOne:
    @pytest.mark.parametrize("name", NAMES)
    def test_on_two_people_seen_by_two_cameras(self, name: str) -> None:
        observations = two_people_two_cameras().observations
        reference, candidate = both(name, ground_plane=CALIBRATED)

        expected = reference.build(observations)
        actual = candidate.build(observations)

        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype == np.float32
        assert np.abs(actual - expected).max() < 1e-6

    @pytest.mark.parametrize("name", NAMES)
    def test_on_an_uncalibrated_site(self, name: str) -> None:
        """No homographies at all, which is the normal state of a site on its first day. The
        geometry must fall open rather than refusing everyone — that is what lets a camera take
        part on appearance alone — and the two backends must fall open the same way."""
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

        actual = candidate.build(())

        assert actual.shape == (0, 0) == reference.build(()).shape

    @pytest.mark.parametrize("name", NAMES)
    def test_on_a_single_track(self, name: str) -> None:
        """One camera, one object: the diagonal is the whole matrix, and it must be zero."""
        cluster = make_cluster({"cam-a": [make_track(camera="cam-a", track_id=1, identity=0)]})
        reference, candidate = both(name, ground_plane=CALIBRATED)

        actual = candidate.build(cluster.observations)

        assert actual.shape == (1, 1)
        assert actual[0, 0] == 0.0
        assert np.array_equal(actual, reference.build(cluster.observations))

    def test_on_a_scene_the_appearance_threshold_rejects(self) -> None:
        """Two different people, so every cross-camera pair is below the bar. The matrix should
        be all ``NEVER_MERGE`` off the diagonal, on both sides — this is the case where an
        implementation that wrote 1.0 for "no evidence" still looks like a distance matrix."""
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


class TestThePropertiesEveryClustererNeeds:
    """Symmetric, zero-diagonal, finite, and exactly ``NEVER_MERGE`` where a pair must not
    group. ``squareform`` has no tolerance for asymmetry — it silently reads the upper triangle
    — and it rejects a non-finite entry outright."""

    @pytest.mark.parametrize("name", NAMES)
    def test_the_compiled_matrix_is_symmetric_with_a_zero_diagonal(self, name: str) -> None:
        observations = two_people_two_cameras().observations
        _, candidate = both(name, ground_plane=CALIBRATED)

        actual = candidate.build(observations)

        assert np.array_equal(actual, actual.T)
        assert np.array_equal(
            np.diagonal(actual), np.zeros(len(observations), dtype=np.float32)
        )

    @pytest.mark.parametrize("name", NAMES)
    def test_nothing_is_infinite(self, name: str) -> None:
        observations = two_people_two_cameras().observations
        _, candidate = both(name)

        assert np.isfinite(candidate.build(observations)).all()

    @pytest.mark.parametrize("name", NAMES)
    def test_two_tracks_in_one_camera_are_never_mergeable(self, name: str) -> None:
        """Even when they are the *same identity's* embedding, which is the adversarial case:
        appearance says yes as loudly as it can, and the answer must still be no. If they were
        the same object the single-camera tracker upstream had one job and failed at it."""
        cluster = make_cluster(
            {
                "cam-a": [
                    make_track(camera="cam-a", track_id=1, identity=0, view=0),
                    make_track(camera="cam-a", track_id=2, identity=0, view=0),
                ]
            }
        )
        _, candidate = both(name, ground_plane=CALIBRATED)

        actual = candidate.build(cluster.observations)

        assert actual[0, 1] == pytest.approx(NEVER_MERGE)
        assert actual[1, 0] == pytest.approx(NEVER_MERGE)


class TestTheIndividualPassesAgree:
    """The matcher-level tests above would pass for a compiled ``to_distance`` that happened to
    cancel out a wrong gate. Each pass is therefore checked against its numpy twin directly."""

    def test_the_threshold_pass_zeroes_exactly_what_numpy_zeroes(self) -> None:
        from shipvision import _C

        rng = np.random.default_rng(3)
        similarity = rng.uniform(-1.0, 1.0, size=(9, 9)).astype(np.float32)

        actual = _C.mtmc_threshold_similarity(similarity, 0.25)

        assert np.array_equal(actual, np.where(similarity > 0.25, similarity, 0.0))

    def test_the_ground_distance_pass_matches_the_numpy_expression(self) -> None:
        from shipvision import _C

        rng = np.random.default_rng(4)
        points = rng.uniform(-5000.0, 5000.0, size=(7, 2)).astype(np.float32)
        known = np.array([1, 1, 0, 1, 1, 0, 1], dtype=np.uint8)

        actual = _C.mtmc_ground_distances(points, known)

        delta = points[:, None, :].astype(np.float64) - points[None, :, :].astype(np.float64)
        expected = np.sqrt(np.sum(delta**2, axis=2))
        expected[~(known.astype(bool)[:, None] & known.astype(bool)[None, :])] = np.inf
        assert np.array_equal(actual, expected)

    def test_the_gate_falls_open_where_it_cannot_judge(self) -> None:
        """The single most consequential line in the spatial half: an uncalibrated camera must
        keep taking part on appearance alone rather than quietly never merging with anyone."""
        from shipvision import _C

        distances = np.array([[0.0, np.inf], [np.inf, 0.0]], dtype=np.float64)

        assert _C.mtmc_spatial_gate(distances, 10.0).all()

    def test_the_veto_produces_exactly_zero_rather_than_a_penalty(self) -> None:
        from shipvision import _C

        similarity = np.full((4, 4), 0.9, dtype=np.float32)
        allowed = np.zeros((4, 4), dtype=bool)

        actual = _C.mtmc_veto(similarity, allowed)

        assert np.array_equal(actual, np.zeros((4, 4), dtype=np.float32))

    def test_the_sentinel_is_the_same_number_on_both_sides(self) -> None:
        """Two constants, one meaning. A C++ side that drifted to 1e6 would still cluster, and
        would still be wrong at every threshold anyone had tuned."""
        from shipvision import _C

        assert _C.MTMC_NEVER_MERGE == pytest.approx(NEVER_MERGE)
