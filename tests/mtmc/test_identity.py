"""Global-id assignment: the stateful part, and the two reference bugs it must not have."""

from __future__ import annotations

import numpy as np
import pytest

from shipvision.errors import ConfigurationError, TrackingError
from shipvision.mtmc import GlobalIdAssigner, TrackKey
from tests.mtmc.conftest import make_cluster, make_track, view_of


def instant(*specs: tuple[str, int, int]) -> tuple:
    """``(camera, track_id, identity)`` triples to observations for one instant."""
    by_camera: dict[str, list] = {}
    for camera, track_id, identity in specs:
        by_camera.setdefault(camera, []).append(
            make_track(camera=camera, track_id=track_id, identity=identity)
        )
    return make_cluster(by_camera).observations


def by_identity(observations: tuple) -> list[int]:
    """Cluster labels that group observations sharing the same source identity.

    The clusterer is not under test here, so the labels are stated rather than computed. That
    is the point of the seam: the assigner's hard cases can be written down as "these two are
    one object, that one is another" without a distance matrix in the way.
    """
    seen: dict[bytes, int] = {}
    labels: list[int] = []
    for observation in observations:
        key = np.asarray(observation.embedding).round(3).tobytes()
        labels.append(seen.setdefault(key, len(seen)))
    return labels


def one_label_per(observations: tuple, groups: list[list[int]]) -> list[int]:
    """Explicit labels: ``groups`` lists the observation indices that form each cluster."""
    labels = [-1] * len(observations)
    for label, members in enumerate(groups):
        for index in members:
            labels[index] = label
    assert all(label >= 0 for label in labels)
    return labels


class TestOneIdentityPerObject:
    """The basic claim: as many global ids as there are objects, no more and no fewer."""

    def test_one_person_on_two_cameras_gets_one_global_id(self) -> None:
        assigner = GlobalIdAssigner(validate_every_step=True)
        observations = instant(("cam-a", 1, 0), ("cam-b", 1, 0))

        assignment = assigner.assign(observations, [0, 0])

        assert len(set(assignment.values())) == 1
        assert len(assigner) == 1

    def test_two_people_on_two_cameras_get_two_global_ids_and_neither_is_none(self) -> None:
        assigner = GlobalIdAssigner(validate_every_step=True)
        observations = instant(
            ("cam-a", 1, 0), ("cam-b", 1, 0), ("cam-a", 2, 1), ("cam-b", 2, 1)
        )

        # Observations flatten view-then-track, so this is cam-a#1, cam-a#2, cam-b#1, cam-b#2.
        assignment = assigner.assign(observations, [0, 1, 0, 1])

        assert len(set(assignment.values())) == 2
        assert len(assignment) == 4
        assert all(value is not None for value in assignment.values())
        assert assignment[TrackKey("cam-a", 1)] == assignment[TrackKey("cam-b", 1)]
        assert assignment[TrackKey("cam-a", 1)] != assignment[TrackKey("cam-a", 2)]

    def test_an_identity_holds_at_most_one_track_per_camera(self) -> None:
        """It is one object. A camera that sees one object twice at one instant has a
        single-camera tracking failure, not a cross-camera one."""
        assigner = GlobalIdAssigner(validate_every_step=True)
        same = view_of(0)
        observations = make_cluster(
            {
                "cam-a": [
                    make_track(camera="cam-a", track_id=1, identity=0, embedding=same),
                    make_track(camera="cam-a", track_id=2, identity=0, embedding=same),
                ]
            }
        ).observations

        # A clusterer would never produce this — the same-camera mask prevents it — so the
        # label is forced, to check the assigner does not rely on that being true.
        assigner.assign(observations, [0, 0])

        for global_id in assigner.global_ids:
            cameras = [key.camera_id for key in assigner.members(global_id)]
            assert len(cameras) == len(set(cameras))

    def test_every_observation_leaves_with_an_id(self) -> None:
        """A track that reached the assigner is one the gate decided to trust. The reference
        left some of them at -1 and published them anyway."""
        assigner = GlobalIdAssigner(validate_every_step=True)
        observations = instant(("cam-a", 1, 0), ("cam-b", 1, 1), ("cam-b", 2, 2))

        assignment = assigner.assign(observations, [0, 1, 2])

        assert set(assignment) == {o.key for o in observations}
        assert all(isinstance(v, int) for v in assignment.values())

    def test_a_track_without_an_embedding_is_a_typed_failure(self) -> None:
        assigner = GlobalIdAssigner()
        observations = instant(("cam-a", 1, 0))
        observations[0].track.embedding = None

        with pytest.raises(TrackingError, match="has no embedding"):
            assigner.assign(observations, [0])

    def test_mismatched_labels_are_refused(self) -> None:
        assigner = GlobalIdAssigner()

        with pytest.raises(ConfigurationError, match="labels for"):
            assigner.assign(instant(("cam-a", 1, 0)), [0, 0])


class TestGlobalIdPersistence:
    """An id is only useful if it is the same id next instant."""

    def test_an_identity_keeps_its_id_across_instants(self) -> None:
        assigner = GlobalIdAssigner(validate_every_step=True)
        first = assigner.assign(instant(("cam-a", 1, 0), ("cam-b", 1, 0)), [0, 0])
        second = assigner.assign(instant(("cam-a", 1, 0), ("cam-b", 1, 0)), [0, 0])

        assert set(first.values()) == set(second.values())
        assert assigner.issued == 1

    def test_a_third_camera_joining_an_identity_adopts_its_id(self) -> None:
        assigner = GlobalIdAssigner(validate_every_step=True)
        first = assigner.assign(instant(("cam-a", 1, 0), ("cam-b", 1, 0)), [0, 0])
        established = next(iter(first.values()))

        second = assigner.assign(
            instant(("cam-a", 1, 0), ("cam-b", 1, 0), ("cam-c", 1, 0)), [0, 0, 0]
        )

        assert set(second.values()) == {established}
        assert len(assigner.members(established)) == 3

    def test_the_largest_cluster_gets_first_claim_on_a_contested_id(self) -> None:
        """reorderCluster. A cluster of three views is much more likely to be the identity
        that owns an id than a cluster of one; processing in arbitrary order lets a single
        spurious view steal it."""
        assigner = GlobalIdAssigner(validate_every_step=True)
        established = assigner.assign(
            instant(("cam-a", 1, 0), ("cam-b", 1, 0), ("cam-c", 1, 0)), [0, 0, 0]
        )
        target = next(iter(established.values()))

        # Next instant the same three are seen, plus a stranger the clusterer wrongly puts
        # with cam-a's track. The trio is processed first, so it keeps the id.
        observations = instant(
            ("cam-a", 1, 0), ("cam-b", 1, 0), ("cam-c", 1, 0), ("cam-d", 9, 5)
        )
        assignment = assigner.assign(
            observations, one_label_per(observations, [[3, 0], [1, 2]])
        )

        assert assignment[TrackKey("cam-b", 1)] == target
        assert assignment[TrackKey("cam-c", 1)] == target

    def test_two_known_identities_that_one_instant_thinks_are_one_are_left_alone(self) -> None:
        """Splitting or merging established identities on one frame's evidence is how ids
        oscillate. Nothing is done until one of them has a track the other wants."""
        assigner = GlobalIdAssigner(validate_every_step=True)
        first = assigner.assign(instant(("cam-a", 1, 0), ("cam-b", 1, 0)), [0, 0])
        second = assigner.assign(instant(("cam-c", 1, 1), ("cam-d", 1, 1)), [0, 0])
        before = len(assigner)

        merged = instant(("cam-a", 1, 0), ("cam-b", 1, 0), ("cam-c", 1, 1), ("cam-d", 1, 1))
        assignment = assigner.assign(merged, [0, 0, 0, 0])

        assert len(assigner) == before
        assert assignment[TrackKey("cam-a", 1)] == next(iter(first.values()))
        assert assignment[TrackKey("cam-c", 1)] == next(iter(second.values()))

    def test_a_replacement_track_id_takes_over_a_stale_camera_slot(self) -> None:
        """A person is occluded, the single-camera tracker drops their track and gives them a
        new id when they reappear. The identity's slot for that camera is held by a track
        nobody has seen this instant, and the live one outranks it."""
        assigner = GlobalIdAssigner(max_age=30, validate_every_step=True)
        first = assigner.assign(instant(("cam-a", 1, 0), ("cam-b", 1, 0)), [0, 0])
        target = next(iter(first.values()))

        # cam-a's track 1 is gone; track 2 is the same person under a new track id.
        observations = instant(("cam-a", 2, 0), ("cam-b", 1, 0))
        assignment = assigner.assign(observations, [0, 0])

        assert assignment[TrackKey("cam-a", 2)] == target
        assert TrackKey("cam-a", 1) not in assigner.members(target)

    def test_the_older_of_two_equal_candidates_wins(self) -> None:
        """selectAppropriateID. Age is consecutive observations, so "oldest" means "has been
        continuously confirmed the longest" — the identity with the most evidence behind it."""
        assigner = GlobalIdAssigner(validate_every_step=True)
        # Identity A is watched for five instants; identity B appears once.
        for _ in range(5):
            long_lived = assigner.assign(instant(("cam-a", 1, 0), ("cam-b", 1, 0)), [0, 0])
        newcomer = assigner.assign(instant(("cam-c", 1, 1), ("cam-d", 1, 1)), [0, 0])
        old_id = next(iter(long_lived.values()))
        new_id = next(iter(newcomer.values()))
        assert old_id != new_id

        # A cluster overlapping both equally (one member each) must take the older one.
        observations = instant(("cam-a", 1, 0), ("cam-c", 1, 1), ("cam-e", 1, 2))
        assignment = assigner.assign(observations, [0, 0, 0])

        assert assignment[TrackKey("cam-e", 1)] == old_id


class TestContestedCamera:
    """Two tracks from one camera both wanting a place in one identity."""

    def test_the_better_looking_track_wins_the_slot_and_the_loser_gets_a_new_id(self) -> None:
        """appearanceSimilarity. The incumbent and the challenger are both live and both from
        cam-a; the identity has room for one, so it keeps whichever looks more like the rest of
        the group right now, and the other becomes its own identity rather than vanishing."""
        assigner = GlobalIdAssigner(validate_every_step=True)
        anchor = view_of(0, view=0)
        good = view_of(0, view=1)
        poor = view_of(0, view=2, jitter=0.9)

        first = make_cluster(
            {
                "cam-a": [make_track(camera="cam-a", track_id=1, identity=0, embedding=poor)],
                "cam-b": [make_track(camera="cam-b", track_id=1, identity=0, embedding=anchor)],
            }
        ).observations
        target = next(iter(assigner.assign(first, [0, 0]).values()))
        assert assigner.members(target) == (TrackKey("cam-a", 1), TrackKey("cam-b", 1))

        # Both cam-a tracks are live this instant, and the clusterer puts the better one with
        # cam-b. cam-a#1 is the incumbent; cam-a#2 looks much more like cam-b#1.
        second = make_cluster(
            {
                "cam-a": [
                    make_track(camera="cam-a", track_id=1, identity=0, embedding=poor),
                    make_track(camera="cam-a", track_id=2, identity=0, embedding=good),
                ],
                "cam-b": [make_track(camera="cam-b", track_id=1, identity=0, embedding=anchor)],
            }
        ).observations
        assignment = assigner.assign(second, one_label_per(second, [[1, 2], [0]]))

        assert assignment[TrackKey("cam-a", 2)] == target
        assert assignment[TrackKey("cam-a", 1)] != target
        assert assignment[TrackKey("cam-a", 1)] is not None
        assigner.validate()

    def test_a_worse_looking_challenger_does_not_displace_the_incumbent(self) -> None:
        """The other half. Without it the rule above would pass on "always prefer the newer
        track", which is how an identity hops between two people standing together."""
        assigner = GlobalIdAssigner(validate_every_step=True)
        anchor = view_of(0, view=0)
        good = view_of(0, view=1)
        poor = view_of(0, view=2, jitter=0.9)

        first = make_cluster(
            {
                "cam-a": [make_track(camera="cam-a", track_id=1, identity=0, embedding=good)],
                "cam-b": [make_track(camera="cam-b", track_id=1, identity=0, embedding=anchor)],
            }
        ).observations
        target = next(iter(assigner.assign(first, [0, 0]).values()))

        second = make_cluster(
            {
                "cam-a": [
                    make_track(camera="cam-a", track_id=1, identity=0, embedding=good),
                    make_track(camera="cam-a", track_id=2, identity=0, embedding=poor),
                ],
                "cam-b": [make_track(camera="cam-b", track_id=1, identity=0, embedding=anchor)],
            }
        ).observations
        assignment = assigner.assign(second, one_label_per(second, [[1, 2], [0]]))

        assert assignment[TrackKey("cam-a", 1)] == target
        assert assignment[TrackKey("cam-a", 2)] != target


class TestTtl:
    """An identity that comes back within the TTL is the same identity. After it, it is not."""

    def test_it_keeps_its_id_when_it_returns_inside_the_ttl(self) -> None:
        assigner = GlobalIdAssigner(max_age=3, validate_every_step=True)
        first = assigner.assign(instant(("cam-a", 1, 0), ("cam-b", 1, 0)), [0, 0])
        target = next(iter(first.values()))

        for _ in range(3):  # absent for exactly max_age instants
            assigner.assign((), [])
        returned = assigner.assign(instant(("cam-a", 1, 0), ("cam-b", 1, 0)), [0, 0])

        assert set(returned.values()) == {target}
        assert assigner.issued == 1

    def test_it_gets_a_new_id_when_it_returns_after_the_ttl(self) -> None:
        assigner = GlobalIdAssigner(max_age=3, validate_every_step=True)
        first = assigner.assign(instant(("cam-a", 1, 0), ("cam-b", 1, 0)), [0, 0])
        target = next(iter(first.values()))

        for _ in range(4):  # one instant too many
            assigner.assign((), [])
        returned = assigner.assign(instant(("cam-a", 1, 0), ("cam-b", 1, 0)), [0, 0])

        assert set(returned.values()) != {target}
        assert assigner.issued == 2

    def test_the_ttl_clock_advances_on_instants_with_no_tracks_at_all(self) -> None:
        """The case that matters: a camera group going quiet is exactly when a non-evicting
        implementation stops evicting."""
        assigner = GlobalIdAssigner(max_age=2, validate_every_step=True)
        assigner.assign(instant(("cam-a", 1, 0), ("cam-b", 1, 0)), [0, 0])
        assert len(assigner) == 1

        for _ in range(3):
            assigner.assign((), [])

        assert len(assigner) == 0
        assert assigner.sizes() == {
            "global_ids": 0,
            "owner": 0,
            "features": 0,
            "hits": 0,
            "last_seen": 0,
            "members_total": 0,
        }

    def test_max_age_is_validated_at_construction(self) -> None:
        with pytest.raises(ConfigurationError, match="max_age must be at least 1"):
            GlobalIdAssigner(max_age=0)


class TestBoundedGrowth:
    """The reference's real bug: storage that only ever grows.

    Its "cleanable" tracker exists to prune, and its name-to-id table maps
    ``"cleanable_aic_tracker"`` to the plain ``aic_tracker`` — so the shipped configuration
    builds the version that never prunes, and its global-id storage and reverse map grow for as
    long as the process lives. Here eviction is not a variant, so this asserts actual lengths.
    """

    @staticmethod
    def churn(assigner: GlobalIdAssigner, instants: int) -> None:
        """A fresh pair of track ids every instant — the worst case for a map keyed on
        (camera, track_id), and what a busy quay with re-identifying MOT actually produces."""
        for step in range(instants):
            observations = instant(
                ("cam-a", step, step % 97),
                ("cam-b", step, step % 97),
                ("cam-c", step + 500_000, (step + 7) % 97),
            )
            assigner.assign(observations, [0, 0, 1])

    def test_every_structure_is_flat_after_thousands_of_churning_instants(self) -> None:
        """Not "it did not crash" and not "it is under some large number": the *same* lengths
        after 200 instants and after 2 200, while forty times as many identities have been
        issued and forgotten."""
        assigner = GlobalIdAssigner(max_age=5, capacity=64, max_tracks=128)

        self.churn(assigner, 200)
        early = assigner.sizes()
        early_issued = assigner.issued
        self.churn(assigner, 2000)
        late = assigner.sizes()

        assert late == early
        # Three new keys per instant, none of them seen again, so at most max_age + 1
        # instants' worth can be alive: 6 * 3 = 18.
        assert late["owner"] == 18
        assert late["features"] == 18
        assert late["hits"] == 18
        assert late["last_seen"] == 18
        assert late["members_total"] == 18
        assert late["global_ids"] == 12
        # And it really did keep discovering identities rather than quietly stopping.
        assert assigner.issued - early_issued == 4000
        assigner.validate()

    def test_the_capacity_evicts_the_least_recently_seen_identity(self) -> None:
        assigner = GlobalIdAssigner(max_age=10_000, capacity=3)
        seen = []
        for step in range(10):
            observations = instant(("cam-a", step, step))
            seen.append(next(iter(assigner.assign(observations, [0]).values())))

        assert len(assigner) == 3
        assert set(assigner.global_ids) == set(seen[-3:])
        assigner.validate()

    def test_max_tracks_evicts_the_least_recently_seen_track(self) -> None:
        assigner = GlobalIdAssigner(max_age=10_000, capacity=10_000, max_tracks=4)
        for step in range(20):
            assigner.assign(instant(("cam-a", step, 0), ("cam-b", step, 0)), [0, 0])

        assert assigner.sizes()["owner"] == 4
        assigner.validate()

    def test_the_capacities_are_validated_at_construction(self) -> None:
        with pytest.raises(ConfigurationError, match="capacity must be positive"):
            GlobalIdAssigner(capacity=0)
        with pytest.raises(ConfigurationError, match="max_tracks must be positive"):
            GlobalIdAssigner(max_tracks=0)


class TestIdCounterOwnership:
    """The reference's other real bug: ``static inline MtmcID m_globalIDCounter``.

    One counter shared by every tracker instance in the process, zeroed by any one instance's
    ``reset()``. Two camera groups in one server therefore interleave their id spaces, and
    after a reset the next new identity is issued id 0 — which is still live, still in a
    database and still on somebody's screen.
    """

    def test_two_assigners_do_not_share_a_counter(self) -> None:
        first = GlobalIdAssigner()
        second = GlobalIdAssigner()

        first.assign(instant(("cam-a", 1, 0), ("cam-b", 1, 0)), [0, 0])
        first.assign(instant(("cam-a", 2, 1), ("cam-b", 2, 1)), [0, 0])
        second_ids = second.assign(instant(("cam-x", 1, 0), ("cam-y", 1, 0)), [0, 0])

        assert first.issued == 2
        assert second.issued == 1
        assert set(second_ids.values()) == {0}

    def test_one_assigners_reset_does_not_touch_another(self) -> None:
        first = GlobalIdAssigner()
        second = GlobalIdAssigner()
        first.assign(instant(("cam-a", 1, 0), ("cam-b", 1, 0)), [0, 0])
        second.assign(instant(("cam-x", 1, 0), ("cam-y", 1, 0)), [0, 0])

        second.reset()

        assert first.issued == 1
        assert len(first) == 1

    def test_reset_forgets_identities_without_rewinding_the_id_space(self) -> None:
        """Published ids are out in the world. Reusing one attaches a stranger to its
        history."""
        assigner = GlobalIdAssigner()
        before = set(
            assigner.assign(instant(("cam-a", 1, 0), ("cam-b", 1, 0)), [0, 0]).values()
        )

        assigner.reset()
        after = set(assigner.assign(instant(("cam-a", 1, 0), ("cam-b", 1, 0)), [0, 0]).values())

        assert len(assigner) == 1
        assert not (before & after)
        assert min(after) > max(before)

    def test_ids_are_monotonic_across_thousands_of_identities(self) -> None:
        assigner = GlobalIdAssigner(max_age=1)
        issued: list[int] = []
        for step in range(200):
            issued.extend(assigner.assign(instant(("cam-a", step, step)), [0]).values())

        assert issued == sorted(set(issued))


class TestInvariant:
    """``_owner`` and ``_members`` are one fact stored twice, and they can disagree."""

    def test_validate_passes_on_a_healthy_assigner(self) -> None:
        assigner = GlobalIdAssigner()
        assigner.assign(instant(("cam-a", 1, 0), ("cam-b", 1, 0)), [0, 0])

        assigner.validate()

    def test_validate_catches_a_desynchronised_reverse_map(self) -> None:
        """A bug that desynchronises them produces output that is wrong but plausible for
        hours: an identity that quietly holds two tracks from one camera, an id nothing can be
        found under."""
        assigner = GlobalIdAssigner()
        assigner.assign(instant(("cam-a", 1, 0), ("cam-b", 1, 0)), [0, 0])

        assigner._owner[TrackKey("cam-a", 1)] = 999  # reaching in is the point of the test

        with pytest.raises(TrackingError, match=r"desynchronised|does not list it"):
            assigner.validate()

    def test_validate_catches_a_track_that_could_never_be_evicted(self) -> None:
        assigner = GlobalIdAssigner()
        assigner.assign(instant(("cam-a", 1, 0), ("cam-b", 1, 0)), [0, 0])

        del assigner._last_seen[TrackKey("cam-a", 1)]

        with pytest.raises(TrackingError, match="no stored feature or last-seen"):
            assigner.validate()
