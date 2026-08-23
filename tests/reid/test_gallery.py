from __future__ import annotations

import numpy as np
import pytest

from shipvision import Embedding
from shipvision.errors import ConfigurationError, DimensionMismatchError
from shipvision.reid import GALLERIES
from tests.reid.conftest import DIM, view_of

GALLERY_NAMES = GALLERIES.names()


def enrol(gallery, identity: int, views: int = 3, camera: str = "cam-a") -> None:
    for v in range(views):
        gallery.add(
            Embedding(
                vector=view_of(identity, view=v),
                identity=f"ship-{identity}",
                camera_id=camera,
                frame_id=v,
            )
        )


# ---------------------------------------------------------------- every gallery must


@pytest.mark.parametrize("name", GALLERY_NAMES)
def test_a_query_finds_the_identity_it_belongs_to(name: str) -> None:
    gallery = GALLERIES.build(name)
    for identity in range(6):
        enrol(gallery, identity)

    result = gallery.query(view_of(3, view=99), top_k=3)

    assert result.best is not None
    assert result.best.identity == "ship-3"


@pytest.mark.parametrize("name", GALLERY_NAMES)
def test_an_empty_gallery_answers_nothing_rather_than_raising(name: str) -> None:
    result = GALLERIES.build(name).query(view_of(0))

    assert not result
    assert result.best is None
    assert len(result) == 0


@pytest.mark.parametrize("name", GALLERY_NAMES)
def test_the_threshold_decides_acceptance_not_ranking(name: str) -> None:
    """A stranger must come back as "ranked but not accepted", not as the nearest ship.

    Returning the best match unconditionally is how a re-identification system gives every
    unknown vessel somebody else's identity.
    """
    gallery = GALLERIES.build(name)
    for identity in range(5):
        enrol(gallery, identity)

    stranger = gallery.query(view_of(999), top_k=3, threshold=0.9)

    assert stranger.best is not None, "it must still rank — the caller may want to see it"
    assert stranger.accepted is None
    assert not stranger


@pytest.mark.parametrize("name", GALLERY_NAMES)
def test_excluding_the_query_camera_removes_those_entries_from_the_ranking(name: str) -> None:
    """The protocol's central rule. Without it a query matches its own camera's last frame,
    which measures tracking and reports it as re-identification."""
    gallery = GALLERIES.build(name)
    gallery.add(Embedding(vector=view_of(1), identity="ship-1", camera_id="cam-a"))
    gallery.add(Embedding(vector=view_of(2), identity="ship-2", camera_id="cam-b"))

    unfiltered = gallery.query(view_of(1, view=5), top_k=2)
    filtered = gallery.query(view_of(1, view=5), top_k=2, exclude_camera="cam-a")

    assert unfiltered.best.identity == "ship-1"
    assert filtered.best is not None
    assert filtered.best.identity == "ship-2", "the only entry left"
    assert all(m.camera_id != "cam-a" for m in filtered.matches)


@pytest.mark.parametrize("name", GALLERY_NAMES)
def test_excluding_every_camera_present_yields_nothing(name: str) -> None:
    gallery = GALLERIES.build(name)
    enrol(gallery, 1, camera="cam-a")

    assert not gallery.query(view_of(1), exclude_camera="cam-a").matches


@pytest.mark.parametrize("name", GALLERY_NAMES)
def test_two_models_feeding_one_gallery_is_a_typed_error(name: str) -> None:
    gallery = GALLERIES.build(name)
    gallery.add(Embedding(vector=np.ones(32, np.float32), identity="a"))

    with pytest.raises(DimensionMismatchError):
        gallery.add(Embedding(vector=np.ones(64, np.float32), identity="b"))
    with pytest.raises(DimensionMismatchError):
        gallery.query(np.ones(64, np.float32))


@pytest.mark.parametrize("name", GALLERY_NAMES)
def test_an_unlabelled_vector_is_refused(name: str) -> None:
    with pytest.raises(ConfigurationError, match="identity"):
        GALLERIES.build(name).add(Embedding(vector=view_of(0)))


@pytest.mark.parametrize("name", GALLERY_NAMES)
def test_removing_an_identity_removes_all_of_it(name: str) -> None:
    gallery = GALLERIES.build(name)
    enrol(gallery, 1, views=4)
    enrol(gallery, 2, views=4)

    removed = gallery.remove_identity("ship-1")

    assert removed >= 1
    assert "ship-1" not in gallery.identities
    assert gallery.query(view_of(1, view=7)).best.identity == "ship-2"
    assert gallery.remove_identity("ship-1") == 0, "removing twice is not an error"


@pytest.mark.parametrize("name", GALLERY_NAMES)
def test_clear_leaves_it_reusable(name: str) -> None:
    gallery = GALLERIES.build(name)
    enrol(gallery, 1)
    gallery.clear()

    assert len(gallery) == 0
    assert gallery.identities == ()

    enrol(gallery, 2)
    assert gallery.query(view_of(2, view=8)).best.identity == "ship-2"


@pytest.mark.parametrize("name", GALLERY_NAMES)
def test_top_k_is_ordered_best_first(name: str) -> None:
    gallery = GALLERIES.build(name)
    for identity in range(8):
        enrol(gallery, identity, views=1)

    matches = gallery.query(view_of(0, view=4), top_k=5).matches

    assert len(matches) == 5
    scores = [m.score for m in matches]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("name", GALLERY_NAMES)
def test_top_k_larger_than_the_gallery_is_clamped_not_padded(name: str) -> None:
    gallery = GALLERIES.build(name)
    enrol(gallery, 1, views=2)

    assert 0 < len(gallery.query(view_of(1), top_k=50).matches) <= 2


# ---------------------------------------------------------------- FlatGallery specifics


def test_per_identity_capacity_stops_one_ship_evicting_the_others() -> None:
    """The starvation this project exists to prevent, one layer down.

    A vessel that sits in view for an hour produces thousands of crops. Without a
    per-identity cap it fills the gallery and every other identity is evicted — so the
    system forgets all fifty ships it might have recognised in order to remember one it
    is already tracking.
    """
    gallery = GALLERIES.build("flat", capacity=100, per_identity=4)
    for identity in range(5):
        enrol(gallery, identity, views=2)
    for view in range(200):
        gallery.add(
            Embedding(vector=view_of(0, view=view), identity="ship-0", camera_id="cam-a")
        )

    assert gallery.count_for("ship-0") == 4
    assert len(gallery) == 4 * 1 + 2 * 4
    for identity in range(1, 5):
        assert gallery.count_for(f"ship-{identity}") == 2
        assert gallery.query(view_of(identity, view=77)).best.identity == f"ship-{identity}"


def test_global_capacity_evicts_the_oldest_entry() -> None:
    gallery = GALLERIES.build("flat", capacity=6, per_identity=6)
    for identity in range(6):
        gallery.add(Embedding(vector=view_of(identity), identity=f"ship-{identity}"))

    gallery.add(Embedding(vector=view_of(90), identity="ship-90"))

    assert len(gallery) == 6
    assert "ship-0" not in gallery.identities, "the first one in is the first one out"
    assert "ship-90" in gallery.identities


def test_eviction_keeps_the_index_bookkeeping_straight() -> None:
    """Rows are compacted by swapping the last live row into the hole, so every index in
    the per-identity map has to be repaired. Get this wrong and a later query returns a
    stale row — the right score attached to the wrong identity."""
    gallery = GALLERIES.build("flat", capacity=50, per_identity=2)
    for round_ in range(20):
        for identity in range(5):
            gallery.add(
                Embedding(
                    vector=view_of(identity, view=round_),
                    identity=f"ship-{identity}",
                    camera_id="cam-a",
                )
            )
    gallery.remove_identity("ship-2")

    assert len(gallery) == 4 * 2
    assert sorted(gallery.identities) == ["ship-0", "ship-1", "ship-3", "ship-4"]
    for identity in (0, 1, 3, 4):
        assert gallery.query(view_of(identity, view=200)).best.identity == f"ship-{identity}"


def test_flat_rejects_nonsense_capacities() -> None:
    with pytest.raises(ConfigurationError):
        GALLERIES.build("flat", capacity=0)
    with pytest.raises(ConfigurationError):
        GALLERIES.build("flat", per_identity=0)


# ------------------------------------------------------------ CentroidGallery specifics


def test_a_centroid_holds_one_vector_per_identity_however_many_views() -> None:
    gallery = GALLERIES.build("centroid")
    enrol(gallery, 1, views=50)

    assert len(gallery) == 1
    assert gallery.observations_for("ship-1") == 50


def test_each_identity_gets_its_own_aggregator() -> None:
    """A shared stateful aggregator would fold every identity into one accumulator and
    hand the same vector back to all of them — so all four would score identically."""
    gallery = GALLERIES.build("centroid", aggregator="mean")
    for identity in range(4):
        enrol(gallery, identity, views=5)

    for identity in range(4):
        result = gallery.query(view_of(identity, view=60), top_k=4)
        assert result.best.identity == f"ship-{identity}"
    scores = [gallery.query(view_of(0, view=61), top_k=4).matches[i].score for i in range(4)]
    assert len(set(np.round(scores, 5))) == 4, "four identities must not share one vector"


def test_a_bad_aggregator_name_fails_where_the_gallery_is_configured() -> None:
    """Not on whichever camera happens to see the first ship."""
    with pytest.raises(ConfigurationError, match="unknown aggregator"):
        GALLERIES.build("centroid", aggregator="nope")


def test_centroid_capacity_evicts_the_least_recently_seen() -> None:
    gallery = GALLERIES.build("centroid", capacity=3)
    for identity in range(3):
        gallery.add(Embedding(vector=view_of(identity), identity=f"ship-{identity}"))
    gallery.add(Embedding(vector=view_of(0, view=1), identity="ship-0"))  # refresh ship-0

    gallery.add(Embedding(vector=view_of(9), identity="ship-9"))

    assert len(gallery) == 3
    assert "ship-1" not in gallery.identities, "least recently observed, not first enrolled"
    assert "ship-0" in gallery.identities


class TestCentroidRowIndicesMeanSomething:
    """``add`` returns a row, and it is the same number ``query`` reports for that identity.

    It returned ``len(self._vectors) - 1`` once — the size of the gallery minus one, which
    is the same value for every identity added while the gallery is full and indexes nothing
    at all. A caller cannot tell a meaningless handle from a meaningful one by looking at it,
    so the contract has to be asserted.
    """

    def test_add_returns_the_row_that_query_reports_for_the_same_identity(self) -> None:
        gallery = GALLERIES.build("centroid", capacity=8)
        rows = {
            f"ship-{i}": gallery.add(
                Embedding(vector=view_of(i), identity=f"ship-{i}", camera_id="cam-a")
            )
            for i in range(5)
        }

        for identity, row in rows.items():
            seed = int(identity.split("-")[1])
            match = gallery.query(view_of(seed, view=3)).best
            assert match is not None and match.identity == identity
            assert match.entry_index == row

    def test_folding_more_views_into_an_identity_keeps_its_row(self) -> None:
        gallery = GALLERIES.build("centroid")
        first = gallery.add(Embedding(vector=view_of(1), identity="ship-1"))

        again = [
            gallery.add(Embedding(vector=view_of(1, view=v), identity="ship-1"))
            for v in range(1, 4)
        ]

        assert again == [first, first, first]
        assert len(gallery) == 1

    def test_distinct_identities_never_share_a_row(self) -> None:
        gallery = GALLERIES.build("centroid", capacity=32)
        rows = [
            gallery.add(Embedding(vector=view_of(i), identity=f"ship-{i}")) for i in range(20)
        ]

        assert sorted(rows) == list(range(20)), "one row each, densely packed"

    def test_a_row_freed_by_eviction_is_handed_to_the_next_identity(self) -> None:
        """The rows stay dense, which is the whole reason the search is one gemm: a hole in
        the middle would either have to be masked on every query or force a reallocation."""
        gallery = GALLERIES.build("centroid", capacity=4)
        for i in range(4):
            gallery.add(Embedding(vector=view_of(i), identity=f"ship-{i}"))

        row = gallery.add(Embedding(vector=view_of(9), identity="ship-9"))

        assert len(gallery) == 4
        assert 0 <= row < 4
        assert "ship-0" not in gallery.identities
        assert gallery.query(view_of(9, view=1)).best.entry_index == row


class TestAGalleryRefusesNonFiniteVectors:
    """``Embedding`` validates on the way in, but it is a mutable dataclass and a gallery is
    the last boundary before a value becomes part of every future score.

    A NaN does not stay in the row it arrived in: it is a unit-looking vector after
    normalisation, it scores NaN against every probe, and a threshold comparison against NaN
    is False — so the entry is never accepted and never rejected either, it just quietly
    corrupts whatever reduction touches it.
    """

    @pytest.mark.parametrize("name", GALLERY_NAMES)
    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_add_refuses_a_vector_that_bypassed_the_embedding_check(
        self, name: str, bad: float
    ) -> None:
        gallery = GALLERIES.build(name)
        smuggled = Embedding(vector=view_of(1), identity="ship-1", camera_id="cam-a")
        smuggled.vector = np.full(DIM, bad, dtype=np.float32)

        with pytest.raises(ConfigurationError, match="non-finite"):
            gallery.add(smuggled)
        assert len(gallery) == 0, "and nothing half-written is left behind"

    @pytest.mark.parametrize("name", GALLERY_NAMES)
    def test_a_non_finite_probe_is_refused_rather_than_accepted_with_a_nan_score(
        self, name: str
    ) -> None:
        """With ``threshold=None`` the best match is accepted unconditionally, so a NaN
        probe used to come back as an accepted identity with a NaN score and a truthy
        result — the worst available outcome, since the caller then assigns that identity.
        """
        gallery = GALLERIES.build(name)
        enrol(gallery, 1)
        probe = view_of(1, view=4)
        probe[3] = np.nan

        with pytest.raises(ConfigurationError, match="non-finite"):
            gallery.query(probe)


class TestFlatGalleryBookkeepingUnderChurn:
    """The two invariants :class:`FlatGallery` documents, asserted directly.

    Both were mutation-tested and both survived the whole suite: turning
    ``sorted(rows, reverse=True)`` into ``sorted(rows)`` in `remove_identity`, and
    ``min(rows, key=sequence)`` into ``max(...)`` in `_make_room`, left every test green.
    They are real bugs — ascending removal order hands the loop a row it has already visited
    under a new number, and `max` trims the newest view instead of the oldest — and the
    end-to-end tests could not see either, because a top-1 answer is still plausible after
    the wrong row has been dropped.

    So these reach into the private index map on purpose. Compaction is the one part of this
    class whose correctness is not observable from outside until it is far too late: a stale
    index returns the right score attached to the wrong identity.
    """

    INVARIANT_CHECKS = 6

    def _assert_bookkeeping(self, gallery, context: str) -> None:
        size = len(gallery)
        rows_by_identity = gallery._by_identity
        stored = gallery._identity

        assert len(stored) == size, f"{context}: identity list and size disagree"
        flat = [row for rows in rows_by_identity.values() for row in rows]
        assert all(0 <= row < size for row in flat), f"{context}: an index is out of range"
        for identity, rows in rows_by_identity.items():
            assert rows, f"{context}: {identity} is an orphaned empty entry"
            for row in rows:
                assert stored[row] == identity, (
                    f"{context}: row {row} is indexed under {identity} but holds "
                    f"{stored[row]} — a query would report the wrong identity"
                )
        assert len(flat) == len(set(flat)), f"{context}: a row is indexed twice"
        assert sorted(flat) == list(range(size)), f"{context}: a live row is unreachable"
        assert set(rows_by_identity) == set(stored), f"{context}: the two views disagree"

    def test_the_index_map_stays_exact_through_four_thousand_mixed_operations(self) -> None:
        """Adds that trip the per-identity cap, adds that trip the global cap, and whole
        identities removed, interleaved — every path that moves a row, mixed rather than
        one at a time, because they only interact badly when they interleave."""
        gallery = GALLERIES.build("flat", capacity=40, per_identity=3)
        rng = np.random.default_rng(20260823)

        for step in range(4_000):
            identity = int(rng.integers(9))
            if rng.random() < 0.12:
                gallery.remove_identity(f"ship-{identity}")
            else:
                gallery.add(
                    Embedding(
                        vector=view_of(identity, view=step),
                        identity=f"ship-{identity}",
                        camera_id=f"cam-{identity % 3}",
                        frame_id=step,
                    )
                )
            self._assert_bookkeeping(gallery, f"step {step}")

        assert len(gallery) > 0, "the churn must not have emptied it by accident"

    def test_removing_an_identity_with_many_rows_leaves_the_rest_intact(self) -> None:
        """`remove_identity` drops its rows highest-index-first because a swap-with-last
        only ever moves a row *down*. Ascending order hands the loop a row number it has
        already visited, now occupied by something else."""
        gallery = GALLERIES.build("flat", capacity=64, per_identity=8)
        for identity in range(6):
            enrol(gallery, identity, views=8)

        removed = gallery.remove_identity("ship-0")

        assert removed == 8
        self._assert_bookkeeping(gallery, "after a wide removal")
        assert len(gallery) == 5 * 8
        for identity in range(1, 6):
            assert gallery.count_for(f"ship-{identity}") == 8
            assert gallery.query(view_of(identity, view=50)).best.identity == f"ship-{identity}"

    def test_per_identity_eviction_drops_the_oldest_view_not_the_newest(self) -> None:
        """The frame ids say which views survived, so this cannot be satisfied by keeping
        the right *number* of rows. Trimming the newest is the plausible-looking mistake:
        the count stays right, the top-1 answer stays right, and the gallery quietly becomes
        a record of what the ship looked like when it arrived.
        """
        gallery = GALLERIES.build("flat", capacity=10, per_identity=3)
        for view in range(10):
            gallery.add(
                Embedding(
                    vector=view_of(1, view=view),
                    identity="ship-1",
                    camera_id="cam-a",
                    frame_id=view,
                )
            )
            surviving = sorted(m.frame_id for m in gallery.query(view_of(1), top_k=3).matches)
            expected = sorted(range(max(0, view - 2), view + 1))
            assert surviving == expected, f"after view {view}"

        assert sorted(m.frame_id for m in gallery.query(view_of(1), top_k=3).matches) == [
            7,
            8,
            9,
        ]
