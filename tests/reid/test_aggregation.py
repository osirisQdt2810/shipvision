from __future__ import annotations

import numpy as np
import pytest

from shipvision.errors import ConfigurationError
from shipvision.reid import AGGREGATORS, cosine_similarity, normalize
from tests.reid.conftest import view_of

NAMES = AGGREGATORS.names()


@pytest.mark.parametrize("name", NAMES)
def test_the_result_is_a_unit_vector(name: str) -> None:
    """Not cosmetic. The mean of unit vectors has length equal to how much they agree, so
    without renormalising, a gallery entry's score would encode how consistent its own
    crops were rather than how much the query resembles it — two identities ranked by their
    internal agreement instead of by appearance."""
    vectors = normalize(np.stack([view_of(1, view=v) for v in range(5)]))

    out = AGGREGATORS.build(name).aggregate(vectors)

    assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-5)


@pytest.mark.parametrize("name", NAMES)
def test_aggregating_one_vector_returns_it(name: str) -> None:
    v = normalize(view_of(2))

    assert np.allclose(AGGREGATORS.build(name).aggregate(v[None, :]), v, atol=1e-5)


@pytest.mark.parametrize("name", NAMES)
def test_the_aggregate_is_closer_to_its_own_views_than_to_a_stranger(name: str) -> None:
    own = normalize(np.stack([view_of(1, view=v) for v in range(6)]))
    stranger = normalize(view_of(2, view=0))

    centroid = AGGREGATORS.build(name).aggregate(own)

    assert cosine_similarity(centroid, own).mean() > cosine_similarity(centroid, stranger)


@pytest.mark.parametrize("name", NAMES)
def test_zero_vectors_cannot_be_aggregated(name: str) -> None:
    with pytest.raises(ConfigurationError):
        AGGREGATORS.build(name).aggregate(np.zeros((0, 8), np.float32))


@pytest.mark.parametrize("name", NAMES)
def test_online_and_batch_agree_when_they_should(name: str) -> None:
    """Folding one at a time must land where aggregating all at once does. If they differ,
    a live track and a rebuilt gallery entry describe the same ship differently."""
    vectors = normalize(np.stack([view_of(3, view=v) for v in range(7)]))

    batch = AGGREGATORS.build(name).aggregate(vectors)

    online = AGGREGATORS.build(name)
    running = None
    for row in vectors:
        running = online.update(running, row)

    assert np.allclose(batch, running, atol=1e-5)


# ------------------------------------------------------------------------- mean


def test_the_mean_is_a_true_running_mean_not_an_accidental_ema() -> None:
    """Averaging the running vector with each new observation weights the newest as
    heavily as the entire history. That is an EMA with an unstated coefficient, and it is
    the single easiest way to get this class wrong."""
    vectors = normalize(np.stack([view_of(4, view=v) for v in range(20)]))

    incremental = AGGREGATORS.build("mean")
    running = None
    for row in vectors:
        running = incremental.update(running, row)

    assert np.allclose(running, normalize(vectors.sum(axis=0)), atol=1e-5)


def test_mean_weights_move_the_result_toward_the_heavier_views() -> None:
    a, b = normalize(view_of(5, view=0)), normalize(view_of(6, view=0))
    vectors = np.stack([a, b])

    even = AGGREGATORS.build("mean").aggregate(vectors)
    toward_a = AGGREGATORS.build("mean").aggregate(vectors, weights=np.array([9.0, 1.0]))

    assert cosine_similarity(toward_a, a[None]) > cosine_similarity(even, a[None])


def test_mean_rejects_weights_that_cannot_mean_anything() -> None:
    vectors = normalize(np.stack([view_of(1), view_of(2)]))
    aggregator = AGGREGATORS.build("mean")

    with pytest.raises(ConfigurationError, match="weights for"):
        aggregator.aggregate(vectors, weights=np.array([1.0]))
    with pytest.raises(ConfigurationError, match="non-negative"):
        aggregator.aggregate(vectors, weights=np.array([1.0, -1.0]))
    with pytest.raises(ConfigurationError, match="positive"):
        aggregator.aggregate(vectors, weights=np.array([0.0, 0.0]))


# -------------------------------------------------------------------------- ema


def test_ema_tracks_the_recent_appearance_where_the_mean_remembers_the_first() -> None:
    """The reason an online tracker wants an EMA. A ship tracked for twenty minutes has
    thousands of crops; under a mean the appearance from when it entered outvotes
    everything since, so the gallery entry describes a vessel the camera no longer sees."""
    old = normalize(np.stack([view_of(7, view=v) for v in range(30)]))
    new = normalize(np.stack([view_of(8, view=v) for v in range(5)]))
    stream = np.concatenate([old, new])

    by_mean = AGGREGATORS.build("mean").aggregate(stream)
    by_ema = AGGREGATORS.build("ema", alpha=0.7).aggregate(stream)

    recent = normalize(view_of(8, view=99))
    assert cosine_similarity(by_ema, recent[None]) > cosine_similarity(by_mean, recent[None])


def test_a_zero_weight_observation_leaves_the_vector_exactly_where_it_was() -> None:
    """Quality scales how far the update moves, not the mixing coefficient. Folding weight
    into alpha instead lets a *low*-quality crop take a bigger step, which is backwards."""
    aggregator = AGGREGATORS.build("ema", alpha=0.9)
    before = aggregator.update(None, normalize(view_of(1)))

    after = aggregator.update(before, normalize(view_of(2)), weight=0.0)

    assert np.allclose(before, after)


def test_a_full_weight_observation_moves_by_exactly_alpha() -> None:
    aggregator = AGGREGATORS.build("ema", alpha=0.75)
    a, b = normalize(view_of(1)), normalize(view_of(2))

    out = aggregator.update(a, b, weight=1.0)

    assert np.allclose(out, normalize(0.75 * a + 0.25 * b), atol=1e-6)


def test_alpha_of_one_is_refused_because_it_would_never_learn() -> None:
    """It would silently turn the whole feature bank into "whatever the first crop looked
    like" — a configuration that runs perfectly and recognises nothing."""
    with pytest.raises(ConfigurationError, match="never incorporate"):
        AGGREGATORS.build("ema", alpha=1.0)


def test_updating_across_a_width_change_is_refused() -> None:
    aggregator = AGGREGATORS.build("ema")
    current = aggregator.update(None, normalize(np.ones(32, np.float32)))

    with pytest.raises(ConfigurationError, match="32-d"):
        aggregator.update(current, normalize(np.ones(64, np.float32)))


def test_an_unknown_name_lists_what_there_is() -> None:
    with pytest.raises(ConfigurationError, match="available"):
        AGGREGATORS.build("bilinear-vibes")


class TestTheEmaRefusesAWeightItCannotMix:
    """`weight` scales how far an update moves, so it is a fraction of the step, not a
    count. `effective = 1 - (1 - alpha) * weight` is only a convex mixture while weight is
    in [0, 1]; above that it goes negative and the update *extrapolates away* from the
    running vector — exactly backwards from what the comment above that line promises, and
    the opposite of what a caller passing a big number for "trust this a lot" expects.
    """

    def test_a_weight_above_one_is_refused_rather_than_extrapolated(self) -> None:
        aggregator = AGGREGATORS.build("ema", alpha=0.9)
        a, b = normalize(view_of(1)), normalize(view_of(2))
        current = aggregator.update(None, a)

        with pytest.raises(ConfigurationError, match="in .0, 1."):
            aggregator.update(current, b, weight=20.0)

    def test_the_old_behaviour_is_the_bug_being_prevented(self) -> None:
        """Kept as a statement of what the guard buys: with `weights=[1, 20]` and alpha 0.9
        the second row produced ``effective = -1.0``, and the result pointed *away* from the
        running vector at cos = -0.497. No weight can legitimately do that."""
        aggregator = AGGREGATORS.build("ema", alpha=0.9)
        rows = normalize(np.eye(2, dtype=np.float32))

        with pytest.raises(ConfigurationError, match="in .0, 1."):
            aggregator.aggregate(rows, weights=np.array([1.0, 20.0]))

    def test_the_whole_weight_vector_is_checked_before_anything_is_folded(self) -> None:
        aggregator = AGGREGATORS.build("ema")
        rows = normalize(np.stack([view_of(1), view_of(2), view_of(3)]))

        with pytest.raises(ConfigurationError, match="non-negative"):
            aggregator.aggregate(rows, weights=np.array([1.0, -0.5, 1.0]))
        with pytest.raises(ConfigurationError, match="positive"):
            aggregator.aggregate(rows, weights=np.zeros(3))

    def test_the_ends_of_the_range_still_mean_what_they_meant(self) -> None:
        """The guard must not move the two values the docstring makes promises about."""
        aggregator = AGGREGATORS.build("ema", alpha=0.75)
        a, b = normalize(view_of(1)), normalize(view_of(2))
        current = aggregator.update(None, a)

        assert np.allclose(aggregator.update(current, b, weight=0.0), current)
        moved = aggregator.update(current, b, weight=1.0)
        assert np.allclose(moved, normalize(0.75 * current + 0.25 * b))
