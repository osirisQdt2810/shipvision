from __future__ import annotations

import numpy as np
import pytest

from shipvision.errors import DimensionMismatchError
from shipvision.reid import (
    cosine_distance,
    cosine_similarity,
    euclidean_distance,
    is_normalized,
    normalize,
)
from tests.reid.conftest import view_of


def test_normalize_makes_unit_rows() -> None:
    x = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    out = normalize(x)

    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
    assert is_normalized(out)
    assert np.allclose(x, [[3.0, 4.0], [1.0, 0.0]]), "copy=True must not touch the input"


def test_a_zero_vector_stays_zero_instead_of_becoming_nan() -> None:
    """One dead crop out of fifty thousand must not poison the whole similarity matrix.

    NaN propagates: a single NaN row makes every score NaN, argsort orders them
    arbitrarily, and the ranking silently becomes noise with no error anywhere.
    """
    out = normalize(np.zeros((1, 8), dtype=np.float32))

    assert not np.isnan(out).any()
    assert np.all(out == 0.0)
    assert is_normalized(out), "a zero row counts as normalised — it has no direction to fix"
    assert float(cosine_similarity(out, normalize(np.ones((1, 8))))[0, 0]) == 0.0


def test_normalize_in_place_when_asked() -> None:
    x = np.array([[3.0, 4.0]], dtype=np.float32)
    out = normalize(x, copy=False)

    assert out is x
    assert np.allclose(x, [[0.6, 0.8]])


def test_integer_input_is_promoted_rather_than_truncated() -> None:
    """np.divide into an int array would floor every component to zero."""
    out = normalize(np.array([[3, 4]], dtype=np.int64))

    assert out.dtype == np.float32
    assert np.allclose(out, [[0.6, 0.8]])


def test_similarity_ranks_the_same_identity_highest() -> None:
    query = normalize(view_of(1, view=0))
    gallery = normalize(np.stack([view_of(2, view=1), view_of(1, view=1), view_of(3, view=1)]))

    scores = cosine_similarity(query, gallery)[0]

    assert int(np.argmax(scores)) == 1


def test_cosine_distance_is_one_minus_similarity() -> None:
    a = normalize(view_of(1))
    b = normalize(np.stack([view_of(1, view=2), view_of(7, view=2)]))

    assert np.allclose(cosine_distance(a, b), 1.0 - cosine_similarity(a, b))


def test_euclidean_self_distance_is_zero_not_nan() -> None:
    """The expanded form can produce a small negative where the true value is 0; without
    the clip, sqrt turns that into NaN on the diagonal."""
    x = normalize(np.stack([view_of(i) for i in range(6)]))

    d = euclidean_distance(x, x)

    assert not np.isnan(d).any()
    assert np.allclose(np.diag(d), 0.0, atol=1e-3)


def test_euclidean_and_cosine_rank_identically_on_unit_vectors() -> None:
    """They are monotone in each other there, so a caller may pick either by taste — and
    if this ever fails, one of them is wrong."""
    query = normalize(view_of(4))
    gallery = normalize(np.stack([view_of(i, view=3) for i in range(10)]))

    by_cosine = np.argsort(-cosine_similarity(query, gallery)[0])
    by_euclidean = np.argsort(euclidean_distance(query, gallery)[0])

    assert np.array_equal(by_cosine, by_euclidean)


def test_mismatched_widths_are_a_typed_error_not_a_broadcast() -> None:
    with pytest.raises(DimensionMismatchError, match="same model"):
        cosine_similarity(np.zeros((1, 128), np.float32), np.zeros((4, 512), np.float32))
