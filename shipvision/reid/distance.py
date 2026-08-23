"""Distances over embeddings, as matrix products.

Every metric here reduces to one BLAS call on L2-normalised rows. That is not a
micro-optimisation: the gallery holds tens of thousands of vectors and is queried at frame
rate, so the difference between ``A @ B.T`` and anything loop-shaped is the difference
between re-identification running and not.

**The normalisation contract.** Cosine similarity is a dot product *only* on unit vectors.
Rather than normalising inside every distance function — which would silently pay for it on
every query against a gallery that is already normalised — the rule is the one stated in
:mod:`shipvision.types`: vectors are normalised once, on the way in, and everything
downstream assumes it. :func:`normalize` is that gate, and :func:`is_normalized` is how a
test or an assertion checks it held.
"""

from __future__ import annotations

import numpy as np

from shipvision.errors import DimensionMismatchError

__all__ = [
    "cosine_distance",
    "cosine_similarity",
    "euclidean_distance",
    "is_normalized",
    "normalize",
]

# Below this the vector carries no direction, only floating-point noise, and dividing by it
# would amplify that noise into a confident-looking unit vector pointing nowhere in
# particular. Such rows are left at zero, which scores 0 similarity against everything —
# the honest answer for an embedding that says nothing.
_MIN_NORM = 1e-12


def normalize(x: np.ndarray, *, copy: bool = True) -> np.ndarray:
    """L2-normalise the rows of ``x``.

    Args:
        x: ``(d,)`` or ``(n, d)``, any float dtype. Integer input is promoted to float32.
        copy: False normalises in place when the array is already float and writeable,
            which matters when the caller is normalising a whole gallery at load time.

    A zero row stays zero rather than becoming NaN. NaN is the worse failure by far: it
    propagates through the similarity matrix into every score, so one bad crop out of
    50 000 turns the entire query into NaN and the ranking silently becomes arbitrary.
    """
    array = np.asarray(x)
    if not np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float32)
    elif copy:
        array = array.copy()

    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    np.divide(array, np.maximum(norms, _MIN_NORM), out=array)
    return array


def is_normalized(x: np.ndarray, *, tolerance: float = 1e-4) -> bool:
    """Whether every row of ``x`` is unit length (a zero row counts as normalised)."""
    norms = np.linalg.norm(np.asarray(x), axis=-1)
    return bool(np.all((np.abs(norms - 1.0) <= tolerance) | (norms <= _MIN_NORM)))


def _check_width(query: np.ndarray, gallery: np.ndarray) -> None:
    if query.shape[-1] != gallery.shape[-1]:
        raise DimensionMismatchError(
            f"query is {query.shape[-1]}-d and the gallery is {gallery.shape[-1]}-d; "
            f"these embeddings did not come from the same model"
        )


def cosine_similarity(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """``(n, m)`` similarities in [-1, 1]. Both arguments must already be normalised.

    Deliberately does not normalise for you. A query is one to a few rows and a gallery is
    tens of thousands; normalising the gallery on every query would dominate the cost of
    the search itself, and it is already normalised because :meth:`BaseGallery.add` did it.
    """
    q = np.atleast_2d(np.asarray(query, dtype=np.float32))
    g = np.atleast_2d(np.asarray(gallery, dtype=np.float32))
    _check_width(q, g)
    return q @ g.T


def cosine_distance(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """``1 - cosine_similarity``, so smaller is more alike. Range [0, 2]."""
    return 1.0 - cosine_similarity(query, gallery)


def euclidean_distance(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """``(n, m)`` euclidean distances, expanded rather than looped.

    ``||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b`` turns the whole matrix into one gemm. The
    expansion is numerically lossy near zero — it can produce a small negative where the
    true distance is 0 — so the result is clipped before the square root, which is what
    stops a self-distance coming back as NaN.

    On normalised vectors this is a monotone function of cosine distance, so it ranks
    identically; it is here for callers whose threshold is expressed in euclidean terms.
    """
    q = np.atleast_2d(np.asarray(query, dtype=np.float32))
    g = np.atleast_2d(np.asarray(gallery, dtype=np.float32))
    _check_width(q, g)
    squared = np.sum(q * q, axis=1, keepdims=True) + np.sum(g * g, axis=1) - 2.0 * (q @ g.T)
    return np.sqrt(np.maximum(squared, 0.0))
