"""Deterministic embeddings that behave like real ones.

Every test here needs vectors where "same identity" and "different identity" have a
known-correct answer. Random vectors will not do: in 512 dimensions two random unit vectors
are nearly orthogonal, so *every* pair looks equally unlike every other and a broken
distance function passes. These fixtures instead give each identity a direction and place
its views as small perturbations around it, which is the structure real embeddings have and
the only structure that can distinguish a working ranking from a coincidence.
"""

from __future__ import annotations

import numpy as np
import pytest

DIM = 32


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(20260823)


def identity_vector(seed: int, dim: int = DIM) -> np.ndarray:
    """A stable direction for one identity."""
    return np.random.default_rng(1000 + seed).normal(size=dim).astype(np.float32)


def view_of(seed: int, jitter: float = 0.1, *, view: int = 0, dim: int = DIM) -> np.ndarray:
    """One observation of identity ``seed``, ``jitter`` away from its true direction.

    `jitter` is the knob these tests turn: small enough and every view of an identity is
    closer to its siblings than to any other identity, which is the regime where ranking
    must be perfect; large enough and it must degrade rather than break.
    """
    base = identity_vector(seed, dim)
    noise = np.random.default_rng(50_000 + seed * 97 + view).normal(size=dim)
    return (base / np.linalg.norm(base) + jitter * noise).astype(np.float32)
