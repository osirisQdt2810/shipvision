"""The gallery contract: what you can put in, and how you ask.

A gallery is the memory of a re-identification system — the set of known appearances a new
observation is compared against. Two properties decide whether it works in a 24/7 server,
and both are contract, not implementation detail:

**It is bounded.** 50 cameras at 20 fps produce roughly 15 000 crops a second. A gallery
that keeps them all is a memory leak with a plausible name. Every implementation takes a
capacity and states what it discards when full.

**A query never matches itself.** The standard protocol excludes gallery entries from the
query's own camera, because a match there is the tracker's job and counting it inflates
every score. That is not an evaluation nicety — a live system that re-identifies a ship
against its own camera's last frame has learned nothing and will report near-perfect
accuracy while doing it.
"""

from __future__ import annotations

import abc
from collections.abc import Iterable, Sequence

import numpy as np

from shipvision.registry import PYTHON, Registry
from shipvision.reid.types import QueryResult
from shipvision.types import Embedding

__all__ = ["GALLERIES", "BaseGallery"]


class BaseGallery(abc.ABC):
    """Store labelled embeddings; return ranked matches for a query.

    Implementations own their own locking. The server calls a gallery from several worker
    threads, and "the caller should lock it" is how two threads end up appending to the
    same row.
    """

    name: str = "gallery"
    backend: str = PYTHON

    @property
    @abc.abstractmethod
    def dim(self) -> int | None:
        """The embedding width, or `None` while the gallery is still empty.

        Fixed by the first vector added and enforced from then on: the failure this
        prevents is two models in one pipeline, and its only other symptom is a similarity
        matrix that cannot be formed — or one that can, because a broadcast succeeded.
        """

    @abc.abstractmethod
    def add(self, embedding: Embedding) -> int:
        """Store one embedding, normalising it, and return its entry index."""

    @abc.abstractmethod
    def query(
        self,
        vector: np.ndarray,
        *,
        top_k: int = 1,
        threshold: float | None = None,
        exclude_camera: str | None = None,
    ) -> QueryResult:
        """Rank the gallery against one vector.

        Args:
            vector: ``(d,)``. Normalised on the way in, so a caller need not.
            top_k: how many ranked matches to return.
            threshold: minimum cosine similarity for :attr:`QueryResult.accepted`. `None`
                accepts the best match unconditionally — appropriate only when the query is
                known to be someone in the gallery.
            exclude_camera: drop entries from this camera before ranking. Pass the query's
                own camera; see the module docstring for why this is not optional in
                evaluation.
        """

    @abc.abstractmethod
    def remove_identity(self, identity: str) -> int:
        """Forget an identity entirely. Returns how many entries went."""

    @abc.abstractmethod
    def clear(self) -> None:
        """Forget everything."""

    @abc.abstractmethod
    def __len__(self) -> int:
        """How many entries are stored."""

    @property
    @abc.abstractmethod
    def identities(self) -> Sequence[str]:
        """Every identity currently represented."""

    def add_many(self, embeddings: Iterable[Embedding]) -> list[int]:
        """Store several. Overridden where a bulk path is cheaper than a loop."""
        return [self.add(e) for e in embeddings]

    def __repr__(self) -> str:
        return f"<{type(self).__name__} entries={len(self)} identities={len(self.identities)}>"


#: An exact search and a centroid search answer the same question with very different memory
#: and very different behaviour on identities with few examples. Which one a deployment wants
#: is decided by its own numbers, so it is selected by name from config.
GALLERIES: Registry[BaseGallery] = Registry("gallery")
