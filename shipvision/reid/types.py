"""What comes out of a gallery query.

Boundary types only — plain dataclasses, no behaviour beyond validation. The vector that
goes *in* is :class:`shipvision.types.Embedding`, which lives in the shared vocabulary
because a detector, a tracker and MTMC all carry one; what comes back out is specific to
re-identification and lives here.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Match", "QueryResult"]


@dataclass(slots=True, frozen=True)
class Match:
    """One ranked answer to a query."""

    identity: str
    score: float
    """Cosine similarity in [-1, 1]. Higher is more alike, always — even where the
    underlying gallery ranks on a distance, so a caller never has to ask which way round
    a particular gallery's numbers run."""
    entry_index: int
    camera_id: str | None = None
    frame_id: int | None = None


@dataclass(slots=True, frozen=True)
class QueryResult:
    """The ranked matches for one query vector, best first.

    Separate from a bare list so the "nothing was similar enough" case has somewhere to
    live: :attr:`accepted` is the answer, `None` when no candidate cleared the threshold.
    Returning an empty list would make the caller re-derive that, and returning the best
    match regardless is how a re-identification system assigns every stranger an identity.
    """

    matches: tuple[Match, ...]
    accepted: Match | None = None

    def __bool__(self) -> bool:
        return self.accepted is not None

    def __len__(self) -> int:
        return len(self.matches)

    @property
    def best(self) -> Match | None:
        """The top-ranked match whether or not it cleared the threshold."""
        return self.matches[0] if self.matches else None
