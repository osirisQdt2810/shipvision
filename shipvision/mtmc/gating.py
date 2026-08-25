"""Which tracks are worth asking about at all.

Two cheap gates, applied before anything expensive, and both of them are about the same
failure: a bad crop produces a confident embedding. A person eight pixels tall, or one the
single-camera tracker has only just noticed and may drop again next frame, gives the re-ID
model too little to work with — but the model does not say so, it returns a unit vector like
any other, and that vector will happily score 0.9 against a stranger. Admitting it does not
add a weak vote to the clustering, it adds a wrong one, and once a global id has been merged
across cameras nothing later un-merges it.

**Height first, then age.** The order is load-bearing and it is the reference's. Age counts
*consecutive frames in which the track was also large enough*, so a figure walking in from the
far distance starts accruing trust only once it is close enough to be worth trusting. Swap the
two and a track banks three frames of age while it is unusably small, then enters the matrix
on its first usable frame with the gate already satisfied.
"""

from __future__ import annotations

from collections.abc import Sequence

from shipvision.errors import ConfigurationError
from shipvision.mtmc.frames import TrackKey, TrackObservation

__all__ = ["ObservationGate"]


class ObservationGate:
    """Drops tracks that are too small, or too new to be trusted yet.

    Holds one piece of state — how many consecutive qualifying frames each track has been
    seen for — and it is bounded by construction rather than by a policy: every key that was
    not seen in the current instant is dropped at the end of the call, so the map can never
    hold more than the number of tracks currently in flight. That also *is* the definition of
    "consecutive": a track that misses one frame starts again from one.
    """

    def __init__(self, *, min_hits: int = 3, min_height_fraction: float = 1.0 / 9.0) -> None:
        """
        Args:
            min_hits: consecutive qualifying observations before a track may take part in
                cross-camera association. The reference's production value is 3.
            min_height_fraction: minimum box height as a fraction of frame height. The
                reference's production value is 0.111, i.e. a person must fill about a ninth
                of the frame's height — which at 1080p is 120 pixels, roughly the smallest
                crop its re-ID model was trained to handle.
        """
        if min_hits < 1:
            raise ConfigurationError(
                f"min_hits must be at least 1; 0 would admit a track on the frame it was "
                f"first seen, got {min_hits}"
            )
        if not 0.0 <= min_height_fraction < 1.0:
            raise ConfigurationError(
                f"min_height_fraction is a fraction of frame height and must be in [0, 1), "
                f"got {min_height_fraction}"
            )
        self.min_hits = int(min_hits)
        self.min_height_fraction = float(min_height_fraction)
        self._hits: dict[TrackKey, int] = {}

    def filter(self, observations: Sequence[TrackObservation]) -> list[TrackObservation]:
        """The observations that may take part in association, in input order."""
        tall_enough = [
            observation
            for observation in observations
            if observation.height_fraction > self.min_height_fraction
        ]

        hits: dict[TrackKey, int] = {}
        admitted: list[TrackObservation] = []
        for observation in tall_enough:
            count = self._hits.get(observation.key, 0) + 1
            hits[observation.key] = count
            if count >= self.min_hits:
                admitted.append(observation)
        # Replacing the map rather than pruning it is what enforces "consecutive", and it is
        # also what keeps the map bounded by the tracks in flight instead of by uptime.
        self._hits = hits
        return admitted

    def hits(self, key: TrackKey) -> int:
        """Consecutive qualifying observations for one track. Zero if it is not being held."""
        return self._hits.get(key, 0)

    def reset(self) -> None:
        self._hits.clear()

    def sizes(self) -> dict[str, int]:
        """Every internal container's length. What a growth test asserts on."""
        return {"hits": len(self._hits)}

    def __len__(self) -> int:
        return len(self._hits)

    def __repr__(self) -> str:
        return (
            f"<ObservationGate min_hits={self.min_hits} "
            f"min_height_fraction={self.min_height_fraction:.4f} held={len(self._hits)}>"
        )
