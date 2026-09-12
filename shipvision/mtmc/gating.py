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

**An instant its camera was not in is not evidence against a track.** "Consecutive" is about
the track, not about the caller's clock: a synchronised instant holds whichever cameras landed
inside its window, and at fleet scale that is a fraction of them. Measured on a 50-camera
deployment: an instant held 11.8 cameras, so a camera appeared in 24% of instants and three
consecutive appearances happened 1.4% of the time — the gate admitted **2.2%** of what it was
offered and every global identity it produced held exactly one track. The same deployment at
twelve cameras held 88% and admitted 74.7%. A streak therefore survives an instant its camera
did not report in, and breaks when the camera *was* there and the track was not -- including
when it was there and saw nothing, which is why the roster comes from the caller rather than
from the observations, where an empty view leaves no trace.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

from shipvision.errors import ConfigurationError
from shipvision.mtmc.frames import TrackKey, TrackObservation

__all__ = ["ObservationGate"]


class ObservationGate:
    """Drops tracks that are too small, or too new to be trusted yet.

    Holds one piece of state — how many consecutive qualifying observations each track has —
    and stays bounded without a policy the caller has to tune: a key is dropped when its
    camera reported and the track did not qualify, and after ``max_absent_instants``
    consecutive instants its camera was not in at all. So the map holds the tracks in flight
    plus, briefly, the tracks of a camera that has just gone quiet.
    """

    def __init__(
        self,
        *,
        min_hits: int = 3,
        min_height_fraction: float = 1.0 / 9.0,
        max_absent_instants: int = 32,
    ) -> None:
        """
        Args:
            min_hits: consecutive qualifying observations before a track may take part in
                cross-camera association. The reference's production value is 3.
            min_height_fraction: minimum box height as a fraction of frame height. The
                reference's production value is 0.111, i.e. a person must fill about a ninth
                of the frame's height — which at 1080p is 120 pixels, roughly the smallest
                crop its re-ID model was trained to handle.
            max_absent_instants: how many consecutive instants a camera may be absent from
                before its tracks' streaks are dropped. Not a tuning knob but a bound: an
                absent camera says nothing about its tracks, and without a limit a camera that
                goes away for good would leave its streaks in the map for the life of the
                process. At a 60 ms window 32 is about two seconds.
        """
        if min_hits < 1:
            raise ConfigurationError(
                f"min_hits must be at least 1; 0 would admit a track on the frame it was "
                f"first seen, got {min_hits}"
            )
        if max_absent_instants < 1:
            raise ConfigurationError(
                f"max_absent_instants must be at least 1; 0 would drop a streak the instant "
                f"its camera missed one instant, which is the behaviour this replaced, "
                f"got {max_absent_instants}"
            )
        if not 0.0 <= min_height_fraction < 1.0:
            raise ConfigurationError(
                f"min_height_fraction is a fraction of frame height and must be in [0, 1), "
                f"got {min_height_fraction}"
            )
        self.min_hits = int(min_hits)
        self.min_height_fraction = float(min_height_fraction)
        self.max_absent_instants = int(max_absent_instants)
        #: Per track: the consecutive qualifying observations, and how many consecutive
        #: instants its camera has been absent from since the last one.
        self._hits: dict[TrackKey, int] = {}
        self._absent: dict[TrackKey, int] = {}

    def filter(
        self,
        observations: Sequence[TrackObservation],
        *,
        cameras: Collection[str] | None = None,
    ) -> list[TrackObservation]:
        """The observations that may take part in association, in input order.

        Args:
            observations: this instant's tracks, from every camera that was in it.
            cameras: the cameras the instant HELD, empty views included. Without it the
                roster is re-derived from the observations, and a camera that reported and
                saw nothing then reads as absent -- its streaks are carried across an instant
                that should have broken them, which is the flicker this gate exists to
                reject. A caller holding a `FrameTrackCluster` passes `cluster.cameras`.
        """
        tall_enough = [
            observation
            for observation in observations
            if observation.height_fraction > self.min_height_fraction
        ]

        # THE CAMERAS THIS INSTANT HELD. A camera that is not here said nothing about its
        # tracks, so its streaks are carried rather than broken -- the module docstring has
        # the measurement that makes this the difference between a gate that admits 2.2% and
        # one that works. Re-derived only when the caller cannot say; see the docstring.
        present = (
            set(cameras)
            if cameras is not None
            else {observation.key.camera_id for observation in observations}
        )

        hits: dict[TrackKey, int] = {}
        absent: dict[TrackKey, int] = {}
        for key, count in self._hits.items():
            if key.camera_id in present:
                continue  # its camera reported: this instant decides, below
            missed = self._absent.get(key, 0) + 1
            if missed <= self.max_absent_instants:
                hits[key] = count
                absent[key] = missed

        admitted: list[TrackObservation] = []
        for observation in tall_enough:
            count = self._hits.get(observation.key, 0) + 1
            hits[observation.key] = count
            absent.pop(observation.key, None)
            if count >= self.min_hits:
                admitted.append(observation)
        # Rebuilding the two maps rather than pruning them is what enforces "consecutive": a
        # track whose camera WAS here and did not qualify is simply not copied across.
        self._hits = hits
        self._absent = absent
        return admitted

    def hits(self, key: TrackKey) -> int:
        """Consecutive qualifying observations for one track. Zero if it is not being held."""
        return self._hits.get(key, 0)

    def reset(self) -> None:
        self._hits.clear()
        self._absent.clear()

    def sizes(self) -> dict[str, int]:
        """Every internal container's length. What a growth test asserts on."""
        return {"hits": len(self._hits), "absent": len(self._absent)}

    def __len__(self) -> int:
        return len(self._hits)

    def __repr__(self) -> str:
        return (
            f"<ObservationGate min_hits={self.min_hits} "
            f"min_height_fraction={self.min_height_fraction:.4f} held={len(self._hits)}>"
        )
