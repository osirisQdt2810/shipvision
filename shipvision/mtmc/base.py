"""The cross-camera tracker contract, and the family it belongs to.

One method, called once per synchronised group of frames: tracks in, global identities out.
Everything a caller needs to know is in that signature — no start-up handshake, no per-camera
registration, no "call this before that". A camera that appears for the first time in the
middle of a call is handled by the call.

**No state updater seam.** The reference implementation has a fourth component,
``StateUpdater``, whose every implementation is an empty ``run()``: a base class, a registry
entry, a config section and a constructor argument that between them do nothing. It is not
ported. An extension point with no implementations is not a design, it is an invitation to
hang per-frame work off the one component nobody has thought about, and the state it was
presumably meant to own already has a home — ageing, eviction and the invariant check belong
to :class:`~shipvision.mtmc.identity.GlobalIdAssigner`, which is where the state actually
lives. If something later needs a genuine post-assignment pass, adding a seam for it will be
a smaller change than removing a wrong one.
"""

from __future__ import annotations

import abc

from shipvision.mtmc.frames import FrameTrackCluster
from shipvision.registry import PYTHON, Registry
from shipvision.types import GlobalTrack

__all__ = ["MTMC", "BaseMTMCTracker"]


class BaseMTMCTracker(abc.ABC):
    """Assigns stable cross-camera identities to the tracks of one synchronised instant."""

    name: str = "mtmc"
    backend: str = PYTHON

    @abc.abstractmethod
    def track(self, cluster: FrameTrackCluster) -> list[GlobalTrack]:
        """One :class:`GlobalTrack` per input track, in input order.

        Every input track comes back, including the ones no identity was assigned to — those
        carry ``global_id=None``. Returning fewer results than were passed in would make the
        caller diff two lists to discover which of its tracks were judged too new or too
        small to identify, and a caller that skips that diff silently loses tracks. `None`
        says "not identified" in a way that fails at first use, which ``-1`` does not.
        """

    @abc.abstractmethod
    def reset(self) -> None:
        """Forget every identity. Must not make a fresh global id collide with a used one."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} backend={self.backend}>"


#: The cross-camera tracker family. Registered by name so a site can be moved from
#: appearance-only to spatially-gated association, or onto something not written yet, by
#: editing config rather than code.
MTMC: Registry[BaseMTMCTracker] = Registry("mtmc tracker")
