"""The cross-camera tracker: four components, one call per synchronised instant.

The decomposition is the reference's and it is worth keeping, because each seam is a place
where a real question gets a separate answer:

1. **Gate** — which tracks are worth asking about (:mod:`shipvision.mtmc.gating`).
2. **Match** — how alike every pair is (:mod:`shipvision.mtmc.matchers`).
3. **Cluster** — which of them are the same object right now
   (:mod:`shipvision.mtmc.clustering`).
4. **Identity** — and which object that is, across time
   (:mod:`shipvision.mtmc.identity`).

Only the fourth carries state between calls, which is the property that makes the other three
testable with literals. The reference had a fifth, ``StateUpdater``, that did nothing; see
:mod:`shipvision.mtmc.base` for why it is not here.
"""

from __future__ import annotations

import inspect
import threading
from collections.abc import Sequence

import numpy as np

from shipvision.errors import ConfigurationError
from shipvision.mtmc.base import BaseMatcher, BaseMTMCTracker
from shipvision.mtmc.clustering import BaseClusterer
from shipvision.mtmc.frames import FrameTrackCluster, TrackKey, TrackObservation
from shipvision.mtmc.gating import ObservationGate
from shipvision.mtmc.identity import GlobalIdAssigner
from shipvision.mtmc.registry import MTMC, MTMC_CLUSTERERS, MTMC_MATCHERS
from shipvision.mtmc.topology import GroundPlane
from shipvision.registry import PYTHON
from shipvision.types import GlobalTrack

__all__ = ["ClusterMTMCTracker"]


@MTMC.register("cluster", backend=PYTHON, aliases=("vtx",))
class ClusterMTMCTracker(BaseMTMCTracker):
    """Cluster the instant, then reconcile those clusters with the identities already known.

    The ``vtx`` alias is the name this algorithm has in the system it is ported from, so an
    operator migrating a configuration can write what they already know.

    Thread-safe: a lock is held for the whole of :meth:`track`, because the four components
    share one logical update and a second thread entering halfway through would see a
    half-assigned identity map. The server runs one instance per camera group, so contention
    is between a call and its own retry, not between cameras.
    """

    def __init__(
        self,
        *,
        matrix_builder: str | BaseMatcher = "gated",
        clusterer: str | BaseClusterer = "agglomerative",
        ground_plane: GroundPlane | None = None,
        appearance_threshold: float = 0.86,
        spatial_threshold: float = 280.0,
        distance_threshold: float = 0.14,
        min_hits: int = 3,
        min_height_fraction: float = 1.0 / 9.0,
        max_age: int = 30,
        capacity: int = 4096,
        max_tracks: int = 8192,
        validate_every_step: bool = False,
    ) -> None:
        """
        Args:
            matrix_builder: a name from
                :data:`~shipvision.mtmc.registry.MTMC_MATCHERS`, or a pre-built instance. The
                default, ``gated``, degrades to appearance-only when no homographies are
                supplied, which is what makes it safe as a default. The keyword kept its
                original spelling when the family was renamed to *matcher*: it is the key a
                deployment writes in config, and renaming a config key in a repackaging is a
                breakage nobody gets anything for.
            clusterer: a name from :data:`~shipvision.mtmc.registry.MTMC_CLUSTERERS`, or an
                instance.
            ground_plane: the camera-to-map homographies. Ignored when ``matrix_builder`` is
                already an instance — that instance has its own.
            appearance_threshold: minimum cosine similarity for a pair to be considered.
            spatial_threshold: maximum ground-plane separation, in map units.
            distance_threshold: the clustering cut.
            min_hits: consecutive qualifying observations before a track may be associated.
            min_height_fraction: minimum box height as a fraction of frame height.
            max_age: instants a track may go unseen before its identity forgets it.
            capacity: maximum live global ids.
            max_tracks: maximum single-camera tracks held across all identities.
            validate_every_step: check the identity maps' invariant after every call.

        Raises:
            ConfigurationError: any argument is out of range, or names something that is not
                registered. All of it at construction — a threshold typo must stop the
                process at start-up, not at frame 40 000.
        """
        self.builder = self._build_matcher(
            matrix_builder,
            ground_plane=ground_plane,
            appearance_threshold=appearance_threshold,
            spatial_threshold=spatial_threshold,
        )
        self.clusterer = self._build_clusterer(clusterer, distance_threshold)
        self.gate = ObservationGate(min_hits=min_hits, min_height_fraction=min_height_fraction)
        self.assigner = GlobalIdAssigner(
            max_age=max_age,
            capacity=capacity,
            max_tracks=max_tracks,
            validate_every_step=validate_every_step,
        )
        self._lock = threading.RLock()

    # -- construction -------------------------------------------------------------------

    @staticmethod
    def _build_matcher(
        spec: str | BaseMatcher,
        *,
        ground_plane: GroundPlane | None,
        appearance_threshold: float,
        spatial_threshold: float,
    ) -> BaseMatcher:
        if isinstance(spec, BaseMatcher):
            return spec
        if not isinstance(spec, str):
            raise ConfigurationError(
                f"matrix_builder must be a registered name or a BaseMatcher, got "
                f"{type(spec).__name__}"
            )
        # Pass only what the named matcher actually accepts. Forwarding all three and
        # letting the constructor reject the surplus would make "appearance" — which has no
        # geometry and therefore no spatial threshold — unselectable from config.
        offered = {
            "appearance_threshold": appearance_threshold,
            "spatial_threshold": spatial_threshold,
            "ground_plane": ground_plane,
        }
        accepted = inspect.signature(MTMC_MATCHERS.get(spec).__init__).parameters
        options = {name: value for name, value in offered.items() if name in accepted}
        return MTMC_MATCHERS.build(spec, **options)

    @staticmethod
    def _build_clusterer(spec: str | BaseClusterer, distance_threshold: float) -> BaseClusterer:
        if isinstance(spec, BaseClusterer):
            return spec
        if not isinstance(spec, str):
            raise ConfigurationError(
                f"clusterer must be a registered name or a BaseClusterer, got "
                f"{type(spec).__name__}"
            )
        return MTMC_CLUSTERERS.build(spec, distance_threshold=distance_threshold)

    # -- the frame path -----------------------------------------------------------------

    def track(self, cluster: FrameTrackCluster) -> list[GlobalTrack]:
        if not isinstance(cluster, FrameTrackCluster):
            raise ConfigurationError(
                f"MTMC consumes a FrameTrackCluster — the tracks of every camera at one "
                f"synchronised instant — not a {type(cluster).__name__}. Handing it one "
                f"camera at a time is how cross-camera association becomes single-camera "
                f"deduplication"
            )
        with self._lock:
            admitted = self.gate.filter(cluster.observations)
            labels = self._cluster(admitted)
            assignment = self.assigner.assign(admitted, labels)
            label_of: dict[TrackKey, int] = {
                observation.key: int(label)
                for observation, label in zip(admitted, labels, strict=True)
            }
            return [
                self._result(observation, assignment.get(observation.key), label_of)
                for observation in cluster.observations
            ]

    def _cluster(self, admitted: Sequence[TrackObservation]) -> np.ndarray:
        """Labels for the admitted observations.

        Nothing to compare below two tracks, so the matcher and the clusterer are
        skipped rather than called with a degenerate input. That is not an optimisation: a
        1x1 distance matrix is a perfectly valid thing to hand scipy and it will refuse it,
        and one visible track is the normal state of a quiet site.
        """
        if len(admitted) < 2:
            return np.zeros(len(admitted), dtype=np.int32)
        return self.clusterer.fit_predict(self.builder.build(admitted))

    def _result(
        self,
        observation: TrackObservation,
        global_id: int | None,
        label_of: dict[TrackKey, int],
    ) -> GlobalTrack:
        if global_id is None:
            # Gated out: too new or too small to identify. Reported rather than dropped, with
            # the reason attached, because "we have no identity for this track" and "this
            # track did not exist" are different things to whoever is looking at the screen.
            return GlobalTrack(
                global_id=None,
                track=observation.track,
                metadata={"gated": True, "hits": self.gate.hits(observation.key)},
            )
        return GlobalTrack(
            global_id=global_id,
            track=observation.track,
            cluster_id=str(label_of[observation.key]),
            members=self.assigner.members(global_id),
        )

    def reset(self) -> None:
        with self._lock:
            self.gate.reset()
            self.assigner.reset()

    # -- introspection ------------------------------------------------------------------

    def sizes(self) -> dict[str, int]:
        """Every internal container's length, across every component.

        The evidence a growth test needs. A 24/7 process is judged on what these numbers do
        over a hundred thousand instants, not on whether one call returned the right answer.
        """
        with self._lock:
            sizes = self.assigner.sizes()
            sizes.update({f"gate_{k}": v for k, v in self.gate.sizes().items()})
            return sizes

    def __repr__(self) -> str:
        return (
            f"<ClusterMTMCTracker builder={type(self.builder).__name__} "
            f"clusterer={type(self.clusterer).__name__} "
            f"identities={len(self.assigner)} backend={self.backend}>"
        )
