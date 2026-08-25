"""The native backend: the fused ``(n, n)`` passes in ``shipvision._C``.

Each class here subclasses the numpy matcher of the same name and replaces exactly the passes
that are worth replacing. It is not a second implementation of cross-camera matching: the
thresholds, the constructor, the ground-plane projection and the definition of "same object"
all stay where they are, and the parity tests compare the two matrices element for element.

**The gemm stays in numpy, deliberately.** ``features @ features.T`` is what BLAS is for —
multithreaded, blocked for the cache — and a hand-written loop in C++ would be slower than the
thing it replaced while looking like an optimisation. What moves is everything *around* it:
threshold, ground distance, gate, veto, and the conversion to a clusterable distance. In numpy
those are five full ``(n, n)`` temporaries, and at fifty cameras with fifteen tracks each that
is 560 000 entries walked five times, once per synchronised instant, a thousand times a second.

The same-camera exclusion crosses as an **integer code per track** rather than as camera names.
The reference implementation this library replaces compared the strings pairwise, which is
560 000 string comparisons per instant — on its own more expensive than the clustering it
feeds.

Importing this module never fails. Only construction does, with
:class:`~shipvision.errors.BackendUnavailableError`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from shipvision.errors import BackendUnavailableError
from shipvision.mtmc.core.appearance import AppearanceMatcher
from shipvision.mtmc.core.gated import GatedMatcher
from shipvision.mtmc.core.spatial import SpatialMatcher
from shipvision.mtmc.frames import TrackObservation
from shipvision.mtmc.registry import MTMC_MATCHERS
from shipvision.mtmc.topology import GroundPlane
from shipvision.registry import NATIVE

__all__ = [
    "NativeAppearanceMatcher",
    "NativeGatedMatcher",
    "NativeSpatialMatcher",
    "native_available",
]

try:  # pragma: no cover - depends on whether the extension was built, not on a branch
    from shipvision import _C

    _IMPORT_ERROR: str | None = None
except ImportError as exc:  # pragma: no cover
    _C = None  # type: ignore[assignment]
    _IMPORT_ERROR = str(exc)

_BUILD_HINT = (
    "Build it with `cmake -S . -B build && cmake --build build -j`, or use backend='python'"
)


def native_available() -> bool:
    """True when ``shipvision._C`` imported and carries the cross-camera helpers.

    Not a device check: these are host passes over an ``(n, n)`` matrix, and a build on a
    machine with no driver runs them perfectly well.
    """
    return _C is not None and hasattr(_C, "mtmc_to_distance")


def _require_extension() -> None:
    """Raise the typed failure, naming the fix.

    Raises:
        BackendUnavailableError: there is no build here, or it predates these entry points.
    """
    if _C is None:
        raise BackendUnavailableError(
            f"shipvision._C is not built: {_IMPORT_ERROR}. {_BUILD_HINT}"
        )
    if not hasattr(_C, "mtmc_to_distance"):
        raise BackendUnavailableError(
            f"shipvision._C is built but has no cross-camera helpers: it predates them. "
            f"Rebuild it — {_BUILD_HINT}"
        )


def _camera_codes(observations: Sequence[TrackObservation]) -> np.ndarray:
    """``(n,)`` int32, equal for two tracks exactly when they share a camera.

    Assigned by first appearance, which is what
    :meth:`~shipvision.mtmc.base.BaseMatcher.mergeable_mask` does — the codes are never
    compared across calls, only within one, so any consistent numbering is the same mask.
    """
    codes: dict[str, int] = {}
    return np.array(
        [codes.setdefault(observation.camera_id, len(codes)) for observation in observations],
        dtype=np.int32,
    )


class _NativeMatcherMixin:
    """The one pass every matcher ends with, in C++.

    Overriding ``build`` rather than only ``similarities`` because the win is in the tail:
    ``to_distance`` is four numpy temporaries (the never-merge rule, the camera mask, the
    symmetrisation, the diagonal) over the same matrix, and fusing them is the whole point.
    """

    def build(self, observations: Sequence[TrackObservation]) -> np.ndarray:
        similarity = np.ascontiguousarray(self.similarities(observations), dtype=np.float32)
        if similarity.size == 0:
            # An instant with no tracks is ordinary input — every camera can be quiet at once —
            # and the shape matters: (0,) turns it into an IndexError three frames later.
            return np.zeros((len(observations), len(observations)), dtype=np.float32)
        return np.asarray(
            _C.mtmc_to_distance(similarity, _camera_codes(observations)), dtype=np.float32
        )


@MTMC_MATCHERS.register("appearance", backend=NATIVE)
class NativeAppearanceMatcher(_NativeMatcherMixin, AppearanceMatcher):
    """Cosine appearance similarity with the threshold and the distance conversion in C++.

    The cosine itself is still numpy's gemm — see the module docstring.
    """

    def __init__(self, *, appearance_threshold: float = 0.86) -> None:
        _require_extension()
        super().__init__(appearance_threshold=appearance_threshold)

    def similarities(self, observations: Sequence[TrackObservation]) -> np.ndarray:
        """See :meth:`~shipvision.mtmc.core.appearance.matcher.AppearanceMatcher.similarities`."""
        from shipvision.mtmc.core.appearance.utils import stack_embeddings
        from shipvision.reid.distance import cosine_similarity

        features = stack_embeddings(observations)
        if features.size == 0:
            return np.zeros((len(observations), len(observations)), dtype=np.float32)
        similarity = np.ascontiguousarray(
            cosine_similarity(features, features), dtype=np.float32
        )
        return np.asarray(
            _C.mtmc_threshold_similarity(similarity, self.appearance_threshold),
            dtype=np.float32,
        )


@MTMC_MATCHERS.register("spatial", backend=NATIVE)
class NativeSpatialMatcher(_NativeMatcherMixin, SpatialMatcher):
    """Ground-plane separation with the pairwise distance, the gate and the conversion in C++.

    The projection stays in Python: it is one homography per *camera*, not per pair, so it is
    a handful of 3x3 multiplies against the O(n^2) work below — and it is the half that has
    OpenCV behind it.
    """

    def __init__(
        self,
        *,
        ground_plane: GroundPlane | None = None,
        spatial_threshold: float = 280.0,
        foot_ratio: float = 1.0,
        aspect_ratio: float = 0.25,
    ) -> None:
        """The numpy matcher's keywords, spelled out rather than forwarded through ``**kwargs``.

        :meth:`shipvision.mtmc.tracker.ClusterMTMCTracker._build_matcher` decides what to pass a
        matcher by inspecting its constructor and offering only what it accepts. A ``**options``
        signature accepts nothing by that test, so a native matcher written that way would be
        built with no ground plane at all — cross-camera tracking with the geometry silently
        switched off, which looks like a tuning problem rather than a bug.
        """
        _require_extension()
        super().__init__(
            ground_plane=ground_plane,
            spatial_threshold=spatial_threshold,
            foot_ratio=foot_ratio,
            aspect_ratio=aspect_ratio,
        )

    def ground_distances(self, observations: Sequence[TrackObservation]) -> np.ndarray:
        """See :meth:`~shipvision.mtmc.core.spatial.matcher.SpatialMatcher.ground_distances`."""
        points, known = self.ground_positions(observations)
        if len(observations) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        return np.asarray(
            _C.mtmc_ground_distances(
                np.ascontiguousarray(points, dtype=np.float32),
                np.ascontiguousarray(known, dtype=np.uint8),
            )
        )

    def similarities(self, observations: Sequence[TrackObservation]) -> np.ndarray:
        """``(n, n)`` similarity from ground separation, for the shared ``build``.

        The numpy class expresses this inside ``build``; naming it here is what lets the two
        share one ``to_distance`` rather than each having its own tail.
        """
        if len(observations) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        ground = self.ground_distances(observations)
        return np.asarray(
            _C.mtmc_spatial_similarity(
                np.ascontiguousarray(ground, dtype=np.float64), self.spatial_threshold
            ),
            dtype=np.float32,
        )

    def gate(self, observations: Sequence[TrackObservation]) -> np.ndarray:
        """See :meth:`~shipvision.mtmc.core.spatial.matcher.SpatialMatcher.gate`."""
        if len(observations) == 0:
            return np.zeros((0, 0), dtype=bool)
        ground = self.ground_distances(observations)
        return np.asarray(
            _C.mtmc_spatial_gate(
                np.ascontiguousarray(ground, dtype=np.float64), self.spatial_threshold
            ),
            dtype=bool,
        )


@MTMC_MATCHERS.register("gated", backend=NATIVE)
class NativeGatedMatcher(_NativeMatcherMixin, GatedMatcher):
    """Appearance vetoed by geometry, with both halves and the veto in C++.

    The composition is unchanged — this class owns no distance function, no mask and no
    threshold logic, only the decision about how two independent pieces of evidence combine.
    What it does is make sure both halves are the native ones: a gated matcher built from a
    native appearance matcher and a numpy spatial one would be a third configuration nobody
    tested.
    """

    def __init__(
        self,
        *,
        ground_plane: GroundPlane | None = None,
        appearance_threshold: float = 0.86,
        spatial_threshold: float = 280.0,
        foot_ratio: float = 1.0,
        aspect_ratio: float = 0.25,
        appearance: AppearanceMatcher | None = None,
        spatial: SpatialMatcher | None = None,
    ) -> None:
        """The numpy matcher's keywords exactly — see :class:`NativeSpatialMatcher` on why.

        ``appearance`` and ``spatial`` still take a pre-built half, which is how one side of the
        gate gets A/B'd without rebuilding the other. Passing a numpy half is legal and means
        what it says.
        """
        _require_extension()
        super().__init__(
            appearance=appearance
            or NativeAppearanceMatcher(appearance_threshold=appearance_threshold),
            spatial=spatial
            or NativeSpatialMatcher(
                ground_plane=ground_plane,
                spatial_threshold=spatial_threshold,
                foot_ratio=foot_ratio,
                aspect_ratio=aspect_ratio,
            ),
        )

    def similarities(self, observations: Sequence[TrackObservation]) -> np.ndarray:
        """See :meth:`~shipvision.mtmc.core.gated.matcher.GatedMatcher.similarities`."""
        similarity = self.appearance.similarities(observations)
        if similarity.size == 0:
            return similarity
        return np.asarray(
            _C.mtmc_veto(
                np.ascontiguousarray(similarity, dtype=np.float32),
                np.ascontiguousarray(self.spatial.gate(observations), dtype=np.uint8),
            ),
            dtype=np.float32,
        )
