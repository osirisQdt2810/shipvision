"""Ground-plane geometry: where each track is standing, and how far apart two are.

Appearance says two crops look alike. Geometry says whether they *can* be the same object,
and it is much harder to fool: two crew members in identical overalls score high on
appearance from any model, and are forty metres apart on the quay.

The estimate has one interesting case, and it is the common one. A person's ground position
is under their feet, so the foot point is the bottom-centre of the box — unless the box is
clipped by the bottom edge of the frame, in which case the feet are *outside* the image and
the bottom-centre is somewhere around the waist. The reference detects that with an aspect
test: a person is roughly four times taller than they are wide, so a box touching the bottom
edge has its foot estimated at ``width / aspect_ratio`` below its top rather than at its own
bottom. Skip that and every track in the near field of every camera projects metres short of
where it is, consistently, which reads as a systematic map offset rather than as a bug.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from shipvision.errors import ConfigurationError
from shipvision.mtmc.frames import TrackObservation
from shipvision.mtmc.matrix.base import MATRIX_BUILDERS, BaseMatrixBuilder
from shipvision.mtmc.topology import GroundPlane
from shipvision.registry import PYTHON

__all__ = ["SpatialMatrixBuilder", "foot_points"]


def foot_points(
    boxes: np.ndarray,
    frame_heights: np.ndarray,
    *,
    foot_ratio: float = 1.0,
    aspect_ratio: float = 0.25,
) -> np.ndarray:
    """``(n, 4)`` xyxy boxes to ``(n, 2)`` image points where each object meets the ground.

    Args:
        boxes: ``(n, 4)`` xyxy, absolute pixels.
        frame_heights: ``(n,)`` the frame height each box was measured in.
        foot_ratio: where the ground is within an un-clipped box, as a fraction of its height
            from the top. 1.0 is its bottom edge, which is right for a person standing.
        aspect_ratio: width-to-height ratio of a whole, un-clipped object — 0.25 meaning a
            person is four times taller than they are wide. Used only to extrapolate a box
            that the bottom of the frame cut off.

    Vectorised over the whole group: this runs once per synchronised instant over every track
    in flight, and the arithmetic is four ufuncs against a thousand Python-level branches.
    """
    box = np.atleast_2d(np.asarray(boxes, dtype=np.float64))
    heights = np.asarray(frame_heights, dtype=np.float64).reshape(-1)
    if box.shape[-1] != 4:
        raise ConfigurationError(f"boxes must be (n, 4) xyxy, got shape {box.shape}")
    if heights.shape[0] != box.shape[0]:
        raise ConfigurationError(
            f"{box.shape[0]} boxes against {heights.shape[0]} frame heights"
        )
    width = box[:, 2] - box[:, 0]
    height = box[:, 3] - box[:, 1]
    truncated = box[:, 3] >= heights - 1.0
    drop = np.where(
        truncated,
        np.maximum(height, width / max(aspect_ratio, 1e-6)),
        height * foot_ratio,
    )
    return np.stack([(box[:, 0] + box[:, 2]) * 0.5, box[:, 1] + drop], axis=1)


@MATRIX_BUILDERS.register("spatial", backend=PYTHON)
class SpatialMatrixBuilder(BaseMatrixBuilder):
    """Euclidean distance between track foot points projected onto a shared ground plane.

    Usable on its own only where every camera is calibrated and the scene is sparse enough
    that position alone identifies an object; its real job is to be the gate inside
    :class:`~shipvision.mtmc.matrix.gated.GatedMatrixBuilder`. The two are kept as separate
    classes so that the projection has a test of its own — a gate whose geometry is wrong and
    whose appearance is right produces output that looks fine until two people walk past each
    other.
    """

    def __init__(
        self,
        *,
        ground_plane: GroundPlane | None = None,
        spatial_threshold: float = 280.0,
        foot_ratio: float = 1.0,
        aspect_ratio: float = 0.25,
    ) -> None:
        """
        Args:
            ground_plane: the camera-to-map homographies. An empty one is legal and means
                nothing can be judged spatially, which the gate handles by falling back to
                appearance.
            spatial_threshold: how far apart, in ground-plane units, two projections may be
                and still be the same object. The reference's production value is 280 map
                pixels. Used by :meth:`build`; the gate reads its own copy.
            foot_ratio: see :func:`foot_points`.
            aspect_ratio: see :func:`foot_points`.
        """
        if spatial_threshold <= 0.0:
            raise ConfigurationError(
                f"spatial_threshold must be positive, got {spatial_threshold}"
            )
        if not 0.0 < aspect_ratio <= 4.0:
            raise ConfigurationError(
                f"aspect_ratio is width over height for a whole object and must be in "
                f"(0, 4], got {aspect_ratio}"
            )
        if not 0.0 < foot_ratio <= 2.0:
            raise ConfigurationError(f"foot_ratio must be in (0, 2], got {foot_ratio}")
        self.ground_plane = ground_plane if ground_plane is not None else GroundPlane()
        self.spatial_threshold = float(spatial_threshold)
        self.foot_ratio = float(foot_ratio)
        self.aspect_ratio = float(aspect_ratio)

    def ground_positions(
        self, observations: Sequence[TrackObservation]
    ) -> tuple[np.ndarray, np.ndarray]:
        """``((n, 2)`` ground points, ``(n,)`` bool "this one is calibrated"``)``.

        Cameras without a homography get a position of ``(nan, nan)`` and a false flag rather
        than ``(0, 0)`` and a side-list. The reference used the origin plus a list of invalid
        indices, and the origin is a real place on the map — one forgotten check away from
        every uncalibrated camera's tracks being coincident with each other.
        """
        count = len(observations)
        points = np.full((count, 2), np.nan, dtype=np.float32)
        known = np.zeros(count, dtype=bool)
        if count == 0:
            return points, known

        by_camera: dict[str, list[int]] = {}
        for index, observation in enumerate(observations):
            by_camera.setdefault(observation.camera_id, []).append(index)

        for camera_id, indices in by_camera.items():
            homography = self.ground_plane.get(camera_id)
            if homography is None:
                continue
            group = [observations[i] for i in indices]
            image_points = foot_points(
                np.stack([o.box for o in group]),
                np.array([o.frame_height for o in group], dtype=np.float64),
                foot_ratio=self.foot_ratio,
                aspect_ratio=self.aspect_ratio,
            )
            # Every observation in this group shares a camera, so one frame size applies.
            points[indices] = homography.project(
                image_points,
                frame_width=group[0].frame_width,
                frame_height=group[0].frame_height,
            )
            known[indices] = True
        return points, known

    def ground_distances(self, observations: Sequence[TrackObservation]) -> np.ndarray:
        """``(n, n)`` euclidean distance on the ground plane; ``inf`` where unknowable.

        ``inf`` is the honest value for a pair where at least one camera is uncalibrated: not
        "far apart" and not "close", but "this builder has nothing to say". The consumer
        decides what that means, and the two consumers decide differently — see
        :meth:`build` and the gate. This is an internal primitive, so the non-finite value
        never reaches a clusterer.
        """
        points, known = self.ground_positions(observations)
        count = len(observations)
        if count == 0:
            return np.zeros((0, 0), dtype=np.float32)
        delta = points[:, None, :] - points[None, :, :]
        distance = np.sqrt(np.sum(delta.astype(np.float64) ** 2, axis=2))
        unknown = ~(known[:, None] & known[None, :])
        distance[unknown] = np.inf
        return distance.astype(np.float64)

    def build(self, observations: Sequence[TrackObservation]) -> np.ndarray:
        """Distances for clustering on position alone.

        An unknowable pair becomes :data:`NEVER_MERGE`, the opposite of what the gate does
        with it, and both are right: with no other evidence in play, "I cannot tell" must not
        become "merge them", whereas the gate still has appearance to fall back on.
        """
        count = len(observations)
        if count == 0:
            return np.zeros((0, 0), dtype=np.float32)
        ground = self.ground_distances(observations)
        # Map to a similarity in (0, 1] so the shared conversion applies: 1.0 at zero
        # separation, falling linearly to 0 at the threshold, and exactly 0 beyond it.
        with np.errstate(invalid="ignore"):
            similarity = np.where(
                np.isfinite(ground) & (ground < self.spatial_threshold),
                1.0 - ground / self.spatial_threshold,
                0.0,
            )
        np.fill_diagonal(similarity, 1.0)
        return self.to_distance(similarity, self.mergeable_mask(observations))

    def gate(self, observations: Sequence[TrackObservation]) -> np.ndarray:
        """``(n, n)`` bool: true where geometry does not object to a merge.

        True when the two projections are within ``spatial_threshold`` **and** when the pair
        cannot be judged at all. Falling open on "cannot judge" is what lets an uncalibrated
        camera keep taking part in cross-camera tracking with appearance alone, instead of
        quietly never merging with anyone.
        """
        ground = self.ground_distances(observations)
        with np.errstate(invalid="ignore"):
            return ~np.isfinite(ground) | (ground < self.spatial_threshold)

    def __repr__(self) -> str:
        return (
            f"<SpatialMatrixBuilder cameras={len(self.ground_plane)} "
            f"spatial_threshold={self.spatial_threshold} backend={self.backend}>"
        )
