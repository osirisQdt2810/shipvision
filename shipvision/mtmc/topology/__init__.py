"""Camera-to-ground-plane geometry: the homography, and how much to trust it.

Two cameras watching the same quay from opposite ends produce boxes that share no pixel
coordinates at all. What they do share is the ground: project both boxes' foot points onto
one map and two views of the same person land in the same place, while two different people
do not. That projection is a homography, and it is the only thing that makes a spatial gate
possible.

Split in two, along the line the frame path draws:

* :mod:`~shipvision.mtmc.topology.homography` — :class:`Homography`, :class:`GroundPlane` and
  :func:`project`. Pure numpy, called once per synchronised instant on every track in flight.
* :mod:`~shipvision.mtmc.topology.calibration` — :func:`calculate_homography`. Needs OpenCV,
  called once when somebody clicks correspondences, and returns *how wrong the result is*
  alongside the result.

**OpenCV is optional**, and that is what the split buys: a deployment that receives its
matrices already calibrated — from a file, from a config service — imports the first module
and never touches cv2.
"""

from __future__ import annotations

from shipvision.mtmc.topology.calibration import calculate_homography
from shipvision.mtmc.topology.homography import GroundPlane, Homography, project

__all__ = ["GroundPlane", "Homography", "calculate_homography", "project"]
