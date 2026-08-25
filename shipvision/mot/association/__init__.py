"""Association: costs, the solver that consumes them, and the appearance policy.

One directory because these three answer one question — *which detection is which track* —
and they are the only part of a tracker that differs between algorithms. Everything else (the
filter, the lifecycle, the camera) is shared.
"""

from shipvision.mot.association.appearance import (
    dynamic_appearance_momentum,
    isolation,
    pairwise_appearance,
)
from shipvision.mot.association.costs import (
    INFEASIBLE,
    appearance_cost,
    direction_cost,
    fuse_score,
    gate_cost,
    gated_iou_cost,
    giou_cost,
    giou_matrix,
    iou_cost,
    min_fuse,
)
from shipvision.mot.association.solver import associate, associate_subset, cascade_associate

__all__ = [
    "INFEASIBLE",
    "appearance_cost",
    "associate",
    "associate_subset",
    "cascade_associate",
    "direction_cost",
    "dynamic_appearance_momentum",
    "fuse_score",
    "gate_cost",
    "gated_iou_cost",
    "giou_cost",
    "giou_matrix",
    "iou_cost",
    "isolation",
    "min_fuse",
    "pairwise_appearance",
]
