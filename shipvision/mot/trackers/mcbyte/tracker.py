# Ported from roboflow/trackers (Apache-2.0), src/trackers/core/mcbyte/tracker.py, commit
# ced34f04886da91dc6bec3dfe02f0a0427231ce8. Changed: BoT-SORT's stages are inherited rather
# than copied, and the mask conditioning the reference fuses in is not part of this class yet.
"""McByte: BoT-SORT, with the pairs nothing else was bidding for decided before the solve.

Mesmer et al., "McByte: Mask-Guided Multi-Object Tracking" — the association half, ported from
roboflow/trackers. The paper's other half conditions the remaining cost on propagated
segmentation masks; this class is the part that needs no mask, and the part the mask half will
be layered onto.

Why locking is not a micro-optimisation. :func:`~shipvision.mot.association.solver.associate`
solves for the cheapest *total* and thresholds afterwards, which is right, and has one sharp
edge: the solver will hand track A's only affordable detection to track B in order to buy a
second pair, and if both of those pairs are over the threshold the frame ends with A unmatched
too. Nothing was gained and an identity was lost. A pair that is the only affordable candidate
in both its row and its column is a pair with no such trade available, so McByte takes it off
the table first and solves what is left.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from shipvision.errors import ConfigurationError
from shipvision.mot.association import associate
from shipvision.mot.registry import TRACKERS
from shipvision.mot.trackers.botsort.tracker import BotSortTracker
from shipvision.mot.trackers.mcbyte.utils import clear_matches, reduce_problem
from shipvision.registry import PYTHON
from shipvision.types import Detection

__all__ = ["McByteTracker"]


def _rebase(
    solved: tuple[list[tuple[int, int]], list[int], list[int]],
    rows: Sequence[int],
    columns: Sequence[int],
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Read a solver's answer back out of one reduction, into the indices it was given."""
    matches, unmatched_rows, unmatched_columns = solved
    return (
        [(rows[row], columns[column]) for row, column in matches],
        [rows[row] for row in unmatched_rows],
        [columns[column] for column in unmatched_columns],
    )


@TRACKERS.register("mcbyte", backend=PYTHON, aliases=("mcb", "mc_byte"))
class McByteTracker(BotSortTracker):
    """BoT-SORT plus clear-match locking, on both association stages.

    A subclass of BoT-SORT because that is what the reference is: it keeps the camera motion
    and the min-fused appearance term, and the diff left over is the paper. That term is inert
    without embeddings, so a box-only stream runs ByteTrack's two stages with locking in front.

    Args:
        lock_clear_matches: off is exactly BoT-SORT, which makes the claim measurable.
        **botsort: everything :class:`BotSortTracker` takes, thresholds and ``cmc`` included.
    """

    def __init__(self, *, lock_clear_matches: bool = True, **botsort: object) -> None:
        """
        Raises:
            ConfigurationError: ``lock_clear_matches`` is not a bool. A YAML file that says
                ``false`` unquoted parses to one, but the string ``"false"`` does not, and
                ``bool("false")`` is ``True`` — the switch this class is measured by, silently
                stuck on.
        """
        super().__init__(**botsort)
        if not isinstance(lock_clear_matches, bool):
            raise ConfigurationError(
                f"lock_clear_matches must be a bool, got {type(lock_clear_matches).__name__} "
                f"({lock_clear_matches!r})"
            )
        self._lock_clear_matches = lock_clear_matches

    def _associate(
        self,
        build_cost: Callable[[Sequence[int], Sequence[int]], np.ndarray],
        max_cost: float,
        rows: Sequence[int],
        columns: Sequence[int],
        detections: Sequence[Detection],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Lock what is already decided, then solve the rest.

        The two index maps out of :func:`~shipvision.mot.trackers.mcbyte.utils.reduce_problem`
        are applied on the way back, and then the caller's own ``rows``/``columns`` on top of
        them: two translations, because there are two reductions. Applying only one is the bug
        that associates the wrong objects with every shape still agreeing.
        """
        if not self._lock_clear_matches:
            return super()._associate(build_cost, max_cost, rows, columns, detections)
        if not rows or not columns:
            return [], list(rows), list(columns)

        cost = build_cost(rows, columns)
        locked = clear_matches(cost, max_cost)
        if not locked:
            # Nothing to lock is a real case — 11 of the 50 stage solves over the busy
            # sequence in the tests — and then BoT-SORT's answer *is* the answer. Reaching
            # it through `super()` would rebuild the cost matrix, a stage's expensive half.
            return _rebase(associate(cost, max_cost), rows, columns)

        reduced, kept_rows, kept_columns = reduce_problem(cost, locked)
        matches, unmatched_rows, unmatched_columns = _rebase(
            associate(reduced, max_cost), kept_rows, kept_columns
        )
        return (
            sorted((rows[row], columns[column]) for row, column in locked + matches),
            [rows[row] for row in unmatched_rows],
            [columns[column] for column in unmatched_columns],
        )

    def describe(self) -> str:
        locking = "locked" if self._lock_clear_matches else "not locked"
        return (
            f"McByte: BoT-SORT (camera motion {self.camera_motion.name}) with clear matches "
            f"{locking} before each assignment"
        )
