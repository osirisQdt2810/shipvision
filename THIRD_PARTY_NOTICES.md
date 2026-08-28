# Third-party notices

shipvision is MIT-licensed. This file lists the third-party code it contains, the licence
that code arrives under, and — as Apache-2.0 §4(b) requires — what was changed.

## roboflow/trackers — Apache License 2.0

    Trackers
    Copyright (c) 2026 Roboflow. All Rights Reserved.
    Licensed under the Apache License, Version 2.0

- **Upstream:** https://github.com/roboflow/trackers
- **Commit:** `ced34f04886da91dc6bec3dfe02f0a0427231ce8`
- **Taken from:** `src/trackers/core/mcbyte/` (the McByte association)
- **Lands in:** `shipvision/mot/trackers/mcbyte/`, and the fixture generator
  `tests/mot/trackers/data/gen_mcbyte_golden.py`

A verbatim copy of the Apache License 2.0 is vendored at `LICENSES/Apache-2.0.txt`, because
§4(a) asks that recipients *receive* the License rather than a link to it — a URL is not a copy,
and an air-gapped deployment is the case that makes the difference concrete. Every file derived
from that source carries a header naming the upstream path and commit.

### Changes made

- **Similarity became cost.** The reference maximises a similarity and thresholds with `>=`;
  this library minimises a cost and thresholds with `<=`. Every predicate is converted:
  `cost = 1 - similarity`, `max_cost = 1 - minimum_similarity`, and "positive IoU" is
  `iou_cost < 1`.
- **The mask conditioning is not ported here.** `condition_similarity_with_masks` fuses
  locking, ambiguity, isolation and mask evidence into one call; the locking and the three
  predicates are split into `mcbyte/utils.py` as separate functions. The mask half — SAM,
  Cutie, the propagation manager and the out-of-memory handling — is not ported: no model runs
  inside `shipvision.mot`.
- **The tracker subclasses BoT-SORT.** The reference reimplements BoT-SORT's stages; here
  McByte inherits them and overrides one association hook, so the diff between the two is the
  paper's contribution and nothing else.
- **Lifecycle, ids and the Kalman state are this library's.** `(cx, cy, aspect, height)` with
  scale-dependent noise rather than `(xc, yc, w, h)`; one shared `TrackPool` rather than a
  per-algorithm tracklet class; process-wide track ids rather than per-instance ones;
  unassigned ids are `None` rather than `-1`. End-to-end numeric parity with the reference is
  therefore not claimed and not attempted.
- **Errors are typed.** `ValueError` became `ConfigurationError` or `TrackingError`, and the
  `warnings.warn` calls were dropped.
- **`scipy.optimize.linear_sum_assignment` is reached through
  `shipvision.mot.association.solver`** rather than called directly.
