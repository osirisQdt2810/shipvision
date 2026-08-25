"""A deterministic, hardware-free detector — the one every other lane tests against.

There is no model here, and that is the point. The tracking, MTMC and pipeline lanes all need
detections whose correct answer is known, over a sequence of frames, with plausible motion —
and loading a real engine to get them would make the offline tier need a GPU, an artefact and
a build.

A mock that returned a fixed box, or a random one, would not do. A fixed box makes every
tracker look perfect: one detection, one track, no association to get wrong. A random box
makes every tracker look broken and, worse, makes a *correct* tracker look broken in a way
that changes between runs. What a tracking test actually needs is the structure real
detections have: **objects that persist across frames, move smoothly, and are indexed by the
camera they are on.**

So this is a synthetic scene rather than a random-number generator:

* Each ``(camera_id, object index)`` owns a **trajectory** — a start position, a constant
  velocity and a size — drawn once from a seeded generator. Different cameras get different
  scenes; the same camera gets the same scene in every process.
* A frame's boxes are that trajectory evaluated at ``frame_id``, reflected off the frame
  border so an object never leaves and the track count stays stable.
* ``jitter`` adds per-frame noise on top, which is what makes an association test meaningful:
  with ``jitter=0`` a nearest-box tracker cannot fail, and the point of a Kalman filter is
  what happens when the measurement is noisy.

Determinism comes from :class:`numpy.random.Generator` seeded on the camera id's stable
64-bit digest, **not** from :func:`hash`, which is salted per process for `str` and would make
the scene differ between runs. A test whose expected answer depends on ``PYTHONHASHSEED`` is
worse than no test.

    detector = DETECTORS.build("mock", objects=3, jitter=1.5)
    frames = [Frame(FrameTag("cam-01", i), image=None, height=1080, width=1920) for i in range(10)]
    tracks = [detector.detect_one(frame) for frame in frames]

Note the ``image=None``: this detector reads only the frame's *extent*, so a lane with no
pixels at all can drive it. That is deliberate — see
:func:`shipvision.detection.base.frame_hw`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import numpy as np

from shipvision.detection.base import DETECTORS, DetectionError, Detector, frame_hw
from shipvision.errors import ConfigurationError
from shipvision.registry import PYTHON
from shipvision.types import Detection, Detections, Frame, FrameTag

__all__ = ["MockDetector"]

#: Mixed into every seed so that ``seed=0`` is not the bare default state of the generator,
#: and so two detectors built with adjacent seeds are properly independent scenes.
_SEED_SALT = 0x5348_4950


def _stable_seed(*parts: object) -> int:
    """A reproducible 64-bit seed from anything, including strings.

    :func:`hash` is salted per process for `str` and `bytes`, so a scene keyed on a camera id
    would differ between runs and between the two workers of the same deployment. blake2b of
    the repr is stable everywhere and costs a microsecond, which is paid once per trajectory
    rather than once per frame.
    """
    digest = hashlib.blake2b("|".join(repr(part) for part in parts).encode(), digest_size=8)
    return int.from_bytes(digest.digest(), "little") ^ _SEED_SALT


@DETECTORS.register("mock", backend=PYTHON, aliases=("fake", "synthetic"))
class MockDetector(Detector):
    """A synthetic scene per camera, evaluated at a frame index.

    Args:
        input_hw: what to report as :attr:`input_hw`. This is the one detector whose input
            extent is *configured*, because it is the one with no artefact to read it from —
            the same exception :class:`~shipvision.reid.extractors.mock.MockExtractor` makes
            for its embedding width. Nothing about the output depends on it; it is here so a
            pipeline built against the mock does not change shape when an engine replaces it.
        objects: how many objects per frame. An ``(low, high)`` pair varies the count per
            camera — inclusive on both ends — which is what makes a *load balancing* test
            meaningful, since the failure ShipInfer exists to fix is a crowded camera
            starving a quiet one.
        class_mix: which class ids appear, and in what proportion. A sequence gives them equal
            weight; a mapping gives ``{class_id: weight}``. Objects are assigned round-robin
            by weight rather than sampled, so the mix is exact for small counts instead of
            approximate.
        jitter: standard deviation, in pixels, of the per-frame noise added to each box
            corner. Zero gives a perfectly smooth trajectory; a few pixels is what a real
            detector does and is what an association test needs.
        speed: pixels per frame of trajectory motion, as a fraction of the frame's smaller
            side. 0.01 on a 1080-high frame is about 11 px/frame, which at 20 fps is a
            realistic walking pace across a scene.
        score_range: ``(low, high)`` confidence. Deterministic per object, not per frame, so a
            score threshold selects the same objects on every frame and a test can predict
            which.
        size_range: ``(low, high)`` box height as a fraction of the frame height. Width
            follows from ``aspect``.
        aspect: box width divided by height.
        fail_every: raise :class:`~shipvision.detection.base.DetectionError` on frames whose
            ``frame_id`` is a multiple of this. `None` never fails. Keyed on the frame id
            rather than on a call counter so that which frames fail is a property of the
            input and not of how the batches happened to be cut — a test can name the frames
            that must fail, and the failure carries the tag of the frame that caused it.
        seed: which scene. Two detectors with different seeds are two different worlds, which
            is how a test builds "the same camera seen by two models" on purpose.
    """

    def __init__(
        self,
        *,
        input_hw: tuple[int, int] = (640, 640),
        objects: int | tuple[int, int] = 3,
        class_mix: Sequence[int] | Mapping[int, float] = (0,),
        jitter: float = 0.0,
        speed: float = 0.01,
        score_range: tuple[float, float] = (0.55, 0.95),
        size_range: tuple[float, float] = (0.08, 0.25),
        aspect: float = 0.5,
        fail_every: int | None = None,
        seed: int = 0,
    ) -> None:
        self._input_hw = _positive_pair(input_hw, "input_hw")
        self._objects = _object_count(objects)
        self._classes, self._weights = _resolve_class_mix(class_mix)
        self._score_range = _ordered_unit_range(score_range, "score_range")
        self._size_range = _ordered_unit_range(size_range, "size_range")

        if jitter < 0.0:
            raise ConfigurationError(f"jitter must be non-negative, got {jitter}")
        if speed < 0.0:
            raise ConfigurationError(f"speed must be non-negative, got {speed}")
        if aspect <= 0.0:
            raise ConfigurationError(f"aspect must be positive, got {aspect}")
        if fail_every is not None and fail_every <= 0:
            raise ConfigurationError(
                f"fail_every must be positive or None, got {fail_every}. Zero would mean "
                f"'fail on no frames', which is what None already says"
            )

        self.jitter = float(jitter)
        self.speed = float(speed)
        self.aspect = float(aspect)
        self.fail_every = None if fail_every is None else int(fail_every)
        self.seed = int(seed)

    # -- introspection ----------------------------------------------------------------

    @property
    def input_hw(self) -> tuple[int, int]:
        return self._input_hw

    # -- the frame path ---------------------------------------------------------------

    def detect(self, frames: Sequence[Frame]) -> list[Detections]:
        """See :meth:`~shipvision.detection.base.Detector.detect`.

        A failing frame aborts the whole call, which is what a real batched backend does: the
        engine executed one batch and the batch failed. The exception names the offending
        frame so the caller can tell which camera to blame, and the frames that would have
        succeeded are *not* returned — a partial result the caller has to re-align by guessing
        is how a detection ends up on the wrong camera.
        """
        return [self._detect_one(frame) for frame in frames]

    def _detect_one(self, frame: Frame) -> Detections:
        tag = frame.tag
        height, width = frame_hw(frame)
        if self.fail_every is not None and tag.frame_id % self.fail_every == 0:
            raise DetectionError(
                f"mock detector configured to fail every {self.fail_every} frames", tag=tag
            )

        count = self._objects_on(tag.camera_id)
        items = [self._object(tag, height, width, index, count) for index in range(count)]
        # Descending score, as every real head returns. Scores are per-object rather than
        # per-frame, so the order is stable from frame to frame and a downstream test can
        # rely on `items[0]` being the same object throughout a sequence.
        items.sort(key=lambda item: (-item.score, item.metadata["mock_object"]))
        return Detections(tag=tag, items=items, height=height, width=width)

    # -- the synthetic scene ----------------------------------------------------------

    def _objects_on(self, camera_id: str) -> int:
        low, high = self._objects
        if low == high:
            return low
        rng = np.random.default_rng(_stable_seed(self.seed, "count", camera_id))
        return int(rng.integers(low, high + 1))

    def _object(
        self, tag: FrameTag, height: int, width: int, index: int, count: int
    ) -> Detection:
        """One object's box on one frame: its trajectory, evaluated at ``frame_id``.

        The trajectory is drawn from a generator seeded on ``(seed, camera_id, index)`` and so
        does not depend on the frame — which is what makes an object *the same object* from
        frame to frame, and is the property a tracker is being tested on.
        """
        rng = np.random.default_rng(_stable_seed(self.seed, "object", tag.camera_id, index))
        box_height = float(rng.uniform(*self._size_range)) * height
        box_width = box_height * self.aspect
        score = float(rng.uniform(*self._score_range))
        class_id = self._class_for(index, count)

        # Position is a reflected ("bounced") walk rather than a wrapped one: wrapping
        # teleports an object from one edge to the other, which no tracker should follow and
        # every tracker would be judged on.
        step = self.speed * min(height, width)
        direction = rng.uniform(-1.0, 1.0, size=2)
        norm = float(np.hypot(*direction)) or 1.0
        velocity = direction / norm * step
        start = np.array(
            [
                rng.uniform(0.0, max(width - box_width, 1.0)),
                rng.uniform(0.0, max(height - box_height, 1.0)),
            ]
        )

        travelled = start + velocity * float(tag.frame_id)
        x = _reflect(travelled[0], max(width - box_width, 1.0))
        y = _reflect(travelled[1], max(height - box_height, 1.0))

        if self.jitter:
            noise = np.random.default_rng(
                _stable_seed(self.seed, "jitter", tag.camera_id, index, tag.frame_id)
            ).normal(scale=self.jitter, size=2)
            x += float(noise[0])
            y += float(noise[1])

        box = np.array([x, y, x + box_width, y + box_height], dtype=np.float32)
        np.clip(box[0::2], 0.0, float(width), out=box[0::2])
        np.clip(box[1::2], 0.0, float(height), out=box[1::2])
        return Detection(
            box=box,
            score=score,
            class_id=class_id,
            metadata={"mock_object": index},
        )

    def _class_for(self, index: int, count: int) -> int:
        """Which class the ``index``-th of ``count`` objects gets.

        An unweighted sequence is plain round-robin. A weighted mapping lays the objects out
        along the cumulative weight instead of sampling from it, because sampling makes the
        answer unstateable: with a 90/10 mix and ten objects, sampling gives "usually one rare
        object, sometimes none, sometimes three" and a test can only assert a distribution.
        Laying them out gives exactly nine and one, every time, which is a scenario with a
        known-correct answer.
        """
        if not self._weights:
            return int(self._classes[index % len(self._classes)])
        total = float(sum(self._weights))
        position = (index + 0.5) / max(count, 1) * total
        cumulative = 0.0
        for class_id, weight in zip(self._classes, self._weights, strict=True):
            cumulative += weight
            if position <= cumulative:
                return int(class_id)
        return int(self._classes[-1])


# ------------------------------------------------------------------------- construction


def _reflect(value: float, limit: float) -> float:
    """``value`` folded into ``[0, limit]`` by reflection, for any distance travelled."""
    period = 2.0 * limit
    folded = float(np.mod(value, period))
    return folded if folded <= limit else period - folded


def _positive_pair(pair: Sequence[int], what: str) -> tuple[int, int]:
    values = tuple(int(v) for v in pair)
    if len(values) != 2 or values[0] <= 0 or values[1] <= 0:
        raise ConfigurationError(f"{what} must be two positive ints, got {pair!r}")
    return values[0], values[1]


def _object_count(objects: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(objects, int):
        if objects < 0:
            raise ConfigurationError(f"objects must be non-negative, got {objects}")
        return objects, objects
    values = tuple(int(v) for v in objects)
    if len(values) != 2 or values[0] < 0 or values[1] < values[0]:
        raise ConfigurationError(
            f"objects must be a count or a (low, high) pair with high >= low >= 0, got "
            f"{objects!r}"
        )
    return values


def _resolve_class_mix(
    class_mix: Sequence[int] | Mapping[int, float],
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """``(class_ids, weights)``; weights are empty for an unweighted sequence."""
    if isinstance(class_mix, Mapping):
        if not class_mix:
            raise ConfigurationError("class_mix must name at least one class")
        classes = tuple(int(k) for k in class_mix)
        weights = tuple(float(v) for v in class_mix.values())
        if any(w <= 0.0 for w in weights):
            raise ConfigurationError(
                f"class_mix weights must be positive, got {list(class_mix.values())}. A "
                f"weight of zero means 'do not include this class' — leave it out"
            )
        return classes, weights
    classes = tuple(int(v) for v in class_mix)
    if not classes:
        raise ConfigurationError("class_mix must name at least one class")
    if any(c < 0 for c in classes):
        raise ConfigurationError(f"class ids must be non-negative, got {list(classes)}")
    return classes, ()


def _ordered_unit_range(pair: Sequence[float], what: str) -> tuple[float, float]:
    values = tuple(float(v) for v in pair)
    if len(values) != 2 or not 0.0 <= values[0] <= values[1] <= 1.0:
        raise ConfigurationError(
            f"{what} must be (low, high) with 0 <= low <= high <= 1, got {pair!r}"
        )
    return values
