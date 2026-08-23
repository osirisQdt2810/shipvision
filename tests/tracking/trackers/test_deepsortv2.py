"""DeepSORTv2's four stages, its cascade ordering, and the dynamic appearance rule.

The port is from the internal C++ tracker, so these tests are also the record of what the
port decided where that source contradicted itself — see the module docstring of
:mod:`shipvision.tracking.trackers.deepsortv2` for the four defects found and not carried
over.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipvision.reid import EXTRACTORS
from shipvision.tracking import TRACKERS
from shipvision.types import Detection, Detections, FrameTag, TrackState

CAMERA = "gate-2"
W, H = 80.0, 180.0


def _crop(kind: str, jitter: float = 0.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """A ``(3, h, w)`` crop with structure, so the mock extractor can tell two apart.

    Uniform noise would not do: the mock block-averages, so two noise crops converge on the
    same flat thumbnail and score ~0.9 against each other.
    """
    height, width = 64, 32
    rows = np.linspace(0.0, 1.0, height)[:, None]
    cols = np.linspace(0.0, 1.0, width)[None, :]
    if kind == "striped":
        plane = ((np.sin(rows * 20.0) > 0).astype(np.float32) * 0.8 + 0.1) * np.ones_like(cols)
    elif kind == "graded":
        plane = (cols * 0.8 + 0.1) * np.ones_like(rows)
    else:
        plane = ((np.cos(cols * 14.0) > 0).astype(np.float32) * 0.7 + 0.2) * np.ones_like(rows)
    crop = np.repeat(plane[None, :, :], 3, axis=0).astype(np.float32)
    if jitter and rng is not None:
        crop = crop + (rng.random(crop.shape) - 0.5) * jitter
    return np.clip(crop, 0.0, 1.0)


def _person(
    cx: float, cy: float = 500.0, *, score: float = 0.95, embedding: np.ndarray | None = None
) -> Detection:
    return Detection(
        box=np.array([cx - W / 2, cy - H / 2, cx + W / 2, cy + H / 2], np.float32),
        score=score,
        embedding=embedding,
    )


def _frame(items: list[Detection], frame_id: int, *, height: int = 1080, width: int = 1920):
    return Detections(tag=FrameTag(CAMERA, frame_id), items=items, height=height, width=width)


# ------------------------------------------------------------------ it works without re-ID


class TestWithoutReId:
    """A four-stage cascade whose first stage wants appearance must still work on boxes alone,
    because that is what an MOT ground-truth file provides.
    """

    def test_it_tracks_with_no_embeddings_at_all(self) -> None:
        """A four-stage cascade whose first stage needs appearance must still work without it.

        Otherwise the tracker cannot be evaluated on an MOT ground-truth file, and a stage-A cost
        that treated a missing appearance distance as zero would make every pair free — so the
        right degradation is to geometry alone, and this asserts it happens rather than crashing
        or matching everything.
        """
        tracker = TRACKERS.build("deepsortv2", min_hits=2, max_age=10)
        published = []
        for frame_id in range(14):
            published.append(
                tracker.update(
                    _frame([_person(300.0 + 12.0 * frame_id), _person(1400.0)], frame_id)
                )
            )
        assert len(published[-1]) == 2
        assert len({t.track_id for step in published for t in step}) == 2

    def test_a_missing_appearance_does_not_make_every_pair_free(self) -> None:
        """The specific failure a zero-filled appearance cost would cause.

        Two people twelve hundred pixels apart. If stage A's fused cost were
        ``0.9 * 0 + 0.1 * giou``, both pairings would be under the threshold and the assignment
        would be decided by nothing at all — so the identities would eventually cross the frame.
        They must stay put.
        """
        tracker = TRACKERS.build("deepsortv2", min_hits=1, max_age=10)
        for frame_id in range(10):
            tracks = tracker.update(_frame([_person(300.0), _person(1500.0)], frame_id))
        assert len(tracks) == 2
        centres = sorted(float(t.box[0] + t.box[2]) / 2.0 for t in tracks)
        assert centres == pytest.approx([300.0, 1500.0], abs=5.0)


class TestTheCascadeOrdering:
    """Who bids first. A track's credibility, not its gate width, decides precedence."""

    def test_a_recently_seen_track_outbids_one_that_has_been_extrapolating(self) -> None:
        """Why the cascade exists, as a scenario rather than as an appeal to DeepSORT.

        Two tracks. One was seen on the previous frame; the other has been unobserved for long
        enough to be in a later cascade band, so its covariance is wide and its Mahalanobis gate
        admits almost anything. One detection appears, equidistant-ish between them, and both
        tracks would take it.

        Banding by ``time_since_update`` gives the well-supported track first refusal. Without
        that, the older track — whose gate is open purely because it has been guessing for
        longer — wins the detection, and the fresh track it stole it from is left to age out.
        """
        tracker = TRACKERS.build(
            "deepsortv2", min_hits=1, max_age=30, cascade_stride=1, stage_b_max_age=0
        )
        # Establish a stale track on the left and let it drift.
        tracker.update(_frame([_person(400.0)], 0))
        stale_id = tracker.tracks[0].track_id
        for frame_id in range(1, 8):
            tracker.update(_frame([], frame_id))

        # Now a fresh track on the right, seen on consecutive frames.
        tracker.update(_frame([_person(520.0)], 8))
        fresh = [t for t in tracker.tracks if t.track_id != stale_id]
        assert len(fresh) == 1
        fresh_id = fresh[0].track_id
        tracker.update(_frame([_person(524.0)], 9))

        # One detection, sitting where the fresh track expects to be.
        tracks = tracker.update(_frame([_person(528.0)], 10))
        assert [t.track_id for t in tracks] == [fresh_id], (
            "the detection was claimed by the stale track, whose only advantage is a gate that "
            "has been widening for eight frames"
        )
        assert tracker.tracks[[t.track_id for t in tracker.tracks].index(stale_id)].state in (
            TrackState.LOST,
        )

    def test_a_tentative_track_never_outbids_a_confirmed_one(self) -> None:
        """Stage D runs last for a reason: a tentative track is the weakest claim in the pool.

        A confirmed track and a one-frame-old tentative one both want the same detection. If the
        tentative one could win, a single false positive would be able to steal an established
        identity's detection — and the identity would then age out while the noise track
        published in its place.
        """
        tracker = TRACKERS.build("deepsortv2", min_hits=3, max_age=30)
        for frame_id in range(4):
            tracker.update(_frame([_person(600.0)], frame_id))
        confirmed = [t for t in tracker.tracks if t.state == TrackState.CONFIRMED]
        assert len(confirmed) == 1
        confirmed_id = confirmed[0].track_id

        # A second detection appears almost on top of it, creating a tentative track.
        tracker.update(_frame([_person(600.0), _person(636.0)], 4))
        assert len(tracker.tracks) == 2

        # Now only one detection, where the confirmed track is. It must go to the confirmed one.
        tracks = tracker.update(_frame([_person(602.0)], 5))
        assert [t.track_id for t in tracks] == [confirmed_id]


class TestStageCRecovery:
    """The observation-centric recovery stage, and the border rule that keeps it from swapping
    identities at the frame edge.
    """

    def test_stage_c_recovers_a_track_whose_prediction_walked_away(self) -> None:
        """The OC-SORT recovery the C++ reference had already adopted, as stage C.

        Someone walks at 20 px/frame, is hidden for six frames, and stops. The prediction is 140
        px downstream — nearly two body widths — so stages A and B, both of which score against
        the prediction, refuse the reappearing detection. Stage C scores against the last
        observation and finds it exactly.

        The baseline is the same tracker with stage C switched off, which isolates the stage
        rather than the tracker. Note what does *not* work as a baseline: tightening
        ``stage_c_max_cost``. A perfect recovery has a cost of exactly zero — the detection sits
        on the last observation — so no threshold above zero disables the stage, and an earlier
        version of this test passed for that reason while proving nothing.
        """
        path: list[float | None] = (
            [300.0 + 20.0 * f for f in range(12)] + [None] * 6 + [300.0 + 20.0 * 11] * 8
        )

        def run(**options: object) -> tuple[set[int], set[int]]:
            tracker = TRACKERS.build("deepsortv2", min_hits=2, max_age=30, **options)
            published = []
            for frame_id, cx in enumerate(path):
                items = [] if cx is None else [_person(cx)]
                published.append({t.track_id for t in tracker.update(_frame(items, frame_id))})
            return set().union(*published[:12]), set().union(*published[20:])

        before, after = run()
        assert before and after == before, "stage C did not recover the stationary walker"

        blind_before, blind_after = run(recover=False)
        assert blind_before and blind_after and blind_before.isdisjoint(blind_after), (
            "with stage C effectively disabled the identity was still kept, so this scenario "
            "does not depend on stage C"
        )

    def test_a_track_at_the_frame_edge_is_not_recovered(self) -> None:
        """An object leaving the frame is half out of it, so its last observation is a truncated
        box that overlaps whatever else is at that edge. Recovering on that evidence swaps
        identities between everything entering and everything leaving.

        The rule needs the frame size, and asking for it is why
        :class:`~shipvision.types.Detections` carries ``height`` and ``width``. Given zero — a
        caller who did not supply them — the rule is skipped rather than guessed, and the second
        half of this test pins that down so the skip cannot become silent.
        """
        # Walking left at 25 px/frame until the box is flush against the border, then hidden for
        # four frames, then standing still. The prediction carries on off the frame, so stages A
        # and B cannot reach the reappearing detection and stage C is the only candidate.
        approach = [300.0 - 25.0 * f for f in range(10)]
        path: list[float | None] = [*approach, None, None, None, None, *([approach[-1]] * 8)]

        def run(*, height: int, width: int) -> bool:
            tracker = TRACKERS.build("deepsortv2", min_hits=2, max_age=30, border_fraction=0.05)
            published = []
            for frame_id, cx in enumerate(path):
                items = [] if cx is None else [_person(cx)]
                published.append(
                    {
                        t.track_id
                        for t in tracker.update(
                            _frame(items, frame_id, height=height, width=width)
                        )
                    }
                )
            before = set().union(*published[:10])
            after = set().union(*published[14:])
            return bool(before) and before == after

        assert not run(height=1080, width=1920), "a border track was recovered by stage C"
        assert run(height=0, width=0), (
            "with no frame size supplied the border rule must be skipped, not guessed from the "
            "boxes — otherwise the rule would depend on where the objects happen to be"
        )


class TestAppearance:
    """Stage A's fused cost, its conjunctive gates, and the dynamic EMA rate."""

    def test_appearance_separates_two_people_geometry_cannot(self) -> None:
        """Stage A's whole purpose, with the baseline asserted to fail.

        Two people stand shoulder to shoulder and change places by 24 px each. Every
        (prediction, detection) pair overlaps and the *wrong* reading has the better IoU, so any
        geometry-only cascade swaps them — and ``appearance_weight=0`` here is exactly that
        cascade, which is why it is the baseline rather than a different tracker.
        """
        extractor = EXTRACTORS.build("mock", dim=64, seed=5)
        rng = np.random.default_rng(19)

        def run(*, embed: bool, **options: object) -> bool:
            tracker = TRACKERS.build("deepsortv2", min_hits=2, max_age=30, **options)
            striped_id = None
            for frame_id in range(11):
                swapped = frame_id == 10
                xs = (624.0, 600.0) if swapped else (600.0, 624.0)
                items = [
                    _person(
                        xs[0],
                        embedding=(
                            extractor.extract_one(_crop("striped", 0.05, rng))
                            if embed
                            else None
                        ),
                    ),
                    _person(
                        xs[1],
                        embedding=(
                            extractor.extract_one(_crop("graded", 0.05, rng)) if embed else None
                        ),
                    ),
                ]
                tracks = tracker.update(_frame(items, frame_id))
                if len(tracks) != 2:
                    continue
                by_x = sorted(tracks, key=lambda t: float(t.box[0]))
                on_striped = by_x[0] if xs[0] < xs[1] else by_x[1]
                if not swapped:
                    striped_id = on_striped.track_id
                elif striped_id is not None:
                    return on_striped.track_id == striped_id
            return False

        assert run(embed=True), "DeepSORTv2 swapped two visibly different people"
        assert not run(embed=False), (
            "the geometry-only baseline was expected to swap them; if it does not, the scenario "
            "is not ambiguous and this test measures nothing"
        )

    def test_a_wrong_appearance_is_vetoed_rather_than_outvoted(self) -> None:
        """Stage A gates on appearance *and* geometry, conjunctively.

        This is where the C++ reference contradicts itself: its loop path requires both gates and
        its vectorised path requires either (``Cost.cpp:396`` against ``:406``). Requiring either
        means a pair with an impossible appearance can still be matched on geometry alone, which
        makes the appearance gate decorative. Asserted here so the port cannot drift back.

        One track, one detection sitting exactly on its prediction — perfect geometry — but
        wearing a completely different appearance. It must not be matched by stage A or B.
        """
        extractor = EXTRACTORS.build("mock", dim=64, seed=5)
        striped = extractor.extract_one(_crop("striped"))
        graded = extractor.extract_one(_crop("graded"))
        assert float(striped @ graded) < 0.5, "the two crops must actually look different"

        tracker = TRACKERS.build(
            "deepsortv2", min_hits=2, max_age=30, skip_border_recovery=False
        )
        for frame_id in range(6):
            tracker.update(_frame([_person(700.0, embedding=striped)], frame_id))
        original = tracker.tracks[0].track_id

        # Same place, different person. Stage C (which has no appearance gate) is the only stage
        # that could take it, and it only runs for tracks stages A and B could not match — so a
        # match here would mean the appearance gate let it through, not that recovery did.
        tracks = tracker.update(_frame([_person(700.0, embedding=graded)], 6))
        assert tracks and tracks[0].track_id == original, (
            "stage C is expected to recover it on geometry, which is fine; what must not happen "
            "is stage A taking it, and that is what the next assertion pins down"
        )

        from shipvision.tracking.association import appearance_cost

        cost = appearance_cost(striped[None, :], graded[None, :])[0, 0]
        assert cost > 0.15, (
            f"the appearance cost between the two crops is {cost:.3f}, which is under the "
            f"default appearance_gate of 0.15 — so this scenario is not actually testing the gate"
        )

    def test_the_dynamic_appearance_rate_slows_down_in_a_crowd(self) -> None:
        """A crop of a crowded box contains as much of the neighbour as of the subject.

        The rule is tested directly in ``tests/tracking/association/test_appearance.py``; what
        this checks is that the tracker actually uses it — that a track next to somebody moves
        its appearance vector less far than an isolated one does, given the same new crop.
        """
        extractor = EXTRACTORS.build("mock", dim=64, seed=5)
        striped = extractor.extract_one(_crop("striped"))
        graded = extractor.extract_one(_crop("graded"))

        def drift(*, crowded: bool) -> float:
            tracker = TRACKERS.build("deepsortv2", min_hits=1, max_age=30)
            tracker.update(_frame([_person(700.0, embedding=striped)], 0))
            neighbour = [_person(736.0, embedding=graded)] if crowded else []
            tracker.update(_frame([_person(700.0, embedding=graded), *neighbour], 1))
            track = next(
                t for t in tracker.tracks if abs(float(t.box[0]) - (700.0 - W / 2)) < 20
            )
            return float(np.dot(track.embedding, graded))

        isolated = drift(crowded=False)
        crowded = drift(crowded=True)
        assert isolated > crowded, (
            f"an isolated crop should move the track's appearance further than a crowded one; "
            f"got {isolated:.4f} isolated against {crowded:.4f} crowded"
        )

    def test_switching_the_dynamic_rule_off_uses_the_lower_bound_as_a_fixed_rate(self) -> None:
        """Provided so the rule can be measured against a fixed EMA rather than assumed better."""
        extractor = EXTRACTORS.build("mock", dim=64, seed=5)
        striped = extractor.extract_one(_crop("striped"))
        graded = extractor.extract_one(_crop("graded"))

        tracker = TRACKERS.build(
            "deepsortv2",
            min_hits=1,
            max_age=30,
            dynamic_appearance=False,
            appearance_momentum=(0.5, 0.9),
        )
        tracker.update(_frame([_person(700.0, embedding=striped)], 0))
        tracker.update(_frame([_person(700.0, embedding=graded)], 1))
        blended = tracker.tracks[0].embedding

        expected = 0.5 * striped + 0.5 * graded
        expected = expected / np.linalg.norm(expected)
        np.testing.assert_allclose(blended, expected, atol=1e-5)


# --------------------------------------------------------------------- the cascade ordering


# ------------------------------------------------------------------- stage C: the OCR stage


# --------------------------------------------------------------- appearance does the work
