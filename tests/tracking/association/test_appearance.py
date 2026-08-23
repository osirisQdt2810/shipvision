"""The dynamic appearance rule: how fast a track's appearance should follow the last crop.

A rate is not a number anyone can eyeball, so this file pins down the *shape* of the rule
instead: which way each input moves it, what the bounds are, and the two conjunction choices
that make it conservative rather than merely configurable.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipvision.errors import ConfigurationError
from shipvision.tracking.association import dynamic_appearance_momentum, isolation


def _box(cx: float, w: float = 60.0, h: float = 140.0) -> list[float]:
    return [cx - w / 2, 500 - h / 2, cx + w / 2, 500 + h / 2]


class TestIsolation:
    """``1 - `` the largest IoU with any *other* box."""

    def test_a_lone_detection_has_the_frame_to_itself(self) -> None:
        """The self-IoU of 1.0 has to be *excluded*, not ignored: a ``max`` over a row that
        includes it is always 1.0, so every detection would look maximally crowded and the
        rule would be a constant."""
        assert isolation(np.array([_box(400)], np.float32))[0] == pytest.approx(1.0)

    def test_overlapping_detections_score_lower_than_separated_ones(self) -> None:
        crowded = isolation(np.array([_box(400), _box(420)], np.float32))
        apart = isolation(np.array([_box(400), _box(1400)], np.float32))
        assert crowded[0] < apart[0]
        assert apart[0] == pytest.approx(1.0)

    def test_it_is_symmetric_for_a_pair(self) -> None:
        scores = isolation(np.array([_box(400), _box(430)], np.float32))
        assert scores[0] == pytest.approx(scores[1])

    def test_an_empty_frame_gives_an_empty_result(self) -> None:
        assert isolation(np.zeros((0, 4), np.float32)).shape == (0,)


class TestDynamicAppearanceMomentum:
    """Retention, where high means "barely update"."""

    def test_a_confident_isolated_crop_moves_a_track_the_furthest(self) -> None:
        boxes = np.array([_box(400)], np.float32)
        confident = dynamic_appearance_momentum(boxes, np.array([0.95], np.float32))
        unsure = dynamic_appearance_momentum(boxes, np.array([0.40], np.float32))
        assert confident[0] < unsure[0]
        assert confident[0] == pytest.approx(0.9, abs=1e-6)

    def test_crowding_holds_a_track_back_even_when_the_detector_is_certain(self) -> None:
        """The conjunction, and the reason it is a maximum rather than a blend.

        A very confident detection in the middle of a crowd would win a weighted average and
        overwrite a track's appearance with a crop that is half somebody else. Taking the
        larger of the two retentions means failing *either* test is enough to hold the track
        back.
        """
        crowded = np.array([_box(400), _box(420)], np.float32)
        scores = np.array([0.99, 0.99], np.float32)
        assert dynamic_appearance_momentum(crowded, scores)[0] > 0.9

    def test_the_result_stays_inside_the_bounds(self) -> None:
        rng = np.random.default_rng(3)
        boxes = np.array([_box(400 + 30 * i) for i in range(6)], np.float32)
        for _ in range(5):
            scores = rng.uniform(0.0, 1.0, size=6).astype(np.float32)
            momentum = dynamic_appearance_momentum(
                boxes, scores, min_momentum=0.85, max_momentum=0.97
            )
            assert np.all((momentum >= 0.85 - 1e-6) & (momentum <= 0.97 + 1e-6))

    def test_the_cap_is_below_one(self) -> None:
        """A track whose appearance can never update at all is a track that will eventually
        fail to match itself, so the worst case still moves a little."""
        crowded = np.array([_box(400), _box(402)], np.float32)
        momentum = dynamic_appearance_momentum(crowded, np.array([0.05, 0.05], np.float32))
        assert momentum[0] < 1.0

    def test_the_bands_clamp_before_they_rescale(self) -> None:
        """0.95 and 0.99 are not meaningfully different qualities of crop. Letting the band
        run to the extremes would spend most of its range distinguishing them."""
        boxes = np.array([_box(400)], np.float32)
        a = dynamic_appearance_momentum(boxes, np.array([0.85], np.float32))
        b = dynamic_appearance_momentum(boxes, np.array([0.99], np.float32))
        assert a[0] == pytest.approx(b[0])

    def test_an_empty_frame_gives_an_empty_result(self) -> None:
        assert dynamic_appearance_momentum(
            np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
        ).shape == (0,)

    def test_bad_bounds_are_refused_at_the_call(self) -> None:
        boxes = np.array([_box(400)], np.float32)
        scores = np.array([0.9], np.float32)
        with pytest.raises(ConfigurationError, match="min_momentum"):
            dynamic_appearance_momentum(boxes, scores, min_momentum=0.9, max_momentum=0.5)
        with pytest.raises(ConfigurationError, match="min_momentum"):
            dynamic_appearance_momentum(boxes, scores, min_momentum=0.5, max_momentum=1.0)
        with pytest.raises(ConfigurationError, match="conf_range"):
            dynamic_appearance_momentum(boxes, scores, conf_range=(0.8, 0.5))

    def test_mismatched_lengths_are_refused(self) -> None:
        """Two models in one pipeline is how this happens, and a broadcast that quietly
        succeeded would produce a plausible-looking rate for the wrong detection."""
        with pytest.raises(ConfigurationError, match="scores"):
            dynamic_appearance_momentum(
                np.array([_box(400), _box(900)], np.float32), np.array([0.9], np.float32)
            )
