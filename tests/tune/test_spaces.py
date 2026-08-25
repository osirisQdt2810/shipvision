"""Search spaces: validated against the constructor, or they do not exist.

The test that matters most is the boring one — that every declared space's parameter names are
names the tracker actually accepts. A typo there tunes nothing, the study still reports an
improvement, and the improvement is the sampler's own noise. Nothing downstream can detect it.
"""

from __future__ import annotations

import pytest

from shipvision.errors import ConfigurationError
from shipvision.mot import TRACKERS
from shipvision.tune.spaces import (
    APPEARANCE_PARAMETERS,
    CategoricalChoice,
    FloatRange,
    IntRange,
    SearchSpace,
    accepted_parameters,
    all_spaces,
    space_for,
)

TRACKER_NAMES = ["sort", "bytetrack", "ocsort", "botsort", "deepsortv2"]


class TestEveryDeclaredSpaceNamesRealParameters:
    def test_there_is_a_space_for_every_registered_tracker(self) -> None:
        """A tracker with no space cannot be tuned, and discovering that at the start of an
        overnight run is worse than discovering it here."""
        assert sorted(all_spaces()) == sorted(TRACKER_NAMES)
        assert sorted(TRACKERS.names()) == sorted(TRACKER_NAMES)

    @pytest.mark.parametrize("name", TRACKER_NAMES)
    def test_every_name_is_a_keyword_the_constructor_accepts(self, name: str) -> None:
        space = space_for(name)
        accepted = accepted_parameters(TRACKERS.get(name))

        assert set(space.names) <= accepted

    @pytest.mark.parametrize("name", TRACKER_NAMES)
    def test_the_midpoint_of_every_range_builds_a_tracker(self, name: str) -> None:
        """The stronger version of the test above: names being accepted is not the same as the
        *values* being accepted, and several constructors validate ranges and relationships."""
        tracker = TRACKERS.build(name, **space_for(name).middle())

        assert tracker.pool_size == 0

    @pytest.mark.parametrize("name", TRACKER_NAMES)
    def test_both_extremes_of_every_range_build_a_tracker(self, name: str) -> None:
        """A sampler will reach the corners. A space whose corners are invalid turns a study
        into a lottery over which trials survive, and the survivors are not a random sample of
        the space."""
        space = space_for(name)
        for corner in ("low", "high"):
            values = {}
            for parameter in space:
                if isinstance(parameter, CategoricalChoice):
                    values[parameter.name] = parameter.choices[0 if corner == "low" else -1]
                else:
                    values[parameter.name] = getattr(parameter, corner)
            TRACKERS.build(name, **values, **dict(space.constants))

    def test_bytetrack_ranges_cannot_violate_the_threshold_ordering(self) -> None:
        """The constructor requires ``low_threshold < track_threshold``. The two ranges are
        disjoint so that every sample is valid by construction rather than by rejection: the
        highest ``low`` is still below the lowest ``track``."""
        space = space_for("bytetrack")
        ranges = {p.name: p for p in space}

        assert ranges["low_threshold"].high < ranges["track_threshold"].low


class TestTheMroWalk:
    """BoT-SORT's constructor forwards ``**byte`` to ByteTrack. A validator that read only the
    subclass would reject ``track_threshold``, which is real; one that saw the ``**kwargs`` and
    gave up would accept anything. Both are worse than no validation."""

    def test_it_finds_the_forwarded_keywords(self) -> None:
        accepted = accepted_parameters(TRACKERS.get("botsort"))

        assert "appearance_weight" in accepted, "botsort's own keyword"
        assert "track_threshold" in accepted, "forwarded to ByteTrack through **byte"
        assert "max_age" in accepted

    def test_it_stops_where_the_forwarding_chain_ends(self) -> None:
        """``BaseTracker.__init__`` takes a ``pool``, which is not a hyperparameter and must not
        become one — ByteTrack has no ``**kwargs``, so the walk stops before it."""
        assert "pool" not in accepted_parameters(TRACKERS.get("botsort"))

    def test_a_tracker_without_forwarding_reports_only_its_own(self) -> None:
        accepted = accepted_parameters(TRACKERS.get("sort"))

        assert accepted == frozenset(
            {"det_threshold", "iou_threshold", "max_age", "min_hits", "gate"}
        )


class TestRefusals:
    def test_a_misspelled_parameter_is_rejected_at_construction(self) -> None:
        """The message names what the tracker does accept, because the next thing anyone does
        after a typo is look for the right spelling."""
        with pytest.raises(ConfigurationError) as raised:
            SearchSpace("sort", (FloatRange("iou_treshold", 0.1, 0.5),))

        message = str(raised.value)
        assert "iou_treshold" in message
        assert "iou_threshold" in message
        assert "tunes nothing" in message

    def test_a_misspelled_constant_is_rejected_too(self) -> None:
        """A pinned parameter is exactly as silent as a sampled one when it is misspelled."""
        with pytest.raises(ConfigurationError, match="does not accept"):
            SearchSpace(
                "sort", (FloatRange("iou_threshold", 0.1, 0.5),), constants={"gaet": True}
            )

    def test_an_empty_space_is_rejected(self) -> None:
        """A study over no parameters evaluates one configuration many times and reports the
        spread as progress — except the spread is zero, so it reports nothing at all."""
        with pytest.raises(ConfigurationError, match="reports the spread as progress"):
            SearchSpace("sort", ())

    def test_a_duplicated_parameter_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="appear twice"):
            SearchSpace(
                "sort",
                (FloatRange("iou_threshold", 0.1, 0.5), FloatRange("iou_threshold", 0.2, 0.6)),
            )

    def test_a_collapsed_float_range_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="constant wearing a parameter"):
            FloatRange("iou_threshold", 0.3, 0.3)

    def test_a_collapsed_int_range_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="must be below high"):
            IntRange("max_age", 30, 30)

    def test_a_log_range_starting_at_zero_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="positive low"):
            FloatRange("iou_threshold", 0.0, 0.5, log=True)

    def test_a_categorical_with_one_choice_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="at least two choices"):
            CategoricalChoice("gate", (True,))

    def test_an_unregistered_tracker_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="unknown tracker"):
            SearchSpace("deepsort_v3", (FloatRange("iou_threshold", 0.1, 0.5),))

    def test_a_tracker_with_no_declared_space_raises_rather_than_guessing(self) -> None:
        """A study over the wrong parameters is worse than no study, because it produces a
        number."""
        with pytest.raises(ConfigurationError, match="rather than tuning a guess"):
            space_for("nonexistent")


class TestAppearanceParametersAreHeldOut:
    """They are real constructor arguments that do nothing when the detections carry no
    embedding, which on a public-detection benchmark is always. Sampling them is the same
    failure as a typo, with a correctly spelled name."""

    @pytest.mark.parametrize("name", sorted(APPEARANCE_PARAMETERS))
    def test_they_are_absent_from_the_default_space(self, name: str) -> None:
        declared = set(space_for(name).names)
        appearance = {p.name for p in APPEARANCE_PARAMETERS[name]}

        assert declared.isdisjoint(appearance)

    @pytest.mark.parametrize("name", sorted(APPEARANCE_PARAMETERS))
    def test_they_are_real_names_the_constructor_accepts(self, name: str) -> None:
        """Held out for being inert, not for being wrong. If one of them were also misspelled
        the exclusion would be hiding a bug rather than a subtlety."""
        accepted = accepted_parameters(TRACKERS.get(name))

        assert {p.name for p in APPEARANCE_PARAMETERS[name]} <= accepted

    @pytest.mark.parametrize("name", sorted(APPEARANCE_PARAMETERS))
    def test_a_caller_with_a_reid_extractor_can_add_them_back(self, name: str) -> None:
        extended = space_for(name).with_parameters(*APPEARANCE_PARAMETERS[name])

        assert len(extended) == len(space_for(name)) + len(APPEARANCE_PARAMETERS[name])
        TRACKERS.build(name, **extended.middle())

    def test_botsort_leaves_camera_motion_out_because_the_benchmark_has_no_pixels(self) -> None:
        """Every estimator returns the identity transform when handed no image, so the choice
        would be free and the study would report the difference as a finding."""
        assert "cmc" not in space_for("botsort").names


class TestConstantsAndDefaults:
    def test_constants_are_applied_to_every_sample(self) -> None:
        """Pinning one parameter while another is tuned is the only way to attribute an
        improvement to one of them."""
        space = SearchSpace(
            "sort", (FloatRange("iou_threshold", 0.1, 0.5),), constants={"max_age": 7}
        )

        assert space.middle() == {"iou_threshold": pytest.approx(0.3), "max_age": 7}

    def test_defaults_are_the_constants_alone_and_not_the_range_midpoints(self) -> None:
        """A baseline built from range midpoints answers "did tuning beat the middle of my own
        guesses", which is a different and much less interesting question than "did tuning beat
        the shipped configuration"."""
        space = SearchSpace(
            "sort", (FloatRange("iou_threshold", 0.1, 0.5),), constants={"max_age": 7}
        )

        assert space.defaults() == {"max_age": 7}

    def test_describe_lists_every_parameter_and_its_range(self) -> None:
        text = space_for("sort").describe()

        assert "det_threshold" in text
        assert "float[0.05, 0.7]" in text
