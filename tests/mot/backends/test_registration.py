"""How the compiled trackers get into the registry, and what happens when there is no build.

Registration and *availability* are different things, and conflating them is the failure this
file guards — the same claim :mod:`tests.imgproc.backends.test_registration` makes for the
image ops, restated for the trackers because the consequence is different here. A tracker is
selected by name from a deployment's config, so "there is no build on this host" must not turn
into "that tracker does not exist": the operator needs to be told which one they got.

Every test here runs with no build, which is the state of a plain CI runner and of any laptop.
"""

from __future__ import annotations

import pytest

import shipvision.mot.backends.native as _backend_base
from shipvision.errors import BackendUnavailableError, ConfigurationError
from shipvision.mot import TRACKERS
from shipvision.mot.trackers.bytetrack.tracker import ByteTrackTracker
from shipvision.mot.trackers.sort.tracker import SortTracker
from shipvision.registry import NATIVE, PYTHON
from tests.mot.backends.conftest import NATIVE_BUILT, NO_BUILD


class TestTheRegistryStatesWhatTheLibraryImplements:
    """Not what this host can run. Making registration conditional would mean
    ``TRACKERS.backends("sort")`` answering differently per machine, and an error message
    losing the ability to say "native exists, it just is not built here"."""

    @pytest.mark.parametrize("name", ["sort", "bytetrack", "ocsort", "botsort", "deepsortv2"])
    def test_the_compiled_trackers_register_with_or_without_a_build(self, name: str) -> None:
        assert TRACKERS.backends(name) == [NATIVE, PYTHON]

    def test_no_compiled_tracker_is_registered_without_its_oracle(self) -> None:
        """The claim that matters is not "there is a native backend" but "every native backend
        has a readable one beside it under the same name". That pairing is what
        ``tests/mot/backends/test_parity.py`` enumerates from the registry, so a compiled
        tracker registered alone would silently be a compiled tracker with nothing to compare
        against — which is a compiled tracker nobody should trust."""
        orphans = [
            name
            for name in TRACKERS.names()
            if NATIVE in TRACKERS.backends(name) and PYTHON not in TRACKERS.backends(name)
        ]

        assert not orphans, f"{orphans} have a compiled backend and no oracle"

    def test_numpy_is_the_floor_under_every_tracker(self) -> None:
        """If this can be skipped, nothing else in the offline tier is a test."""
        assert all(PYTHON in TRACKERS.backends(name) for name in TRACKERS.names())

    def test_a_compiled_tracker_does_not_become_a_new_algorithm(self) -> None:
        """``sort`` is one algorithm with two backends, not two entries. A registry that
        listed ``sort_native`` would put the choice of implementation into every config file
        that names a tracker."""
        assert sorted(TRACKERS.names()) == [
            "botsort",
            "bytetrack",
            "deepsortv2",
            "mcbyte",
            "ocsort",
            "sort",
        ]


class TestResolutionWhenThereIsNoBuild:
    """``TRACKERS.build("sort")`` must return a working tracker on a laptop. That is the
    promise in the README and in CLAUDE.md — "fastest available, numpy as the floor" — and
    until the registry could fall back it was true only where nothing faster was registered."""

    def test_an_unpinned_build_falls_back_past_a_backend_that_cannot_be_built(
        self, monkeypatch
    ) -> None:
        """Simulated by hiding the extension rather than by trusting this machine not to have
        one: a fallback only tested where it cannot fail is not tested."""
        monkeypatch.setattr(_backend_base, "_C", None)

        assert isinstance(TRACKERS.build("sort", min_hits=1), SortTracker)
        assert isinstance(TRACKERS.build("bytetrack", min_hits=1), ByteTrackTracker)

    def test_the_fallback_reports_the_backend_it_actually_built(self, monkeypatch) -> None:
        """A log line that said "native" for numpy work would make every later measurement
        unattributable."""
        monkeypatch.setattr(_backend_base, "_C", None)

        assert TRACKERS.build("sort", min_hits=1).backend == PYTHON

    def test_pinning_native_raises_instead_of_silently_giving_numpy(self, monkeypatch) -> None:
        """A deployment that asked for ``native`` and quietly got numpy would be a large
        throughput regression reported as a successful start-up."""
        monkeypatch.setattr(_backend_base, "_C", None)

        with pytest.raises(BackendUnavailableError, match="not built"):
            TRACKERS.build("sort", backend=NATIVE)

    def test_the_refusal_names_the_command_that_fixes_it(self, monkeypatch) -> None:
        monkeypatch.setattr(_backend_base, "_C", None)

        with pytest.raises(BackendUnavailableError, match="cmake --build"):
            TRACKERS.build("bytetrack", backend=NATIVE)

    def test_an_extension_older_than_the_trackers_is_refused_by_name(self, monkeypatch) -> None:
        """The realistic half of "there is no build": an ``_C`` compiled before these entry
        points existed imports perfectly and then fails on an attribute, which reads as a
        typo rather than as a stale build."""
        monkeypatch.setattr(_backend_base, "_C", object())

        with pytest.raises(BackendUnavailableError, match="predates"):
            TRACKERS.build("sort", backend=NATIVE)

    def test_pinning_python_is_unaffected_by_any_of_this(self) -> None:
        """Every config that pinned the reference implementation must keep working unchanged."""
        assert TRACKERS.build("sort", backend=PYTHON).backend == PYTHON

    def test_asking_for_a_backend_nobody_implements_still_lists_what_there_is(self) -> None:
        """Asked of the camera-motion family, because every *tracker* now has both backends.

        The claim is about the registry's refusal, not about tracking: a name that exists with
        a backend that does not must say which backends it does have, or an operator reading
        the log cannot tell "I typo'd the backend" from "this host has no build"."""
        from shipvision.mot.motion.cmc import CAMERA_MOTION

        with pytest.raises(ConfigurationError, match=r"has no 'native' backend"):
            CAMERA_MOTION.build("none", backend=NATIVE)


class TestAvailabilityIsNotADeviceQuestion:
    """The trackers are host C++. An association over fifteen boxes has nothing to gain from a
    GPU, so requiring a visible device before using them would skip the parity tests on exactly
    the machines where they are cheapest to run."""

    def test_availability_asks_whether_the_trackers_are_there(self, monkeypatch) -> None:
        monkeypatch.setattr(_backend_base, "_C", None)

        assert _backend_base.native_available() is False

    @pytest.mark.native
    @pytest.mark.skipif(not NATIVE_BUILT, reason=NO_BUILD)
    def test_a_build_with_no_device_still_reports_available(self) -> None:
        assert _backend_base.native_available() is True
