"""How the compiled matchers get into the registry, and what happens with no build.

The same claim as :mod:`tests.tracking.backends.test_registration`, for the cross-camera half.
It is worth making twice because the consequence differs: a matcher is chosen per *site* — an
uncalibrated one runs ``appearance``, a calibrated one runs ``gated`` — so a registry that
answered differently depending on whether the host had a compiler would make one site's config
mean something else on another site's box.

Every test here runs with no build.
"""

from __future__ import annotations

import pytest

from shipvision.errors import BackendUnavailableError
from shipvision.mtmc import MTMC_MATCHERS
from shipvision.mtmc.backends import native
from shipvision.mtmc.core.appearance import AppearanceMatcher
from shipvision.mtmc.core.gated import GatedMatcher
from shipvision.registry import NATIVE, PYTHON

NAMES = ["appearance", "gated", "spatial"]


class TestTheRegistryStatesWhatTheLibraryImplements:
    @pytest.mark.parametrize("name", NAMES)
    def test_every_matcher_registers_in_both_backends(self, name: str) -> None:
        assert MTMC_MATCHERS.backends(name) == [NATIVE, PYTHON]

    def test_a_compiled_matcher_does_not_become_a_new_strategy(self) -> None:
        """Three strategies, six implementations. A registry that listed ``gated_native``
        would put the choice of implementation into every site's config file."""
        assert sorted(MTMC_MATCHERS.names()) == NAMES

    @pytest.mark.parametrize("name", NAMES)
    def test_an_alias_still_reaches_the_same_algorithm(self, name: str) -> None:
        """Aliases are how a migrating operator writes the name their old config used, and a
        second backend must not claim or shadow one."""
        assert MTMC_MATCHERS.get("aic", PYTHON) is MTMC_MATCHERS.get("appearance", PYTHON)
        assert MTMC_MATCHERS.get("spatial_gating", PYTHON) is MTMC_MATCHERS.get("gated", PYTHON)


class TestResolutionWhenThereIsNoBuild:
    @pytest.mark.parametrize("name", NAMES)
    def test_an_unpinned_build_falls_back_to_numpy(self, monkeypatch, name: str) -> None:
        """Simulated by hiding the extension rather than by trusting this machine not to have
        one: a fallback only tested where it cannot fail is not tested."""
        monkeypatch.setattr(native, "_C", None)

        assert MTMC_MATCHERS.build(name).backend == PYTHON

    def test_pinning_native_raises_instead_of_silently_giving_numpy(self, monkeypatch) -> None:
        monkeypatch.setattr(native, "_C", None)

        with pytest.raises(BackendUnavailableError, match="not built"):
            MTMC_MATCHERS.build("gated", backend=NATIVE)

    def test_an_extension_older_than_the_matchers_is_refused_by_name(self, monkeypatch) -> None:
        monkeypatch.setattr(native, "_C", object())

        with pytest.raises(BackendUnavailableError, match="predates"):
            MTMC_MATCHERS.build("appearance", backend=NATIVE)


class TestTheCompiledMatcherIsTheSameAlgorithm:
    """Subclassing rather than reimplementing is what makes that true by construction, and it
    is load-bearing for the tracker: it decides what to pass a matcher by inspecting the
    constructor, so a native matcher whose signature drifted would silently be built without
    its ground plane — cross-camera tracking with the geometry switched off, which reads as a
    tuning problem rather than a bug."""

    def test_a_compiled_matcher_is_an_instance_of_the_readable_one(self) -> None:
        assert issubclass(MTMC_MATCHERS.get("appearance", NATIVE), AppearanceMatcher)
        assert issubclass(MTMC_MATCHERS.get("gated", NATIVE), GatedMatcher)

    @pytest.mark.parametrize("name", NAMES)
    def test_the_two_backends_accept_exactly_the_same_keywords(self, name: str) -> None:
        import inspect

        reference = inspect.signature(MTMC_MATCHERS.get(name, PYTHON).__init__).parameters
        candidate = inspect.signature(MTMC_MATCHERS.get(name, NATIVE).__init__).parameters

        assert set(candidate) == set(reference)
        for keyword, parameter in reference.items():
            assert candidate[keyword].default == parameter.default, (
                f"{name}: the compiled matcher defaults {keyword} to "
                f"{candidate[keyword].default!r} where the readable one defaults it to "
                f"{parameter.default!r}; a site that omitted the key would be configured "
                f"differently depending on whether the host had a build"
            )
