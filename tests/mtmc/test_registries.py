"""The three MTMC registries, and the names the old spellings still resolve to.

Two claims are pinned here, and they decay in opposite ways.

**A new cross-camera strategy is a package plus a decorator.** That is easy to state and easy
to lose: the first time somebody adds a strategy by threading an ``if name == ...`` through
the tracker, the registry is still there, still passing its own tests, and no longer the way
anything gets added. So the test registers a matcher the shipped package has never heard of
and then selects it through the tracker's ordinary config path.

**The pre-repackaging names are aliases, not copies.** ``MATRIX_BUILDERS`` and
``MTMC_MATCHERS`` have to be one object. Two registries would each hold half the strategies,
and the symptom is a config string that resolves in one process and raises in another
depending on which name the caller reached for.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import pytest

import shipvision
import shipvision.mtmc as mtmc
import shipvision.mtmc.clustering as clustering_package
import shipvision.mtmc.matrix as matrix_shim
import shipvision.mtmc.topology as topology_package
from shipvision.errors import ConfigurationError
from shipvision.mtmc import (
    MTMC,
    MTMC_CLUSTERERS,
    MTMC_MATCHERS,
    BaseClusterer,
    BaseMatcher,
    ClusterMTMCTracker,
    TrackObservation,
)
from shipvision.registry import PYTHON, Registry

#: Everything ``shipvision.mtmc`` exported before the matchers moved into ``mtmc/matchers/``.
#: Hard-coded rather than read from the package, because reading it from the package is what
#: a test of a compatibility promise must not do — it would pass whatever the package says
#: today.
LEGACY_EXPORTS = (
    "CLUSTERERS",
    "MATRIX_BUILDERS",
    "MTMC",
    "NEVER_MERGE",
    "AgglomerativeClusterer",
    "AppearanceMatrixBuilder",
    "BaseClusterer",
    "BaseMTMCTracker",
    "BaseMatrixBuilder",
    "CameraTracks",
    "ClusterMTMCTracker",
    "FrameTrackCluster",
    "GatedMatrixBuilder",
    "GlobalIdAssigner",
    "GroundPlane",
    "Homography",
    "ObservationGate",
    "SpatialMatrixBuilder",
    "TrackKey",
    "TrackObservation",
    "calculate_homography",
    "foot_points",
    "project",
)

#: The old spelling of each matcher class, and the name it is now defined under.
RENAMED = (
    ("BaseMatrixBuilder", "BaseMatcher"),
    ("AppearanceMatrixBuilder", "AppearanceMatcher"),
    ("SpatialMatrixBuilder", "SpatialMatcher"),
    ("GatedMatrixBuilder", "GatedMatcher"),
)


@pytest.fixture
def restored_matchers() -> Iterator[Registry]:
    """:data:`MTMC_MATCHERS` with whatever the test registers removed afterwards.

    A registry is deliberately append-only — registering a name twice raises, which is what
    catches two implementations claiming one config string — so there is no public way to take
    an entry out again. A test that adds a strategy therefore has to put the registry back
    itself, or every later test in the session sees a matcher that does not exist in the
    shipped package.
    """
    entries = dict(MTMC_MATCHERS._entries)
    aliases = dict(MTMC_MATCHERS._aliases)
    lazy = dict(MTMC_MATCHERS._lazy)
    try:
        yield MTMC_MATCHERS
    finally:
        MTMC_MATCHERS._entries = entries
        MTMC_MATCHERS._aliases = aliases
        MTMC_MATCHERS._lazy = lazy


class TestMatcherRegistry:
    """Adding a cross-camera association strategy is a new package plus a decorator."""

    @pytest.mark.parametrize("name", MTMC_MATCHERS.names())
    def test_every_registered_matcher_implements_the_contract(self, name: str) -> None:
        """The backend is pinned. Unpinned, what comes back depends on whether
        ``shipvision._C`` is built on the machine running the suite — which is the intended
        behaviour and would make this assertion say something different in CI than on a build
        host."""
        built = MTMC_MATCHERS.build(name, backend=PYTHON)

        assert isinstance(built, BaseMatcher)
        assert built.name == name
        assert built.backend == PYTHON

    @pytest.mark.parametrize(
        ("alias", "name"),
        [("aic", "appearance"), ("spatial_gating", "gated")],
    )
    def test_an_alias_resolves_to_the_same_class(self, alias: str, name: str) -> None:
        """Aliases are how a migrating operator writes the name their old config used."""
        assert MTMC_MATCHERS.get(alias) is MTMC_MATCHERS.get(name)

    def test_a_strategy_the_package_has_never_heard_of_becomes_selectable(
        self, restored_matchers: Registry
    ) -> None:
        """The whole point of the seam: a decorator, and the tracker selects it by name with
        nothing in the tracker changed."""

        @restored_matchers.register("everything_matches", backend=PYTHON, aliases=("em",))
        class EverythingMatches(BaseMatcher):
            """A matcher that finds every cross-camera pair identical."""

            def build(self, observations: Sequence[TrackObservation]) -> np.ndarray:
                return self.to_distance(
                    np.ones((len(observations), len(observations)), dtype=np.float32),
                    self.mergeable_mask(observations),
                )

        tracker = ClusterMTMCTracker(matrix_builder="em")

        assert isinstance(tracker.builder, EverythingMatches)
        assert "everything_matches" in MTMC_MATCHERS.names()

    def test_the_registration_is_undone_between_tests(self) -> None:
        """The fixture above is doing real work; without this, a leak would be invisible."""
        assert "everything_matches" not in MTMC_MATCHERS.names()

    def test_an_unknown_name_is_a_typed_failure_that_lists_what_exists(self) -> None:
        with pytest.raises(ConfigurationError, match="unknown mtmc matcher"):
            MTMC_MATCHERS.build("telepathy")


class TestClustererRegistry:
    """The clusterer family, selected the same way and from the same table."""

    @pytest.mark.parametrize("name", MTMC_CLUSTERERS.names())
    def test_every_registered_clusterer_implements_the_contract(self, name: str) -> None:
        built = MTMC_CLUSTERERS.build(name)

        assert isinstance(built, BaseClusterer)
        # One track is the normal state of a quiet site, and it needs no solver — so this
        # stays in the offline tier rather than depending on scipy being installed.
        assert built.fit_predict(np.zeros((1, 1))).tolist() == [0]

    @pytest.mark.parametrize("alias", ["average_linkage", "aic"])
    def test_an_alias_resolves_to_the_same_class(self, alias: str) -> None:
        assert MTMC_CLUSTERERS.get(alias) is MTMC_CLUSTERERS.get("agglomerative")


class TestTheFamiliesAreSeparate:
    """Three registries, not one shared table under three names."""

    def test_the_three_registries_are_three_objects(self) -> None:
        assert len({id(MTMC), id(MTMC_MATCHERS), id(MTMC_CLUSTERERS)}) == 3

    def test_a_matcher_name_does_not_resolve_as_a_clusterer(self) -> None:
        """A copy-paste slip in `registry.py` would make ``MTMC_CLUSTERERS.build("gated")``
        succeed and hand the tracker a matcher where a clusterer belongs."""
        assert "gated" in MTMC_MATCHERS
        assert "gated" not in MTMC_CLUSTERERS

    def test_each_family_names_itself_in_its_errors(self) -> None:
        """The family string is what an operator reads when a config key is wrong; three
        registries saying "mtmc tracker" would send them to the wrong file."""
        assert MTMC.family == "mtmc tracker"
        assert MTMC_MATCHERS.family == "mtmc matcher"
        assert MTMC_CLUSTERERS.family == "mtmc clusterer"


class TestLegacyNames:
    """Everything the package exported before the repackaging still resolves, to the same
    object it always did."""

    @pytest.mark.parametrize("name", LEGACY_EXPORTS)
    def test_the_legacy_export_still_resolves(self, name: str) -> None:
        assert hasattr(mtmc, name), f"shipvision.mtmc lost {name} in the repackaging"
        assert name in mtmc.__all__

    @pytest.mark.parametrize(("old", "new"), RENAMED)
    def test_a_renamed_class_is_an_alias_and_not_a_copy(self, old: str, new: str) -> None:
        """`is`, not `==`: a second class would make ``isinstance`` answer differently
        depending on which name the caller imported."""
        assert getattr(mtmc, old) is getattr(mtmc, new)
        assert getattr(matrix_shim, old) is getattr(mtmc, new)

    def test_the_registries_are_one_object_under_both_names(self) -> None:
        assert mtmc.MATRIX_BUILDERS is MTMC_MATCHERS
        assert matrix_shim.MATRIX_BUILDERS is MTMC_MATCHERS
        assert mtmc.CLUSTERERS is MTMC_CLUSTERERS
        assert clustering_package.CLUSTERERS is MTMC_CLUSTERERS

    def test_the_tracker_registry_still_lives_at_its_old_import_path(self) -> None:
        import shipvision.mtmc.base as base_module

        assert base_module.MTMC is MTMC
        assert "MTMC" in base_module.__all__

    def test_the_matrix_module_still_exports_what_the_matrix_package_did(self) -> None:
        for name in (
            "MATRIX_BUILDERS",
            "NEVER_MERGE",
            "AppearanceMatrixBuilder",
            "BaseMatrixBuilder",
            "GatedMatrixBuilder",
            "SpatialMatrixBuilder",
            "foot_points",
            "stack_embeddings",
        ):
            assert hasattr(matrix_shim, name), f"shipvision.mtmc.matrix lost {name}"

    def test_topology_still_exports_its_four_names_after_becoming_a_package(self) -> None:
        for name in ("GroundPlane", "Homography", "calculate_homography", "project"):
            assert hasattr(topology_package, name)

    @pytest.mark.parametrize(
        "name", ["MTMC", "MTMC_MATCHERS", "MTMC_CLUSTERERS", "MATRIX_BUILDERS", "CLUSTERERS"]
    )
    def test_both_spellings_resolve_from_the_top_level(self, name: str) -> None:
        """`shipvision.MATRIX_BUILDERS` goes through the lazy `__getattr__` table, which is a
        second place the rename could have been half-applied."""
        resolved = getattr(shipvision, name)

        assert isinstance(resolved, Registry)
