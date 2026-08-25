"""The clusterer, exercised on matrices small enough to reason about by hand."""

from __future__ import annotations

import numpy as np
import pytest

from shipvision.errors import ConfigurationError, TrackingError
from shipvision.mtmc import CLUSTERERS, NEVER_MERGE, AgglomerativeClusterer

pytest.importorskip("scipy", reason="hierarchical clustering is scipy's job, not ours")


def groups(labels: np.ndarray) -> set[frozenset[int]]:
    """Labels to a set of index groups, so a test asserts the partition and not the naming.

    Cluster labels are arbitrary integers — only equality between them carries meaning — so a
    test that compared them elementwise would be pinning an implementation detail of scipy.
    """
    by_label: dict[int, set[int]] = {}
    for index, label in enumerate(labels.tolist()):
        by_label.setdefault(int(label), set()).add(index)
    return {frozenset(members) for members in by_label.values()}


class TestNeverMergeSentinel:
    """Why the sentinel is a large finite number and not infinity."""

    def test_it_stops_a_transitive_merge(self) -> None:
        """A(cam-1) and B(cam-2) are alike, B and C(cam-1) are alike, A and C are same-camera.
        With average linkage the sentinel drags the mean for any group containing both A and C
        to ~5e4, so C is never pulled in on B's coat-tails. This is the mechanism the whole
        design rests on."""
        distances = np.array(
            [
                [0.0, 0.05, NEVER_MERGE],
                [0.05, 0.0, 0.06],
                [NEVER_MERGE, 0.06, 0.0],
            ]
        )

        labels = AgglomerativeClusterer(distance_threshold=0.14).fit_predict(distances)

        assert groups(labels) == {frozenset({0, 1}), frozenset({2})}

    def test_an_infinite_distance_is_a_typed_failure_naming_the_sentinel(self) -> None:
        """inf is the obvious way to say "never merge" and it is wrong: scipy rejects a
        non-finite condensed matrix, and any averaging linkage that got past that would
        compute inf - inf and produce NaN — a dendrogram whose merges are arbitrary rather
        than an error."""
        distances = np.array([[0.0, np.inf], [np.inf, 0.0]])

        with pytest.raises(TrackingError, match="NEVER_MERGE"):
            AgglomerativeClusterer().fit_predict(distances)


class TestAgglomerativeCut:
    """The cut is a distance, and the linkage is average. Both are decisions."""

    def test_the_cut_is_a_distance_not_a_cluster_count(self) -> None:
        """Same matrix, two thresholds, two different numbers of clusters — because the number
        of identities in front of a camera group is the answer, not an input."""
        distances = np.array(
            [
                [0.0, 0.10, 0.30],
                [0.10, 0.0, 0.32],
                [0.30, 0.32, 0.0],
            ]
        )

        tight = AgglomerativeClusterer(distance_threshold=0.14)
        loose = AgglomerativeClusterer(distance_threshold=0.40)

        assert len(groups(tight.fit_predict(distances))) == 2
        assert len(groups(loose.fit_predict(distances))) == 1

    def test_average_linkage_rather_than_single_linkage_refuses_a_chain(self) -> None:
        """Single linkage would merge all three: 0-1 at 0.10 and 1-2 at 0.10 while 0-2 is
        0.30. Average linkage judges the candidate merge on its mean (0.20) and declines at a
        cut of 0.14 — which is the behaviour a threshold is chosen against."""
        distances = np.array(
            [
                [0.0, 0.10, 0.30],
                [0.10, 0.0, 0.10],
                [0.30, 0.10, 0.0],
            ]
        )

        labels = AgglomerativeClusterer(distance_threshold=0.14).fit_predict(distances)

        assert len(groups(labels)) == 2

    def test_the_threshold_is_validated_at_construction(self) -> None:
        with pytest.raises(ConfigurationError, match="distance_threshold must be positive"):
            AgglomerativeClusterer(distance_threshold=0.0)


class TestClustererContract:
    """The shapes and the edge cases every clusterer has to survive."""

    def test_one_track_is_one_cluster_and_no_tracks_is_no_labels(self) -> None:
        """A quiet site is the normal case. scipy refuses a 1x1 condensed matrix, so this is
        handled before scipy is reached."""
        clusterer = AgglomerativeClusterer()

        assert clusterer.fit_predict(np.zeros((1, 1))).tolist() == [0]
        assert clusterer.fit_predict(np.zeros((0, 0))).shape == (0,)

    def test_labels_are_zero_based_int32(self) -> None:
        """scipy labels from 1. Normalising here keeps "label 0" from meaning two things
        depending on which library produced it."""
        labels = AgglomerativeClusterer().fit_predict(np.zeros((3, 3)))

        assert labels.min() == 0
        assert labels.dtype == np.int32

    def test_a_non_square_matrix_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="must be square"):
            AgglomerativeClusterer().fit_predict(np.zeros((2, 3)))

    def test_a_tiny_asymmetry_out_of_blas_does_not_change_the_answer(self) -> None:
        """squareform's symmetry check is exact and, with checks off, it silently keeps the
        upper triangle. Symmetrising explicitly is what makes the result the matrix the
        builder meant rather than half of it."""
        distances = np.array([[0.0, 0.10], [0.10 + 3e-16, 0.0]])

        labels = AgglomerativeClusterer(distance_threshold=0.14).fit_predict(distances)

        assert groups(labels) == {frozenset({0, 1})}

    def test_the_clusterer_is_selectable_by_name_and_by_alias(self) -> None:
        """Comparing two linkages on one recorded stream must not require a code change."""
        assert isinstance(CLUSTERERS.build("agglomerative"), AgglomerativeClusterer)
        assert isinstance(CLUSTERERS.build("average_linkage"), AgglomerativeClusterer)
