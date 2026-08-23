"""Appearance re-identification: extractors, galleries, aggregation, ranking metrics.

Four separable concerns, and keeping them separable is what makes re-ID measurable:

* **Extraction** (:data:`EXTRACTORS`) — crops to embeddings. One implementation per runtime,
  plus a mock so that everything below can be tested with no model at all.
* **The gallery** (:data:`GALLERIES`) — bounded memory of known appearances, searched with
  one gemm.
* **Aggregation** (:data:`AGGREGATORS`) — several views of one identity folded into the one
  vector that represents it.
* **Measurement** (:func:`evaluate_ranking`, :func:`rerank`) — CMC and mAP under the standard
  protocol, and the largest accuracy gain available without touching the model.

Everything but the extractors is numpy alone, which is deliberate: re-identification quality
is decided by rank-1 and mAP on recorded data, and that measurement must be runnable in
seconds on a laptop or it will not be run.

    from shipvision import Embedding
    from shipvision.reid import EXTRACTORS, GALLERIES

    extractor = EXTRACTORS.build("mock", dim=128)
    gallery = GALLERIES.build("flat", per_identity=8)

    vector = extractor.extract_one(crop)
    gallery.add(Embedding(vector=vector, identity="ship-14", camera_id="cam-03"))

    result = gallery.query(probe, top_k=5, threshold=0.55, exclude_camera="cam-03")
    if result:
        print(result.accepted.identity, result.accepted.score)

``exclude_camera`` is not optional decoration — see :mod:`shipvision.reid.gallery.base`.
"""

from __future__ import annotations

from shipvision.reid.aggregation import (
    AGGREGATORS,
    EmaAggregator,
    FeatureAggregator,
    MeanAggregator,
)
from shipvision.reid.base import EXTRACTORS, FeatureExtractor
from shipvision.reid.distance import (
    cosine_distance,
    cosine_similarity,
    euclidean_distance,
    is_normalized,
    normalize,
)
from shipvision.reid.extractors import MockExtractor
from shipvision.reid.gallery import GALLERIES, BaseGallery, CentroidGallery, FlatGallery
from shipvision.reid.metrics import (
    RankingResult,
    cmc_curve,
    evaluate_ranking,
    mean_average_precision,
)
from shipvision.reid.rerank import rerank
from shipvision.reid.types import Match, QueryResult

__all__ = [
    "AGGREGATORS",
    "EXTRACTORS",
    "GALLERIES",
    "BaseGallery",
    "CentroidGallery",
    "EmaAggregator",
    "FeatureAggregator",
    "FeatureExtractor",
    "FlatGallery",
    "Match",
    "MeanAggregator",
    "MockExtractor",
    "QueryResult",
    "RankingResult",
    "cmc_curve",
    "cosine_distance",
    "cosine_similarity",
    "euclidean_distance",
    "evaluate_ranking",
    "is_normalized",
    "mean_average_precision",
    "normalize",
    "rerank",
]
