"""Turning several views of one identity into the vector that represents it."""

from __future__ import annotations

from shipvision.reid.aggregation.base import AGGREGATORS, FeatureAggregator
from shipvision.reid.aggregation.ema import EmaAggregator
from shipvision.reid.aggregation.mean import MeanAggregator

__all__ = [
    "AGGREGATORS",
    "EmaAggregator",
    "FeatureAggregator",
    "MeanAggregator",
]
