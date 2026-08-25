"""The TensorRT detector: engine loading, binding introspection, execution and calibration.

A package rather than a module because it has four separable concerns and only one of them
needs a driver:

:mod:`~shipvision.detection.backends.tensorrt.bindings`
    What an engine says about itself — shapes, dtypes, optimisation profiles, and the two
    incompatible TensorRT IO APIs reconciled into one. **Imports nothing**; the engine and the
    ``tensorrt`` module are passed in, so all of it is tested with no GPU.
:mod:`~shipvision.detection.backends.tensorrt.engine`
    :class:`~shipvision.detection.backends.tensorrt.engine.TensorRTDetector` — buffers from
    torch, execution from TensorRT.
:mod:`~shipvision.detection.backends.tensorrt.calibration`
    INT8 calibration, which no reference in this project has. Read its module docstring before
    using it: the calibration data must go through the same preprocessing as inference, and
    getting that wrong produces an engine that is fast and quietly inaccurate.

Engine *building* is one level up, in :mod:`shipvision.detection.engine_build`, because it is a
deployment step rather than part of the frame path — and because it is what a caller reaches for
before any of this.
"""

from __future__ import annotations

from shipvision.detection.backends.tensorrt.bindings import Binding, EngineBindings
from shipvision.detection.backends.tensorrt.calibration import (
    CalibrationBatchFeeder,
    CalibrationCache,
    build_int8_calibrator,
)
from shipvision.detection.backends.tensorrt.engine import TensorRTDetector

__all__ = [
    "Binding",
    "CalibrationBatchFeeder",
    "CalibrationCache",
    "EngineBindings",
    "TensorRTDetector",
    "build_int8_calibrator",
]
