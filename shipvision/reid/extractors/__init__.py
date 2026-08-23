"""Turning crops into embeddings — one implementation per runtime.

The registered *name* is what the algorithm is, and the *backend* is what executes it. That
puts every artefact-driven extractor under one name, ``generic``, on purpose: this library
does not own a re-ID architecture. What it owns is "run a trained crop-to-embedding artefact
over this batch and normalise the result", which is genuinely one algorithm — a CLIP-ReID
engine, an OSNet engine and a scripted ResNet trunk differ in the file they load, not in
what this package does with them. So a deployment writes ``backend: tensorrt`` and a parity
test writes ``backend: torch`` against the same crops, which is the comparison the registry
exists to make possible.

``mock`` is a separate name because it is a separate algorithm: it computes a feature map
rather than running a model, and pretending it is the same thing with a different backend
would let ``build("generic")`` silently fall back to it on a machine with no runtime. A
missing engine must be an error, never a mock.

The two artefact backends are registered **lazily**. Importing tensorrt to discover that
tensorrt is absent costs a second and an exception on every start-up of every process that
never uses it, and importing torch costs more.
"""

from __future__ import annotations

from shipvision.registry import TENSORRT, TORCH
from shipvision.reid.base import EXTRACTORS, FeatureExtractor
from shipvision.reid.extractors.mock import MockExtractor

EXTRACTORS.register_lazy(
    "generic",
    "shipvision.reid.extractors.tensorrt_extractor:TensorRTExtractor",
    backend=TENSORRT,
)
EXTRACTORS.register_lazy(
    "generic",
    "shipvision.reid.extractors.torch_extractor:TorchExtractor",
    backend=TORCH,
)

__all__ = [
    "EXTRACTORS",
    "FeatureExtractor",
    "MockExtractor",
]
