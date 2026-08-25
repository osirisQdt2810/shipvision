"""The extractor contract: crops in, normalised embeddings out.

Everything else in this package works on vectors. This is where the vectors come from, and
the family exists so that the *rest* of re-identification can be exercised without a model:
the mock extractor is a real implementation of this interface, so a gallery test, a tracking
test and an MTMC test all run with no engine, no GPU and no build.

Two parts of the interface are load-bearing rather than convenient:

**`dim` is discovered, never configured.** A TensorRT engine states its embedding width in
its output binding and a TorchScript module states it by what it returns; the reference C++
implementation reads it from the binding for exactly this reason. A configured width that
disagrees with the artefact is a silent correctness bug — the gallery is allocated one way,
the model writes the other, and the only symptom is a similarity matrix nobody can form or,
worse, one a broadcast quietly produced.

**Output is L2-normalised on the way out.** :mod:`shipvision.reid.distance` assumes it, the
gallery assumes it, and doing it once here rather than inside every distance function is the
difference between a similarity search costing one gemm and costing a gemm plus a full pass
over the gallery.
"""

from __future__ import annotations

import abc

import numpy as np

from shipvision.errors import ConfigurationError, DimensionMismatchError
from shipvision.registry import PYTHON, Registry

__all__ = ["EXTRACTORS", "FeatureExtractor"]


class FeatureExtractor(abc.ABC):
    """Turns crops into embeddings. Registered in :data:`EXTRACTORS`."""

    name: str = "extractor"
    backend: str = PYTHON

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        """Embedding width. Discovered from the artefact, not configured.

        See the module docstring: a configured value that disagrees with the artefact is a
        silent correctness bug, so implementations that have an artefact read it from there
        and fail at load if it cannot be read.
        """

    @abc.abstractmethod
    def extract(self, crops: np.ndarray) -> np.ndarray:
        """``(n, 3, h, w)`` float32 crops to ``(n, dim)`` float32, L2-normalised.

        An empty batch returns ``(0, dim)``, not ``(0,)``. The shape matters: a frame with
        no detections is ordinary input, and ``(0,)`` breaks every downstream ``[:, k]``
        with an IndexError instead of yielding an empty result.
        """

    def extract_one(self, crop: np.ndarray) -> np.ndarray:
        """``(3, h, w)`` to ``(dim,)``. For tests and single-shot calls, not the frame path.

        Batching is most of what makes an accelerator worth having — 15 000 crops a second
        arrive as batches and must stay that way — so this is deliberately a convenience
        rather than the interface anything hot goes through.
        """
        return self.extract(np.asarray(crop)[None])[0]

    # -- shared machinery -----------------------------------------------------------------

    def _as_batch(self, crops: np.ndarray, *, channels: int = 3) -> np.ndarray:
        """Validate and coerce a crop batch to contiguous ``(n, c, h, w)`` float32.

        Shared rather than repeated per implementation because the failure it catches is
        the same everywhere and is easy to miss: an ``(n, h, w, 3)`` batch — the layout
        OpenCV hands back — has the right rank and the right element count, so a model runs
        on it and returns confident nonsense. Only the channel axis says otherwise.
        """
        batch = np.asarray(crops, dtype=np.float32)
        if batch.ndim != 4:
            raise ConfigurationError(
                f"crops must be (n, {channels}, h, w); got shape {batch.shape}. A single "
                f"crop needs a leading axis — use extract_one for that"
            )
        if batch.shape[1] != channels:
            raise DimensionMismatchError(
                f"crops must be channels-first with {channels} channels; got shape "
                f"{batch.shape}. An (n, h, w, c) batch has the same element count and will "
                f"run without complaint, so this is checked rather than assumed"
            )
        return np.ascontiguousarray(batch)

    def _empty(self) -> np.ndarray:
        """``(0, dim)`` float32 — the answer for a frame with no detections."""
        return np.zeros((0, self.dim), dtype=np.float32)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} dim={self.dim} backend={self.backend}>"


#: The extractor family. A mock for tests, and one implementation per runtime that can
#: execute a trained artefact — which is the same algorithm ("embed these crops") at
#: different speeds, so they share a name and differ by backend.
EXTRACTORS: Registry[FeatureExtractor] = Registry("feature extractor")
