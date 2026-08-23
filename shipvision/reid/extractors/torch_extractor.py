"""Run a TorchScript embedding artefact over crops.

For prototyping, for numeric parity against the TensorRT path, and for a deployment with no
engine build. The model architecture is deliberately *not* here: what is loaded is a
serialised :func:`torch.jit.script` / :func:`torch.jit.trace` artefact, so a CLIP-ReID
checkpoint, an OSNet or a bare ResNet trunk are all the same object to this class. Vendoring
somebody's model definition to load their weights is how a library acquires a dependency on
a research repository's refactors.

**Nothing torch is imported at module scope.** The module must import cleanly on a machine
with no torch so that :data:`~shipvision.reid.base.EXTRACTORS` can list it, and constructing
it there must raise :class:`~shipvision.errors.BackendUnavailableError` — "torch is not
installed here" is a deployment fact an operator can act on, where an ``ImportError`` out of
a registry lookup is a stack trace they have to read this file to interpret.

**`input_size` is configuration and `dim` is discovered.** A TorchScript artefact carries no
input-shape metadata — unlike a TensorRT engine, whose bindings state it — so the crop size
has to be given. The embedding width is then probed by one forward pass at load time rather
than configured, because a configured width that disagrees with the artefact is a silent
correctness bug, and because probing fails at start-up where the disagreement would
otherwise surface as an unformable similarity matrix on frame 40 000.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from shipvision.errors import (
    BackendUnavailableError,
    ConfigurationError,
    DimensionMismatchError,
    InferenceError,
    ModelLoadError,
)
from shipvision.reid.base import FeatureExtractor
from shipvision.reid.distance import normalize

__all__ = ["TorchExtractor"]


def _require_torch() -> Any:
    """Import torch, or say so in the project's own vocabulary."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised by monkeypatching sys.modules
        raise BackendUnavailableError(
            "the torch feature extractor needs torch; install the 'torch' extra "
            "(pip install 'shipvision[torch]') or select a different backend"
        ) from exc
    return torch


def _squeeze_trailing(tensor: Any) -> Any:
    """Drop trailing singleton axes, so a CNN's ``(n, d, 1, 1)`` pooled output is ``(n, d)``."""
    while getattr(tensor, "ndim", 0) > 2 and tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)
    return tensor


class TorchExtractor(FeatureExtractor):
    """A scripted module, run batched under ``no_grad`` and L2-normalised on the way out.

    Args:
        path: the ``.pt`` / ``.ts`` TorchScript artefact.
        input_size: ``(height, width)`` the artefact expects. Required — see the module
            docstring.
        device: any torch device string. ``"cpu"`` is the default so that a parity test
            runs on a machine with no accelerator.
        batch_size: how many crops go through in one forward pass. Chunking rather than one
            call per crop is the whole reason this is affordable: at 15 000 crops a second
            the per-call Python and launch overhead is the budget, not the arithmetic.
        half: run in float16. Roughly halves the time on a tensor-core GPU and is the usual
            choice for re-ID, where the embedding is normalised afterwards and the lost
            mantissa bits do not survive into the ranking anyway.
        channels: channels per crop, 3 by every convention in this library.
    """

    def __init__(
        self,
        *,
        path: str | Path,
        input_size: tuple[int, int],
        device: str = "cpu",
        batch_size: int = 32,
        half: bool = False,
        channels: int = 3,
    ) -> None:
        torch = _require_torch()

        if batch_size <= 0:
            raise ConfigurationError(f"batch_size must be positive, got {batch_size}")
        if channels <= 0:
            raise ConfigurationError(f"channels must be positive, got {channels}")
        size = tuple(int(v) for v in input_size)
        if len(size) != 2 or any(v <= 0 for v in size):
            raise ConfigurationError(
                f"input_size must be (height, width) with positive values, got {input_size}"
            )

        self.path = Path(path)
        self.input_size: tuple[int, int] = (size[0], size[1])
        self.device = device
        self.batch_size = int(batch_size)
        self.half = bool(half)
        self._channels = int(channels)
        self._torch = torch

        if not self.path.is_file():
            raise ModelLoadError(f"no TorchScript artefact at {self.path}")
        try:
            module = torch.jit.load(str(self.path), map_location=device)
        except Exception as exc:
            raise ModelLoadError(
                f"{self.path} is not a loadable TorchScript module on device {device!r}: "
                f"{exc}"
            ) from exc

        module.eval()
        self._module = module.half() if self.half else module
        self._dtype = torch.float16 if self.half else torch.float32
        self._dim = self._probe_dim()

    # -- introspection --------------------------------------------------------------------

    @property
    def dim(self) -> int:
        return self._dim

    def _probe_dim(self) -> int:
        """One forward pass over zeros, to learn what the artefact actually returns."""
        torch = self._torch
        probe = torch.zeros(
            (1, self._channels, *self.input_size), dtype=self._dtype, device=self.device
        )
        try:
            with torch.no_grad():
                output = self._module(probe)
        except Exception as exc:
            raise ModelLoadError(
                f"{self.path} raised on a {tuple(probe.shape)} probe: {exc}. Either "
                f"input_size={self.input_size} is not what it expects, or it is not a "
                f"crop-to-embedding model"
            ) from exc
        return int(self._as_matrix(output).shape[1])

    def _as_matrix(self, output: Any) -> Any:
        """Coerce whatever the artefact returned into ``(n, d)``.

        Two shapes are accepted beyond the obvious one, and both come from real models:

        * A tuple or list of tensors is concatenated along the feature axis. CLIP-ReID's
          own forward does exactly this — its ViT-B-16 embedding is the 768-wide bottleneck
          output and the 512-wide projection output joined into one 1280-wide vector — so a
          module scripted from a checkpoint that stops one step earlier returns the parts.
        * Trailing singleton axes are dropped, which is what a CNN trunk's pooled output
          ``(n, d, 1, 1)`` looks like.

        Anything else raises rather than being reshaped. ``reshape(n, -1)`` on a genuine
        spatial feature map would succeed and hand the gallery a 100 000-wide "embedding",
        which is the sort of thing that only shows up as an out-of-memory much later.
        """
        torch = self._torch
        if isinstance(output, (list, tuple)):
            if not output:
                raise ModelLoadError(f"{self.path} returned an empty tuple")
            parts = [_squeeze_trailing(t) for t in output]
            output = torch.cat(parts, dim=-1)
        else:
            output = _squeeze_trailing(output)

        if not isinstance(output, torch.Tensor):
            raise ModelLoadError(
                f"{self.path} returned {type(output).__name__}, not a tensor or a tuple of "
                f"tensors"
            )
        if output.ndim != 2:
            raise ModelLoadError(
                f"{self.path} returned shape {tuple(output.shape)}; an embedding model must "
                f"return (n, dim). A spatial feature map is not an embedding — pool it "
                f"inside the artefact, where the pooling is part of the model"
            )
        return output

    # -- inference ------------------------------------------------------------------------

    def extract(self, crops: np.ndarray) -> np.ndarray:
        torch = self._torch
        batch = self._as_batch(crops, channels=self._channels)
        if batch.shape[0] == 0:
            return self._empty()
        if batch.shape[2:] != self.input_size:
            raise DimensionMismatchError(
                f"crops are {batch.shape[2]}x{batch.shape[3]} but the artefact was probed "
                f"at {self.input_size[0]}x{self.input_size[1]}. Resize at the imgproc "
                f"boundary: a convolutional trunk accepts the wrong size and returns an "
                f"embedding of a different width, or of the same width and a different "
                f"meaning"
            )

        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, batch.shape[0], self.batch_size):
                piece = batch[start : start + self.batch_size]
                tensor = torch.from_numpy(piece).to(device=self.device, dtype=self._dtype)
                try:
                    output = self._module(tensor)
                except Exception as exc:
                    raise InferenceError(
                        f"{self.path} failed on a batch of {piece.shape[0]}: {exc}"
                    ) from exc
                chunks.append(self._as_matrix(output).float().cpu().numpy())

        features = np.concatenate(chunks, axis=0)
        if features.shape[1] != self._dim:
            raise InferenceError(
                f"{self.path} returned {features.shape[1]}-d embeddings but was probed at "
                f"{self._dim}-d; the artefact's width depends on its input and cannot be "
                f"trusted by a gallery"
            )
        return normalize(features.astype(np.float32), copy=False)
