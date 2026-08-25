"""Run a TorchScript detector — for prototyping, for parity, and where no engine is built.

The model architecture is deliberately not here: what is loaded is a serialised
:func:`torch.jit.script` or :func:`torch.jit.trace` artefact, so a YOLO26 export and a YOLO26
segmentation export are the same object to this class. Vendoring somebody's model definition
in order to load their weights is how a library acquires a dependency on a research
repository's refactors.

**Nothing torch is imported at module scope.** The module must import cleanly on a machine with
no torch so that :data:`~shipvision.detection.base.DETECTORS` can list the backend, and
constructing it there must raise :class:`~shipvision.errors.BackendUnavailableError` — "torch
is not installed here" is a deployment fact an operator can act on, where an ``ImportError``
out of a registry lookup is a stack trace they have to read this file to interpret.

**``input_hw`` is the one thing this backend cannot discover, and it is verified rather than
trusted.** A TorchScript artefact carries no input-shape metadata; a TensorRT engine states it
in its binding, which is why
:class:`~shipvision.detection.backends.tensorrt.engine.TensorRTDetector` reads it and refuses a
caller who disagrees. Here the given extent is *probed* at load: the artefact is run once on
zeros of that shape, and a refusal is a :class:`~shipvision.errors.ModelLoadError` at start-up
rather than a shape error on frame 40 000. Be aware of what a probe cannot catch — a
fully-convolutional detector accepts any extent divisible by its total stride and returns
plausible boxes at the wrong scale — so for those artefacts the probe checks that the export
*is* a detector, and the extent remains the caller's responsibility.

The probe earns its keep a second way: the output shapes it returns are what
:func:`shipvision.detection.heads.resolve_head` uses to decide whether this artefact segments.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from shipvision.detection.artefact import ArtefactDetector
from shipvision.detection.base import DetectionError
from shipvision.detection.heads import resolve_head
from shipvision.errors import (
    BackendUnavailableError,
    ConfigurationError,
    ModelLoadError,
)
from shipvision.registry import TORCH
from shipvision.types import FrameTag

__all__ = ["TorchDetector"]


def _require_torch() -> Any:
    """Import torch, or say so in the project's own vocabulary."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised by monkeypatching sys.modules
        raise BackendUnavailableError(
            "the torch detector needs torch; install the 'torch' extra "
            "(pip install 'shipvision[torch]') or select a different backend"
        ) from exc
    return torch


class TorchDetector(ArtefactDetector):
    """A scripted detector, run batched under ``no_grad``.

    Args:
        path: the ``.pt`` / ``.ts`` TorchScript artefact.
        input_hw: ``(height, width)`` the artefact expects. Required, and probed — see the
            module docstring.
        device: any torch device string. ``"cpu"`` by default so a parity run needs no
            accelerator.
        half: run in float16. Only meaningful on a CUDA device.
        head: pin the decode by name, or `None` to read it off the artefact's outputs.
        head_options: forwarded to the head's constructor — ``conf_threshold``,
            ``nms_method``, ``max_detections`` and the rest. A separate mapping rather than
            ``**kwargs`` so that a typo lands in the head's own validation instead of being
            swallowed as an unknown detector argument.
        max_batch: frames per forward pass.
        **kwargs: the pre-processing arguments of
            :class:`~shipvision.detection.artefact.ArtefactDetector` — ``pad_value``,
            ``mean``, ``std``, ``image_ops``.
    """

    # Declared here rather than left to the registry: `register_lazy` claims the
    # (name, backend) pair without importing the class, so unlike `@register` it has nothing
    # to stamp these on. Stating them keeps `repr` and any log line honest.
    name: str = "yolo26"
    backend: str = TORCH

    def __init__(
        self,
        *,
        path: str | Path,
        input_hw: tuple[int, int],
        device: str = "cpu",
        half: bool = False,
        head: str | None = None,
        head_options: Mapping[str, Any] | None = None,
        max_batch: int = 1,
        **kwargs: Any,
    ) -> None:
        # Arguments before the runtime: see the note in `reid/extractors/torch_extractor.py`.
        extent = tuple(int(v) for v in input_hw)
        if len(extent) != 2 or extent[0] <= 0 or extent[1] <= 0:
            raise ConfigurationError(
                f"input_hw must be (height, width) with positive values, got {input_hw!r}"
            )

        torch = _require_torch()
        self._torch = torch
        self.path = Path(path)
        self.device = device
        self.half = bool(half)

        if not self.path.is_file():
            raise ModelLoadError(f"no TorchScript artefact at {self.path}")

        try:
            module = torch.jit.load(str(self.path), map_location=device)
        except Exception as exc:
            raise ModelLoadError(
                f"{self.path} is not a loadable TorchScript module on device {device!r}: {exc}"
            ) from exc
        module.eval()
        self._module = module.half() if self.half else module
        self._dtype = torch.float16 if self.half else torch.float32

        output_shapes = self._probe(extent)
        super().__init__(
            input_hw=(extent[0], extent[1]),
            head=resolve_head(
                output_shapes,
                name=head,
                artefact=str(self.path),
                **dict(head_options or {}),
            ),
            max_batch=max_batch,
            **kwargs,
        )

    # -- load-time discovery ----------------------------------------------------------

    def _probe(self, extent: tuple[int, int]) -> list[tuple[int, ...]]:
        """Run the artefact once on zeros, and return its output shapes.

        Two things at once, on purpose: it is the only check available that ``input_hw`` is
        something this artefact accepts, and it is where the head is discovered. Doing both in
        one forward pass means the cost is paid once, at start-up, and the two answers cannot
        disagree with each other.
        """
        torch = self._torch
        probe = torch.zeros((1, 3, *extent), dtype=self._dtype, device=self.device)
        try:
            with torch.no_grad():
                output = self._module(probe)
        except Exception as exc:
            raise ModelLoadError(
                f"{self.path} raised on a {tuple(probe.shape)} probe: {exc}. Either "
                f"input_hw={extent} is not what it expects, or it is not a detector"
            ) from exc
        return [tuple(int(v) for v in array.shape) for array in self._as_outputs(output)]

    def _as_outputs(self, output: Any) -> list[Any]:
        """Whatever the artefact returned, as a flat list of tensors.

        A detector returns one tensor, a segmentation model returns two, and an exporter may
        wrap either in a tuple or nest a tuple inside one — Ultralytics' segmentation export
        returns ``(detections, (protos, ...))``. Flattening one level of nesting handles that
        without guessing which element is which, because
        :func:`shipvision.detection.heads.resolve_head` identifies them by rank afterwards.
        """
        torch = self._torch
        flat: list[Any] = []
        pending = list(output) if isinstance(output, (list, tuple)) else [output]
        for item in pending:
            if isinstance(item, (list, tuple)):
                flat.extend(t for t in item if isinstance(t, torch.Tensor))
            elif isinstance(item, torch.Tensor):
                flat.append(item)
        if not flat:
            raise ModelLoadError(
                f"{self.path} returned {type(output).__name__} with no tensors in it; a "
                f"detector returns a tensor, or a tuple of them"
            )
        return flat

    # -- execution --------------------------------------------------------------------

    def _execute(self, batch: np.ndarray, tags: Sequence[FrameTag]) -> list[np.ndarray]:
        """See :meth:`~shipvision.detection.artefact.ArtefactDetector._execute`."""
        torch = self._torch
        tensor = torch.from_numpy(np.ascontiguousarray(batch)).to(
            device=self.device, dtype=self._dtype
        )
        try:
            with torch.no_grad():
                output = self._module(tensor)
        except Exception as exc:
            raise DetectionError(
                f"{self.path} failed on a batch of {batch.shape[0]}: {exc}",
                tag=tags[0] if tags else None,
            ) from exc
        return [array.detach().float().cpu().numpy() for array in self._as_outputs(output)]
