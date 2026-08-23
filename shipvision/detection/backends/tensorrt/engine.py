"""Run a TensorRT engine — the production path.

The engine is the source of truth about its own shapes and this class treats it that way:
:class:`~shipvision.detection.backends.tensorrt.bindings.EngineBindings` reads the input
extent, the batch bound and the output shapes off the bindings, and a caller who configured a
different input extent is refused rather than overridden. That is the whole reason this backend
exists in the shape it does — see
:meth:`~shipvision.detection.backends.tensorrt.bindings.EngineBindings.resolve_input_hw`.

**Device memory comes from torch, execution from TensorRT.** Torch's caching allocator, its
streams and its host-to-device copies are already the fastest and most tested versions of those
things reachable from Python, and re-implementing them over ``cuda-python`` would add a few
hundred lines whose only distinguishing feature is being newer. TensorRT does the one thing
only it can do. That means this backend needs torch as well as tensorrt, which is not a
hardship: a machine with an engine to run has torch.

**Buffers are allocated once, at ``max_batch``, in the constructor.** Nothing on the frame path
allocates. A caching allocator makes a per-call allocation cheap rather than free, and at a
thousand frames a second "cheap" is still two allocations per batch that never needed to
happen.

Neither import happens at module scope, so this file imports cleanly on a laptop and
constructing it there raises :class:`~shipvision.errors.BackendUnavailableError`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from shipvision.detection.artefact import ArtefactDetector
from shipvision.detection.backends.tensorrt.bindings import Binding, EngineBindings
from shipvision.detection.base import DetectionError
from shipvision.detection.heads import resolve_head
from shipvision.errors import (
    BackendUnavailableError,
    ConfigurationError,
    ModelLoadError,
)
from shipvision.registry import TENSORRT
from shipvision.types import FrameTag

__all__ = ["TensorRTDetector"]


def _require_runtime() -> tuple[Any, Any]:
    """Import tensorrt and torch, naming whichever is missing."""
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise BackendUnavailableError(
            "the tensorrt detector needs tensorrt; install the 'tensorrt' extra "
            "(pip install 'shipvision[tensorrt]') or select the torch or mock backend"
        ) from exc
    try:
        import torch
    except ImportError as exc:
        raise BackendUnavailableError(
            "the tensorrt detector uses torch for device memory and streams; install the "
            "'torch' extra alongside tensorrt"
        ) from exc
    return trt, torch


class TensorRTDetector(ArtefactDetector):
    """A serialised engine, its shapes read from its own bindings.

    Args:
        path: the serialised ``.engine`` / ``.plan`` / ``.trt``. **Not built here** — building
            is hardware- and TensorRT-version-specific and takes minutes, so it belongs in a
            deployment step. :mod:`shipvision.detection.engine_build` is that step, in process
            and without a subprocess.
        device: CUDA device ordinal. Explicit and per-instance because the failure ShipInfer
            exists to fix is a predecessor that never called ``cudaSetDevice`` and ran sixteen
            GPUs' worth of work on GPU 0.
        input_hw: normally omitted. Given, it is *checked* against the engine rather than
            applied — a disagreement is a :class:`~shipvision.errors.ModelLoadError`.
        head: pin the decode by name, or `None` to read it off the engine's output arity.
        head_options: forwarded to the head's constructor.
        max_batch: cap on frames per execution, clamped down to what the engine allows. `None`
            uses the engine's own maximum.
        **kwargs: the pre-processing arguments of
            :class:`~shipvision.detection.artefact.ArtefactDetector`.
    """

    # Declared here rather than left to the registry: `register_lazy` claims the
    # (name, backend) pair without importing the class, so unlike `@register` it has nothing
    # to stamp these on. Stating them keeps `repr` and any log line honest.
    name: str = "yolo26"
    backend: str = TENSORRT

    def __init__(
        self,
        *,
        path: str | Path,
        device: int = 0,
        input_hw: tuple[int, int] | None = None,
        head: str | None = None,
        head_options: Mapping[str, Any] | None = None,
        max_batch: int | None = None,
        **kwargs: Any,
    ) -> None:
        trt, torch = _require_runtime()
        if device < 0:
            raise ConfigurationError(f"device must be a non-negative ordinal, got {device}")
        if max_batch is not None and max_batch <= 0:
            raise ConfigurationError(f"max_batch must be positive or None, got {max_batch}")
        if not torch.cuda.is_available():
            raise BackendUnavailableError(
                "tensorrt needs a CUDA device and torch reports none; this is a deployment "
                "problem, not an artefact one"
            )

        self._trt = trt
        self._torch = torch
        self.path = Path(path)
        self.device = int(device)

        if not self.path.is_file():
            raise ModelLoadError(f"no TensorRT engine at {self.path}")

        self._engine, self._context = self._load(trt)
        self._bindings = EngineBindings.read(self._engine, trt, artefact=str(self.path))
        extent = self._bindings.resolve_input_hw(input_hw, artefact=str(self.path))

        engine_batch = self._bindings.max_batch
        batch = engine_batch if max_batch is None else min(int(max_batch), engine_batch)

        super().__init__(
            input_hw=extent,
            head=resolve_head(
                self._bindings.output_shapes(batch),
                name=head,
                artefact=str(self.path),
                **dict(head_options or {}),
            ),
            max_batch=batch,
            **kwargs,
        )
        self._allocate(extent)

    # -- load -------------------------------------------------------------------------

    def _load(self, trt: Any) -> tuple[Any, Any]:
        logger = trt.Logger(trt.Logger.WARNING)
        # Plugins first: an engine that uses one deserialises to None without them, and the
        # error TensorRT gives for that is indistinguishable from a corrupt file.
        trt.init_libnvinfer_plugins(logger, "")
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(self.path.read_bytes())
        if engine is None:
            raise ModelLoadError(
                f"{self.path} did not deserialise. An engine is specific to the TensorRT "
                f"version and the GPU architecture it was built on — rebuild it here with "
                f"shipvision.detection.engine_build"
            )
        context = engine.create_execution_context()
        if context is None:
            raise ModelLoadError(f"{self.path} deserialised but gave no execution context")
        # The runtime owns the engine's memory and must outlive it; a local would be collected
        # the moment this returns and take the engine with it.
        self._runtime = runtime
        return engine, context

    def _allocate(self, extent: tuple[int, int]) -> None:
        """One device buffer per binding, sized at ``max_batch``, for the life of the process."""
        torch = self._torch
        image = self._bindings.image_input
        with torch.cuda.device(self.device):
            self._stream = torch.cuda.Stream()
            self._device_in = self._buffer(image, (self.max_batch, image.shape[1], *extent))
            self._device_out = {
                binding.name: self._buffer(binding, binding.sized(self.max_batch))
                for binding in self._bindings.outputs
            }

    def _buffer(self, binding: Binding, shape: Sequence[int]) -> Any:
        torch = self._torch
        dtype = self._torch_dtype(binding)
        return torch.empty(
            tuple(int(v) for v in shape), dtype=dtype, device=f"cuda:{self.device}"
        )

    def _torch_dtype(self, binding: Binding) -> Any:
        torch = self._torch
        try:
            return getattr(torch, np.dtype(binding.dtype).name)
        except AttributeError as exc:
            raise ModelLoadError(
                f"{self.path} binding {binding.name!r} is {binding.dtype}, which has no torch "
                f"equivalent. An int8 engine still exposes float bindings — one that does not "
                f"needs its own quantisation-aware wrapper"
            ) from exc

    # -- introspection ----------------------------------------------------------------

    @property
    def bindings(self) -> EngineBindings:
        """What the engine said about itself. Read once, at load."""
        return self._bindings

    # -- execution --------------------------------------------------------------------

    def _execute(self, batch: np.ndarray, tags: Sequence[FrameTag]) -> list[np.ndarray]:
        """See :meth:`~shipvision.detection.artefact.ArtefactDetector._execute`."""
        torch = self._torch
        rows = int(batch.shape[0])
        tag = tags[0] if tags else None

        with torch.cuda.device(self.device), torch.cuda.stream(self._stream):
            self._device_in[:rows].copy_(
                torch.from_numpy(np.ascontiguousarray(batch, dtype=np.float32)).to(
                    self._device_in.dtype
                ),
                non_blocking=True,
            )
            self._enqueue(rows, tag)
            # Synchronise before reading: the copies below are ordered on this stream but numpy
            # is not, and handing a head a half-written tensor is a bug that only appears under
            # load.
            self._stream.synchronize()
            return [
                self._device_out[binding.name][:rows].float().cpu().numpy()
                for binding in self._bindings.outputs
            ]

    def _enqueue(self, rows: int, tag: FrameTag | None) -> None:
        """Bind the addresses and run, through whichever execution API the engine has."""
        image = self._bindings.image_input
        if self._bindings.named_api:
            if self._bindings.dynamic_batch and not self._context.set_input_shape(
                image.name, (rows, *self._device_in.shape[1:])
            ):
                raise DetectionError(
                    f"{self.path} refused a batch of {rows}; its profile allows at most "
                    f"{self.max_batch}",
                    tag=tag,
                )
            self._context.set_tensor_address(image.name, self._device_in.data_ptr())
            for binding in self._bindings.outputs:
                self._context.set_tensor_address(
                    binding.name, self._device_out[binding.name].data_ptr()
                )
            if not self._context.execute_async_v3(self._stream.cuda_stream):
                raise DetectionError(
                    f"{self.path} failed to enqueue a batch of {rows}", tag=tag
                )
            return

        # TensorRT 9 and earlier: addresses in binding-index order.
        if self._bindings.dynamic_batch:
            self._context.set_binding_shape(image.index, (rows, *self._device_in.shape[1:]))
        addresses = [0] * (len(self._bindings.inputs) + len(self._bindings.outputs))
        addresses[image.index] = self._device_in.data_ptr()
        for binding in self._bindings.outputs:
            addresses[binding.index] = self._device_out[binding.name].data_ptr()
        if not self._context.execute_async_v2(addresses, self._stream.cuda_stream):
            raise DetectionError(f"{self.path} failed to enqueue a batch of {rows}", tag=tag)
