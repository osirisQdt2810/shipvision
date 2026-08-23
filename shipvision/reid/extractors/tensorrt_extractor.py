"""Run a TensorRT engine over crops — the production path.

The engine is the source of truth about its own shapes, and this class treats it that way.
The internal C++ implementation this replaces does the same thing
(``BaseFeatureExtractor::loadInputOutputInformation``): it walks the engine's IO tensors and
takes the max batch size, the channel count and the crop height and width from the input
binding, and the **embedding width from the output binding's second dimension**. Nothing
about the artefact is configured here, because every one of those numbers has a version of
itself in a config file somewhere, and the two disagreeing is a bug with no symptom — the
gallery is sized one way, the engine writes the other, and the ranking is quietly wrong.

**Device memory comes from torch, execution from TensorRT.** Torch's caching allocator,
stream and host-to-device copy are already the fastest and most tested versions of those
things available in Python, and re-implementing them over ``cuda-python`` would add a few
hundred lines whose only distinguishing feature is being newer. TensorRT is used for the one
thing only it can do: running the engine. That does mean this backend needs torch as well as
tensorrt; a machine that has an engine to run has torch.

Neither import happens at module scope, so this file imports cleanly on a laptop and
constructing it there raises :class:`~shipvision.errors.BackendUnavailableError`.
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
from shipvision.registry import TENSORRT
from shipvision.reid.base import FeatureExtractor
from shipvision.reid.distance import normalize

__all__ = ["TensorRTExtractor"]


def _require_runtime() -> tuple[Any, Any]:
    """Import tensorrt and torch, naming whichever is missing."""
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise BackendUnavailableError(
            "the tensorrt feature extractor needs tensorrt; install the 'tensorrt' extra "
            "(pip install 'shipvision[tensorrt]') or select the torch or mock backend"
        ) from exc
    try:
        import torch
    except ImportError as exc:
        raise BackendUnavailableError(
            "the tensorrt feature extractor uses torch for device memory and streams; "
            "install the 'torch' extra alongside tensorrt"
        ) from exc
    return trt, torch


class TensorRTExtractor(FeatureExtractor):
    """A serialised engine, its shapes read from its own bindings.

    Args:
        path: the serialised ``.engine`` / ``.plan``. Not built here: engine building is
            hardware- and TensorRT-version-specific, takes minutes, and belongs in the
            deployment step rather than in a constructor on the frame path.
        device: CUDA device ordinal. Explicit and per-instance because the whole reason
            ShipInfer exists is that its predecessor never called ``cudaSetDevice`` and ran
            sixteen GPUs' worth of work on GPU 0.
        max_batch: cap on crops per execution, clamped down to what the engine's profile
            allows. `None` uses the engine's maximum.

    Attributes:
        input_size: ``(height, width)`` read from the input binding — what crops must be
            resized to, so the caller does not have to be told separately.
    """

    # Declared here rather than left to the registry: Registry.register_lazy claims the
    # (name, backend) pair without importing the class, so unlike @register it has nothing
    # to stamp these on. Stating them keeps `repr` and any log line honest about which
    # entry produced the instance.
    name: str = "generic"
    backend: str = TENSORRT

    def __init__(
        self,
        *,
        path: str | Path,
        device: int = 0,
        max_batch: int | None = None,
    ) -> None:
        trt, torch = _require_runtime()
        if device < 0:
            raise ConfigurationError(f"device must be a non-negative ordinal, got {device}")
        if max_batch is not None and max_batch <= 0:
            raise ConfigurationError(f"max_batch must be positive, got {max_batch}")
        if not torch.cuda.is_available():
            raise BackendUnavailableError(
                "tensorrt needs a CUDA device and torch reports none; this is a deployment "
                "problem, not an artefact one"
            )

        self.path = Path(path)
        self.device = int(device)
        self._trt = trt
        self._torch = torch

        if not self.path.is_file():
            raise ModelLoadError(f"no TensorRT engine at {self.path}")

        self._logger = trt.Logger(trt.Logger.WARNING)
        # Plugins first: an engine that uses one deserialises to None without it, and the
        # error TensorRT gives for that is indistinguishable from a corrupt file.
        trt.init_libnvinfer_plugins(self._logger, "")
        blob = self.path.read_bytes()
        runtime = trt.Runtime(self._logger)
        engine = runtime.deserialize_cuda_engine(blob)
        if engine is None:
            raise ModelLoadError(
                f"{self.path} did not deserialise. An engine is specific to the TensorRT "
                f"version and the GPU architecture it was built on — rebuild it here"
            )
        context = engine.create_execution_context()
        if context is None:
            raise ModelLoadError(f"{self.path} deserialised but gave no execution context")

        self._runtime = runtime
        self._engine = engine
        self._context = context

        self._input_name, self._output_name = self._io_names()
        (
            self._channels,
            self._input_size,
            engine_batch,
            self._dynamic_batch,
        ) = self._read_input_binding()
        self._dim = self._read_output_binding()
        self.max_batch = engine_batch if max_batch is None else min(max_batch, engine_batch)

        with torch.cuda.device(self.device):
            self._stream = torch.cuda.Stream()
            # Allocated once at the maximum, not per call. A caching allocator makes a
            # per-call allocation cheap rather than free, and at 15 000 crops a second
            # "cheap" is still two allocations per batch that never needed to happen.
            self._device_in = torch.empty(
                (self.max_batch, self._channels, *self._input_size),
                dtype=self._torch_dtype(self._input_name),
                device=f"cuda:{self.device}",
            )
            self._device_out = torch.empty(
                (self.max_batch, self._dim),
                dtype=self._torch_dtype(self._output_name),
                device=f"cuda:{self.device}",
            )

    # -- introspection --------------------------------------------------------------------

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def input_size(self) -> tuple[int, int]:
        return self._input_size

    def _io_names(self) -> tuple[str, str]:
        trt = self._trt
        inputs, outputs = [], []
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            target = (
                inputs
                if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                else outputs
            )
            target.append(name)
        if len(inputs) != 1 or len(outputs) != 1:
            raise ModelLoadError(
                f"{self.path} has inputs {inputs} and outputs {outputs}; a feature "
                f"extractor is one crop batch in and one embedding batch out. An engine "
                f"with several outputs needs a class that knows which one is the embedding"
            )
        return inputs[0], outputs[0]

    def _read_input_binding(self) -> tuple[int, tuple[int, int], int, bool]:
        """``(channels, (height, width), max_batch, dynamic_batch)`` from the input binding.

        Whether the batch axis is dynamic is not a detail: a fixed-batch engine cannot be
        told to run fewer rows, so the last chunk of a batch has to be executed at full
        width with the surplus discarded. Guessing wrong in either direction is a runtime
        error on the first partial batch, which is to say on the first quiet frame.
        """
        shape = tuple(self._engine.get_tensor_shape(self._input_name))
        if len(shape) != 4:
            raise ModelLoadError(
                f"{self.path} takes shape {shape}; a crop batch is (n, c, h, w)"
            )
        batch, channels, height, width = shape
        if height <= 0 or width <= 0:
            height, width = self._profile_max()[2:]
        if channels <= 0:
            raise ModelLoadError(
                f"{self.path} has a dynamic channel count, which no crop layout produces"
            )
        dynamic = batch <= 0
        if dynamic:
            batch = self._profile_max()[0]
        return int(channels), (int(height), int(width)), int(batch), dynamic

    def _profile_max(self) -> tuple[int, int, int, int]:
        """The engine's optimisation-profile maximum, for whatever it left dynamic."""
        try:
            _, _, maximum = self._engine.get_tensor_profile_shape(self._input_name, 0)
        except Exception as exc:
            raise ModelLoadError(
                f"{self.path} has a dynamic input shape but no readable optimisation "
                f"profile: {exc}"
            ) from exc
        values = tuple(int(v) for v in maximum)
        if len(values) != 4:
            raise ModelLoadError(f"{self.path} profile maximum is {values}, not 4-D")
        return values  # type: ignore[return-value]

    def _read_output_binding(self) -> int:
        """The embedding width, from the output binding's second dimension.

        This is the line the C++ implementation gets right and every config file gets
        wrong: ``embeddingDim = outputDims[BINDING_FEATURE_OUTPUT].d[1]``.
        """
        shape = tuple(self._engine.get_tensor_shape(self._output_name))
        if len(shape) != 2:
            raise ModelLoadError(
                f"{self.path} returns shape {shape}; an embedding batch is (n, dim). A "
                f"spatial output means the pooling was left out of the engine"
            )
        dim = int(shape[1])
        if dim <= 0:
            raise ModelLoadError(
                f"{self.path} declares a dynamic embedding width ({shape}). A gallery is "
                f"allocated against this number, so an engine that will not commit to it "
                f"cannot be used"
            )
        return dim

    def _torch_dtype(self, name: str) -> Any:
        trt, torch = self._trt, self._torch
        mapping = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
        }
        dtype = self._engine.get_tensor_dtype(name)
        if dtype not in mapping:
            raise ModelLoadError(
                f"{self.path} binding {name!r} is {dtype}; this extractor handles float32 "
                f"and float16 bindings. An int8 engine still exposes float bindings — one "
                f"that does not needs its own quantisation-aware wrapper"
            )
        return mapping[dtype]

    # -- inference ------------------------------------------------------------------------

    def extract(self, crops: np.ndarray) -> np.ndarray:
        torch = self._torch
        batch = self._as_batch(crops, channels=self._channels)
        if batch.shape[0] == 0:
            return self._empty()
        if batch.shape[2:] != self._input_size:
            raise DimensionMismatchError(
                f"crops are {batch.shape[2]}x{batch.shape[3]} but the engine's input "
                f"binding is {self._input_size[0]}x{self._input_size[1]}; resize at the "
                f"imgproc boundary"
            )

        chunks: list[np.ndarray] = []
        with torch.cuda.device(self.device), torch.cuda.stream(self._stream):
            for start in range(0, batch.shape[0], self.max_batch):
                piece = batch[start : start + self.max_batch]
                count = piece.shape[0]
                self._device_in[:count].copy_(torch.from_numpy(piece), non_blocking=True)
                # A fixed-batch engine gets no say in how many rows it runs: the shape is
                # baked into the plan, so the surplus rows run on whatever was left in the
                # buffer and their embeddings are sliced off below. Only a dynamic engine
                # is told the real count, which is also the only one that saves the work.
                if self._dynamic_batch and not self._context.set_input_shape(
                    self._input_name, (count, self._channels, *self._input_size)
                ):
                    raise InferenceError(
                        f"{self.path} refused a batch of {count}; its profile allows at "
                        f"most {self.max_batch}"
                    )
                self._context.set_tensor_address(self._input_name, self._device_in.data_ptr())
                self._context.set_tensor_address(self._output_name, self._device_out.data_ptr())
                if not self._context.execute_async_v3(self._stream.cuda_stream):
                    raise InferenceError(f"{self.path} failed to enqueue a batch of {count}")
                # Synchronise before reading: the copy back below is ordered on this stream,
                # but numpy is not, and handing the gallery a half-written matrix is the
                # kind of bug that only appears under load.
                self._stream.synchronize()
                chunks.append(self._device_out[:count].float().cpu().numpy())

        return normalize(np.concatenate(chunks, axis=0).astype(np.float32), copy=False)
