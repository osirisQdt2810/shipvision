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
        max_batch: cap on crops per execution. `None` uses the engine's maximum, which is
            always the safe answer. A value is clamped down to the profile maximum on a
            **dynamic** engine and refused outright on a fixed-batch one, where it cannot
            be honoured in either direction — see :meth:`_resolve_max_batch`.

    Attributes:
        input_size: ``(height, width)`` read from the input binding — what crops must be
            resized to, so the caller does not have to be told separately.
    """

    def __init__(
        self,
        *,
        path: str | Path,
        device: int = 0,
        max_batch: int | None = None,
    ) -> None:
        # Arguments before the runtime: see the note in `torch_extractor.py`. A negative
        # `device` is the caller's mistake whether or not TensorRT is installed, and it must
        # be reported the same way on a runner with neither.
        if device < 0:
            raise ConfigurationError(f"device must be a non-negative ordinal, got {device}")
        if max_batch is not None and max_batch <= 0:
            raise ConfigurationError(f"max_batch must be positive, got {max_batch}")

        trt, torch = _require_runtime()
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
        self.max_batch = self._resolve_max_batch(max_batch, engine_batch)

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

    def _resolve_max_batch(self, requested: int | None, engine_batch: int) -> int:
        """How many rows a batch may hold, given what the engine will actually run.

        The buffers below are allocated at this number and the engine writes into them, so
        it has to be the row count the plan runs — not merely a cap on what the caller
        submits. On a **dynamic** engine those are the same thing: the context is told the
        real count per execution, so clamping to the profile maximum is safe. On a
        **fixed-batch** engine they are not. The plan's row count is baked in and
        ``set_input_shape`` is correctly skipped, so a smaller ``max_batch`` would leave
        the engine reading and writing past the end of both buffers — a silent corruption
        of whatever the caching allocator handed out next, or a sticky illegal access that
        poisons the context for the life of the process.

        So a fixed-batch engine refuses any ``max_batch`` but its own. Quietly running 32
        rows for a caller who asked for 8 is the other wrong answer: they sized something
        against that number too.
        """
        if requested is None:
            return engine_batch
        if self._dynamic_batch:
            return min(requested, engine_batch)
        if requested != engine_batch:
            raise ConfigurationError(
                f"{self.path} has a fixed batch of {engine_batch} rows baked into its plan "
                f"and max_batch={requested} was asked for. A fixed-batch plan runs "
                f"{engine_batch} rows whatever it is handed, so device buffers sized for "
                f"{requested} would be read and written out of bounds. Pass "
                f"max_batch={engine_batch}, leave it as None, or rebuild the engine with a "
                f"dynamic batch axis and an optimisation profile"
            )
        return engine_batch

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
