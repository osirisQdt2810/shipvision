"""ONNX to a serialised TensorRT engine, **in process**.

Every reference in this project shells out. ``Yolo26Detector::buildEngine``
(``src/process/Yolo26Detector.cpp:66-124``) searches a directory for an ONNX, recovers its own
build configuration by running two regular expressions over the *filename* —
``mbs(\\d+)`` for the maximum batch size and ``imgsz(\\d+)`` for the input extent — and then
calls ``system("python3 pytools/onnx2trt.py ...")``. That is unshippable from a library, for
four separate reasons:

* **The filename is not configuration.** A file renamed on the way to the deployment box
  silently builds a 640x640 engine with a batch of 64 whatever the model actually is, and the
  regex failure path only logs a warning.
* **A subprocess cannot report a build failure usefully.** The return status is one bit; the
  parser errors that would say *why* the ONNX was rejected go to the child's stderr and are
  lost.
* **It needs a Python interpreter, a script at a fixed path relative to a compile-time
  ``WORKSPACE`` macro, and a matching tensorrt in that interpreter.** Three deployment
  assumptions to satisfy in order to call a Python API that was already importable.
* **No cancellation, no timeout, no progress.** A TensorRT build takes minutes and the caller
  has no handle on it.

So this module uses the TensorRT Python API directly: ``Builder``, ``OnnxParser``,
``IBuilderConfig``, an explicit :class:`OptimisationProfile` with shapes passed as *arguments*,
and ``build_serialized_network``. No filename parsing and no subprocess.

    from shipvision.detection.engine_build import OptimisationProfile, build_engine

    profile = OptimisationProfile.for_batch(input_hw=(640, 640), max_batch=16)
    build_engine("yolo26n.onnx", "yolo26n.engine", profiles=[profile], fp16=True)

INT8 is a first-class argument here and is absent from every reference — see
:mod:`shipvision.detection.backends.tensorrt.calibration`, and read its trap before using it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shipvision.detection.backends.tensorrt.calibration import (
    CalibrationBatchFeeder,
    CalibrationCache,
    build_int8_calibrator,
)
from shipvision.errors import (
    BackendUnavailableError,
    ConfigurationError,
    ModelLoadError,
)

__all__ = ["OptimisationProfile", "build_engine"]

DEFAULT_WORKSPACE = 1 << 32
"""4 GiB of scratch for tactic selection. Not the engine's runtime footprint — this is the
budget the builder may use while *choosing* kernels, and starving it makes the builder silently
skip the fastest tactics rather than fail."""


@dataclass(slots=True, frozen=True)
class OptimisationProfile:
    """The shape range one input may take, as ``(min, opt, max)``.

    A profile is not optional for a dynamic engine and it is not a formality. ``opt`` is the
    shape TensorRT selects kernels for; running at ``min`` or ``max`` is legal and slower, often
    by a lot. The reference hard-codes ``opt`` to ``max_batch // 2``
    (``pytools/onnx2trt.py:107``), which is the right shape for nothing in particular — a
    deployment that batches four frames should say ``opt_batch=4``.

    Attributes:
        input_name: the engine input this profile is for. It is checked against the parsed
            network's actual input names, because a profile for an input that does not exist
            builds an engine with **no** profile for the input that does, and that failure
            surfaces at inference.
        minimum: the smallest shape, as a full ``(n, c, h, w)``.
        optimum: the shape to tune for.
        maximum: the largest shape.
    """

    input_name: str
    minimum: tuple[int, ...]
    optimum: tuple[int, ...]
    maximum: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.input_name:
            raise ConfigurationError("a profile needs the name of the input it applies to")
        ranks = {len(self.minimum), len(self.optimum), len(self.maximum)}
        if len(ranks) != 1:
            raise ConfigurationError(
                f"profile shapes for {self.input_name!r} have different ranks: "
                f"{self.minimum}, {self.optimum}, {self.maximum}"
            )
        if any(v <= 0 for shape in self._shapes for v in shape):
            raise ConfigurationError(
                f"profile shapes for {self.input_name!r} must be fully specified and positive; "
                f"got {self.minimum}, {self.optimum}, {self.maximum}. A -1 belongs in the ONNX, "
                f"not in the profile that resolves it"
            )
        for axis, (low, mid, high) in enumerate(zip(*self._shapes, strict=True)):
            if not low <= mid <= high:
                raise ConfigurationError(
                    f"profile for {self.input_name!r} is not ordered on axis {axis}: "
                    f"min={low}, opt={mid}, max={high}. TensorRT accepts this and then refuses "
                    f"every shape at build time, with a message about the axis rather than the "
                    f"ordering"
                )

    @property
    def _shapes(self) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return self.minimum, self.optimum, self.maximum

    @classmethod
    def for_batch(
        cls,
        *,
        input_hw: tuple[int, int],
        max_batch: int,
        min_batch: int = 1,
        opt_batch: int | None = None,
        channels: int = 3,
        input_name: str = "images",
    ) -> OptimisationProfile:
        """The common case: a fixed input extent and a dynamic batch.

        ``opt_batch`` defaults to ``max_batch``, not to half of it. A detector instance in this
        system is fed by a batcher that fills up to its cap, so the batch it actually runs is
        the maximum far more often than half of it.
        """
        height, width = (int(v) for v in input_hw)
        if height <= 0 or width <= 0:
            raise ConfigurationError(f"input_hw must be positive, got {input_hw!r}")
        if channels <= 0:
            raise ConfigurationError(f"channels must be positive, got {channels}")
        optimum = max_batch if opt_batch is None else int(opt_batch)
        return cls(
            input_name=input_name,
            minimum=(int(min_batch), channels, height, width),
            optimum=(optimum, channels, height, width),
            maximum=(int(max_batch), channels, height, width),
        )


def build_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    *,
    profiles: Sequence[OptimisationProfile] = (),
    fp16: bool = False,
    int8: bool = False,
    int8_calibration: CalibrationBatchFeeder | None = None,
    calibration_cache: str | Path | None = None,
    workspace_bytes: int = DEFAULT_WORKSPACE,
    timing_cache: str | Path | None = None,
    overwrite: bool = True,
    verbose: bool = False,
) -> Path:
    """Parse an ONNX, build an engine for *this* machine, and serialise it.

    Args:
        onnx_path: the ONNX to parse.
        engine_path: where to write the serialised plan. Parent directories are created.
        profiles: one :class:`OptimisationProfile` per dynamic input. A fully static ONNX needs
            none; a dynamic one without a profile fails at build, which is the correct place for
            it to fail.
        fp16: allow half precision. Roughly halves the latency on a tensor-core GPU. Refused
            rather than ignored when the platform has no fast fp16 — a deployment that asked for
            it and silently got fp32 is a large regression reported as a successful start-up.
        int8: allow INT8. Needs either ``int8_calibration`` or an ONNX that already carries
            Q/DQ nodes from quantisation-aware training. **Read
            :mod:`shipvision.detection.backends.tensorrt.calibration` first**: calibration data
            that does not go through the same preprocessing as inference produces an engine that
            builds, runs at full speed, and is quietly wrong.
        int8_calibration: the calibration batch source. Only meaningful with ``int8``.
        calibration_cache: where to persist the calibrated scales, so that changing an unrelated
            builder flag does not re-run calibration. Delete it when the ONNX changes; the
            scales belong to a network.
        workspace_bytes: tactic-selection scratch. See :data:`DEFAULT_WORKSPACE`.
        timing_cache: persist the builder's kernel timings. Cuts a rebuild of the same network
            on the same GPU from minutes to seconds and is safe to share between builds on
            identical hardware.
        overwrite: refuse to replace an existing engine when `False`.
        verbose: put the TensorRT logger at ``VERBOSE``. Worth it exactly once, when a build
            fails for a reason the parser will not say at ``WARNING``.

    Returns:
        The path written.

    Raises:
        ConfigurationError: the arguments cannot work. Checked **before** tensorrt is imported,
            so a config mistake is reported the same way on a build machine and on a laptop.
        BackendUnavailableError: no tensorrt here, or the platform cannot do what was asked.
        ModelLoadError: the ONNX would not parse, or the builder produced no engine.
    """
    onnx = Path(onnx_path)
    engine = Path(engine_path)

    if int8_calibration is not None and not int8:
        raise ConfigurationError(
            "int8_calibration was given but int8 is False, so the calibrator would never be "
            "asked for a batch. A calibrator that is not used is a silent no-op — the build "
            "would succeed and produce an fp32 engine"
        )
    if workspace_bytes <= 0:
        raise ConfigurationError(f"workspace_bytes must be positive, got {workspace_bytes}")
    if not onnx.is_file():
        raise ModelLoadError(f"no ONNX at {onnx}")
    if engine.exists() and not overwrite:
        raise ConfigurationError(
            f"{engine} already exists and overwrite=False. An engine is specific to this GPU "
            f"and this TensorRT, so replacing one is normal — say so explicitly"
        )

    trt = _require_tensorrt()
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)

    # `_parser` is bound, not discarded: the network holds memory the parser owns, so letting
    # it be collected here is a use-after-free that surfaces as a crash later in the build.
    # Named with a leading underscore because it is never read — its lifetime is the point.
    network, _parser = _parse(trt, builder, logger, onnx)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_bytes))

    _apply_precision(
        trt,
        builder,
        config,
        fp16=fp16,
        int8=int8,
        calibration=int8_calibration,
        calibration_cache=calibration_cache,
    )
    _apply_profiles(builder, config, network, profiles)
    cache_blob = _load_timing_cache(config, timing_cache)

    serialised = builder.build_serialized_network(network, config)
    if serialised is None:
        raise ModelLoadError(
            f"TensorRT built no engine from {onnx}. Run again with verbose=True: the builder "
            f"logs the layer it could not implement, which a returned None cannot"
        )

    if cache_blob is not None:
        _save_timing_cache(config, timing_cache)
    engine.parent.mkdir(parents=True, exist_ok=True)
    engine.write_bytes(bytes(serialised))
    return engine


# ------------------------------------------------------------------------------- steps


def _require_tensorrt() -> Any:
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise BackendUnavailableError(
            "building an engine needs tensorrt; install the 'tensorrt' extra "
            "(pip install 'shipvision[tensorrt]'). An engine must be built on the machine it "
            "will run on — it is specific to the GPU architecture and the TensorRT version"
        ) from exc
    return trt


def _validate_is_onnx(onnx: Path) -> None:
    """Refuse anything that is not a serialised ONNX ModelProto, before TensorRT sees it.

    **TensorRT's ONNX parser segmentation-faults on malformed input.** Measured on 10.14.1:
    both ``parser.parse(bytes)`` and ``parser.parse_from_file(path)`` dump core on a text file
    named ``*.onnx``. There is no return value to check and no exception to catch — the process
    is gone.

    That matters far more than a bad error message. The server builds engines from ONNX on
    demand, so a truncated download or a Git-LFS pointer left unresolved in a model repository
    would take the whole worker down at start-up, and the operator would see a core dump
    instead of "this file is not ONNX".

    So the file is checked here and TensorRT is only ever handed something valid. `onnx` is
    used when importable because its parser reports *why*; the fallback is the ModelProto
    header, which catches every case that is not a protobuf at all — the realistic failures.
    """
    # A backstop: `build_engine` checks existence first, so this fires only for a direct
    # caller. Kept because this function is what stands between TensorRT and a core dump, and
    # a validator that assumes its caller already checked is one refactor from being wrong.
    if not onnx.is_file():
        raise ModelLoadError(f"ONNX not found: {onnx}")
    if onnx.stat().st_size == 0:
        raise ModelLoadError(f"ONNX is empty: {onnx}")

    try:
        import onnx as onnx_module
    except ImportError:
        # Field 1 of ModelProto is `ir_version`, a varint, so a valid file begins with 0x08.
        # Crude, and it is the difference between a typed error and a core dump.
        head = onnx.read_bytes()[:1]
        if head != b"\x08":
            raise ModelLoadError(
                f"{onnx} does not begin like a serialised ONNX ModelProto (first byte "
                f"{head!r}, expected b'\\x08'). Install `onnx` for a precise diagnosis; "
                f"TensorRT's parser crashes rather than reporting on malformed input, so "
                f"this file is refused here."
            ) from None
        return

    try:
        onnx_module.load(str(onnx))
    except Exception as exc:
        raise ModelLoadError(
            f"{onnx} is not a readable ONNX model: {type(exc).__name__}: {exc}. Refused "
            f"before parsing because TensorRT's ONNX parser crashes on malformed input "
            f"rather than reporting it."
        ) from exc


def _parse(trt: Any, builder: Any, logger: Any, onnx: Path) -> tuple[Any, Any]:
    """A parsed network **and the parser that owns it**, or every parser error in one exception.

    The reference logs the errors and carries on to build from a half-populated network
    (``pytools/onnx2trt.py:96-99``), which produces either a mysterious build failure or an
    engine missing layers. Collecting them into the exception is the difference between "the
    build failed" and "opset 19 LayerNormalization is unsupported by this TensorRT".

    The parser is returned rather than dropped because the network holds memory the parser
    owns; letting it be collected while the network is still in use is a use-after-free that
    presents as a crash somewhere later in the build.
    """
    flags = 0
    explicit_batch = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)
    if explicit_batch is not None:
        # Required up to TensorRT 9 and a no-op afterwards, where every network is explicit
        # batch. Reading the flag off the enum rather than assuming a version is what makes one
        # call site work on both.
        flags = 1 << int(explicit_batch)
    _validate_is_onnx(onnx)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx.read_bytes()):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise ModelLoadError(
            f"{onnx} did not parse as ONNX for this TensorRT: " + "; ".join(errors)
        )
    return network, parser


def _apply_precision(
    trt: Any,
    builder: Any,
    config: Any,
    *,
    fp16: bool,
    int8: bool,
    calibration: CalibrationBatchFeeder | None,
    calibration_cache: str | Path | None,
) -> None:
    if fp16:
        if not builder.platform_has_fast_fp16:
            raise BackendUnavailableError(
                "fp16 was requested but this platform reports no fast fp16. Building anyway "
                "would emulate it and be slower than fp32, while reporting success"
            )
        config.set_flag(trt.BuilderFlag.FP16)
    if not int8:
        return

    if not builder.platform_has_fast_int8:
        raise BackendUnavailableError(
            "int8 was requested but this platform reports no fast int8"
        )
    config.set_flag(trt.BuilderFlag.INT8)
    if calibration is None:
        # Legal, and only for a quantisation-aware-trained ONNX that already carries Q/DQ
        # nodes. On a plain float ONNX the builder will refuse for want of scales, which is the
        # correct failure and a clearer one than anything guessed here.
        return
    config.int8_calibrator = build_int8_calibrator(
        trt, calibration, cache=CalibrationCache(calibration_cache)
    )


def _apply_profiles(
    builder: Any, config: Any, network: Any, profiles: Sequence[OptimisationProfile]
) -> None:
    """Add one optimisation profile per declaration, after checking the input names exist."""
    if not profiles:
        return
    declared = {network.get_input(index).name for index in range(network.num_inputs)}
    unknown = [p.input_name for p in profiles if p.input_name not in declared]
    if unknown:
        raise ModelLoadError(
            f"optimisation profile(s) name input(s) {unknown} but this network's inputs are "
            f"{sorted(declared)}. A profile for an input that does not exist leaves the real "
            f"input with no profile, and that only fails at inference"
        )
    for declaration in profiles:
        profile = builder.create_optimization_profile()
        profile.set_shape(
            declaration.input_name,
            declaration.minimum,
            declaration.optimum,
            declaration.maximum,
        )
        config.add_optimization_profile(profile)


def _load_timing_cache(config: Any, path: str | Path | None) -> bytes | None:
    if path is None:
        return None
    location = Path(path)
    blob = location.read_bytes() if location.is_file() else b""
    cache = config.create_timing_cache(blob)
    # ignore_mismatch=False: a cache from a different GPU must be rejected rather than used,
    # because a tactic that was fastest elsewhere is merely a tactic here.
    config.set_timing_cache(cache, False)
    return blob


def _save_timing_cache(config: Any, path: str | Path | None) -> None:
    if path is None:
        return
    cache = config.get_timing_cache()
    if cache is None:
        return
    location = Path(path)
    location.parent.mkdir(parents=True, exist_ok=True)
    location.write_bytes(bytes(cache.serialize()))
