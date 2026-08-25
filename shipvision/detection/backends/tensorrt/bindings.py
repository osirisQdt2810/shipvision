"""What an engine says about itself: names, shapes, dtypes and optimisation profiles.

Separated from execution for two reasons, and the second one is why it is worth a file.

The first is that TensorRT has two incompatible IO APIs. Up to 9 an engine is a list of
*bindings* addressed by index (``num_bindings``, ``get_binding_shape``, ``enqueueV2``); from
8.5 it is a set of named IO *tensors* (``num_io_tensors``, ``get_tensor_shape``,
``setTensorAddress``, ``enqueueV3``). Both are in the field. Handling both in one place means
the execution path below has one shape of engine to talk to, rather than a branch at every
call site — which is how the reference wrapper in ``references/counting-simulation``
(``trt_backend.py:44-84``) ends up with the version test repeated in five places.

The second is that **this module imports nothing**. Reading an engine's bindings is the part of
the TensorRT path with all the correctness decisions in it — which dimension is the batch,
whether a spatial extent is dynamic, what happens when a caller's configured input size
disagrees with the binding — and none of that needs a GPU or a driver to be exercised. Passing
the engine and the ``tensorrt`` module in as arguments is what lets the whole of it be tested
on a machine with neither, against a stub that answers like an engine. The alternative, an
``import tensorrt`` at the top, would mean the only tests of this logic are the ones nobody can
run on the machine they are writing it on.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from shipvision.errors import ModelLoadError

__all__ = ["Binding", "EngineBindings"]

#: The input every YOLO export names its image tensor. Used only to disambiguate an engine with
#: several inputs; a single-input engine is not required to use it.
IMAGE_INPUT = "images"


@dataclass(slots=True, frozen=True)
class Binding:
    """One engine IO tensor, as the engine describes it.

    Attributes:
        name: the tensor name. Under the pre-10 API this is the binding name, which is the
            same string.
        index: position in the engine's IO list. Load-bearing for the pre-10 API, whose
            ``execute_v2`` takes addresses in exactly this order.
        is_input: which direction.
        shape: the declared shape. ``-1`` marks a dimension the profile decides.
        dtype: the numpy dtype the engine reads or writes.
        profile: ``(min, opt, max)`` for an input with any dynamic dimension, else `None`.
    """

    name: str
    index: int
    is_input: bool
    shape: tuple[int, ...]
    dtype: np.dtype
    profile: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] | None = None

    @property
    def is_dynamic(self) -> bool:
        return any(dimension < 0 for dimension in self.shape)

    def sized(self, batch: int) -> tuple[int, ...]:
        """The shape with every dynamic dimension resolved, at a given batch.

        The leading axis takes ``batch``; any *other* dynamic axis takes its profile maximum,
        because a buffer has to be big enough for the largest thing the engine may write and
        there is nothing else to size it from. An output with no profile and a dynamic
        non-batch axis cannot be sized at all, and says so rather than allocating something
        that will be overrun.
        """
        resolved: list[int] = []
        for axis, dimension in enumerate(self.shape):
            if dimension >= 0:
                resolved.append(int(dimension))
            elif axis == 0:
                resolved.append(int(batch))
            elif self.profile is not None:
                resolved.append(int(self.profile[2][axis]))
            else:
                raise ModelLoadError(
                    f"binding {self.name!r} has shape {self.shape} with a dynamic axis "
                    f"{axis} and no optimisation profile to bound it. A buffer sized by "
                    f"guesswork is a buffer overrun waiting for a busy frame"
                )
        return tuple(resolved)


@dataclass(slots=True, frozen=True)
class EngineBindings:
    """An engine's IO, read once at load and never asked again.

    Attributes:
        inputs: every input binding, in engine order.
        outputs: every output binding, in engine order. That order is preserved and passed to
            the head, which identifies the detection and prototype tensors by *rank* — an
            exporter's output order is not something to depend on.
        named_api: `True` when the engine exposes the TensorRT 8.5+ named-tensor API. Recorded
            rather than re-tested, because the two APIs also differ in how execution is
            enqueued.
    """

    inputs: tuple[Binding, ...]
    outputs: tuple[Binding, ...]
    named_api: bool

    # -- reading ----------------------------------------------------------------------

    @classmethod
    def read(cls, engine: Any, trt: Any, *, artefact: str = "engine") -> EngineBindings:
        """Walk an engine's IO through whichever API it has.

        Args:
            engine: a deserialised ``ICudaEngine``, or anything that answers like one.
            trt: the ``tensorrt`` module — for ``nptype`` and ``TensorIOMode``. Injected rather
                than imported so this can be exercised without a driver; see the module
                docstring.
            artefact: what to call the engine in an error message.
        """
        named = hasattr(engine, "num_io_tensors")
        entries = cls._read_named(engine, trt) if named else cls._read_indexed(engine, trt)
        inputs = tuple(b for b in entries if b.is_input)
        outputs = tuple(b for b in entries if not b.is_input)
        if not inputs or not outputs:
            raise ModelLoadError(
                f"{artefact} has {len(inputs)} input(s) and {len(outputs)} output(s); a "
                f"detector needs at least one of each"
            )
        return cls(inputs=inputs, outputs=outputs, named_api=named)

    @staticmethod
    def _read_named(engine: Any, trt: Any) -> list[Binding]:
        """TensorRT 8.5+: IO tensors addressed by name."""
        entries: list[Binding] = []
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            shape = tuple(int(v) for v in engine.get_tensor_shape(name))
            profile = None
            if is_input and any(v < 0 for v in shape):
                profile = _profile_of(engine.get_tensor_profile_shape, name, 0, name)
            entries.append(
                Binding(
                    name=name,
                    index=index,
                    is_input=is_input,
                    shape=shape,
                    dtype=np.dtype(trt.nptype(engine.get_tensor_dtype(name))),
                    profile=profile,
                )
            )
        return entries

    @staticmethod
    def _read_indexed(engine: Any, trt: Any) -> list[Binding]:
        """TensorRT 9 and earlier: bindings addressed by index.

        Kept because engines are not portable across TensorRT major versions and a deployment
        on an older base image cannot simply rebuild them — the machine that has the ONNX and
        the machine that has the driver are often not the same machine.
        """
        entries: list[Binding] = []
        for index in range(engine.num_bindings):
            name = engine.get_binding_name(index)
            is_input = bool(engine.binding_is_input(index))
            shape = tuple(int(v) for v in engine.get_binding_shape(index))
            profile = None
            if is_input and any(v < 0 for v in shape):
                profile = _profile_of(engine.get_profile_shape, 0, index, name)
            entries.append(
                Binding(
                    name=name,
                    index=index,
                    is_input=is_input,
                    shape=shape,
                    dtype=np.dtype(trt.nptype(engine.get_binding_dtype(index))),
                    profile=profile,
                )
            )
        return entries

    # -- the image input --------------------------------------------------------------

    @property
    def image_input(self) -> Binding:
        """The input the frames go into.

        A single-input engine is unambiguous. With several inputs, the one named ``images`` is
        taken — that is what every YOLO export calls it — and anything else raises rather than
        picking the first, because "the first input" is a property of the exporter's graph
        traversal and not of the model.
        """
        if len(self.inputs) == 1:
            return self.inputs[0]
        for binding in self.inputs:
            if binding.name == IMAGE_INPUT:
                return binding
        raise ModelLoadError(
            f"this engine has inputs {[b.name for b in self.inputs]} and none of them is "
            f"{IMAGE_INPUT!r}. A detector with several inputs needs a class that knows which "
            f"one takes pixels; picking the first would be guessing"
        )

    @property
    def dynamic_batch(self) -> bool:
        return self.image_input.shape[0] < 0

    @property
    def max_batch(self) -> int:
        """The largest batch this engine will accept, from the binding or from the profile."""
        binding = self.image_input
        if binding.shape[0] >= 0:
            return int(binding.shape[0])
        if binding.profile is None:
            raise ModelLoadError(
                f"input {binding.name!r} has a dynamic batch and no optimisation profile, so "
                f"there is no upper bound to size buffers against"
            )
        return int(binding.profile[2][0])

    def resolve_input_hw(
        self, requested: tuple[int, int] | None = None, *, artefact: str = "engine"
    ) -> tuple[int, int]:
        """The network input extent, from the engine — and a caller who disagrees is refused.

        This is the method the whole module exists for. A configured 640x640 against a 512x512
        engine does not crash: the letterbox produces the configured size, and either TensorRT
        rejects it or, with a dynamic profile, accepts it and hands back boxes scaled by
        640/512 on every frame forever. So the engine's number wins, and a caller who passed a
        different one is told rather than overridden — being overridden silently is how the
        config file and the artefact stay disagreeing for a year.

        A **dynamic** spatial extent has no single answer, so the profile's *optimum* is used:
        that is the shape the engine's kernels were selected for, and running anything else is
        a measurable loss even when it is legal. A caller may name a different extent, which is
        then checked against the profile's ``min`` and ``max``.

        Raises:
            ModelLoadError: the request contradicts a static binding, or falls outside a
                dynamic profile.
        """
        binding = self.image_input
        if len(binding.shape) != 4:
            raise ModelLoadError(
                f"{artefact} input {binding.name!r} has shape {binding.shape}; an image batch "
                f"is (n, c, h, w)"
            )
        height, width = binding.shape[2], binding.shape[3]

        if height >= 0 and width >= 0:
            engine_hw = (int(height), int(width))
            if requested is not None and tuple(int(v) for v in requested) != engine_hw:
                raise ModelLoadError(
                    f"{artefact} takes {engine_hw[0]}x{engine_hw[1]} but input_hw="
                    f"{tuple(requested)} was configured. The engine is the artefact and wins; "
                    f"a letterbox at the configured size would scale every box by "
                    f"{engine_hw[0]}/{int(requested[0])} with no other symptom"
                )
            return engine_hw

        if binding.profile is None:
            raise ModelLoadError(
                f"{artefact} input {binding.name!r} has a dynamic spatial extent "
                f"{binding.shape} and no optimisation profile, so there is nothing to "
                f"letterbox to"
            )
        minimum, optimum, maximum = binding.profile
        if requested is None:
            return int(optimum[2]), int(optimum[3])

        wanted = tuple(int(v) for v in requested)
        if not (
            minimum[2] <= wanted[0] <= maximum[2] and minimum[3] <= wanted[1] <= maximum[3]
        ):
            raise ModelLoadError(
                f"{artefact} accepts spatial extents from {minimum[2]}x{minimum[3]} to "
                f"{maximum[2]}x{maximum[3]}; input_hw={wanted} is outside its profile"
            )
        return wanted[0], wanted[1]

    def output_shapes(self, batch: int) -> list[tuple[int, ...]]:
        """Every output's shape at a given batch, for sizing buffers and picking a head."""
        return [binding.sized(batch) for binding in self.outputs]

    def __repr__(self) -> str:
        api = "named" if self.named_api else "indexed"
        return (
            f"<EngineBindings {api} in={[b.name for b in self.inputs]} "
            f"out={[b.name for b in self.outputs]}>"
        )


def _profile_of(
    accessor: Any, first: Any, second: Any, name: str
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """``(min, opt, max)`` from either profile accessor, or a typed refusal.

    The two APIs take their arguments in opposite orders — ``get_tensor_profile_shape(name,
    profile)`` against ``get_profile_shape(profile, index)`` — which is exactly the kind of
    detail that is fine once and a bug the third time it is written out.
    """
    try:
        minimum, optimum, maximum = accessor(first, second)
    except Exception as exc:
        raise ModelLoadError(
            f"binding {name!r} is dynamic but its optimisation profile could not be read: "
            f"{exc}. An engine built without a profile for a dynamic input cannot run one"
        ) from exc
    shapes = tuple(tuple(int(v) for v in shape) for shape in (minimum, optimum, maximum))
    if len({len(shape) for shape in shapes}) != 1:
        raise ModelLoadError(f"binding {name!r} has profile shapes of differing rank: {shapes}")
    return shapes  # type: ignore[return-value]


def as_shape(values: Sequence[int]) -> tuple[int, ...]:
    """``tuple(int, ...)``, for turning a ``trt.Dims`` into something comparable."""
    return tuple(int(v) for v in values)
