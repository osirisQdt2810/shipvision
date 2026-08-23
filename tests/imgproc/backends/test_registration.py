"""How the three backends get into the registry, and what happens when one cannot be built.

Registration and *availability* are different things, and conflating them is the failure this
file guards. The native backend registers on a machine with no CUDA toolchain — that is
deliberate, so nothing above it needs a try/import dance — which means only its constructor
knows the truth, and ``build_image_ops`` has to act on that. Meanwhile the torch backend must
be reachable by name without torch being imported by anyone who never asks for it.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from shipvision.errors import BackendUnavailableError, ConfigurationError
from shipvision.imgproc import IMGPROC, NumpyImageOps, build_image_ops
from shipvision.imgproc.backends import native_ops, torch_ops
from shipvision.registry import NATIVE, PYTHON, TORCH
from tests.imgproc.conftest import NATIVE_BUILT, TORCH_INSTALLED


def test_the_numpy_backend_is_always_available() -> None:
    """The floor the whole registry rests on. If this can be skipped, nothing else is a test."""
    assert PYTHON in IMGPROC.backends("default")


def test_all_three_backends_register_regardless_of_what_is_installed() -> None:
    """Including on a machine with no build and no torch.

    The registry is a statement about what this library *implements*, not about what this host
    can run. Making it conditional would mean ``IMGPROC.backends("default")`` returned a
    different answer per machine, and an error message listing what is available would stop
    being able to say "native exists, it just is not built here".
    """
    assert IMGPROC.backends("default") == [NATIVE, TORCH, PYTHON]


def test_the_resolution_order_puts_numpy_last() -> None:
    """Fastest first, numpy as the floor rather than the default."""
    order = IMGPROC.backends("default")

    assert order.index(NATIVE) < order.index(TORCH) < order.index(PYTHON)


def test_importing_the_package_does_not_import_torch() -> None:
    """The lazy registration, checked the only way it can be: in a fresh interpreter.

    ``shipvision.imgproc`` registers a torch backend, and importing torch costs about a second
    and several hundred megabytes. If that cost leaks into ``import shipvision.imgproc`` then
    every numpy-only process pays it, the offline test tier stops being a second long, and the
    claim in the package docstring is quietly false. Nothing else in this suite can catch it,
    because by the time these tests run something else has already imported torch.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import shipvision.imgproc, sys; "
            "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(not TORCH_INSTALLED, reason="torch and torchvision are not installed")
def test_a_lazily_registered_backend_still_knows_what_it_is() -> None:
    """``name`` and ``backend`` are stamped when the registry resolves the lazy target.

    They are what a log line prints, so a backend that inherited the wrong pair would report
    "python" for work that ran on a GPU — and the whole point of the two-backend rule is being
    able to tell which one produced a number.
    """
    ops = build_image_ops(backend=TORCH)

    assert (ops.name, ops.backend) == ("default", TORCH)
    assert repr(ops) == "<TorchImageOps name='default' backend='torch'>"


def test_pinning_an_unbuildable_backend_raises_instead_of_falling_back(monkeypatch) -> None:
    """Asking for ``native`` and silently getting numpy would be a large throughput regression
    reported as a successful start-up. An explicit backend never falls back."""
    monkeypatch.setattr(native_ops, "_C", None)

    with pytest.raises(BackendUnavailableError, match="not built"):
        build_image_ops(backend=NATIVE)


def test_resolution_falls_back_past_a_backend_that_cannot_be_built(monkeypatch) -> None:
    """With no ``_C``, ``build_image_ops()`` must land on the next backend, not raise.

    Simulated by hiding the extension rather than by trusting this machine not to have one:
    the fallback has to be tested on the box that *does* have a build, or it is only ever
    exercised where it cannot fail.
    """
    monkeypatch.setattr(native_ops, "_C", None)

    ops = build_image_ops()

    assert ops.backend != NATIVE


def test_resolution_reaches_numpy_when_nothing_else_can_be_built(monkeypatch) -> None:
    """The floor, reached for real: every faster runtime hidden at once.

    That is the state of a plain CI runner, and the answer must still be a working ImageOps
    rather than an exception — it is what lets the whole offline tier exist. Simulated by
    hiding the two runtimes rather than by trusting this machine to lack them, because a
    fallback only tested where it cannot fail is not tested.
    """
    monkeypatch.setattr(native_ops, "_C", None)
    monkeypatch.setattr(torch_ops, "torch", None)

    ops = build_image_ops()

    assert isinstance(ops, NumpyImageOps)
    assert ops.backend == PYTHON


def test_an_unknown_algorithm_name_is_refused_by_name() -> None:
    with pytest.raises(ConfigurationError, match="unknown image ops"):
        build_image_ops("bilinear_but_faster")


def test_an_unknown_backend_for_a_known_name_is_refused() -> None:
    with pytest.raises(ConfigurationError, match="has no"):
        build_image_ops(backend="opencl")


@pytest.mark.native
@pytest.mark.skipif(not NATIVE_BUILT, reason="shipvision._C is not built, or there is no GPU")
def test_the_native_backend_reports_the_device_it_is_bound_to() -> None:
    """One instance, one device, for the life of the instance — the staging ring is
    per-instance, so an instance that wandered between devices would race itself."""
    ops = build_image_ops(backend=NATIVE, device_index=0)

    assert ops.device_index == 0
    assert set(ops.scratch_bytes()) >= {"staging_ring", "output", "nms"}
