"""One module per execution backend for the image ops.

Deliberately empty of imports. Importing a backend module is what runs its registration
decorator, so *which* backends get imported is a policy decision — and it belongs in
:mod:`shipvision.imgproc`, which owns the public surface, rather than being an invisible side
effect of touching this directory. In particular the torch backend must not be imported until
something asks for it: torch costs about a second and several hundred megabytes, and the
offline test tier is a second long in total.

The three, in preference order:

:mod:`~shipvision.imgproc.backends.native_ops`
    the fused CUDA/HIP kernels in ``shipvision._C``. Registered unconditionally; construction
    is what fails when there is no build.
:mod:`~shipvision.imgproc.backends.torch_ops`
    ``F.interpolate``, ``F.grid_sample`` and ``torchvision.ops.nms``. Registered lazily.
:mod:`~shipvision.imgproc.backends.numpy_ops`
    the oracle. Always importable, needs no build and no device, and is what the parity tests
    compare the other two against.
"""

from __future__ import annotations
