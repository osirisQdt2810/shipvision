"""Detection runtimes: one per way of executing a trained artefact, plus a mock.

The registered *name* is the model family whose output layout is decoded — ``yolo26`` — and the
*backend* is what executes it. That puts the TensorRT and TorchScript paths under one name on
purpose: they are the same algorithm answering the same question at different speeds, which is
exactly the comparison the registry exists to make possible::

    fast      = DETECTORS.build("yolo26", backend="tensorrt", path="yolo26n.engine")
    reference = DETECTORS.build("yolo26", backend="torch", path="yolo26n.ts", input_hw=(640, 640))

Which *head* runs is read from the artefact rather than from the name, because the artefact
knows: a detection export has one output and a segmentation export has two. So
``yolo26_seg`` is an alias of ``yolo26`` rather than a separate entry — naming it selects the
same class, and the class then discovers that the engine segments. A config file that claims
otherwise is refused at load, not silently preferred over the file on disk.

``mock`` is a separate *name*, not a backend of ``yolo26``, and that distinction matters: if it
were a backend, ``DETECTORS.build("yolo26")`` on a machine with no engine would resolve to it
and a deployment would report a successful start-up while detecting nothing real. A missing
engine must be an error.

Both artefact backends are registered **lazily**. Importing tensorrt to discover that tensorrt
is absent costs a second and an exception on every start-up of every process that never uses
it, and importing torch costs more.
"""

from __future__ import annotations

from shipvision.detection.backends.mock import MockDetector
from shipvision.detection.base import DETECTORS
from shipvision.registry import TENSORRT, TORCH

DETECTORS.register_lazy(
    "yolo26",
    "shipvision.detection.backends.tensorrt.engine:TensorRTDetector",
    backend=TENSORRT,
    aliases=("yolo26_seg", "yolo26-seg"),
)
DETECTORS.register_lazy(
    "yolo26",
    "shipvision.detection.backends.torch_backend:TorchDetector",
    backend=TORCH,
)

__all__ = ["DETECTORS", "MockDetector"]
