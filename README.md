# shipvision

Maritime computer-vision algorithms: detection, segmentation, re-identification,
single-camera tracking and cross-camera tracking, with the fused CUDA/HIP kernels they need.

The algorithm half of [ShipInfer](https://github.com/osirisQdt2810/shipinfer). ShipInfer owns
the system — fifty RTSP cameras, sixteen GPUs, scheduling, serving. This library owns the
algorithms and imports nothing from it. The relationship is vLLM↔AITER.

## Why it is its own repository

An algorithm is judged by HOTA, IDF1, rank-1 and mAP on recorded footage. That measurement
has to run in seconds, with no GPU, no model repository and no engine to load — otherwise it
does not get run, and the algorithm that shipped first wins by default instead of by
evidence. Keeping the algorithms outside the server is what makes that possible.

## Install

```bash
pip install -e ".[dev]"                 # numpy backends only — no GPU, no build needed
pip install -e ".[dev,torch,tensorrt]"  # plus the accelerated backends
cmake -S . -B build && cmake --build build -j   # the compiled shipvision._C
```

## Use

```python
from shipvision import Detection, Detections, FrameTag
from shipvision.mot import TRACKERS

tracker = TRACKERS.build("bytetrack", track_threshold=0.5)

for frame_id, (boxes, scores) in enumerate(stream):
    dets = Detections(
        tag=FrameTag(camera_id="cam-01", frame_id=frame_id),
        items=[Detection(box=b, score=s) for b, s in zip(boxes, scores)],
    )
    for track in tracker.update(dets):
        print(track.track_id, track.box)
```

## Every algorithm exists at least twice

A compiled `native` backend for production, and a readable `python` backend in numpy. Same
name, same registry, chosen by config:

```python
fast      = TRACKERS.build("bytetrack", backend="native")
reference = TRACKERS.build("bytetrack", backend="python")
resolved  = TRACKERS.build("bytetrack")   # fastest available; numpy is the floor
```

The numpy one is not a toy. It is the oracle the compiled one is checked against — a fused
kernel nobody can compare against is a fused kernel nobody can trust — and it is what runs
on a machine with no build.

## What is here

| Package | Contents |
|---|---|
| `imgproc/` | letterbox, crop, colour conversion, normalisation, NMS — fused into one pass over the pixels |
| `detection/` | YOLO26 detection and segmentation, TensorRT and torch backends, postprocessing |
| `reid/` | embedding extractors, bounded galleries, feature aggregation, CMC/mAP, k-reciprocal re-ranking |
| `mot/` | SORT, ByteTrack, BoT-SORT, OC-SORT, McByte, DeepSORTv2 over one shared track pool, one package per algorithm (`tracking/` is a deprecated alias) |
| `mtmc/` | appearance + ground-plane matrix builders, agglomerative clustering, camera topology, global-id assignment with TTL |
| `tune/` | Optuna search spaces and objectives — a tracker's thresholds are an empirical question |
| `eval/` | HOTA/MOTA/IDF1 for MOT and MTMC, CMC/mAP for re-ID |

## Design

Three decisions worth knowing about.

**One registry, keyed on `(name, backend)`.** Adding an implementation is a file and a
decorator, never an edit to a switch statement. See `shipvision/registry.py`.

**One shared vocabulary.** `Detection`, `Track`, `GlobalTrack`, `Embedding`, `FrameTag` in
`shipvision/types.py`, with the conventions fixed once: boxes are `xyxy` float32; the
`(camera_id, frame_id)` tag travels with everything including error paths; embeddings are
stored normalised; unassigned ids are `None`, never `-1`.

**One C++ source tree for two vendors.** Every device call goes through the `gpu*` aliases
in `csrc/shipvision/core/platform.h`, so the ROCm build is the same code. A raw
`cudaMalloc` anywhere else breaks it silently, which is why a test guards for exactly that.

## Tests

```bash
scripts/run_tests.sh   # offline tier, as CI runs it: CUDA_VISIBLE_DEVICES=""
pytest -m native       # compiled backends and the parity tests
pytest -m gpu          # real devices
```

## Licence

MIT, with one exception: the `mcbyte` tracker is ported from
[roboflow/trackers](https://github.com/roboflow/trackers) under the Apache License 2.0.
The notice and the list of changes are in `THIRD_PARTY_NOTICES.md`, and every ported file
carries the upstream path and commit in its header.

No code is taken from the AGPL-licensed boxmot trackers in the reference tree — see
`CLAUDE.md` for why that boundary matters more than the Apache one.
