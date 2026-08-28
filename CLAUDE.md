# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

**shipvision** — the maritime computer-vision *algorithms*: detection, segmentation,
re-identification, single-camera tracking (MOT) and cross-camera tracking (MTMC), plus the
fused image pre/post-processing kernels they all need.

It is consumed by [ShipInfer](https://github.com/osirisQdt2810/shipinfer), which owns the
*system*: reading ~50 RTSP cameras, scheduling across 16 GPUs, serving, observing. The
relationship is vLLM↔AITER: ShipInfer calls in here, never the reverse. **This library must
not import anything from ShipInfer**, and there is an architecture test that says so.

Sizing that decides every design choice: **50 cameras × 20 fps = 1000 frames/s**, 10–20
objects per frame → **~15 000 crops/s**, on 16 GPUs. At that sizing the box is GPU-rich:
detection is ~63 fps/GPU. The bottleneck is **load balance** and **end-to-end latency**, not
raw throughput — and per-frame Python overhead, which at 1000 fps is the whole budget.

## The one rule that shapes everything: the numpy backend is the floor

Every algorithm has a readable `python` backend in numpy. An algorithm that has been made
fast also has a compiled one (`native`, C++/CUDA/HIP through `shipvision._C`) registered
under the same name in the same registry — which is the ordinary case, and is what "Adding
an algorithm" step 6 means by *if* you also add a `native` backend.

The asymmetry is deliberate: numpy is what makes a compiled twin checkable and is what runs
where there is no build, so it cannot be the optional half. Landing an algorithm in numpy
first and compiling it when the frame budget asks is the intended order, not a shortcut —
and the gap is recorded rather than assumed, below.

**Outstanding compiled twins.** `mcbyte` (added Aug 2026) is `python`-only: it is BoT-SORT
plus a pre-assignment lock, so the C++ work is one predicate over the cost matrix inside the
existing `BotSortTracker` association loop, and it joins `tests/mot/backends/test_parity.py`
by construction the day it registers. It is the only entry in `TRACKERS` without a twin.
`CAMERA_MOTION` and the MTMC clusterers are `python`-only and are *not* gaps: they are thin
wrappers over OpenCV and scipy, which are compiled already — the ponytail principle, not a
missing port.

This is not redundancy:
- **A fused kernel nobody can compare against is a fused kernel nobody can trust.** The
  numpy version is the oracle in `tests/**/test_*_parity.py`.
- **The numpy version is what runs where there is no build** — which is how the offline test
  tier stays GPU-free and runnable on a laptop in under a second.
- Selecting by name from config is what lets a deployment A/B two algorithms on one stream
  without a code change. Every reference implementation this library replaces uses a
  hand-written `if/elif` instead, and every one has the same consequence: the algorithm that
  shipped first wins by default rather than by measurement.

```python
from shipvision import TRACKERS
fast      = TRACKERS.build("bytetrack", backend="native")
reference = TRACKERS.build("bytetrack", backend="python")
resolved  = TRACKERS.build("bytetrack")   # fastest available, numpy as the floor
```

## The ponytail principle

**Reuse basic, powerful, already-highly-optimised libraries. Do not reimplement them.**
numpy/BLAS for anything that is a matrix product. scipy for `linear_sum_assignment` and
hierarchical clustering. torch where a tensor should stay on the device. TensorRT for
engines. OpenCV for `findHomography`. If you are writing a Hungarian solver, a Kalman
filter's Cholesky, a gemm, or a resize — stop.

What *does* belong in `csrc/`: several memory-bound passes fused into one, and association
inner loops where a Python call per track is the frame budget. Nothing else.

## Non-negotiable conventions

- **Boxes are `xyxy` float32, absolute pixels**, at every boundary. Convert at the edges of
  a model, never in the middle. The Kalman state is `(cx, cy, aspect, height)` — aspect and
  **height**. A converter that writes width there tracks square objects perfectly and falls
  apart on a ship.
- **`(camera_id, frame_id)` survives every path, including error paths.** A mis-tagged
  result is worse than a dropped one: dropped is counted, mis-tagged is a real-looking
  detection on a camera where nothing happened. Use `FrameTag`; never pass the camera
  alongside a bare list.
- **Embeddings are stored L2-normalised**, once, on the way in.
- **Unassigned ids are `None`, never `-1`.** `-1` compares, sorts and serialises as an
  ordinary id, so it flows downstream looking assigned.
- **Failures are typed** (`shipvision.errors`). Never signal failure by returning an empty
  list — a dropped frame, a saturated queue and a dead GPU are three different events.
- **Nothing grows without bound.** Galleries, track pools and MTMC global-id maps all take a
  capacity and state what they evict. A process here runs for weeks.
- **C++: `gpu*` aliases from `core/platform.h` only.** A raw `cudaMalloc` breaks the ROCm
  build silently — `tests/test_architecture.py::TestVendorApiBoundary` is the guard, and it
  strips comments first so prose explaining *why* an alias exists is allowed.
- **C++: nothing here touches the GIL** (operator decision, V70). No `py::gil_scoped_release`
  and no `gil_scoped_acquire` anywhere — `tests/test_architecture.py` guards it. A library
  transforms data; GIL policy belongs to the server that embeds it. Where a stateful session
  can race — a tracker's `track()` — one `std::mutex` on the session is the whole answer.
- **C++ style** (`.clang-format`): `NamespaceIndentation: All`, `IndentAccessModifiers: true`,
  `PointerAlignment: Left` — so `float* dst`, not `float *dst`.
- English for all documentation, comments and commit messages.

## Layout

```
shipvision/            the Python package, at the repository root
├── registry.py        the one Registry primitive every family uses
├── types.py           Detection / Detections / Track / GlobalTrack / Embedding / FrameTag
├── errors.py          ShipVisionError and its children
├── imgproc/           @IMGPROC   letterbox, crop, colour, normalise, NMS
├── detection/         @DETECTORS yolo26 (tensorrt / torch), postprocess
├── reid/              @EXTRACTORS @GALLERIES @AGGREGATORS + metrics, re-ranking
├── mot/               @TRACKERS  sort, bytetrack, botsort, ocsort, mcbyte, deepsortv2 — one per algorithm
├── mtmc/              @MTMC      matrix builders, clustering, topology, global id
├── tune/              Optuna search spaces + objectives
└── eval/              HOTA/MOTA/IDF1 for MOT and MTMC; CMC/mAP for re-ID
csrc/
├── include/shipvision/{core,imgproc,detection,reid,tracking,mtmc}/
├── src/*.cu           one tree, compiled by CUDA or HIP
└── bindings/          pybind11 → shipvision._C
```

## Adding an algorithm

1. New file under the family's directory. Subclass the family's base class.
2. `@FAMILY.register("<name>", backend=PYTHON, aliases=(...))`; import it in the family
   `__init__.py` so registration happens on package import.
3. Validate constructor arguments and raise `ConfigurationError`. A bad config must fail at
   start-up, not on frame 40 000.
4. Reuse the family's shared machinery (the track pool, the gallery, `associate()`). If you
   are re-deriving the lifecycle, change it *in the shared component* behind a parameter.
5. **A test with a scenario that has a known-correct answer.** "It produced output" proves
   nothing. If a test claims one algorithm beats another, it must also assert the baseline
   fails — and the baseline must be configured the way its paper describes, or the
   comparison measures a handicap rather than an algorithm.
6. If you also add a `native` backend, add a parity test against the `python` one.

## Licensing — read before porting anything

The reference repositories under `shipinfer/references/` are internal and carry no LICENSE,
**except** `gitea-multi-object-tracking-pyservice/app/pysrc/core/`, where **47 files are
boxmot under AGPL-3.0**. That covers its BoT-SORT, ByteTrack, DeepOCSORT, HybridSORT,
OCSort, StrongSORT and DeepSort, plus `motion/cmc/` and the Kalman adapters.

**Do not copy, adapt, or consult those files while writing an implementation.** This library
is MIT; AGPL propagates even to network use. Every tracker here is written from the
published papers or ported from the internal C++ (`gitea-generic-multi-object-tracking-cpp`,
`mtmc-tracker`), which is ours.

One permissive exception exists, and it sets the pattern for any other: `mcbyte` is ported
from roboflow/trackers under Apache-2.0. That licence allows it and §4 asks two things back —
retain the notice, state the changes — so every ported file carries a three-line header naming
the upstream path and commit, and `THIRD_PARTY_NOTICES.md` carries the notice and the list.

## Testing

```bash
pytest                    # offline tier: numpy backends, no GPU, no build, under a second
pytest -m native          # the compiled backends and the parity tests
pytest -m gpu             # real devices
```

**Tests are organised into classes, never bare module-level functions.** One class per
coherent claim, so the class name and the method name read as a sentence together:

```python
class TestLetterboxInversion:
    """A box decoded from network space must land back where it started in image pixels."""

    def test_it_round_trips_at_a_non_integer_scale(self) -> None: ...
    def test_an_odd_total_pad_splits_bottom_and_right(self) -> None: ...
```

Two collection rules that bite: the class name **must** start with `Test` or pytest does not
collect it at all — a silently-uncollected file is worse than no file — and the class must
not define `__init__`, or pytest skips it with a warning. Use fixtures or `setup_method` for
shared state. Parametrize, fixtures and markers all work unchanged on methods, and
`conftest.py` fixtures are still injected as method arguments. After any restructuring,
check the collected **count** is unchanged; a drop means a class was misnamed and its tests
vanished quietly.

The offline tier must stay green **and stay GPU-free**. Deselecting a marker is not the same
guarantee as having no device: an unmarked test can take a CUDA path by accident, pass on a
dev box and fail on the runner. `scripts/run_tests.sh` exports `CUDA_VISIBLE_DEVICES=""` for
exactly that reason — use it rather than bare `pytest` before pushing.

## PRs

Branch (`feat/…`, `fix/…`, `chore/…`, `docs/…`), push, PR against `main` with
`.github/pull_request_template.md` filled in — every heading kept, non-applicable ones set
to `N/A — <reason>`. Add the `automerge` label. CI runs tests, then the specialist reviewer
in `.github/reviewer-prompt.md`, then merges on `VERDICT: APPROVE` + green tests + the label.

Remotes are SSH. Co-author trailer on large feature commits only.
