"""The torch backend: ``F.interpolate``, ``F.grid_sample`` and ``torchvision.ops.nms``.

Nothing here is hand-rolled. A resize, a sub-pixel gather and an O(n^2) overlap test are
exactly the operations torch and torchvision have spent years vectorising, fusing and
tuning per architecture, and a numpy-shaped reimplementation of any of them would be slower
*and* a second opinion on the half-pixel convention. What this module actually does is
translate this library's conventions — the geometry ones in
:mod:`shipvision.imgproc.geometry`, the colour and normalisation one in
:mod:`shipvision.imgproc.base` — into the arguments those three primitives want:

* the resize is ``F.interpolate(..., mode="bilinear", align_corners=False)``, which *is*
  convention 1;
* the crop is ``F.grid_sample(..., padding_mode="border", align_corners=False)`` fed the
  coordinates from :func:`~shipvision.imgproc.geometry.crop_centres`, because a crop samples a
  continuous sub-region and ``interpolate`` cannot express one;
* classic NMS is ``torchvision.ops.nms``, which agrees with the CUDA kernel on both the
  strict ``iou >`` test and the stable tie order. The soft and neighbourhood methods have no
  torch primitive and are a sequential loop over survivors either way, so they come from
  :mod:`shipvision.imgproc.nms` — the same code the numpy backend runs.

This backend exists to prototype and to give a numeric second opinion without a build. It
returns numpy, so it pays a device-to-host copy when ``device`` is a GPU; the path that
keeps a preprocessed batch on the device belongs to the native backend and to whatever
allocates the engine's input binding.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from shipvision.errors import BackendUnavailableError, ConfigurationError
from shipvision.imgproc.base import (
    DEFAULT_PAD_VALUE,
    DeviceBuffer,
    ImageOps,
    as_image_batch,
    nchw_nbytes,
    resolve_normalisation,
    validate_boxes,
    validate_image,
    validate_pad_value,
)
from shipvision.imgproc.geometry import (
    LetterboxGeometry,
    clamp_boxes_to_frame,
    crop_centres,
    validate_target_hw,
)
from shipvision.imgproc.nms import CLASSIC, prepare, suppress, validate_max_output

__all__ = ["TorchImageOps"]

try:  # pragma: no cover - exercised by whether the machine has torch, not by a branch
    import torch
    import torchvision
    from torch.nn import functional as torch_functional

    _IMPORT_ERROR: str | None = None
except ImportError as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    torchvision = None  # type: ignore[assignment]
    torch_functional = None  # type: ignore[assignment]
    _IMPORT_ERROR = str(exc)


class TorchImageOps(ImageOps):
    """Letterbox, crop and NMS through torch ops.

    Registered lazily from the package ``__init__`` (see
    :func:`shipvision.registry.Registry.register_lazy`), so importing ``shipvision.imgproc``
    on a machine without torch costs nothing and raises nothing — which is the only reason the
    offline test tier can be a second long. The registry stamps ``name`` and ``backend`` when
    it resolves the lazy target, so there is nothing to declare here.
    """

    def __init__(self, *, device: str = "cpu") -> None:
        """
        Args:
            device: any torch device string. ``"cpu"`` by default, deliberately: this
                backend's job is to be runnable and comparable everywhere, and a default of
                ``"cuda"`` would make the parity tests need a GPU.

        Raises:
            BackendUnavailableError: torch or torchvision is not installed, or ``device``
                names an accelerator this machine does not have.
        """
        if torch is None:
            raise BackendUnavailableError(
                f"the torch image-ops backend needs torch and torchvision: {_IMPORT_ERROR}. "
                f"Install shipvision[torch], or use backend='python'"
            )
        try:
            self._device = torch.device(device)
        except (RuntimeError, TypeError) as exc:
            raise ConfigurationError(f"not a torch device: {device!r}") from exc
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise BackendUnavailableError(
                f"torch reports no CUDA device, so image ops cannot run on {device!r}"
            )

    @property
    def device(self) -> str:
        """The device tensors are built on, as a string."""
        return str(self._device)

    # -- pre-processing ---------------------------------------------------------------

    def letterbox(
        self,
        images: Sequence[np.ndarray] | np.ndarray,
        target_hw: tuple[int, int],
        *,
        pad_value: int = DEFAULT_PAD_VALUE,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        swap_rb: bool = True,
    ) -> tuple[np.ndarray, list[LetterboxGeometry]]:
        """See :meth:`ImageOps.letterbox`. The device-to-host copy is the last line."""
        canvas, geometries = self._letterbox_tensor(
            images, target_hw, pad_value=pad_value, mean=mean, std=std, swap_rb=swap_rb
        )
        return canvas.cpu().numpy(), geometries

    def _letterbox_tensor(
        self,
        images: Sequence[np.ndarray] | np.ndarray,
        target_hw: tuple[int, int],
        *,
        pad_value: int,
        mean: Sequence[float] | None,
        std: Sequence[float] | None,
        swap_rb: bool,
    ) -> tuple[torch.Tensor, list[LetterboxGeometry]]:
        """The batch, still on this backend's device.

        Split out of :meth:`letterbox` so that :meth:`letterbox_into` is the same arithmetic
        with a different destination. Two copies of the canvas assembly would be two chances
        for the fast path to drift from the one the parity suite checks.
        """
        frames = as_image_batch(images)
        target_h, target_w = validate_target_hw(target_hw)
        mean_array, std_array = resolve_normalisation(mean, std)
        pad_value = validate_pad_value(pad_value)
        mean_t = self._as_channel_vector(mean_array)
        std_t = self._as_channel_vector(std_array)

        bars = (np.float32(pad_value) - mean_array) / std_array
        canvas = torch.empty(
            (len(frames), 3, target_h, target_w), dtype=torch.float32, device=self._device
        )
        canvas[:] = self._as_channel_vector(bars)

        geometries: list[LetterboxGeometry] = []
        for index, frame in enumerate(frames):
            geometry = LetterboxGeometry.plan(frame.shape[:2], (target_h, target_w))
            geometries.append(geometry)
            source = self._as_planar_float(frame)
            resized = torch_functional.interpolate(
                source,
                size=(geometry.resized_height, geometry.resized_width),
                mode="bilinear",
                align_corners=False,
            )
            # flip(1) on a three-channel axis is the BGR -> RGB swap, and it happens before
            # normalisation so mean/std stay in destination order — the order the output has,
            # which is BGR when the caller turned the swap off.
            ordered = resized.flip(1) if swap_rb else resized
            normalised = (ordered - mean_t) / std_t
            canvas[
                index : index + 1,
                :,
                geometry.pad_top : geometry.pad_top + geometry.resized_height,
                geometry.pad_left : geometry.pad_left + geometry.resized_width,
            ] = normalised
        return canvas, geometries

    def crop_batch(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        target_hw: tuple[int, int],
        *,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        swap_rb: bool = True,
    ) -> np.ndarray:
        """See :meth:`ImageOps.crop_batch`. The device-to-host copy is the last line."""
        crops = self._crop_tensor(image, boxes, target_hw, mean=mean, std=std, swap_rb=swap_rb)
        return crops.cpu().numpy()

    def _crop_tensor(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        target_hw: tuple[int, int],
        *,
        mean: Sequence[float] | None,
        std: Sequence[float] | None,
        swap_rb: bool,
    ) -> torch.Tensor:
        """The crops, still on this backend's device. See :meth:`_letterbox_tensor`."""
        frame = validate_image(image)
        box_array = validate_boxes(boxes)
        target_h, target_w = validate_target_hw(target_hw)
        mean_array, std_array = resolve_normalisation(mean, std)

        count = box_array.shape[0]
        if count == 0:
            return torch.zeros(
                (0, 3, target_h, target_w), dtype=torch.float32, device=self._device
            )

        height, width = frame.shape[:2]
        clamped = clamp_boxes_to_frame(box_array, height, width)
        grid, degenerate = _crop_grid(clamped, height, width, target_h, target_w)

        source = self._as_planar_float(frame)
        # One grid_sample call for the whole batch, with every crop's rows stacked into one
        # tall grid. The alternative is `source.expand(n, ...)`, which asks torch to see n
        # copies of a 6 MB frame; stacking the *grids* instead keeps the frame single and
        # the launch count at one, which is what matters at 15 000 crops a second.
        stacked = torch.from_numpy(grid.reshape(1, count * target_h, target_w, 2)).to(
            self._device
        )
        sampled = torch_functional.grid_sample(
            source,
            stacked,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        crops = sampled.view(3, count, target_h, target_w).permute(1, 0, 2, 3)

        # A box with no area reads a black crop, matching the CUDA kernel: killing a whole
        # batch over one bad box is the wrong trade, and so is leaving it uninitialised.
        crops = crops.contiguous()
        if degenerate.any():
            crops[torch.from_numpy(degenerate).to(self._device)] = 0.0

        # As in `_letterbox_tensor`: flip(1) is the BGR -> RGB swap, skipped when the caller
        # asked for BGR out, and mean/std are indexed by destination plane either way.
        ordered = crops.flip(1) if swap_rb else crops
        return (ordered - self._as_channel_vector(mean_array)) / (
            self._as_channel_vector(std_array)
        )

    # -- pre-processing, straight to the device ---------------------------------------

    @property
    def supports_device_output(self) -> bool:
        """Only on an accelerator. A CPU tensor's pointer is host memory, and saying yes for it
        would make "device output" mean nothing."""
        return self._device.type == "cuda"

    def letterbox_into(
        self,
        images: Sequence[np.ndarray] | np.ndarray,
        target_hw: tuple[int, int],
        out: DeviceBuffer,
        *,
        pad_value: int = DEFAULT_PAD_VALUE,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        swap_rb: bool = True,
    ) -> list[LetterboxGeometry]:
        """See :meth:`ImageOps.letterbox_into`.

        Torch cannot build a tensor over a foreign device pointer from Python, so this writes
        through :attr:`~shipvision.imgproc.base.DeviceBuffer.owner` — a device-to-device copy
        inside torch, which still keeps the batch off the host. A descriptor built from a bare
        pointer is refused rather than silently downgraded to a host round trip.
        """
        canvas, geometries = self._letterbox_tensor(
            images, target_hw, pad_value=pad_value, mean=mean, std=std, swap_rb=swap_rb
        )
        self._write_through_owner(canvas, out, what="letterbox output")
        return geometries

    def crop_batch_into(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        target_hw: tuple[int, int],
        out: DeviceBuffer,
        *,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        swap_rb: bool = True,
    ) -> None:
        """See :meth:`ImageOps.crop_batch_into`."""
        crops = self._crop_tensor(image, boxes, target_hw, mean=mean, std=std, swap_rb=swap_rb)
        if crops.numel() == 0:
            return
        self._write_through_owner(crops, out, what="crop output")

    def _write_through_owner(
        self, values: torch.Tensor, out: DeviceBuffer, *, what: str
    ) -> None:
        """Copy ``values`` into the tensor the descriptor was built from, on the device."""
        if not self.supports_device_output:
            raise BackendUnavailableError(
                f"these torch image ops are on {self.device!r}, so there is no device output "
                f"path; build them with device='cuda:N'"
            )
        target = out.owner
        if target is None or not hasattr(target, "copy_"):
            raise BackendUnavailableError(
                "the torch backend writes through a tensor rather than a raw address, because "
                "torch cannot wrap a foreign device pointer from Python. Build the buffer with "
                "DeviceBuffer.from_tensor(tensor), or use the native backend for a bare pointer"
            )
        out.require(
            nchw_nbytes(values.shape[0], values.shape[2:]), self._device_ordinal(), what=what
        )
        target.view(-1)[: values.numel()].copy_(values.reshape(-1))

    def _device_ordinal(self) -> int:
        """``torch.device("cuda")`` carries no index; it means the current device, which is 0
        unless something set it, and a descriptor has to be compared against a number."""
        index = self._device.index
        return int(torch.cuda.current_device() if index is None else index)

    # -- post-processing --------------------------------------------------------------

    def nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        *,
        iou_threshold: float,
        method: str = CLASSIC,
        sigma: float = 0.5,
        score_threshold: float = 0.0,
        min_neighbors: int = 0,
        min_score_sum: float = 0.0,
        max_output: int | None = None,
    ) -> np.ndarray:
        """See :meth:`ImageOps.nms`. ``"classic"`` goes to ``torchvision.ops.nms``."""
        if method != CLASSIC:
            return suppress(
                boxes,
                scores,
                iou_threshold=iou_threshold,
                method=method,
                sigma=sigma,
                score_threshold=score_threshold,
                min_neighbors=min_neighbors,
                min_score_sum=min_score_sum,
                max_output=max_output,
            )[0]

        box_array, score_array, order = prepare(
            boxes,
            scores,
            iou_threshold=iou_threshold,
            method=method,
            sigma=sigma,
            score_threshold=score_threshold,
            max_output=max_output,
        )
        if order.size == 0:
            return np.zeros(0, dtype=np.int64)
        # torchvision sorts internally and returns descending-score order, so `order` is
        # only used to apply the score threshold and to map back to input indices.
        kept = torchvision.ops.nms(
            torch.from_numpy(box_array[order]).to(self._device),
            torch.from_numpy(score_array[order]).to(self._device),
            float(iou_threshold),
        )
        # The cap is a slice of torchvision's own output, which is already descending score.
        # `torchvision.ops.nms` has no budget argument, so there is nothing to push down into
        # it — and taking the head of a sorted result is exactly what the CUDA sweep does when
        # it stops early, so the two backends truncate the same list at the same place.
        # Normalised for the same reason as `suppress()`: a whole-valued float cap passes
        # validation and must slice as the int it names, not raise (#12 round 1).
        max_output = validate_max_output(max_output)
        if max_output is not None:
            kept = kept[:max_output]
        return order[kept.cpu().numpy()].astype(np.int64)

    # -- helpers ----------------------------------------------------------------------

    def _as_channel_vector(self, values: np.ndarray) -> torch.Tensor:
        """``(3,)`` numpy -> ``(1, 3, 1, 1)`` tensor, ready to broadcast over NCHW."""
        return torch.from_numpy(np.ascontiguousarray(values)).to(self._device).view(1, 3, 1, 1)

    def _as_planar_float(self, frame: np.ndarray) -> torch.Tensor:
        """``(h, w, 3)`` uint8 HWC -> ``(1, 3, h, w)`` float32 on this backend's device."""
        writable = frame if frame.flags.writeable else frame.copy()
        tensor = torch.from_numpy(writable).to(self._device)
        return tensor.permute(2, 0, 1).unsqueeze(0).to(torch.float32)


def _crop_grid(
    clamped: np.ndarray, height: int, width: int, target_h: int, target_w: int
) -> tuple[np.ndarray, np.ndarray]:
    """The ``grid_sample`` grid for a batch of crops, plus a mask of zero-area boxes.

    ``grid_sample`` with ``align_corners=False`` reads a coordinate as
    ``((g + 1) * extent - 1) / 2``, so the pixel coordinates from
    :func:`~shipvision.imgproc.geometry.crop_centres` are handed over as
    ``g = (2 * coordinate + 1) / extent - 1``. Building the grid from those coordinates rather
    than from a normalised box is what keeps this backend on convention 1 instead of on
    grid_sample's own idea of a half pixel.
    """
    grid = np.zeros((clamped.shape[0], target_h, target_w, 2), dtype=np.float32)
    degenerate = np.zeros(clamped.shape[0], dtype=bool)
    for index, box in enumerate(clamped):
        x1, y1, x2, y2 = (float(v) for v in box)
        if x2 <= x1 or y2 <= y1:
            degenerate[index] = True
            continue
        ys = crop_centres(y1, y2, target_h)
        xs = crop_centres(x1, x2, target_w)
        grid[index, :, :, 0] = (np.float32(2.0) * xs + np.float32(1.0)) / np.float32(
            width
        ) - np.float32(1.0)
        grid[index, :, :, 1] = (
            (np.float32(2.0) * ys + np.float32(1.0)) / np.float32(height) - np.float32(1.0)
        )[:, None]
    return grid, degenerate
