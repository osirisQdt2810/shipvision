"""A deterministic, hardware-free extractor — the one every other test depends on.

There is no model here, and that is the point: a gallery test, a tracking test and an MTMC
test all need embeddings whose "same object" and "different object" answers are known, and
loading a real engine to get them would make the offline tier need a GPU.

A mock that returns zeros, or one that returns noise, would not do either. Zeros make every
similarity identical, so a broken ranking passes. Noise makes every pair of crops equally
unlike, so a broken *tracker* passes — in high dimensions two random unit vectors are
nearly orthogonal whatever produced them. What the rest of the library actually needs is the
structure real embeddings have: **crops that look alike must embed alike, and crops that do
not must not.**

So this is a real feature map rather than a hash. Each crop is reduced to a small
block-mean thumbnail, and that thumbnail is put through random Fourier features (Rahimi &
Recht, NIPS 2007): ``cos(W s + b)`` with ``W`` gaussian and ``b`` uniform gives, after
normalisation, a cosine similarity that approximates ``exp(-||s - s'||^2 / 2 sigma^2)``.
That yields exactly the three properties the tests want, for the price of one small gemm:

* identical crops score 1.0, exactly, every run and every process;
* similar crops score high, decaying smoothly with how different they are;
* unrelated crops score near 0, the way real re-ID embeddings behave.

Determinism comes from a seeded :class:`numpy.random.Generator`, not from :func:`hash`,
which is salted per process for `str` and `bytes` and would make results differ between
runs — and a test whose expected answer depends on ``PYTHONHASHSEED`` is worse than no test.

**The usable regime, because outside it this class does the opposite of what it says.** Two
things have to hold, and both are checked or asserted rather than left as advice.

*Crops must be on a scale that `bandwidth` can resolve.* `bandwidth` is a per-pixel
intensity difference, so it only means anything next to the numbers the crops actually
carry. Crops left in [0, 255] against the default 0.15 do not degrade — they **invert**:
measured over six objects at three jittered views each, two views of one object score -0.135
to 0.073 while unrelated crops reach 0.085, a margin of -0.220 the wrong way round. Every
appearance test built on that silently becomes a geometry-only test and still passes. So
:meth:`MockExtractor.extract` refuses a batch whose peak-to-peak spread is more than 64
bandwidths and names the value to pass instead (for [0, 255] that is 38, which reproduces
the [0, 1] embeddings to 1e-5). Peak-to-peak rather than maximum on purpose: the random
Fourier construction is shift-invariant, so crops of 1000.5 +/- 0.5 are as well behaved as
crops of 0.5 +/- 0.5 — 0.904 margin against 0.879 — and a guard on `max()` would refuse them
for no reason.

*Crops must carry high-frequency structure.* A crop is reduced to a block-mean thumbnail
before anything else happens, so whatever survives that is all this class can see. Measured
maximum similarity between crops of six unrelated objects: textured 0.12, flat colour 0.84,
smooth gradient 0.98. **Independent uniform noise is not two different crops** either —
block-averaging noise converges on the same flat thumbnail whatever the noise was, and two
noise images score around 0.9. Those are the right answers to the only question this class
can be asked, and they are rarely the question a test meant to ask: a synthetic gradient
crop with an appearance gate at 0.8 matches everything against everything and proves nothing
while passing. Give crops structure and unrelated ones drop to near zero.
"""

from __future__ import annotations

import numpy as np

from shipvision.errors import ConfigurationError
from shipvision.registry import PYTHON
from shipvision.reid.base import EXTRACTORS, FeatureExtractor
from shipvision.reid.distance import normalize

__all__ = ["MockExtractor"]

#: Mixed into `seed` so that the default mock is not the same map as a caller's `seed=0`
#: would suggest, and so two extractors built with adjacent seeds are properly independent.
_SEED_SALT = 0x5348_4950

#: How many bandwidths of peak-to-peak spread a batch may carry before it is refused. Chosen
#: from the measured same-object-versus-unrelated margin over six objects at three views:
#: 0.88 at a ratio of 7, 0.82 at 53, 0.78 at 64, then 0.57 at 107, 0.06 at 213 and *negative*
#: from 255 up. Everything the measurement calls healthy is allowed and everything it calls
#: degrading is refused, which is the useful place for the line — a mock whose ordering is
#: merely compressed is still a mock nobody should be building an oracle on.
_MAX_SPREAD_IN_BANDWIDTHS = 64.0


def _reduce_axis(x: np.ndarray, axis: int, target: int) -> np.ndarray:
    """Average ``x`` down to ``target`` bins along ``axis``, whatever its length.

    ``np.add.reduceat`` rather than a reshape because a reshape only works when the length
    divides evenly, and crops do not have a fixed size — a person crop is tall, a ship crop
    is wide, and both must reduce to the same summary or their embeddings are not
    comparable. Where ``target`` exceeds the length, repeated bin starts make reduceat
    replicate the single element, which is nearest-neighbour upsampling and is what we want.
    """
    length = x.shape[axis]
    starts = (np.arange(target) * length) // target
    counts = np.maximum(np.diff(np.append(starts, length)), 1)
    summed = np.add.reduceat(x, starts, axis=axis)
    shape = [1] * x.ndim
    shape[axis] = target
    return summed / counts.reshape(shape)


def _thumbnail(batch: np.ndarray, grid: int) -> np.ndarray:
    """``(n, c, h, w)`` to ``(n, c * grid * grid)`` block means, in float64.

    float64 throughout so that the same crop passed in a batch of one and in a batch of a
    hundred lands on the same embedding. In float32 the gemm below picks a different
    blocking for different batch shapes and the last bit or two disagree, which is a
    perfectly reasonable thing for BLAS to do and a miserable thing to have to write a
    tolerance for in every test that uses this class.
    """
    reduced = _reduce_axis(batch.astype(np.float64), axis=2, target=grid)
    reduced = _reduce_axis(reduced, axis=3, target=grid)
    return reduced.reshape(batch.shape[0], -1)


@EXTRACTORS.register("mock", backend=PYTHON, aliases=("fake",))
class MockExtractor(FeatureExtractor):
    """Content-sensitive embeddings with no model, no GPU and no build.

    Args:
        dim: embedding width. The one implementation whose width *is* configured, because
            it is the one with no artefact to read it from — 512 by default so that a
            pipeline built against the mock does not change shape when a real engine
            replaces it.
        grid: side of the block-mean thumbnail. Bigger sees finer detail and is slower;
            8 (a 192-value summary for a 3-channel crop) is enough to tell crops apart
            without being sensitive to a one-pixel shift.
        bandwidth: the **per-pixel** intensity difference at which two crops stop looking
            alike. Per-pixel rather than per-vector so the value does not have to be
            retuned when `grid` changes. The default suits crops scaled to roughly [0, 1],
            which is what a preprocessed batch is; crops left in [0, 255] need 38, and are
            refused rather than silently inverted if they arrive without it.
        channels: how many channels a crop has. 3 is the convention at every boundary in
            this library; it is a parameter so a single-channel infrared stream can be
            mocked too.
        seed: which map. Two extractors with different seeds are two different "models",
            which is how a test builds the two-models-one-gallery failure on purpose.
    """

    def __init__(
        self,
        *,
        dim: int = 512,
        grid: int = 8,
        bandwidth: float = 0.15,
        channels: int = 3,
        seed: int = 0,
    ) -> None:
        if dim <= 0:
            raise ConfigurationError(f"dim must be positive, got {dim}")
        if grid <= 0:
            raise ConfigurationError(f"grid must be positive, got {grid}")
        if channels <= 0:
            raise ConfigurationError(f"channels must be positive, got {channels}")
        if bandwidth <= 0.0:
            raise ConfigurationError(f"bandwidth must be positive, got {bandwidth}")

        self._dim = int(dim)
        self._grid = int(grid)
        self._channels = int(channels)
        self.bandwidth = float(bandwidth)
        self.seed = int(seed)

        features = self._channels * self._grid * self._grid
        rng = np.random.default_rng(_SEED_SALT + self.seed)
        # sqrt(features) in the scale is what makes `bandwidth` per-pixel: the squared
        # distance between two thumbnails grows linearly with how many values they have, so
        # without it the same bandwidth would mean something different at every grid size.
        scale = 1.0 / (self.bandwidth * np.sqrt(features))
        self._projection = rng.normal(scale=scale, size=(features, self._dim))
        self._phase = rng.uniform(0.0, 2.0 * np.pi, size=self._dim)

    @property
    def dim(self) -> int:
        return self._dim

    def extract(self, crops: np.ndarray) -> np.ndarray:
        batch = self._as_batch(crops, channels=self._channels)
        if batch.shape[0] == 0:
            return self._empty()
        self._check_scale(batch)
        summary = _thumbnail(batch, self._grid)
        features = np.cos(summary @ self._projection + self._phase)
        return normalize(features.astype(np.float32), copy=False)

    def _check_scale(self, batch: np.ndarray) -> None:
        """Refuse a batch whose spread `bandwidth` cannot resolve. See the module docstring.

        Raised rather than auto-scaled. Auto-scaling would make the same crops embed
        differently depending on what else happened to be in their batch, which is the one
        property everything downstream of this class relies on not changing.
        """
        spread = float(batch.max() - batch.min())
        limit = _MAX_SPREAD_IN_BANDWIDTHS * self.bandwidth
        if spread > limit:
            raise ConfigurationError(
                f"these crops span {spread:.3g} but bandwidth is {self.bandwidth:g}, which "
                f"resolves differences up to about {limit:.3g}. Past that the kernel has "
                f"underflowed for every pair and the ranking is the phase noise of the "
                f"random projection — two views of one object come out *less* alike than "
                f"two unrelated crops, so an appearance test keeps passing while measuring "
                f"nothing. Either scale the crops to roughly [0, 1], which is what a "
                f"preprocessed batch is, or build this extractor with "
                f"bandwidth={0.15 * spread:.3g}"
            )
