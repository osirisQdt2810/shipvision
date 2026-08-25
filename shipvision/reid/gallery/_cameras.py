"""Camera ids as small integers, so the protocol's camera filter is a numpy mask.

Excluding the query's own camera is not optional (see :mod:`shipvision.reid.gallery.base`),
which means it runs on *every* query. Written over the stored strings it is a Python
comparison per gallery entry — about 4 ms at 10 000 entries, two orders of magnitude more
than the matrix product it is filtering. Interned to an ``int32`` code it is one vectorised
``==`` over a contiguous array, and it disappears from the profile.

Both galleries need exactly this and nothing more of each other, so this is the one piece
they share.
"""

from __future__ import annotations

from shipvision.errors import ConfigurationError

__all__ = ["NO_CAMERA", "CameraCodec"]

#: The code for "this entry has no camera". Negative so it can never collide with a real
#: code, and a distinct value rather than `None` so the whole column stays one integer array.
NO_CAMERA = -1

#: Camera ids are a deployment's RTSP streams — fifty of them at this project's sizing, and a
#: number that changes when someone edits a config file, not while the process runs. The cap
#: is two orders of magnitude above that so nothing legitimate reaches it; what reaches it is
#: a caller minting an id per frame or per track, which would otherwise show up only as an
#: unexplained few megabytes per hundred thousand frames.
_DEFAULT_MAX_CAMERAS = 4096


class CameraCodec:
    """A bounded two-way map between camera ids and dense ``int32`` codes.

    Codes are assigned in first-seen order and never reused while the codec lives, because
    a stored column of codes would otherwise start meaning something else. :meth:`clear`
    is the only reset, and a gallery calls it from its own ``clear`` — the table is derived
    state, and a gallery that has forgotten everything must not still be holding the names.
    """

    def __init__(self, *, max_cameras: int = _DEFAULT_MAX_CAMERAS) -> None:
        if max_cameras <= 0:
            raise ConfigurationError(f"max_cameras must be positive, got {max_cameras}")
        self.max_cameras = int(max_cameras)
        self._codes: dict[str, int] = {}
        self._names: list[str] = []

    def __len__(self) -> int:
        return len(self._names)

    def code_for(self, camera_id: str | None) -> int:
        """The code for ``camera_id``, assigning a new one if this is the first sighting."""
        if camera_id is None:
            return NO_CAMERA
        code = self._codes.get(camera_id)
        if code is None:
            if len(self._names) >= self.max_cameras:
                raise ConfigurationError(
                    f"more than {self.max_cameras} distinct camera ids in one gallery "
                    f"(newest: {camera_id!r}). A deployment has as many camera ids as it "
                    f"has streams; an unbounded stream of them means ids are being minted "
                    f"per frame or per track, and the table would grow all day"
                )
            code = len(self._names)
            self._codes[camera_id] = code
            self._names.append(camera_id)
        return code

    def lookup(self, camera_id: str | None) -> int | None:
        """The code for ``camera_id``, or `None` if it was never stored.

        Deliberately does not intern. A query naming a camera the gallery has never seen
        must not add it to the table — queries arrive at frame rate, and interning on the
        read path is how the table grows without any entry ever being stored.
        """
        if camera_id is None:
            return None
        return self._codes.get(camera_id)

    def name_for(self, code: int) -> str | None:
        """The camera id behind a code, or `None` for :data:`NO_CAMERA`."""
        return None if code == NO_CAMERA else self._names[code]

    def clear(self) -> None:
        self._codes.clear()
        self._names.clear()

    def __repr__(self) -> str:
        return f"<CameraCodec cameras={len(self._names)}/{self.max_cameras}>"
