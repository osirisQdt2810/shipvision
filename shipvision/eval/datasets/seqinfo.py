"""``seqinfo.ini``: the only place that knows how long a sequence really is.

Every per-frame rate in a report divides by the sequence length, and the length is *not*
derivable from the annotation files. A sequence whose ground truth ends on frame 500 of 837
has 337 frames in which a false positive is possible and every one of them belongs in the
denominator. Guessing the length from the highest annotated frame — which is what a loader
without this file has to do — silently reports a false-positive rate up to 40% too high.

The file is INI, so :mod:`configparser` reads it. Writing a parser for four key-value pairs
would be four times the code and would not handle the comment style.
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

from shipvision.errors import ConfigurationError

__all__ = ["SeqInfo", "read_seqinfo"]


@dataclass(frozen=True, slots=True)
class SeqInfo:
    """What ``seqinfo.ini`` says about one sequence.

    Attributes:
        name: the sequence name, e.g. ``MOT17-09-FRCNN``. Used as the ``camera_id`` on every
            frame, so a tracker built for one sequence refuses a frame from another — which
            is the same guard that stops one tracker instance serving two real cameras.
        length: frames the camera produced.
        frame_rate: frames per second, needed to turn a ``max_age`` in frames into seconds
            when comparing a 30 fps sequence with a 14 fps one.
        width: frame width in pixels.
        height: frame height in pixels.
        image_dir: the directory of frames, relative to the sequence root.
        extension: image file extension, including the dot.
    """

    name: str
    length: int
    frame_rate: float = 0.0
    width: int = 0
    height: int = 0
    image_dir: str = "img1"
    extension: str = ".jpg"

    def __post_init__(self) -> None:
        if self.length < 1:
            raise ConfigurationError(
                f"sequence {self.name!r} claims {self.length} frames; a length of zero makes "
                f"every per-frame rate in a report a division by zero"
            )

    def image_path(self, root: Path, frame_id: int) -> Path:
        """Where frame ``frame_id`` lives. MOTChallenge numbers from 1 with six digits."""
        return Path(root) / self.image_dir / f"{frame_id:06d}{self.extension}"


def read_seqinfo(path: Path) -> SeqInfo:
    """Parse ``seqinfo.ini``, or say which key was missing.

    Raises:
        ConfigurationError: the file is absent, has no ``[Sequence]`` section, or omits
            ``seqLength``. Every one of those is a mis-laid-out dataset directory, which is a
            start-up problem and must not become a plausible-looking number later.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigurationError(
            f"no seqinfo.ini at {path}. Without it the sequence length has to be guessed "
            f"from the annotations, which understates it and inflates every per-frame rate"
        )
    parser = configparser.ConfigParser()
    parser.read(path)
    if not parser.has_section("Sequence"):
        raise ConfigurationError(
            f"{path} has no [Sequence] section; sections: {parser.sections()}"
        )
    section = parser["Sequence"]
    if "seqlength" not in section:
        raise ConfigurationError(f"{path} has no seqLength; keys: {sorted(section)}")
    return SeqInfo(
        name=section.get("name", path.parent.name),
        length=int(section["seqlength"]),
        frame_rate=float(section.get("framerate", 0.0)),
        width=int(section.get("imwidth", 0)),
        height=int(section.get("imheight", 0)),
        image_dir=section.get("imdir", "img1"),
        extension=section.get("imext", ".jpg"),
    )
