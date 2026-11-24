"""Unicode-safe image helpers.

OpenCV's ``cv2.imread``/``cv2.imwrite`` use the operating system's ANSI code
page on Windows.  A person's name (and therefore the image path) may contain
non-ASCII characters, which makes those functions fail even though the path is
valid.  Reading and writing bytes through Python's ``pathlib`` avoids the
problem on Windows, Linux and macOS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def imread(path: Path | str) -> Optional[np.ndarray]:
    """Read a BGR image from ``path``, returning ``None`` on failure."""

    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except (OSError, ValueError):
        return None

    if data.size == 0:
        return None

    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return image


def imwrite(path: Path | str, image: np.ndarray) -> bool:
    """Write a BGR image to ``path``.

    Returns ``True`` on success.  The file extension decides the format, just
    like ``cv2.imwrite``.
    """

    target = Path(path)
    extension = target.suffix.lower() or ".jpg"

    try:
        success, buffer = cv2.imencode(extension, image)
    except cv2.error:
        return False

    if not success:
        return False

    try:
        target.write_bytes(buffer.tobytes())
    except OSError:
        return False

    return True
