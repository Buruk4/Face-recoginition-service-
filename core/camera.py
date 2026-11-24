"""Webcam helpers.

The correct OpenCV capture backend differs between operating systems.  Using
``cv2.VideoCapture(index)`` with the default backend is unreliable on Windows
and can be very slow on some machines, so this module tries the native backend
for the current OS and falls back to OpenCV's automatic choice.
"""

from __future__ import annotations

import platform
import time
from typing import Optional

import cv2

from config import CAMERA_BACKEND, CAMERA_INDEX


def _backend_constant(name: str) -> Optional[int]:
    """Return an OpenCV backend constant if this build defines it."""

    return getattr(cv2, f"CAP_{name.upper()}", None)


def _candidate_backends() -> list[Optional[int]]:
    """Return capture backends to try, in order."""

    requested = (CAMERA_BACKEND or "auto").strip().lower()

    named = {
        "any": _backend_constant("any"),
        "dshow": _backend_constant("dshow"),
        "msmf": _backend_constant("msmf"),
        "v4l2": _backend_constant("v4l2"),
        "avfoundation": _backend_constant("avfoundation"),
        "gstreamer": _backend_constant("gstreamer"),
    }

    if requested not in {"auto", *named}:
        raise ValueError(
            f"Unknown FACE_CAMERA_BACKEND={CAMERA_BACKEND!r}. "
            f"Choose one of: auto, {', '.join(sorted(named))}."
        )

    if requested != "auto":
        backend = named[requested]
        return [backend] if backend is not None else [None]

    system = platform.system()
    if system == "Windows":
        order = ["dshow", "msmf", "any"]
    elif system == "Darwin":
        order = ["avfoundation", "any"]
    elif system == "Linux":
        order = ["v4l2", "any"]
    else:
        order = ["any"]

    backends: list[Optional[int]] = []
    for name in order:
        backend = named[name]
        if backend is not None and backend not in backends:
            backends.append(backend)

    # OpenCV's automatic choice (``None``) is only needed when CAP_ANY is not
    # available; otherwise it would try the same backend twice.
    if None not in backends and named["any"] not in backends:
        backends.append(None)

    return backends


def _warm_up(camera: cv2.VideoCapture, attempts: int = 15) -> bool:
    """Read a few frames so slow cameras have time to start."""

    for _ in range(attempts):
        try:
            success, frame = camera.read()
        except cv2.error:
            return False

        if success and frame is not None and frame.size > 0:
            return True

        time.sleep(0.05)

    return False


def open_camera(index: Optional[int] = None) -> cv2.VideoCapture:
    """Open the webcam and return a ready-to-read ``VideoCapture``.

    Raises ``RuntimeError`` with an actionable message when no camera can be
    opened.
    """

    camera_index = CAMERA_INDEX if index is None else int(index)
    attempts: list[str] = []

    for backend in _candidate_backends():
        backend_name = "default" if backend is None else str(backend)

        try:
            if backend is None:
                camera = cv2.VideoCapture(camera_index)
            else:
                camera = cv2.VideoCapture(camera_index, backend)
        except cv2.error as error:
            attempts.append(f"backend {backend_name}: {error}")
            continue

        if not camera.isOpened():
            camera.release()
            attempts.append(f"backend {backend_name}: could not be opened")
            time.sleep(0.15)
            continue

        # A small buffer keeps the preview close to real time.  Some backends
        # ignore this property, so failures are not fatal.
        try:
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except cv2.error:
            pass

        if _warm_up(camera):
            return camera

        camera.release()
        attempts.append(f"backend {backend_name}: opened but returned no frames")
        time.sleep(0.15)

    details = "\n  ".join(attempts) if attempts else "no backends were tried"
    raise RuntimeError(
        f"Could not open webcam index {camera_index}.\n"
        f"Tried:\n  {details}\n"
        "Close other programs that use the camera, check the camera privacy "
        "settings, and try FACE_CAMERA_INDEX=1 (or another index)."
    )


def release_camera(camera: Optional[cv2.VideoCapture]) -> None:
    """Release a camera and close any OpenCV windows, ignoring cleanup errors."""

    if camera is not None:
        try:
            camera.release()
        except cv2.error:
            pass

    try:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    except cv2.error:
        pass


def probe_camera(index: Optional[int] = None) -> tuple[bool, str]:
    """Check whether a camera can be opened, without keeping it open."""

    camera = None
    try:
        camera = open_camera(index)
        return True, f"Camera index {CAMERA_INDEX if index is None else index} works."
    except Exception as error:  # noqa: BLE001 - reported to the user
        return False, str(error)
    finally:
        release_camera(camera)
