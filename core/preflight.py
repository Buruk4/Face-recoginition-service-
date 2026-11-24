"""Environment checks with friendly, actionable messages.

This module only uses the standard library at import time.  Heavy packages are
imported inside the individual checks, which means ``python main.py --check``
can report a missing dependency instead of crashing with an ImportError.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from config import (
    BASE_DIR,
    DATABASE_PATH,
    DEEPFACE_HOME,
    DEEPFACE_WEIGHTS_DIR,
    DETECTOR_BACKEND,
    MODEL_NAME,
    ensure_database_path,
)

MIN_PYTHON = (3, 10)
MAX_PYTHON = (3, 12)

ASSET_PATH = BASE_DIR / "assets" / "canonical_face_model.obj"

# import name -> (distribution name, expected version)
REQUIRED_PACKAGES = {
    "numpy": ("numpy", "1.26.4"),
    "pandas": ("pandas", "2.2.3"),
    "cv2": ("opencv-contrib-python", "4.11.0.86"),
    "mediapipe": ("mediapipe", "0.10.21"),
    "onnxruntime": ("onnxruntime", "1.23.2"),
    "deepface": ("deepface", "0.0.103"),
    "werkzeug": ("werkzeug", "3.1.9"),
    "matplotlib": ("matplotlib", "3.10.8"),
}

OPENCV_DISTRIBUTIONS = {
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "opencv-contrib-python-headless",
}


def ensure_utf8() -> None:
    """Force UTF-8 console output.

    DeepFace logs emoji characters.  On a default Windows console (code page
    1252) printing those characters raises ``UnicodeEncodeError``.  Replacing
    the stream encoding here makes the application portable to every console.
    """

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


def python_version_string() -> str:
    return ".".join(str(part) for part in sys.version_info[:3])


def python_version_supported() -> bool:
    return MIN_PYTHON <= sys.version_info[:2] <= MAX_PYTHON


def check_python_version() -> List[str]:
    """Return an error message when the Python version is unsupported."""

    if python_version_supported():
        return []

    if os.environ.get("FACE_SKIP_PYTHON_CHECK", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return []

    minimum = ".".join(str(part) for part in MIN_PYTHON)
    maximum = ".".join(str(part) for part in MAX_PYTHON)
    return [
        f"Unsupported Python version: {python_version_string()}. "
        f"This project supports Python {minimum} to {maximum} (64-bit). "
        "Install a supported version from https://www.python.org/downloads/ "
        "and recreate the virtual environment."
    ]


def _installed_version(distribution: str) -> Optional[str]:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:  # noqa: BLE001 - metadata can be broken by partial installs
        return None


def check_dependencies() -> Tuple[List[str], List[str]]:
    """Return ``(errors, warnings)`` for missing or unexpected packages."""

    errors: List[str] = []
    warnings: List[str] = []

    for import_name, (distribution, expected) in REQUIRED_PACKAGES.items():
        try:
            found = importlib.util.find_spec(import_name) is not None
        except (ImportError, ValueError):
            found = False

        if not found:
            errors.append(
                f"Missing package: {distribution} (import name {import_name!r})."
            )
            continue

        version = _installed_version(distribution)
        if version is None:
            warnings.append(
                f"{distribution} is importable but its version metadata is missing."
            )
        elif version != expected:
            warnings.append(
                f"{distribution} {version} is installed; this project was tested "
                f"with {expected}."
            )

    # Detect the classic OpenCV conflict: several distributions provide cv2 and
    # whichever was installed last wins.
    try:
        installed_distributions = {
            distribution.metadata["Name"].lower()
            for distribution in importlib.metadata.distributions()
            if distribution.metadata["Name"]
        }
    except Exception:  # noqa: BLE001
        installed_distributions = set()

    conflicting = sorted(OPENCV_DISTRIBUTIONS & installed_distributions)
    if len(conflicting) > 1:
        warnings.append(
            "Multiple OpenCV packages are installed: "
            + ", ".join(conflicting)
            + ". Only opencv-contrib-python should be installed. Run "
            "`python scripts/install.py` to repair the environment."
        )

    return errors, warnings


def check_opencv_gui() -> List[str]:
    """Return errors when the installed OpenCV build cannot show windows."""

    try:
        import cv2
    except Exception as error:  # noqa: BLE001
        return [f"OpenCV could not be imported: {error}"]

    errors: List[str] = []

    try:
        build_info = cv2.getBuildInformation()
    except Exception:  # noqa: BLE001
        build_info = ""

    gui_line = next(
        (line.strip() for line in build_info.splitlines() if line.strip().startswith("GUI:")),
        "",
    )

    if not gui_line or gui_line == "GUI:" or gui_line.upper().endswith("NONE"):
        errors.append(
            "This OpenCV build has no GUI support (cv2.imshow will fail). "
            "Install the regular OpenCV build with:\n"
            "  python scripts/install.py"
        )

    if not hasattr(cv2, "imshow"):
        errors.append("cv2.imshow is missing from the installed OpenCV build.")

    return errors


def desktop_session_available() -> bool:
    """Return ``False`` when a Linux machine has no graphical session.

    Windows and macOS always provide a desktop session to normal applications.
    On Linux, ``cv2.imshow`` needs an X11 or Wayland display.
    """

    if not sys.platform.startswith("linux"):
        return True

    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def check_assets() -> List[str]:
    """Verify that the bundled 3D face mesh is present and complete."""

    if not ASSET_PATH.is_file():
        return [
            f"Missing 3D face mesh asset:\n  {ASSET_PATH}\n"
            "Re-download the project or restore assets/canonical_face_model.obj."
        ]

    try:
        vertices = 0
        with ASSET_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("v "):
                    vertices += 1
    except OSError as error:
        return [f"Could not read the 3D face mesh asset:\n  {error}"]

    if vertices < 468:
        return [
            f"The 3D face mesh asset is incomplete "
            f"({vertices} vertices, expected 468):\n  {ASSET_PATH}"
        ]

    return []


def check_database() -> List[str]:
    """Return errors when the face database folder cannot be used."""

    try:
        ensure_database_path()
    except RuntimeError as error:
        return [str(error)]

    test_file = DATABASE_PATH / f".write_test_{os.getpid()}"
    try:
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink()
    except OSError as error:
        return [
            f"The face database folder is not writable:\n  {DATABASE_PATH}\n"
            f"Reason: {error}"
        ]

    return []


def check_model() -> List[str]:
    """Load the ArcFace model (downloading weights on first use)."""

    try:
        from deepface.commons import backend_utils
        from deepface.modules import modeling

        backend = backend_utils.get_backend_engine()
        print(f"  DeepFace backend: {backend}")

        modeling.build_model(task="facial_recognition", model_name=MODEL_NAME)
        return []
    except Exception as error:  # noqa: BLE001
        return [
            f"Could not load the {MODEL_NAME} model: {error}\n"
            "The first run needs internet access to download the model. "
            f"Model files are stored in: {DEEPFACE_WEIGHTS_DIR}"
        ]


def check_mediapipe() -> List[str]:
    """Initialize MediaPipe Face Mesh on a blank frame."""

    try:
        import cv2
        import mediapipe as mp
        import numpy as np

        blank = np.zeros((64, 64, 3), dtype=np.uint8)
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as face_mesh:
            face_mesh.process(cv2.cvtColor(blank, cv2.COLOR_BGR2RGB))
        return []
    except Exception as error:  # noqa: BLE001
        return [f"MediaPipe Face Mesh could not be initialized: {error}"]


def check_camera() -> List[str]:
    """Try to open the configured webcam once."""

    try:
        from core.camera import probe_camera

        ok, message = probe_camera()
        if ok:
            print(f"  {message}")
            return []
        return [message]
    except Exception as error:  # noqa: BLE001
        return [f"Camera check failed: {error}"]


def print_environment() -> None:
    print("Environment")
    print(f"  Python:        {python_version_string()} ({platform.architecture()[0]})")
    print(f"  Platform:      {platform.platform()}")
    print(f"  Executable:    {sys.executable}")
    print(f"  Project folder:{BASE_DIR}")
    print(f"  Face database: {DATABASE_PATH}")
    print(f"  DeepFace home: {DEEPFACE_HOME}")
    print(f"  Model weights: {DEEPFACE_WEIGHTS_DIR}")
    print(f"  Model:         {MODEL_NAME} / {DETECTOR_BACKEND}")


def _print_errors(title: str, errors: Iterable[str]) -> None:
    for error in errors:
        print(f"[FAIL] {title}: {error}")


def _print_warnings(warnings: Iterable[str]) -> None:
    for warning in warnings:
        print(f"[WARN] {warning}")


def run_self_test(check_camera_flag: bool = False, require_gui: bool = True) -> int:
    """Run the installation self-test and return a process exit code."""

    ensure_utf8()

    print("=" * 62)
    print(" FACE RECOGNITION SERVICE - INSTALLATION SELF-TEST")
    print("=" * 62)

    print_environment()
    print()

    errors: List[str] = []
    warnings: List[str] = []

    print("[1/7] Python version")
    python_errors = check_python_version()
    if python_errors:
        errors.extend(python_errors)
        _print_errors("Python", python_errors)
    else:
        print(f"  OK: Python {python_version_string()}")

    print("\n[2/7] Python packages")
    dependency_errors, dependency_warnings = check_dependencies()
    errors.extend(dependency_errors)
    warnings.extend(dependency_warnings)
    if dependency_errors:
        _print_errors("Dependencies", dependency_errors)
    else:
        print("  OK: all required packages are importable")
    _print_warnings(dependency_warnings)

    print("\n[3/7] Face database")
    database_errors = check_database()
    errors.extend(database_errors)
    if database_errors:
        _print_errors("Database", database_errors)
    else:
        print(f"  OK: {DATABASE_PATH} is writable")

    print("\n[4/7] Camera preview (OpenCV GUI)")
    gui_errors = check_opencv_gui()
    if not desktop_session_available():
        gui_errors.append(
            "No graphical desktop session was detected (DISPLAY/WAYLAND_DISPLAY "
            "is not set). The camera preview needs a desktop session."
        )
    if gui_errors:
        if require_gui and desktop_session_available():
            errors.extend(gui_errors)
            _print_errors("GUI", gui_errors)
        else:
            warnings.extend(gui_errors)
            _print_warnings(gui_errors)
    else:
        print("  OK: OpenCV has a GUI backend")

    print("\n[5/7] Bundled 3D face mesh")
    asset_errors = check_assets()
    errors.extend(asset_errors)
    if asset_errors:
        _print_errors("Assets", asset_errors)
    else:
        print(f"  OK: {ASSET_PATH.name} is present")

    print("\n[6/7] ArcFace model")
    model_errors = check_model()
    errors.extend(model_errors)
    if model_errors:
        _print_errors("Model", model_errors)
    else:
        print("  OK: ArcFace model loaded")

    print("\n[7/7] MediaPipe Face Mesh")
    mediapipe_errors = check_mediapipe()
    errors.extend(mediapipe_errors)
    if mediapipe_errors:
        _print_errors("MediaPipe", mediapipe_errors)
    else:
        print("  OK: MediaPipe Face Mesh initialized")

    if check_camera_flag:
        print("\n[extra] Webcam")
        camera_errors = check_camera()
        if camera_errors:
            warnings.extend(camera_errors)
            _print_warnings(camera_errors)
        else:
            print("  OK: webcam opened successfully")

    print("\n" + "-" * 62)
    if errors:
        print(f"RESULT: {len(errors)} check(s) failed.")
        print("Fix the [FAIL] items above, then run this command again.")
        print("If you have not installed the project yet, run: python scripts/install.py")
        return 1

    if warnings:
        print(f"RESULT: all required checks passed ({len(warnings)} warning(s)).")
    else:
        print("RESULT: all checks passed.")
    print("Start the application with: python main.py")
    return 0


def preflight_for_app() -> bool:
    """Validate the environment before starting the interactive application."""

    ensure_utf8()

    errors: List[str] = []
    warnings: List[str] = []

    errors.extend(check_python_version())

    dependency_errors, dependency_warnings = check_dependencies()
    errors.extend(dependency_errors)
    warnings.extend(dependency_warnings)

    errors.extend(check_database())
    errors.extend(check_opencv_gui())
    errors.extend(check_assets())

    if not desktop_session_available():
        errors.append(
            "No graphical desktop session was detected (DISPLAY/WAYLAND_DISPLAY "
            "is not set). The camera preview needs a desktop session."
        )

    for warning in warnings:
        print(f"Warning: {warning}", file=sys.stderr)

    if errors:
        print("\nThe application cannot start because of these problems:\n")
        for error in errors:
            print(f"  - {error}")
        print(
            "\nInstall or repair the environment with:\n"
            "  python scripts/install.py\n"
            "Then verify it with:\n"
            "  python main.py --check"
        )
        return False

    return True
