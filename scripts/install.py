#!/usr/bin/env python3
"""Cross-platform installer for the Face Recognition Service.

Run this script *inside an activated virtual environment*:

    Windows:      py -3.12 -m venv .venv
                  .\\.venv\\Scripts\\Activate.ps1
                  python scripts\\install.py

    Linux/macOS:  python3.12 -m venv .venv
                  source .venv/bin/activate
                  python scripts/install.py

The script installs pinned dependencies, removes conflicting OpenCV builds and
installs the machine-learning packages with --no-deps (see
requirements-ml.txt for the reason).
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence

ROOT = Path(__file__).resolve().parent.parent
MIN_PYTHON = (3, 10)
MAX_PYTHON = (3, 12)

OPENCV_VERSION = "4.11.0.86"
OPENCV_PACKAGE = "opencv-contrib-python"
CONFLICTING_OPENCV = [
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python-headless",
]

# `pip check` reports these because the ML packages are installed with
# --no-deps on purpose.  They do not affect this application.
EXPECTED_PIP_CHECK_WARNINGS = [
    "opencv-contrib-python-headless",
    "retina-face",
    "jax",
    "jaxlib",
    "sounddevice",
    "sentencepiece",
]


def run(command: Sequence[str], *, check: bool = True, quiet: bool = False):
    printable = " ".join(str(part) for part in command)
    print(f"\n$ {printable}")

    kwargs = {"check": check}
    if quiet:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.STDOUT
        kwargs["text"] = True

    result = subprocess.run(command, **kwargs)
    if quiet:
        return result
    return result


def pip(*arguments: str, check: bool = True, quiet: bool = False):
    return run(
        [sys.executable, "-m", "pip", *arguments],
        check=check,
        quiet=quiet,
    )


def check_python() -> bool:
    version = sys.version_info[:2]
    minimum = ".".join(str(part) for part in MIN_PYTHON)
    maximum = ".".join(str(part) for part in MAX_PYTHON)

    print(f"Python:   {sys.version.split()[0]} ({platform.architecture()[0]})")
    print(f"Platform: {platform.platform()}")
    print(f"Project:  {ROOT}")

    if not (MIN_PYTHON <= version <= MAX_PYTHON):
        print(
            f"\nERROR: Python {minimum} to {maximum} is required. "
            f"You are using {sys.version.split()[0]}."
        )
        print("Install a supported 64-bit Python from https://www.python.org/downloads/")
        return False

    return True


def check_virtual_environment(allow_global: bool) -> bool:
    in_venv = sys.prefix != sys.base_prefix

    if in_venv:
        print(f"Virtual environment: {sys.prefix}")
        return True

    if allow_global:
        print("\nWARNING: installing into the global Python (--allow-global).")
        return True

    print(
        "\nERROR: this installer must run inside a virtual environment.\n"
        "Create and activate one first:\n"
        "\n  Windows (PowerShell):\n"
        "    py -3.12 -m venv .venv\n"
        "    .\\.venv\\Scripts\\Activate.ps1\n"
        "\n  Linux / macOS:\n"
        "    python3.12 -m venv .venv\n"
        "    source .venv/bin/activate\n"
        "\nThen run:\n"
        "    python scripts/install.py\n"
        "\nUse --allow-global only if you really know what you are doing."
    )
    return False


def install_dependencies() -> bool:
    if not (ROOT / "requirements.txt").is_file():
        print(f"ERROR: requirements.txt not found in {ROOT}")
        return False
    if not (ROOT / "requirements-ml.txt").is_file():
        print(f"ERROR: requirements-ml.txt not found in {ROOT}")
        return False

    try:
        pip("install", "--upgrade", "pip")
        pip("install", "-r", str(ROOT / "requirements.txt"))
        pip("install", "--no-deps", "-r", str(ROOT / "requirements-ml.txt"))
    except subprocess.CalledProcessError as error:
        print(f"\nERROR: dependency installation failed (exit code {error.returncode}).")
        return False

    return True


def repair_opencv() -> bool:
    """Make opencv-contrib-python the only package that provides cv2."""

    print("\nEnsuring a single, GUI-capable OpenCV installation...")

    for package in CONFLICTING_OPENCV:
        # Uninstalling a package that is not installed is not an error.
        pip("uninstall", "-y", package, check=False, quiet=True)

    try:
        pip(
            "install",
            "--no-deps",
            "--force-reinstall",
            f"{OPENCV_PACKAGE}=={OPENCV_VERSION}",
        )
    except subprocess.CalledProcessError as error:
        print(f"\nERROR: could not install {OPENCV_PACKAGE} (exit code {error.returncode}).")
        return False

    return True


def verify_installation() -> bool:
    code = r"""
from importlib.metadata import version
import cv2
import mediapipe

print("numpy       ", version("numpy"))
print("pandas      ", version("pandas"))
print("opencv      ", cv2.__version__)
print("mediapipe   ", mediapipe.__version__)
print("onnxruntime ", version("onnxruntime"))
print("deepface    ", version("deepface"))
print("werkzeug    ", version("werkzeug"))

gui = [line.strip() for line in cv2.getBuildInformation().splitlines()
       if line.strip().startswith("GUI:")]
print("OpenCV", gui[0] if gui else "GUI: unknown")

if not cv2.__version__.startswith("4.11."):
    raise SystemExit("Unexpected OpenCV version: " + cv2.__version__)

if not gui or gui[0].upper().endswith("NONE"):
    raise SystemExit("OpenCV was installed without GUI support.")
"""

    print("\nVerifying the installed packages...")
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
        )
    except OSError as error:
        print(f"ERROR: could not run the verification: {error}")
        return False

    if result.returncode != 0:
        print(
            "\nERROR: verification failed. Read the message above, then run:\n"
            "  python main.py --check"
        )
        return False

    return True


def report_pip_check() -> None:
    print("\nChecking dependency metadata...")
    result = pip("check", check=False, quiet=True)
    output = (result.stdout or "").strip()

    if not output or "No broken requirements found" in output:
        print("  OK: pip check found no problems.")
        return

    unexpected: List[str] = []
    expected: List[str] = []

    for line in output.splitlines():
        if any(package in line for package in EXPECTED_PIP_CHECK_WARNINGS):
            expected.append(line)
        else:
            unexpected.append(line)

    if expected:
        print(
            "  Known, harmless metadata warnings (packages installed with --no-deps):"
        )
        for line in expected:
            print(f"    - {line}")

    if unexpected:
        print("  WARNING: unexpected dependency metadata problems:")
        for line in unexpected:
            print(f"    - {line}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-global",
        action="store_true",
        help="allow installation outside a virtual environment (not recommended)",
    )
    args = parser.parse_args(argv)

    print("=" * 62)
    print(" FACE RECOGNITION SERVICE - INSTALLER")
    print("=" * 62)

    if not check_python():
        return 1

    if not check_virtual_environment(args.allow_global):
        return 1

    if not install_dependencies():
        return 1

    if not repair_opencv():
        return 1

    if not verify_installation():
        return 1

    report_pip_check()

    print("\n" + "=" * 62)
    print(" INSTALLATION COMPLETE")
    print("=" * 62)
    print("Next steps:")
    print("  1. Verify the installation (downloads the ArcFace model on first run):")
    print("       python main.py --check")
    print("  2. Test the webcam as well:")
    print("       python main.py --check --camera")
    print("  3. Start the application:")
    print("       python main.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
