"""Central configuration for the Face Recognition Service.

Every value can be overridden with an environment variable, so the application
can be moved to another computer without editing this file.  Paths are resolved
relative to the project folder by default, which means the project works no
matter where it is cloned or extracted.

See ``.env.example`` and the README for the full list of variables.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Absolute path of the folder that contains this file (the project root).
BASE_DIR = Path(__file__).resolve().parent


def _load_dotenv(path: Path) -> None:
    """Load a simple KEY=VALUE .env file without adding a dependency.

    Real environment variables always win.  Lines starting with ``#`` are
    ignored, and surrounding single or double quotes are removed.
    """

    if not path.is_file():
        return

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        if line.startswith("export "):
            line = line[len("export ") :].lstrip()

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        os.environ.setdefault(key, value)


_load_dotenv(BASE_DIR / ".env")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _resolve_path(value: str, default: Path) -> Path:
    """Turn a user supplied path into an absolute path.

    Relative paths are interpreted relative to the project folder rather than
    the current working directory.  This keeps ``python main.py`` and
    ``python -m main`` behaviour identical from any directory.
    """

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return _resolve_path(value, default) if value else default.resolve()


DATABASE_PATH = _env_path("FACE_DATABASE_PATH", BASE_DIR / "faces")

# DeepFace stores model weights in ``<DEEPFACE_HOME>/.deepface/weights``.  The
# variable therefore points at the user's home folder, not at the ``.deepface``
# folder itself.  Defaulting to the real home keeps the location stable across
# operating systems.
DEEPFACE_HOME = _env_path("DEEPFACE_HOME", Path.home())
DEEPFACE_WEIGHTS_DIR = DEEPFACE_HOME / ".deepface" / "weights"
os.environ["DEEPFACE_HOME"] = str(DEEPFACE_HOME)

# Optional embedding cache.  Cached embeddings make recognition start almost
# instantly after the first run.  It can be disabled with
# FACE_EMBEDDING_CACHE=0.
CACHE_PATH = _env_path(
    "FACE_EMBEDDING_CACHE_PATH", DATABASE_PATH / ".embeddings_cache.npz"
)

# ---------------------------------------------------------------------------
# Model / recognition settings
# ---------------------------------------------------------------------------

# DeepFace backend.  ``onnx`` is used because it is lightweight and produces
# the same ArcFace embeddings as the TensorFlow backend (verified on the same
# images), without the ~2 GB TensorFlow installation.
#
# Advanced users can set DEEPFACE_BACKEND_ENGINE=tensorflow after installing
# tensorflow and tf-keras themselves.
os.environ.setdefault("DEEPFACE_BACKEND_ENGINE", "onnx")

# Keep MediaPipe's native logs quiet; real errors are still printed.
os.environ.setdefault("GLOG_minloglevel", "2")

MODEL_NAME = os.environ.get("FACE_MODEL_NAME", "ArcFace").strip() or "ArcFace"
DETECTOR_BACKEND = (
    os.environ.get("FACE_DETECTOR_BACKEND", "opencv").strip() or "opencv"
)

# Cosine-distance threshold for ArcFace.  DeepFace's published ArcFace
# threshold is 0.68.  Distances <= threshold are considered a match.
MATCH_THRESHOLD = 0.68

# Difference required between the best and second-best identity.  This stops
# the system from accepting a person when two registered people look alike.
MATCH_MARGIN = 0.05

# ---------------------------------------------------------------------------
# Camera settings
# ---------------------------------------------------------------------------

CAMERA_INDEX = 0

# ``auto`` tries the native backend for the current OS first.  You can force a
# specific OpenCV backend with FACE_CAMERA_BACKEND (dshow, msmf, v4l2,
# avfoundation, any).
CAMERA_BACKEND = "auto"

# ---------------------------------------------------------------------------
# Registration settings
# ---------------------------------------------------------------------------

# Number of consecutive frames required before saving a pose.
STABLE_FRAMES = 8

# Head pose thresholds in degrees.
YAW_THRESHOLD = 15.0
PITCH_THRESHOLD = 12.0

# Safety limit so a registration cannot run forever.  Set to 0 to disable.
REGISTRATION_TIMEOUT = 300.0

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Print full tracebacks for unexpected errors.  Useful while debugging.
DEBUG = os.environ.get("FACE_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}

EMBEDDING_CACHE_ENABLED = os.environ.get(
    "FACE_EMBEDDING_CACHE", "1"
).strip().lower() not in {"0", "false", "no", "off"}

# When the configured detector cannot find a face, try a MediaPipe face crop
# before giving up.  This makes recognition work in dim webcam images.
MEDIAPIPE_FALLBACK_ENABLED = os.environ.get(
    "FACE_MEDIAPIPE_FALLBACK", "1"
).strip().lower() not in {"0", "false", "no", "off"}


def _env_number(name: str, default, cast, minimum=None, maximum=None):
    """Read a numeric environment variable, falling back safely.

    Invalid values produce a warning instead of crashing the application.
    """

    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default

    try:
        value = cast(str(raw).strip())
    except (TypeError, ValueError):
        print(
            f"Warning: {name}={raw!r} is not a valid {cast.__name__}; "
            f"using {default!r}.",
            file=sys.stderr,
        )
        return default

    if minimum is not None and value < minimum:
        print(
            f"Warning: {name} must be >= {minimum}; using {default!r}.",
            file=sys.stderr,
        )
        return default
    if maximum is not None and value > maximum:
        print(
            f"Warning: {name} must be <= {maximum}; using {default!r}.",
            file=sys.stderr,
        )
        return default

    return value


# Environment overrides for numeric settings.  They are applied here so the
# rest of the code can keep using the simple names above.
MATCH_THRESHOLD = _env_number(
    "FACE_MATCH_THRESHOLD", MATCH_THRESHOLD, float, 0.0, 2.0
)
MATCH_MARGIN = _env_number("FACE_MATCH_MARGIN", MATCH_MARGIN, float, 0.0, 2.0)
CAMERA_INDEX = _env_number("FACE_CAMERA_INDEX", CAMERA_INDEX, int, 0, 100)
STABLE_FRAMES = _env_number("FACE_STABLE_FRAMES", STABLE_FRAMES, int, 1, 10000)
YAW_THRESHOLD = _env_number(
    "FACE_YAW_THRESHOLD", YAW_THRESHOLD, float, 1.0, 90.0
)
PITCH_THRESHOLD = _env_number(
    "FACE_PITCH_THRESHOLD", PITCH_THRESHOLD, float, 1.0, 90.0
)
REGISTRATION_TIMEOUT = _env_number(
    "FACE_REGISTRATION_TIMEOUT", REGISTRATION_TIMEOUT, float, 0.0, 86400.0
)
CAMERA_BACKEND = (
    os.environ.get("FACE_CAMERA_BACKEND", CAMERA_BACKEND).strip() or "auto"
)


def ensure_database_path() -> Path:
    """Create the face database folder if needed and return it.

    Raises a ``RuntimeError`` with an actionable message when the folder cannot
    be created (for example because of permissions or a bad path).
    """

    try:
        DATABASE_PATH.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise RuntimeError(
            f"Cannot create the face database folder:\n  {DATABASE_PATH}\n"
            f"Reason: {error}\n"
            "Set FACE_DATABASE_PATH to a writable folder and try again."
        ) from error

    if not os.access(DATABASE_PATH, os.W_OK):
        raise RuntimeError(
            f"The face database folder is not writable:\n  {DATABASE_PATH}\n"
            "Set FACE_DATABASE_PATH to a writable folder and try again."
        )

    return DATABASE_PATH
