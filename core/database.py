"""Face database: embedding extraction, loading, caching and comparison."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import (
    CACHE_PATH,
    DATABASE_PATH,
    DETECTOR_BACKEND,
    EMBEDDING_CACHE_ENABLED,
    IMAGE_EXTENSIONS,
    MEDIAPIPE_FALLBACK_ENABLED,
    MODEL_NAME,
)
from core.image_io import imread

# Increment this when the cache layout or embedding pipeline changes.
CACHE_VERSION = 1

Embedding = np.ndarray
Database = Dict[str, List[Embedding]]
CacheEntry = Tuple[int, int, Embedding]

# Margin (as a fraction of the face size) added around a MediaPipe crop.
MEDIAPIPE_CROP_MARGIN = 0.25


def _is_face_not_detected(error: Exception) -> bool:
    """Return ``True`` for DeepFace's "no face" error, across versions."""

    try:
        from deepface.modules.exceptions import FaceNotDetected
    except ImportError:
        FaceNotDetected = ()  # type: ignore[assignment]

    if FaceNotDetected and isinstance(error, FaceNotDetected):
        return True

    message = str(error).lower()
    return "could not be detected" in message or "no face" in message


def _mediapipe_crop(image: np.ndarray) -> Optional[np.ndarray]:
    """Return a face crop found by MediaPipe, or ``None``.

    DeepFace's OpenCV detector is fast but needs reasonably bright, frontal
    images.  When it fails, MediaPipe's Face Mesh usually still finds the face.
    The crop is passed to ArcFace with DeepFace's ``skip`` detector.
    """

    try:
        import mediapipe as mp
    except ImportError:
        return None

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return None

    try:
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
        ) as face_mesh:
            result = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    except (cv2.error, ValueError):
        return None

    if not result.multi_face_landmarks:
        return None

    landmarks = result.multi_face_landmarks[0].landmark
    xs = [point.x * width for point in landmarks]
    ys = [point.y * height for point in landmarks]

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    face_width = x2 - x1
    face_height = y2 - y1

    if face_width <= 1 or face_height <= 1:
        return None

    x1 -= face_width * MEDIAPIPE_CROP_MARGIN
    x2 += face_width * MEDIAPIPE_CROP_MARGIN
    y1 -= face_height * MEDIAPIPE_CROP_MARGIN
    y2 += face_height * MEDIAPIPE_CROP_MARGIN

    left = max(0, int(x1))
    top = max(0, int(y1))
    right = min(width, int(x2))
    bottom = min(height, int(y2))

    if right - left < 8 or bottom - top < 8:
        return None

    return image[top:bottom, left:right]


def _represent(image: np.ndarray, detector_backend: str, enforce_detection: bool):
    from deepface import DeepFace

    return DeepFace.represent(
        img_path=image,
        model_name=MODEL_NAME,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
    )


def extract_embedding(image: np.ndarray) -> Embedding:
    """Return the 512-dimensional ArcFace embedding for one face.

    ``image`` must be a BGR image (the format produced by OpenCV and by the
    camera module).  Exactly one face must be visible.

    If the configured detector cannot find a face (common in dim webcam
    images), MediaPipe Face Mesh is used to locate and crop the face, and the
    crop is passed to ArcFace directly.  This fallback can be disabled with
    ``FACE_MEDIAPIPE_FALLBACK=0``.
    """

    if image is None:
        raise ValueError("No image was provided.")
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("The image is empty or not a NumPy array.")

    try:
        result = _represent(image, DETECTOR_BACKEND, True)
    except Exception as error:  # noqa: BLE001 - only detection failures fall back
        if not MEDIAPIPE_FALLBACK_ENABLED or not _is_face_not_detected(error):
            raise

        crop = _mediapipe_crop(image)
        if crop is None:
            raise ValueError(
                "No face was detected. Move closer, improve the lighting, "
                "and keep only one face in view."
            ) from error

        print("Face detector missed the face; retrying with a MediaPipe crop.")
        result = _represent(crop, "skip", False)

    if isinstance(result, dict):
        result = [result]

    if not result:
        raise ValueError("No face was detected.")

    if len(result) != 1:
        raise ValueError(
            f"Exactly one face must be visible, but {len(result)} were detected."
        )

    raw_embedding = result[0].get("embedding")
    if raw_embedding is None:
        raise ValueError("DeepFace did not return an embedding.")

    embedding = np.asarray(raw_embedding, dtype=np.float32).reshape(-1)
    if embedding.size == 0 or not np.all(np.isfinite(embedding)):
        raise ValueError("DeepFace returned an invalid embedding.")

    return embedding


def cosine_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Return the cosine distance between two embeddings (0 = identical)."""

    first = np.asarray(embedding1, dtype=np.float64).reshape(-1)
    second = np.asarray(embedding2, dtype=np.float64).reshape(-1)

    if first.shape != second.shape:
        return 1.0

    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0 or not np.isfinite(denominator):
        return 1.0

    similarity = float(np.dot(first, second) / denominator)
    return float(np.clip(1.0 - similarity, 0.0, 2.0))


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------


def _cache_key(path: Path) -> str:
    """Return a stable, OS-independent cache key for an image path."""

    return path.relative_to(DATABASE_PATH).as_posix()


def _iter_image_entries() -> List[Tuple[str, Path, int, int]]:
    """Return ``(person, path, mtime_ns, size)`` for every registered image."""

    if not DATABASE_PATH.is_dir():
        return []

    entries: List[Tuple[str, Path, int, int]] = []

    try:
        person_folders = sorted(
            (item for item in DATABASE_PATH.iterdir() if item.is_dir()),
            key=lambda item: item.name.casefold(),
        )
    except OSError as error:
        print(f"Could not read the face database: {error}")
        return []

    for person_folder in person_folders:
        if person_folder.name.startswith("."):
            continue

        try:
            files = sorted(
                (item for item in person_folder.iterdir() if item.is_file()),
                key=lambda item: item.name.casefold(),
            )
        except OSError as error:
            print(f"Skipped folder {person_folder.name}: {error}")
            continue

        for path in files:
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            try:
                stat = path.stat()
            except OSError:
                continue

            entries.append((person_folder.name, path, stat.st_mtime_ns, stat.st_size))

    return entries


def _load_cache() -> Dict[str, CacheEntry]:
    """Load the embedding cache, returning an empty dict if it is unusable."""

    if not CACHE_PATH.is_file():
        return {}

    try:
        with np.load(CACHE_PATH, allow_pickle=False) as data:
            version = int(data["version"][0])
            model = str(data["model"][0])
            detector = str(data["detector"][0])
            fallback = bool(int(data["fallback"][0]))

            if (
                version != CACHE_VERSION
                or model != MODEL_NAME
                or detector != DETECTOR_BACKEND
                or fallback != MEDIAPIPE_FALLBACK_ENABLED
            ):
                return {}

            keys = [str(key) for key in data["keys"].tolist()]
            embeddings = np.asarray(data["embeddings"], dtype=np.float32)
            mtimes = np.asarray(data["mtimes"], dtype=np.int64).tolist()
            sizes = np.asarray(data["sizes"], dtype=np.int64).tolist()

    except (OSError, ValueError, KeyError, EOFError, TypeError):
        return {}
    except Exception as error:  # noqa: BLE001 - a corrupt cache must not stop the app
        print(f"Ignoring unreadable embedding cache: {error}")
        return {}

    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    if (
        len(keys) != len(mtimes)
        or len(keys) != len(sizes)
        or embeddings.shape[0] != len(keys)
    ):
        return {}

    cache: Dict[str, CacheEntry] = {}
    for index, key in enumerate(keys):
        cache[key] = (
            int(mtimes[index]),
            int(sizes[index]),
            embeddings[index].astype(np.float32),
        )

    return cache


def _save_cache(cache: Dict[str, CacheEntry]) -> None:
    """Atomically save the embedding cache, ignoring write failures."""

    try:
        if not cache:
            CACHE_PATH.unlink(missing_ok=True)
            return

        keys = sorted(cache)
        embeddings = np.vstack(
            [cache[key][2].reshape(1, -1) for key in keys]
        ).astype(np.float32)
        mtimes = np.asarray([cache[key][0] for key in keys], dtype=np.int64)
        sizes = np.asarray([cache[key][1] for key in keys], dtype=np.int64)

        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = CACHE_PATH.with_suffix(".tmp.npz")

        np.savez(
            temporary,
            version=np.asarray([CACHE_VERSION], dtype=np.int64),
            model=np.asarray([MODEL_NAME]),
            detector=np.asarray([DETECTOR_BACKEND]),
            fallback=np.asarray(
                [1 if MEDIAPIPE_FALLBACK_ENABLED else 0], dtype=np.int64
            ),
            keys=np.asarray(keys),
            embeddings=embeddings,
            mtimes=mtimes,
            sizes=sizes,
        )
        os.replace(temporary, CACHE_PATH)

    except (OSError, ValueError, TypeError) as error:
        print(f"Warning: could not save the embedding cache: {error}")


# ---------------------------------------------------------------------------
# Database loading
# ---------------------------------------------------------------------------


def load_database(use_cache: Optional[bool] = None) -> Database:
    """Load every registered person and their embeddings.

    Embeddings are cached in ``faces/.embeddings_cache.npz``.  The cache is
    validated against each image's modification time and size, so replacing or
    editing a photo automatically recomputes only that photo.
    """

    if use_cache is None:
        use_cache = EMBEDDING_CACHE_ENABLED

    entries = _iter_image_entries()
    if not entries:
        # Remove stale embeddings if every photo was deleted.  This also keeps
        # biometric data from lingering after the database is emptied.
        if use_cache and CACHE_PATH.is_file():
            _save_cache({})
        return {}

    cache = _load_cache() if use_cache else {}
    new_cache: Dict[str, CacheEntry] = {}
    database: Database = {}

    computed = 0
    reused = 0
    skipped = 0

    for person_name, image_path, mtime_ns, size in entries:
        key = _cache_key(image_path)
        cached = cache.get(key)

        if cached is not None and cached[0] == mtime_ns and cached[1] == size:
            embedding = cached[2]
            reused += 1
        else:
            image = imread(image_path)

            if image is None:
                print(f"Skipped unreadable image: {key}")
                skipped += 1
                continue

            try:
                embedding = extract_embedding(image)
            except Exception as error:  # noqa: BLE001 - skip a single bad photo
                print(f"Skipped {key}: {error}")
                skipped += 1
                continue

            computed += 1
            print(f"Embedded {key}")

        database.setdefault(person_name, []).append(embedding)
        new_cache[key] = (mtime_ns, size, embedding)

    if use_cache and (computed > 0 or set(new_cache) != set(cache)):
        _save_cache(new_cache)

    total = sum(len(embeddings) for embeddings in database.values())
    summary = (
        f"Loaded {total} embedding(s) for {len(database)} person(s) "
        f"({reused} from cache, {computed} computed"
    )
    if skipped:
        summary += f", {skipped} skipped"
    summary += ")."
    print(summary)

    return database
