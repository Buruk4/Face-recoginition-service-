"""Head-pose estimation from MediaPipe Face Mesh landmarks.

The pose is estimated with OpenCV's ``solvePnP``: the 2D face-mesh landmarks are
matched against MediaPipe's neutral 3D face mesh (``assets/canonical_face_model.obj``)
and the resulting rotation is decomposed into yaw and pitch angles.

Sign convention (matching the registration instructions):

* **yaw > 0**  the person turned their head to *their own left*
  (the face points towards the right side of the unmirrored camera image)
* **pitch > 0** the person is looking *up*
* **pitch < 0** the person is looking *down*

The camera preview is mirrored for the user, but all pose calculations use the
unmirrored frame, so the instructions match what the user actually does.

If the 3D mesh asset is missing, a small generic six-point model is used
instead.  It is less accurate but keeps registration usable; the installation
self-test reports the missing asset.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from config import BASE_DIR

MODEL_PATH = BASE_DIR / "assets" / "canonical_face_model.obj"

# Landmarks used by the generic fallback model.
FALLBACK_INDEX = [1, 152, 33, 263, 61, 291]

# Approximate 3D face model (millimetres) for the fallback.  Coordinates are in
# the MediaPipe frame (x right, y up, z towards the camera) and are converted to
# OpenCV's frame (x right, y down, z away from the camera) below.
_FALLBACK_MODEL = np.array(
    [
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -63.6, -12.5),  # Chin
        (-43.3, 32.7, -26.0),  # Right eye outer corner
        (43.3, 32.7, -26.0),  # Left eye outer corner
        (-28.9, -28.9, -24.1),  # Right mouth corner
        (28.9, -28.9, -24.1),  # Left mouth corner
    ],
    dtype=np.float64,
) * np.array([1.0, -1.0, -1.0], dtype=np.float64)

_FALLBACK_WARNED = False


@lru_cache(maxsize=1)
def _canonical_model() -> Optional[np.ndarray]:
    """Load and cache the 468-point canonical face mesh.

    Returns ``None`` when the asset is missing or malformed.
    """

    try:
        vertices = []
        with MODEL_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.startswith("v "):
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                vertices.append(tuple(float(value) for value in parts[1:4]))
    except OSError:
        return None

    if len(vertices) < 468:
        return None

    model = np.asarray(vertices[:468], dtype=np.float64)

    # The OBJ file uses x right, y down, z towards the camera (a left-handed
    # frame).  OpenCV's camera frame is x right, y down, z away from the camera,
    # so flipping y and z produces a proper right-handed object frame.
    model = model * np.array([1.0, -1.0, -1.0], dtype=np.float64)

    if not np.all(np.isfinite(model)):
        return None

    return model


def _camera_matrix(width: int, height: int) -> np.ndarray:
    # A rough focal length is sufficient for pose classification.
    focal = float(max(width, height))
    return np.array(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _decompose(rotation_vector: np.ndarray) -> Tuple[float, float]:
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles = cv2.RQDecomp3x3(rotation_matrix)[0]
    return float(angles[1]), float(angles[0])  # yaw, pitch


def get_head_pose(
    face_landmarks,
    width: int,
    height: int,
) -> Tuple[Optional[float], Optional[float]]:
    """Return ``(yaw, pitch)`` in degrees, or ``(None, None)`` on failure.

    ``face_landmarks`` is a MediaPipe ``NormalizedLandmarkList``.  A single
    malformed frame must never abort a registration session, so every failure
    is reported as ``(None, None)``.
    """

    global _FALLBACK_WARNED

    if face_landmarks is None or width <= 0 or height <= 0:
        return None, None

    try:
        landmarks = face_landmarks.landmark
        if len(landmarks) < 468:
            return None, None

        camera = _camera_matrix(width, height)
        distortion = np.zeros((5, 1), dtype=np.float64)
        model = _canonical_model()

        if model is not None:
            points = np.array(
                [
                    (landmark.x * width, landmark.y * height)
                    for landmark in landmarks[:468]
                ],
                dtype=np.float64,
            )

            if not np.all(np.isfinite(points)):
                return None, None

            try:
                success, rotation_vector, _ = cv2.solvePnP(
                    model,
                    points,
                    camera,
                    distortion,
                    flags=cv2.SOLVEPNP_SQPNP,
                )
            except cv2.error:
                success = False

            if success:
                yaw, pitch = _decompose(rotation_vector)
                if np.isfinite(yaw) and np.isfinite(pitch):
                    return yaw, pitch

        # Fallback: six well-known landmarks and a generic 3D model.
        if model is None and not _FALLBACK_WARNED:
            _FALLBACK_WARNED = True
            print(
                "Warning: the 3D face mesh asset is missing "
                f"({MODEL_PATH}). Head-pose accuracy will be reduced. "
                "Re-download the project or restore assets/canonical_face_model.obj."
            )

        if len(landmarks) <= max(FALLBACK_INDEX):
            return None, None

        points = np.array(
            [
                (landmarks[index].x * width, landmarks[index].y * height)
                for index in FALLBACK_INDEX
            ],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(points)):
            return None, None

        success, rotation_vector, _ = cv2.solvePnP(
            _FALLBACK_MODEL,
            points,
            camera,
            np.zeros((4, 1), dtype=np.float64),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return None, None

        yaw, pitch = _decompose(rotation_vector)
        if not (np.isfinite(yaw) and np.isfinite(pitch)):
            return None, None

        return yaw, pitch

    except (cv2.error, ValueError, IndexError, TypeError):
        return None, None
