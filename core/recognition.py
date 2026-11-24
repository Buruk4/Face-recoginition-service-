"""Interactive face recognition."""

from __future__ import annotations

import time
import traceback
from typing import Dict, List, Optional, Tuple

import cv2

from config import DEBUG, MATCH_MARGIN, MATCH_THRESHOLD
from core.camera import open_camera, release_camera
from core.database import (
    Database,
    cosine_distance,
    extract_embedding,
    load_database,
)

WINDOW_NAME = "Face Recognition"


def find_match(
    query_embedding,
    database: Database,
) -> Tuple[str, Optional[float]]:
    """Find the best matching identity.

    Returns ``(name, distance)``.  ``name`` is ``"Unknown"`` when no identity
    is close enough (or when the best two identities are too similar), and
    ``"No registered faces"`` when the database is empty.
    """

    if not database:
        return "No registered faces", None

    if query_embedding is None:
        return "Unknown", None

    person_scores: Dict[str, float] = {}

    for person_name, embeddings in database.items():
        distances = [
            cosine_distance(query_embedding, stored_embedding)
            for stored_embedding in embeddings
        ]

        if distances:
            # Each identity's score is its best matching photo.
            person_scores[person_name] = min(distances)

    if not person_scores:
        return "Unknown", None

    ranked: List[Tuple[str, float]] = sorted(
        person_scores.items(), key=lambda item: item[1]
    )

    best_name, best_distance = ranked[0]

    print("\nRecognition scores (lower is better):")
    for name, distance in ranked:
        marker = "  <-- best" if name == best_name else ""
        print(f"  {name}: {distance:.4f}{marker}")

    if best_distance > MATCH_THRESHOLD:
        print(
            f"  Best distance {best_distance:.4f} is above the "
            f"threshold {MATCH_THRESHOLD:.2f}."
        )
        return "Unknown", best_distance

    if len(ranked) > 1:
        second_distance = ranked[1][1]
        if second_distance - best_distance < MATCH_MARGIN:
            print(
                f"  The best two identities are too close "
                f"(margin {second_distance - best_distance:.4f} < "
                f"{MATCH_MARGIN:.2f})."
            )
            return "Unknown", best_distance

    return best_name, best_distance


def recognize_frame(frame, database: Database) -> Tuple[str, Optional[float]]:
    """Recognize the face in one BGR frame (used by the UI and by tests)."""

    embedding = extract_embedding(frame)
    return find_match(embedding, database)


def _window_closed(window_name: str) -> bool:
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return False


def check_face() -> None:
    """Run the interactive recognition workflow."""

    print("\nLoading registered faces...")

    try:
        database = load_database()
    except Exception as error:  # noqa: BLE001 - reported to the user
        print("Could not load the face database:", error)
        if DEBUG:
            traceback.print_exc()
        return

    if not database:
        print(
            "No registered faces found. Register someone first "
            "(option 1 in the main menu)."
        )
        return

    camera = None

    try:
        camera = open_camera()

        print("\nCamera opened.")
        print("Look at the camera and press SPACE to recognize a face.")
        print("Press Q or Esc to close the camera.")

        result_text: Optional[str] = None
        result_color = (0, 255, 0)
        result_until = 0.0

        while True:
            success, frame = camera.read()

            if not success or frame is None:
                print("Could not read from the camera.")
                break

            display = cv2.flip(frame, 1)

            cv2.putText(
                display,
                "Press SPACE to recognize, Q to quit",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if result_text and time.monotonic() < result_until:
                cv2.putText(
                    display,
                    result_text,
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    result_color,
                    2,
                    cv2.LINE_AA,
                )

            try:
                cv2.imshow(WINDOW_NAME, display)
            except cv2.error as error:
                raise RuntimeError(
                    "OpenCV cannot open a camera preview window. "
                    "Install the regular (non-headless) OpenCV build:\n"
                    "  python -m pip install --force-reinstall --no-deps "
                    "opencv-contrib-python==4.11.0.86"
                ) from error

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if key == ord(" "):
                try:
                    # The preview is mirrored for a natural selfie view; the
                    # unmirrored frame is used for recognition.
                    name, distance = recognize_frame(frame, database)

                    if name == "Unknown":
                        print("\nUnknown person.")
                        result_text = "Unknown person"
                        result_color = (0, 0, 255)
                    elif name == "No registered faces":
                        print("\nNo registered faces.")
                        result_text = "No registered faces"
                        result_color = (0, 165, 255)
                    else:
                        print(f"\nRecognized: {name}")
                        result_text = f"Recognized: {name}"
                        result_color = (0, 255, 0)

                    if distance is not None:
                        print(f"Best distance: {distance:.4f}")

                    result_until = time.monotonic() + 3.0

                except Exception as error:  # noqa: BLE001 - keep the camera open
                    print("Recognition error:", error)
                    if DEBUG:
                        traceback.print_exc()
                    message = str(error).lower()
                    if "detect" in message or "visible" in message:
                        result_text = "No face detected"
                    else:
                        result_text = "Recognition failed"
                    result_color = (0, 0, 255)
                    result_until = time.monotonic() + 2.0

            if _window_closed(WINDOW_NAME):
                break

    except KeyboardInterrupt:
        print("\nRecognition cancelled.")

    except Exception as error:  # noqa: BLE001 - reported to the user
        print("Camera error:", error)
        if DEBUG:
            traceback.print_exc()

    finally:
        release_camera(camera)
