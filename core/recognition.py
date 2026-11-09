import cv2
import numpy as np

from config import (
    MATCH_THRESHOLD,
    MATCH_MARGIN
)

from core.camera import open_camera, release_camera
from core.database import (
    extract_embedding,
    cosine_distance,
    load_database
)


def find_match(query_embedding, database):

    if not database:
        return "No registered faces", None

    person_scores = {}

    for person_name, embeddings in database.items():

        distances = []

        for stored_embedding in embeddings:

            distance = cosine_distance(
                query_embedding,
                stored_embedding
            )

            distances.append(distance)

        if distances:
            # Each identity's score is its best matching photo.
            person_scores[person_name] = min(distances)

    if not person_scores:
        return "Unknown", None

    ranked = sorted(
        person_scores.items(),
        key=lambda item: item[1]
    )

    best_name, best_distance = ranked[0]

    print("\nRecognition scores:")

    for name, distance in ranked:
        print(f"{name}: {distance:.4f}")

    if best_distance >= MATCH_THRESHOLD:
        return "Unknown", best_distance

    if len(ranked) > 1:

        second_distance = ranked[1][1]

        if second_distance - best_distance < MATCH_MARGIN:
            return "Unknown", best_distance

    return best_name, best_distance


def check_face():

    print("\nLoading registered faces...")

    database = load_database()

    if not database:
        print("No registered faces found.")
        return

    camera = None

    try:

        camera = open_camera()

        print("\nCamera opened.")
        print("Look at the camera. Press Q to cancel.")

        while True:

            success, frame = camera.read()

            if not success:
                break

            frame = cv2.flip(frame, 1)

            display = frame.copy()

            cv2.putText(
                display,
                "Press SPACE to recognize",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.imshow("Face Recognition", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord(" "):

                try:

                    # Use the unmirrored frame for processing.
                    original_frame = cv2.flip(frame, 1)

                    query_embedding = extract_embedding(
                        original_frame
                    )

                    name, distance = find_match(
                        query_embedding,
                        database
                    )

                    if name == "Unknown":
                        print("\nUnknown person.")

                    elif name == "No registered faces":
                        print("\nNo registered faces.")

                    else:
                        print(
                            f"\nRecognized: {name}"
                        )

                    if distance is not None:
                        print(
                            f"Best distance: {distance:.4f}"
                        )

                except Exception as error:

                    print("Recognition error:", error)

    except Exception as error:
        print("Camera error:", error)

    finally:
        release_camera(camera)