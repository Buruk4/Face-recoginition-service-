import os
import cv2
import numpy as np

from deepface import DeepFace

from config import (
    DATABASE_PATH,
    MODEL_NAME,
    DETECTOR_BACKEND,
    IMAGE_EXTENSIONS
)


def extract_embedding(image):

    result = DeepFace.represent(
        img_path=image,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=True
    )

    if len(result) != 1:
        raise ValueError("Exactly one face must be detected.")

    return np.array(
        result[0]["embedding"],
        dtype=np.float32
    )


def load_database():

    database = {}

    if not os.path.exists(DATABASE_PATH):
        return database

    for person_name in os.listdir(DATABASE_PATH):

        person_folder = os.path.join(
            DATABASE_PATH,
            person_name
        )

        if not os.path.isdir(person_folder):
            continue

        embeddings = []

        for filename in os.listdir(person_folder):

            if not filename.lower().endswith(IMAGE_EXTENSIONS):
                continue

            image_path = os.path.join(
                person_folder,
                filename
            )

            image = cv2.imread(image_path)

            if image is None:
                print("Could not read:", image_path)
                continue

            try:

                embedding = extract_embedding(image)

                embeddings.append(embedding)

                print(
                    f"Loaded {person_name}: {filename}"
                )

            except Exception as error:

                print(
                    f"Skipped {filename}: {error}"
                )

        if embeddings:
            database[person_name] = embeddings

    return database


def cosine_distance(embedding1, embedding2):

    denominator = (
        np.linalg.norm(embedding1) *
        np.linalg.norm(embedding2)
    )

    if denominator == 0:
        return 1.0

    similarity = np.dot(
        embedding1,
        embedding2
    ) / denominator

    return float(1.0 - similarity)