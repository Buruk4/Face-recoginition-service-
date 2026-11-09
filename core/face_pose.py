import cv2
import numpy as np


# MediaPipe facial landmark indices
LANDMARKS = [1, 152, 33, 263, 61, 291]

# Approximate 3D facial model points
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -63.6, -12.5),      # Chin
    (-43.3, 32.7, -26.0),     # Left eye corner
    (43.3, 32.7, -26.0),      # Right eye corner
    (-28.9, -28.9, -24.1),    # Left mouth corner
    (28.9, -28.9, -24.1)      # Right mouth corner
], dtype=np.float64)



def get_head_pose(face_landmarks, width, height):
    image_points = []

    for index in LANDMARKS:
        landmark = face_landmarks.landmark[index]
        x = landmark.x * width
        y = landmark.y * height
        image_points.append((x, y))

    image_points = np.array(image_points, dtype=np.float64)

    focal_length = width

    camera_matrix = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    distortion = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None, None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    angles = cv2.RQDecomp3x3(rotation_matrix)[0]

    pitch = float(angles[0])
    yaw = float(angles[1])

    # Normalize pitch so that angles near -180° or +180°
    # are represented around 0°.
    if pitch < -90:
        pitch += 180
    elif pitch > 90:
        pitch -= 180

    return yaw, pitch
