import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATABASE_PATH = os.path.join(BASE_DIR, "faces")

MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "opencv"

# Initial experimental threshold. Calibrate with real samples.
MATCH_THRESHOLD = 0.40

# Difference required between the best and second-best identity.
MATCH_MARGIN = 0.05

CAMERA_INDEX = 0

# Number of consecutive frames required before saving a pose.
STABLE_FRAMES = 8

# Head pose thresholds in degrees.
YAW_THRESHOLD = 15
PITCH_THRESHOLD = 12

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

os.makedirs(DATABASE_PATH, exist_ok=True)