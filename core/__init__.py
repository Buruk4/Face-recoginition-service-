"""Core building blocks for the Face Recognition Service.

Modules
-------
cache         Small cache-deletion helper used by the command line.
camera        Webcam handling with per-platform backends.
database      ArcFace embedding extraction, matching and the embedding cache.
face_pose     Head-pose estimation from MediaPipe Face Mesh landmarks.
image_io      Unicode-safe image reading/writing helpers.
preflight     Dependency, GUI, camera and model checks.
recognition   Interactive recognition workflow.
registration  Interactive five-pose registration workflow.
"""

__version__ = "2.0.0"
