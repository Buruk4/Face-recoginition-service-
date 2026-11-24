# Third-party assets

## `canonical_face_model.obj`

- **Source:** <https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj>
- **Project:** MediaPipe (Google)
- **License:** Apache License 2.0 (<https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE>)
- **Used by:** `core/face_pose.py`

The file is a neutral 3D face mesh with 468 vertices. Vertex `N` (1-based)
corresponds to MediaPipe Face Mesh landmark `N - 1`. `core/face_pose.py` uses
the vertices as the 3D object points for `cv2.solvePnP`, which makes head-pose
estimation (yaw and pitch) accurate and stable. The mesh is not modified.

If you redistribute this project, keep this notice together with the asset.
