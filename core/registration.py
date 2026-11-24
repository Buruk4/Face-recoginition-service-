"""Interactive five-pose face registration."""

from __future__ import annotations

import re
import shutil
import time
import traceback
from pathlib import Path
from typing import Optional

import cv2

from config import (
    DATABASE_PATH,
    DEBUG,
    PITCH_THRESHOLD,
    REGISTRATION_TIMEOUT,
    STABLE_FRAMES,
    YAW_THRESHOLD,
    ensure_database_path,
)
from core.camera import open_camera, release_camera
from core.face_pose import get_head_pose
from core.image_io import imwrite

POSES = ["front", "left", "right", "up", "down"]

INSTRUCTIONS = {
    "front": "Look straight at the camera",
    "left": "Slowly turn your head LEFT",
    "right": "Slowly turn your head RIGHT",
    "up": "Look UP",
    "down": "Look DOWN",
}

WINDOW_NAME = "Automatic Face Registration"

# Names that Windows does not allow as folder names.
_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


def get_pose_instruction(pose: str) -> str:
    return INSTRUCTIONS[pose]


# A natural "straight ahead" pose is not always exactly 0 degrees: a laptop
# camera can sit above or below eye level.  The front capture therefore uses
# generous limits, and its measured angles become the baseline for the other
# four poses.  This makes registration work for different faces and cameras.
FRONT_YAW_LIMIT = 15.0
FRONT_PITCH_LIMIT = 25.0


def pose_detected(
    pose: str,
    yaw: Optional[float],
    pitch: Optional[float],
    baseline_yaw: float = 0.0,
    baseline_pitch: float = 0.0,
) -> bool:
    """Return ``True`` when the current head pose matches the requested pose.

    Yaw is positive when the person turns to *their own left*; pitch is positive
    when they look up.  The baseline angles are measured during the front
    capture, so the other poses are judged relative to the person's neutral
    position instead of an absolute zero.
    """

    if yaw is None or pitch is None:
        return False

    if pose == "front":
        return abs(yaw) < FRONT_YAW_LIMIT and abs(pitch) < FRONT_PITCH_LIMIT

    relative_yaw = yaw - baseline_yaw
    relative_pitch = pitch - baseline_pitch

    if pose == "left":
        return relative_yaw > YAW_THRESHOLD

    if pose == "right":
        return relative_yaw < -YAW_THRESHOLD

    if pose == "up":
        return relative_pitch > PITCH_THRESHOLD

    if pose == "down":
        return relative_pitch < -PITCH_THRESHOLD

    return False


def update_stable_count(is_pose_detected: bool, stable_count: int) -> int:
    if is_pose_detected:
        return stable_count + 1

    return max(0, stable_count - 1)


def validate_person_name(name: str) -> Optional[str]:
    """Return an error message for an invalid name, or ``None`` if valid."""

    if not name:
        return "The name cannot be empty."

    if len(name) > 64:
        return "The name is too long (maximum 64 characters)."

    if name in {".", ".."} or name.startswith("."):
        return "Names cannot start with a dot."

    if not re.fullmatch(r"[\w][\w \-]*", name, flags=re.UNICODE):
        return (
            "Use only letters, numbers, spaces, hyphens and underscores. "
            "The name must start with a letter or number."
        )

    if name[-1] in ". ":
        return "The name cannot end with a dot or a space."

    if name.upper() in _WINDOWS_RESERVED:
        return f"{name!r} is reserved by Windows and cannot be used as a folder name."

    if Path(name).name != name:
        return "The name cannot contain path separators."

    return None


def _draw_text(frame, text: str, y: int, color, scale: float = 0.7) -> None:
    cv2.putText(
        frame,
        text,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )


def _window_closed(window_name: str) -> bool:
    """Return ``True`` if the user closed the OpenCV window."""

    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return False


def register_face() -> None:
    """Run the interactive registration workflow."""

    try:
        name = input("\nEnter person's name: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nRegistration cancelled.")
        return

    error = validate_person_name(name)
    if error:
        print(f"Invalid name: {error}")
        return

    try:
        ensure_database_path()
    except RuntimeError as error:
        print(error)
        return

    person_folder = DATABASE_PATH / name

    if person_folder.exists():
        print("This person is already registered.")
        print(f"Remove the folder below to register them again:\n  {person_folder}")
        return

    try:
        person_folder.mkdir(parents=False, exist_ok=False)
    except FileExistsError:
        print("This person is already registered.")
        return
    except OSError as error:
        print(f"Could not create the person's folder: {error}")
        return

    camera = None
    completed = False

    try:
        # Imported here so a missing MediaPipe install produces a friendly
        # pre-flight message instead of an import error at startup.
        import mediapipe as mp

        camera = open_camera()

        saved_count = 0
        stable_count = 0
        pose_index = 0
        started_at = time.monotonic()
        last_pose_print = 0.0
        last_status = ""
        last_status_at = 0.0
        baseline_yaw = 0.0
        baseline_pitch = 0.0

        print("\nStarting automatic face registration...")
        print(f"Required stable frames: {STABLE_FRAMES}")
        print(f"Yaw threshold: {YAW_THRESHOLD} degrees")
        print(f"Pitch threshold: {PITCH_THRESHOLD} degrees")
        print("The front photo calibrates your neutral head position.")
        print("Press Q or Esc in the camera window to cancel.\n")

        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        ) as face_mesh:

            while pose_index < len(POSES):

                if (
                    REGISTRATION_TIMEOUT > 0
                    and time.monotonic() - started_at > REGISTRATION_TIMEOUT
                ):
                    print(
                        f"\nRegistration timed out after "
                        f"{REGISTRATION_TIMEOUT:.0f} seconds."
                    )
                    break

                success, frame = camera.read()

                if not success or frame is None:
                    print("\nCould not read from the camera.")
                    break

                height, width = frame.shape[:2]
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)

                current_pose = POSES[pose_index]
                instruction = get_pose_instruction(current_pose)
                display_frame = cv2.flip(frame, 1)

                _draw_text(display_frame, f"Registration: {name}", 35, (0, 255, 0), 0.8)
                _draw_text(display_frame, instruction, 75, (0, 255, 255))
                _draw_text(
                    display_frame,
                    f"Captured: {saved_count}/{len(POSES)}",
                    115,
                    (255, 255, 0),
                )

                current_pose_detected = False
                yaw = None
                pitch = None
                face_count = (
                    len(results.multi_face_landmarks)
                    if results.multi_face_landmarks
                    else 0
                )

                if face_count == 0:
                    _draw_text(display_frame, "No face detected", 155, (0, 0, 255))
                    message = "No face detected. Move into the frame."
                elif face_count != 1:
                    _draw_text(
                        display_frame,
                        "Only one person should be visible",
                        155,
                        (0, 0, 255),
                        0.6,
                    )
                    message = f"Multiple faces detected: {face_count}. Ask others to step aside."
                else:
                    face_landmarks = results.multi_face_landmarks[0]
                    yaw, pitch = get_head_pose(face_landmarks, width, height)

                    if yaw is None or pitch is None:
                        _draw_text(
                            display_frame,
                            "Head pose estimation failed",
                            155,
                            (0, 0, 255),
                            0.6,
                        )
                        message = "Head pose estimation failed. Hold still and face the camera."
                    else:
                        current_pose_detected = pose_detected(
                            current_pose,
                            yaw,
                            pitch,
                            baseline_yaw,
                            baseline_pitch,
                        )
                        _draw_text(
                            display_frame,
                            f"Yaw: {yaw:.1f}  Pitch: {pitch:.1f}",
                            155,
                            (255, 255, 255),
                            0.6,
                        )

                        now = time.monotonic()
                        if now - last_pose_print >= 0.5:
                            print(
                                f"Pose: {current_pose.upper():<6} | "
                                f"Yaw: {yaw:7.2f} | "
                                f"Pitch: {pitch:7.2f} | "
                                f"Detected: {current_pose_detected}"
                            )
                            last_pose_print = now

                        message = (
                            "Hold position..."
                            if current_pose_detected
                            else "Align with the requested pose..."
                        )

                stable_count = update_stable_count(
                    current_pose_detected, stable_count
                )

                status = f"{message} {stable_count}/{STABLE_FRAMES}"
                now = time.monotonic()
                if message != last_status or now - last_status_at >= 2.0:
                    print(status)
                    last_status = message
                    last_status_at = now

                _draw_text(
                    display_frame,
                    f"{message} {stable_count}/{STABLE_FRAMES}",
                    195,
                    (0, 255, 0) if current_pose_detected else (0, 165, 255),
                )

                if stable_count >= STABLE_FRAMES:
                    print(f"\nCAPTURE TRIGGERED: {current_pose.upper()}")

                    filename = person_folder / f"{current_pose}.jpg"

                    if not imwrite(filename, frame):
                        raise OSError(f"Could not save captured image: {filename}")

                    if current_pose == "front":
                        baseline_yaw = yaw if yaw is not None else 0.0
                        baseline_pitch = pitch if pitch is not None else 0.0
                        print(
                            f"Neutral pose calibrated: "
                            f"yaw={baseline_yaw:.1f}, pitch={baseline_pitch:.1f}"
                        )

                    print(f"SUCCESS: Saved {filename}")

                    saved_count += 1
                    pose_index += 1
                    stable_count = 0

                    print(f"Progress: {saved_count}/{len(POSES)}")

                    time.sleep(0.8)

                try:
                    cv2.imshow(WINDOW_NAME, display_frame)
                except cv2.error as error:
                    raise RuntimeError(
                        "OpenCV cannot open a camera preview window. "
                        "Install the regular (non-headless) OpenCV build:\n"
                        "  python -m pip install --force-reinstall --no-deps "
                        "opencv-contrib-python==4.11.0.86"
                    ) from error

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    print("\nRegistration cancelled.")
                    break

                if _window_closed(WINDOW_NAME):
                    print("\nRegistration window closed.")
                    break

        completed = saved_count == len(POSES)

        if completed:
            print(f"\nSuccessfully registered {name}!")
            print(f"Photos saved in: {person_folder}")
        else:
            print(
                f"\nRegistration incomplete: "
                f"{saved_count}/{len(POSES)} photos."
            )

    except KeyboardInterrupt:
        print("\nRegistration cancelled.")

    except Exception as error:  # noqa: BLE001 - reported to the user
        print("\nRegistration error:", error)
        if DEBUG:
            traceback.print_exc()

    finally:
        release_camera(camera)

        if not completed:
            shutil.rmtree(person_folder, ignore_errors=True)
            print("Partial registration data was removed.")
