
import os
import re
import cv2
import time

import mediapipe as mp

from config import (
    DATABASE_PATH,
    STABLE_FRAMES,
    YAW_THRESHOLD,
    PITCH_THRESHOLD
)

from core.camera import open_camera, release_camera
from core.face_pose import get_head_pose


POSES = [
    "front",
    "left",
    "right",
    "up",
    "down"
]


def get_pose_instruction(pose):

    instructions = {
        "front": "Look straight at the camera",
        "left": "Slowly turn your head LEFT",
        "right": "Slowly turn your head RIGHT",
        "up": "Look UP",
        "down": "Look DOWN"
    }

    return instructions[pose]


def pose_detected(pose, yaw, pitch):

    if yaw is None or pitch is None:
        return False

    if pose == "front":
        return abs(yaw) < 10 and abs(pitch) < 10

    if pose == "left":
        return yaw < -YAW_THRESHOLD

    if pose == "right":
        return yaw > YAW_THRESHOLD

    if pose == "up":
        return pitch < -PITCH_THRESHOLD

    if pose == "down":
        return pitch > PITCH_THRESHOLD

    return False


def update_stable_count(is_pose_detected, stable_count):

    if is_pose_detected:
        return stable_count + 1

    return max(0, stable_count - 1)


def register_face():

    name = input("\nEnter person's name: ").strip()

    if not name or not re.fullmatch(r"[\w -]+", name):
        print("Invalid name.")
        return

    person_folder = os.path.join(DATABASE_PATH, name)

    if os.path.exists(person_folder):
        print("This person is already registered.")
        return

    os.makedirs(person_folder)

    camera = None

    try:

        camera = open_camera()

        mp_face_mesh = mp.solutions.face_mesh

        saved_count = 0
        stable_count = 0
        pose_index = 0

        print("\nStarting automatic face registration...")
        print(f"Required stable frames: {STABLE_FRAMES}")
        print(f"Yaw threshold: {YAW_THRESHOLD}")
        print(f"Pitch threshold: {PITCH_THRESHOLD}")

        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        ) as face_mesh:

            while pose_index < len(POSES):

                success, frame = camera.read()

                if not success:
                    print("Could not read camera.")
                    break

                height, width = frame.shape[:2]

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                results = face_mesh.process(rgb_frame)

                current_pose = POSES[pose_index]

                instruction = get_pose_instruction(current_pose)

                display_frame = cv2.flip(frame, 1)

                cv2.putText(
                    display_frame,
                    f"Registration: {name}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    display_frame,
                    instruction,
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    display_frame,
                    f"Captured: {saved_count}/{len(POSES)}",
                    (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

                current_pose_detected = False

                face_count = (
                    len(results.multi_face_landmarks)
                    if results.multi_face_landmarks
                    else 0
                )

                if face_count == 0:

                    print("No face detected.")

                    cv2.putText(
                        display_frame,
                        "No face detected",
                        (20, 155),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

                elif face_count != 1:

                    print(f"Multiple faces detected: {face_count}")

                    cv2.putText(
                        display_frame,
                        "Only one person should be visible",
                        (20, 155),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

                else:

                    face_landmarks = results.multi_face_landmarks[0]

                    yaw, pitch = get_head_pose(
                        face_landmarks,
                        width,
                        height
                    )

                    if yaw is not None and pitch is not None:

                        current_pose_detected = pose_detected(
                            current_pose,
                            yaw,
                            pitch
                        )

                        print(
                            f"Pose: {current_pose.upper():<6} | "
                            f"Yaw: {yaw:7.2f} | "
                            f"Pitch: {pitch:7.2f} | "
                            f"Detected: {current_pose_detected}"
                        )

                        cv2.putText(
                            display_frame,
                            f"Yaw: {yaw:.1f}  Pitch: {pitch:.1f}",
                            (20, 155),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2
                        )

                    else:
                        print("Head pose estimation failed.")

                stable_count = update_stable_count(
                    current_pose_detected,
                    stable_count
                )

                print(
                    f"Stable frames: {stable_count}/{STABLE_FRAMES}",
                    end="\r",
                    flush=True
                )

                status = (
                    "Hold position..."
                    if current_pose_detected
                    else "Align with the requested pose..."
                )

                cv2.putText(
                    display_frame,
                    f"{status} {stable_count}/{STABLE_FRAMES}",
                    (20, 195),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0) if current_pose_detected else (0, 165, 255),
                    2
                )

                if stable_count >= STABLE_FRAMES:

                    print(f"\n\nCAPTURE TRIGGERED: {current_pose.upper()}")

                    filename = os.path.join(
                        person_folder,
                        f"{current_pose}.jpg"
                    )

                    if not cv2.imwrite(filename, frame):
                        raise OSError(
                            f"Could not save captured image: {filename}"
                        )

                    print(f"SUCCESS: Saved {filename}")

                    saved_count += 1
                    pose_index += 1
                    stable_count = 0

                    print(f"Progress: {saved_count}/{len(POSES)}")

                    time.sleep(0.8)

                cv2.imshow(
                    "Automatic Face Registration",
                    display_frame
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("\nRegistration cancelled.")
                    break

        if saved_count == len(POSES):
            print(f"\nSuccessfully registered {name}!")

        else:
            print(
                f"\nRegistration incomplete: "
                f"{saved_count}/{len(POSES)} photos."
            )

            import shutil
            shutil.rmtree(person_folder, ignore_errors=True)

    except Exception as error:

        print("\nRegistration error:", error)

        import shutil
        shutil.rmtree(person_folder, ignore_errors=True)

    finally:
        release_camera(camera)
