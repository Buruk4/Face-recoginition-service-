import cv2

from config import CAMERA_INDEX


def open_camera():

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    return camera


def release_camera(camera):

    if camera is not None:
        camera.release()

    cv2.destroyAllWindows()