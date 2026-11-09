# Face Recognition Service

A small Python app for registering people with a webcam and recognizing them later. Registration automatically takes five photos at different head poses. Recognition compares the camera image with the saved faces.

## What you need

- Windows 10 or 11
- Python 3.12 (64-bit)
- A working webcam
- Internet access the first time you install packages and run recognition (DeepFace may download its ArcFace model)

## Get it running

Open PowerShell in the project folder (the folder containing `main.py`), then run these commands.

### 1. Create and activate a virtual environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell says scripts are disabled, allow activation just for this PowerShell window and try again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

When it is active, your prompt should start with `(.venv)`.

### 2. Install the project packages

```powershell
python -m pip install --upgrade pip
python -m pip install "numpy==1.26.4" "pandas==2.2.3" "protobuf==4.25.9" "mediapipe==0.10.21" "tensorflow==2.19.1" "tf-keras==2.19.0" "deepface==0.0.103" "onnxruntime==1.31.1" "opencv-contrib-python==4.11.0.86"
python -m pip install --force-reinstall --no-deps "opencv-contrib-python==4.11.0.86"
```

TensorFlow and MediaPipe need compatible protobuf versions, so keep the versions above together. DeepFace also installs a headless OpenCV package; the last command makes sure the regular (camera-window capable) OpenCV build is active.

### 3. Start the app

```powershell
python main.py
```

## Using the app

### Register someone

1. Choose `1` in the terminal.
2. Enter their name.
3. Keep one face in view and follow the instruction in the camera window: front, left, right, up, then down.
4. Hold each pose until the counter reaches `8/8`. The app saves that photo and automatically moves to the next pose.
5. Press `Q` in the camera window to cancel.

Photos are saved under `faces\<person name>\`. A name can only be registered once; to register them again, remove that person's folder first.

### Recognize someone

1. Choose `2` in the terminal.
2. Press `Space` in the camera window to check the face.
3. Press `Q` in the camera window to close it.

The first recognition may take longer while the face model is downloaded and loaded.

## If something goes wrong

- **The camera will not open:** Close other apps using it. The default camera is index `0` in `config.py`; change `CAMERA_INDEX` if your webcam uses a different index.
- **The pose counter keeps going up and down:** Face the camera clearly, use good lighting, keep only one person in view, and hold the requested pose until the counter reaches `8/8`.
- **No face is detected:** Move closer, make sure your face is lit, and remove anything blocking it.
- **You see TensorFlow or MediaPipe warnings:** Startup warnings can be normal. If the program stops, look for the final traceback/error message.

## Sharing with the team

Share the project files and this README, but do not share the `.venv`/`venv` folder; each teammate should create their own environment using the steps above. The `faces` folder contains personal face images, so do not share it unless everyone pictured has agreed. Teammates can register their own faces after starting the app.

The app's main settings, including the camera index and pose thresholds, are in `config.py`.
