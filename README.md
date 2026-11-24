# Face Recognition Service

A desktop Python application that registers people with a webcam and recognises
them later. Registration is automatic: the app asks for five head poses and
saves one photo per pose. Recognition compares a live camera frame with the
saved photos and prints the best match.

The project uses **DeepFace's ArcFace model running on ONNX Runtime** for face
embeddings and **MediaPipe Face Mesh** for head-pose guidance. TensorFlow is not
required.

---

## Contents

1. [Features](#features)
2. [How it works](#how-it-works)
3. [Requirements](#requirements)
4. [Quick start](#quick-start)
   - [Windows](#windows)
   - [Linux](#linux)
   - [macOS](#macos)
5. [First run and model download](#first-run-and-model-download)
6. [Using the application](#using-the-application)
7. [Configuration](#configuration)
8. [Project structure](#project-structure)
9. [Data, privacy and resetting](#data-privacy-and-resetting)
10. [Testing and verification](#testing-and-verification)
11. [Troubleshooting](#troubleshooting)
12. [Known limitations](#known-limitations)
13. [Future improvements](#future-improvements)
14. [Dependency notes](#dependency-notes)
15. [License and attribution](#license-and-attribution)

---

## Features

- **Five-pose registration** – front, left, right, up and down.
- **Automatic capture** – a pose is saved after eight stable frames; no key
  press is needed for each photo.
- **Head-pose guidance** – MediaPipe Face Mesh landmarks plus OpenCV
  `solvePnP` estimate yaw and pitch. The front photo calibrates the person's
  neutral position, so the workflow works with different faces and camera
  heights.
- **ArcFace recognition** – 512-dimensional embeddings and cosine distance.
- **Sensible match rules** – a distance threshold plus a margin between the
  best and second-best identity.
- **Low-light fallback** – if the configured detector misses a face,
  MediaPipe locates and crops it, then ArcFace processes the crop. This keeps
  recognition usable in dim webcam images.
- **Embedding cache** – recognition starts almost instantly after the first
  run. The cache is invalidated automatically when a photo, model or detector
  changes.
- **Portable paths** – the project works from any folder; no hard-coded
  machine-specific paths.
- **Environment configuration** – every setting can be changed with an
  environment variable or a `.env` file.
- **Unicode-safe image handling** – names such as `José` or `日本語` work on
  Windows.
- **Installation self-test** – `python main.py --check` checks Python,
  packages, the database folder, OpenCV GUI support, the bundled 3D mesh, the
  ArcFace model and MediaPipe.
- **Single OpenCV installation** – avoids the classic conflict between
  `opencv-python`, `opencv-contrib-python` and their headless variants.

---

## How it works

```
                    REGISTRATION
  webcam frame ──► MediaPipe Face Mesh ──► solvePnP ──► yaw / pitch
        │                                      │
        │                                      ▼
        │                         "turn left / right / up / down"
        │                                      │
        ▼                                      ▼
  stable for 8 frames? ──yes──► save faces/<name>/<pose>.jpg
        │
        no
        └──► keep showing guidance

                    RECOGNITION
  webcam frame ──► DeepFace ArcFace (ONNX) ──► 512-D embedding
                                                     │
                                                     ▼
                                      cosine distance to every saved photo
                                                     │
                                                     ▼
                              best person + threshold + margin check
```

### Registration

1. The person's name is validated and a folder `faces/<name>/` is created.
2. MediaPipe tracks the face and estimates its orientation.
3. The app asks for **front**, then **left**, **right**, **up**, **down**.
4. For each pose, a counter must reach `STABLE_FRAMES` (default `8`) before the
   photo is saved.
5. The **front** photo records the person's neutral yaw and pitch. The other
   four poses are judged relative to that neutral position.
6. If registration is cancelled or fails, the partial folder is removed.

### Recognition

1. Every registered photo is converted to an ArcFace embedding. Embeddings are
   cached in `faces/.embeddings_cache.npz`.
2. When the user presses **Space**, the current (unmirrored) camera frame is
   converted to an embedding. If the OpenCV detector cannot find a face, the
   app retries with a MediaPipe face crop (see `FACE_MEDIAPIPE_FALLBACK`).
3. The cosine distance to every saved embedding is calculated. Each person's
   score is their closest photo.
4. A match is accepted only when:
   - the best distance is `<= MATCH_THRESHOLD` (default `0.68`), and
   - the second-best person is at least `MATCH_MARGIN` (default `0.05`) farther
     away. This avoids choosing between two look-alike registered people.

The camera preview is mirrored for a natural "selfie" view, but recognition and
pose calculations always use the unmirrored frame.

---

## Requirements

### Supported operating systems

| Operating system | Status in this audit |
| --- | --- |
| Windows 10 / 11 (64-bit) | **Tested** on Windows 11 |
| Linux desktop (64-bit) | **Clean install verified** under WSL Ubuntu 24.04 (Python 3.12.3); runtime self-test and camera preview were not run |
| macOS 13 or newer (Intel / Apple silicon) | Installation and commands provided; **not tested** here |

The camera preview needs a graphical desktop session. A headless server can run
the checks, but it cannot display the registration or recognition windows.

### Python

- **Python 3.10, 3.11 or 3.12**, 64-bit.
- **Python 3.12 is recommended and was used for testing.**
- Python 3.13 is not supported because MediaPipe 0.10.21 has no 3.13 wheels.
- Python 3.9 and older are not supported by the pinned package set.

### Hardware and disk space

- A working webcam.
- About 1 GB of free disk space for the virtual environment and model files.
- The ArcFace ONNX model is about **137 MB** and is downloaded on first use.
- A CPU is sufficient. No GPU is required.

### Internet

- Needed once to install packages.
- Needed once to download the ArcFace model, unless you copy the model file
  manually (see [First run and model download](#first-run-and-model-download)).

---

## Quick start

The repository URL from the original project is:

```text
https://github.com/Buruk4/Face-recoginition-service-.git
```

> **Important:** install into a virtual environment. The installer refuses to
> modify the global Python unless you pass `--allow-global`.

### Windows

Open **PowerShell** in the folder where you want the project.

```powershell
git clone https://github.com/Buruk4/Face-recoginition-service-.git
cd Face-recoginition-service-

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python scripts\install.py

python main.py --check
python main.py --check --camera
python main.py
```

If PowerShell blocks activation, allow it for the current window only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Your prompt should start with `(.venv)` after activation.

### Linux

These commands assume a Debian/Ubuntu-style distribution. On other
distributions, install the equivalent packages.

```bash
git clone https://github.com/Buruk4/Face-recoginition-service-.git
cd Face-recoginition-service-

sudo apt update
sudo apt install -y python3.12 python3.12-venv libgl1 libglib2.0-0

python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python scripts/install.py

python main.py --check
python main.py --check --camera
python main.py
```

Notes:

- The camera preview needs an X11 or Wayland session (`DISPLAY` or
  `WAYLAND_DISPLAY` must be set). It does not work over a plain SSH session.
- If the camera cannot be opened, make sure your user is in the `video` group
  and log out/in again:
  ```bash
  sudo usermod -aG video "$USER"
  ```
- On Wayland, OpenCV may use XWayland. If the preview fails, try
  `FACE_CAMERA_BACKEND=v4l2`.

### macOS

```bash
git clone https://github.com/Buruk4/Face-recoginition-service-.git
cd Face-recoginition-service-

brew install python@3.12

python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python scripts/install.py

python main.py --check
python main.py --check --camera
python main.py
```

Notes:

- macOS 13 or newer is required by the OpenCV 4.11 wheels. On older macOS,
  install a newer macOS or use a newer OpenCV build and adjust the pins.
- Grant camera access to **Terminal** (or your IDE) in
  **System Settings → Privacy & Security → Camera**.
- Apple silicon and Intel Macs are both supported by the pinned wheels.

### What the installer does

`scripts/install.py` performs these steps:

1. Checks the Python version (3.10–3.12) and that a virtual environment is
   active.
2. Installs `requirements.txt` (NumPy, pandas, OpenCV, ONNX Runtime,
   MediaPipe's runtime libraries, DeepFace's runtime libraries).
3. Installs `requirements-ml.txt` with `--no-deps`:
   - `deepface==0.0.103`
   - `mediapipe==0.10.21`
4. Removes every other OpenCV distribution and force-reinstalls
   `opencv-contrib-python==4.11.0.86`, which provides both `cv2` and the GUI
   window support.
5. Verifies that the packages import and that OpenCV has a GUI backend.
6. Prints known, harmless `pip check` warnings caused by the `--no-deps`
   install.

### Manual installation

If you prefer to run the steps yourself, use exactly this order:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -r requirements-ml.txt
python -m pip install --force-reinstall --no-deps opencv-contrib-python==4.11.0.86
python main.py --check
```

The `--no-deps` steps are deliberate. See [Dependency notes](#dependency-notes)
for the reasons.

---

## First run and model download

The first time the ArcFace model is needed, DeepFace downloads it from the
internet:

```text
arcface_weights.onnx  (~137 MB)
```

The file is stored in:

```text
<your home folder>/.deepface/weights/arcface_weights.onnx
```

On Windows this is usually:

```text
C:\Users\<you>\.deepface\weights\arcface_weights.onnx
```

`python main.py --check` downloads the model and reports whether it loaded
correctly. Run it once before using the camera.

If the automatic download fails:

1. Download `arcface_weights.onnx` manually from one of these URLs:
   - <https://github.com/serengil/deepface_models/releases/download/v1.0/arcface_weights.onnx>
   - <https://huggingface.co/serengil/deepface/resolve/main/arcface_weights.onnx>
2. Create the weights folder if it does not exist.
3. Place the file at
   `<your home folder>/.deepface/weights/arcface_weights.onnx`.
4. Run `python main.py --check` again.

If you want to store models somewhere else, set `DEEPFACE_HOME` to the folder
that should contain `.deepface` (usually your home folder), not to `.deepface`
itself. See [Configuration](#configuration).

---

## Using the application

Start the menu:

```bash
python main.py
```

```text
==============================
     FACE RECOGNITION SYSTEM
==============================
1. Register New Person
2. Recognize Person
3. Exit
```

### Register a person

1. Choose `1`.
2. Enter a name. Letters, numbers, spaces, hyphens and underscores are allowed.
   Names such as `José` and `日本語` work.
3. Follow the instruction in the camera window:
   - Look straight at the camera.
   - Slowly turn your head left.
   - Slowly turn your head right.
   - Look up.
   - Look down.
4. Hold each pose until the on-screen counter reaches `8/8`. The photo is saved
   automatically and the app moves to the next pose.
5. Press `Q` or `Esc`, or close the window, to cancel. Partial data is removed.

Tips:

- Use good, even lighting.
- Keep only one person in view.
- Move your whole head, not just your eyes.
- If the counter does not advance, make the movement a little larger, then hold
  still.

A name can only be registered once. To register the same person again, delete
their folder first (see [Data, privacy and resetting](#data-privacy-and-resetting)).

### Recognise a person

1. Choose `2`.
2. The app loads the registered embeddings (from the cache when possible).
3. Look at the camera and press **Space**.
4. The result is printed in the terminal and shown in the camera window for a
   few seconds.
5. Press `Q`, `Esc`, or close the window to stop.

Possible results:

- `Recognized: <name>`
- `Unknown person` – the best distance is above the threshold, or two
  registered people are too similar.
- `No registered faces` – the database is empty.

### Command-line options

```bash
python main.py                  # start the interactive menu
python main.py --check          # verify the installation, no camera needed
python main.py --check --camera # also try to open the webcam
python main.py --clear-cache    # delete cached embeddings
python main.py --version        # print the project version
```

`--check` exits with code `0` when all required checks pass and `1` when
something must be fixed. On a headless Linux machine, the GUI check is reported
as a warning instead of a failure.

---

## Configuration

Settings can be provided in three ways, in this order of precedence:

1. Real environment variables.
2. A `.env` file in the project folder.
3. The defaults in `config.py`.

Copy `.env.example` to `.env` and uncomment the values you need:

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

The `.env` file is ignored by git.

### Settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `FACE_DATABASE_PATH` | `faces` | Folder containing one sub-folder per registered person. Relative paths are resolved from the project folder. |
| `DEEPFACE_HOME` | your home folder | DeepFace creates `<DEEPFACE_HOME>/.deepface/weights` for model files. |
| `FACE_EMBEDDING_CACHE_PATH` | `<database>/.embeddings_cache.npz` | Location of the embedding cache. |
| `FACE_EMBEDDING_CACHE` | `1` | Set to `0` to disable the cache. |
| `FACE_MEDIAPIPE_FALLBACK` | `1` | Set to `0` to disable the MediaPipe crop fallback when the detector misses a face. |
| `FACE_MATCH_THRESHOLD` | `0.68` | Maximum cosine distance for a match. Lower is stricter. |
| `FACE_MATCH_MARGIN` | `0.05` | Required distance gap between the best and second-best person. |
| `FACE_MODEL_NAME` | `ArcFace` | DeepFace recognition model. |
| `FACE_DETECTOR_BACKEND` | `opencv` | DeepFace face detector. `opencv` needs no extra downloads. |
| `DEEPFACE_BACKEND_ENGINE` | `onnx` | DeepFace backend. `onnx` needs only ONNX Runtime. |
| `FACE_CAMERA_INDEX` | `0` | Webcam index. Try `1`, `2`, … if index `0` is wrong. |
| `FACE_CAMERA_BACKEND` | `auto` | `auto`, `dshow`, `msmf`, `v4l2`, `avfoundation`, `gstreamer` or `any`. |
| `FACE_STABLE_FRAMES` | `8` | Consecutive matching frames required before a pose is saved. |
| `FACE_YAW_THRESHOLD` | `15` | Degrees of left/right turn required (relative to the front pose). |
| `FACE_PITCH_THRESHOLD` | `12` | Degrees of up/down tilt required (relative to the front pose). |
| `FACE_REGISTRATION_TIMEOUT` | `300` | Maximum registration time in seconds. `0` disables the limit. |
| `FACE_DEBUG` | `0` | Set to `1` to print full tracebacks for unexpected errors. |
| `FACE_SKIP_PYTHON_CHECK` | `0` | Set to `1` to bypass the Python version check (not recommended). |

Example `.env`:

```dotenv
FACE_CAMERA_INDEX=1
FACE_MATCH_THRESHOLD=0.65
FACE_STABLE_FRAMES=10
FACE_DATABASE_PATH=D:/face-data
```

### Tuning recognition

- **False negatives** (a registered person is rejected): raise
  `FACE_MATCH_THRESHOLD` slightly, for example to `0.72`.
- **False positives** (an unknown person is accepted): lower the threshold, for
  example to `0.60`, and keep the margin at `0.05` or higher.
- Register several photos per person in different lighting for better results.
  The current workflow saves one photo per pose, which is a good start.

The default `0.68` is DeepFace's published ArcFace threshold. On the test
images used during this audit, genuine matches scored about `0.50` and
different people scored about `0.87–0.92`.

---

## Project structure

```text
.
├── main.py                     # menu and command-line options
├── config.py                   # paths, defaults and environment variables
├── requirements.txt            # core Python packages (pinned)
├── requirements-ml.txt         # DeepFace/MediaPipe, installed with --no-deps
├── .env.example                # documented configuration template
├── README.md
├── assets/
│   ├── canonical_face_model.obj  # MediaPipe neutral 3D face mesh (468 vertices)
│   └── NOTICE.md               # asset source and licence
├── core/
│   ├── __init__.py             # version and package documentation
│   ├── cache.py                # small cache-deletion helper
│   ├── camera.py               # per-platform webcam backends
│   ├── database.py             # embeddings, cosine distance, cache
│   ├── face_pose.py            # yaw/pitch from Face Mesh landmarks
│   ├── image_io.py             # Unicode-safe imread/imwrite
│   ├── preflight.py            # installation and environment checks
│   ├── recognition.py          # interactive recognition
│   └── registration.py         # interactive five-pose registration
├── scripts/
│   └── install.py              # cross-platform installer
└── faces/                      # created at runtime; ignored by git
    ├── Alice/
    │   ├── front.jpg
    │   ├── left.jpg
    │   ├── right.jpg
    │   ├── up.jpg
    │   └── down.jpg
    └── .embeddings_cache.npz
```

---

## Data, privacy and resetting

### Where data is stored

- **Photos:** `faces/<person name>/<pose>.jpg`
- **Embeddings cache:** `faces/.embeddings_cache.npz`
- **Model weights:** `<your home folder>/.deepface/weights/`

Face photos and embeddings are **biometric data**. Treat the `faces/` folder as
private. It is ignored by git by default.

### Unregister a person

Delete their folder:

```bash
# Linux / macOS
rm -rf "faces/Alice"
```

```powershell
# Windows PowerShell
Remove-Item -Recurse -Force "faces\Alice"
```

The cache automatically drops entries for photos that no longer exist on the
next recognition run.

### Rebuild the cache

```bash
python main.py --clear-cache
```

The next recognition run recalculates all embeddings. This is useful after
changing the model, the detector or a photo.

### Move the database

Set `FACE_DATABASE_PATH` to a folder outside the project, for example:

```dotenv
FACE_DATABASE_PATH=D:/secure/face-data
```

---

## Testing and verification

This section separates what was actually executed from what is provided but
unverified.

### Verified on Windows 11

- **Environment:** Windows 11 (`10.0.26200`), Python `3.12.6`, 64-bit, fresh
  virtual environment.
- **Clean install:** `python scripts/install.py` completed successfully with the
  pinned versions.
- **Self-test:** `python main.py --check` passed all seven checks (Python,
  packages, database folder, OpenCV GUI, bundled mesh, ArcFace model, MediaPipe).
- **Imports:** `numpy 1.26.4`, `pandas 2.2.3`, `opencv 4.11.0`,
  `mediapipe 0.10.21`, `onnxruntime 1.23.2`, `deepface 0.0.103`,
  `werkzeug 3.1.9`, `matplotlib 3.10.8`.
- **OpenCV GUI:** `WIN32UI` is present, so `cv2.imshow` works.
- **ONNX vs TensorFlow equivalence:** ArcFace embeddings from the ONNX backend
  and the TensorFlow backend were compared on four test photos. Cosine
  distances agreed to about `1e-7`:
  - same person: `0.4969–0.5078`
  - different people: `0.8653–0.9216`
- **Recognition logic:** `img2` matched the registered `img1/img2/img4` person
  with distance `0.0000` (identical photo) and the other person at `0.8653`;
  `img3` matched the other person. The default threshold `0.68` separates the
  two groups correctly.
- **Embedding cache:** the first load computed four embeddings in about
  `8.7 s`; the second load reused all four in about `0.02 s`; touching one
  photo caused only that photo to be recomputed.
- **Unicode paths:** a database folder and image named with `ë` and Japanese
  characters loaded and saved correctly.
- **Head-pose signs:** synthetic yaw/pitch warps were used to check the
  direction conventions. A `+20°` yaw warp was classified as "left"; a frontal
  face was accepted as "front".
- **Live camera capture:** the built-in webcam opened through the camera helper,
  read 15 frames at `640×480`, and MediaPipe detected a face in the captured
  frame (yaw `2.4°`, pitch `9.8°`). An OpenCV preview window displayed live
  frames successfully. The frame was dim; DeepFace's OpenCV detector missed it,
  but the MediaPipe crop fallback produced an embedding. Compared with the
  registered test people, the live frame scored `0.81–0.92`, so it was
  correctly reported as unknown.
- **Dark-image fallback:** the same dim frame was registered as a one-person
  database and then queried again. Both the registration embedding and the
  query embedding used the MediaPipe crop fallback, and the query matched with
  distance `0.0000`.
- **Dependency metadata:** `pip check` reports only the expected
  `--no-deps` warnings for DeepFace (`opencv-contrib-python-headless`,
  `retina-face`) and MediaPipe (`jax`, `jaxlib`, `sounddevice`,
  `sentencepiece`). These packages are intentionally not installed.

### Verified on Linux (WSL Ubuntu 24.04)

- **Environment:** WSL 2, Ubuntu 24.04, Python `3.12.3`, 64-bit.
- **Clean install:** `python scripts/install.py` completed successfully. It
  installed the pinned Linux wheels, removed conflicting OpenCV builds,
  verified the imports (`numpy`, `pandas`, `cv2`, `mediapipe`, `onnxruntime`,
  `deepface`, `werkzeug`) and confirmed that the OpenCV build reports a GUI
  backend.
- **Not run on Linux:** `python main.py --check`, the ArcFace model download,
  recognition and the camera preview. The WSL distribution was shut down after
  the installer finished, so the Windows results above remain the reference for
  runtime behaviour.

### Not verified in this audit

- **Interactive camera windows:** an OpenCV preview window was created and
  displayed live camera frames, but the actual five-pose registration session
  and key presses were not automated in this environment. Camera capture and
  the underlying recognition/pose functions were exercised directly.
- **Linux runtime:** only the clean install was verified (WSL Ubuntu 24.04).
  `main.py --check`, the ArcFace model download, recognition and the camera
  preview were not run on Linux.
- **macOS:** not tested.
- **Python 3.10 and 3.11:** not tested. The pinned packages publish wheels for
  both versions, but only 3.12 was exercised.
- **TensorFlow backend inside this project:** not used. The equivalence test
  above compared embeddings from both backends outside the final application.
- **Accuracy on a large face dataset:** not measured. The four test photos are
  only a smoke test, not a benchmark.

If you run the live camera workflow on another platform, please record the
result and update this section.

---

## Troubleshooting

### The installer says it must run inside a virtual environment

Create and activate a virtual environment first:

```powershell
# Windows
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
python3.12 -m venv .venv
source .venv/bin/activate
```

Then run `python scripts/install.py`.

### `UnicodeEncodeError: 'charmap' codec ...`

DeepFace logs emoji characters, and some Windows consoles use code page 1252.
The application reconfigures stdout/stderr to UTF-8 automatically. If you still
see the error, set:

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
python main.py
```

### The camera will not open

- Close other applications that use the camera (Teams, Zoom, OBS, browsers).
- Check Windows **Settings → Privacy & security → Camera**.
- Try another index:
  ```dotenv
  FACE_CAMERA_INDEX=1
  ```
- Try another backend:
  ```dotenv
  FACE_CAMERA_BACKEND=dshow
  ```
  On Linux try `v4l2`; on macOS try `avfoundation`.
- Run `python main.py --check --camera` to see the exact backend errors.

### `cv2.imshow` fails or the window is blank

The installed OpenCV build has no GUI support (a "headless" build). Repair it:

```bash
python scripts/install.py
```

or:

```bash
python -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python-headless
python -m pip install --force-reinstall --no-deps opencv-contrib-python==4.11.0.86
```

### Multiple OpenCV packages are installed

Check:

```powershell
python -m pip list | Select-String opencv
```

```bash
python -m pip list | grep opencv
```

Only `opencv-contrib-python==4.11.0.86` should be present. If others appear, run
`python scripts/install.py` or uninstall them manually.

### `pip check` reports missing packages

These warnings are expected after the intentional `--no-deps` install:

```text
deepface 0.0.103 requires opencv-contrib-python-headless, which is not installed.
deepface 0.0.103 requires retina-face, which is not installed.
mediapipe 0.10.21 requires jax, which is not installed.
mediapipe 0.10.21 requires jaxlib, which is not installed.
mediapipe 0.10.21 requires sentencepiece, which is not installed.
mediapipe 0.10.21 requires sounddevice, which is not installed.
```

- `opencv-contrib-python-headless` is intentionally replaced by the GUI build.
- `retina-face` imports TensorFlow at runtime and is not used by the default
  OpenCV detector.
- `jax`, `jaxlib`, `sentencepiece` and `sounddevice` are declared by MediaPipe
  but are not used by Face Mesh.

If `pip check` reports anything else, reinstall with
`python scripts/install.py`.

### `ImportError: C extension: None not built` or a pandas/NumPy error

This happens when an incompatible pandas version (for example pandas 3.x) is
mixed with NumPy 1.26. Repair the environment:

```bash
python -m pip install --force-reinstall "numpy==1.26.4" "pandas==2.2.3"
```

or simply run `python scripts/install.py` again.

### The ArcFace model download fails

- Check your internet connection and proxy settings.
- Make sure there is enough disk space.
- Download the model manually (see
  [First run and model download](#first-run-and-model-download)).
- Confirm the file is at
  `<your home folder>/.deepface/weights/arcface_weights.onnx`.

### `ModuleNotFoundError: No module named 'retina_face'`

RetinaFace is not part of this installation. It requires TensorFlow and is not
needed for the default OpenCV detector. Keep
`FACE_DETECTOR_BACKEND=opencv`.

### "No face detected" during recognition

- Move closer to the camera and improve the lighting.
- Keep only one face in view.
- Make sure `FACE_MEDIAPIPE_FALLBACK` is not set to `0`; the fallback retries
  with a MediaPipe face crop when the OpenCV detector misses the face.
- If the room is very dark, a brighter lamp usually helps more than changing
  settings.
- Run `python main.py --check --camera` to confirm the camera itself works.

### The pose counter keeps going up and down

- Face the camera clearly and use good lighting.
- Keep only one face in view.
- Hold the requested pose still for a moment.
- Make the movement a little larger if the threshold is not reached.
- The front photo calibrates the neutral pose; if the first pose is hard to
  capture, check that your face is fully visible and the camera is roughly at
  eye level.

### "This person is already registered"

A folder with that name already exists. Delete it to register the person again:

```powershell
Remove-Item -Recurse -Force "faces\Alice"
```

```bash
rm -rf "faces/Alice"
```

### The `faces` folder is not writable

Set `FACE_DATABASE_PATH` to a writable location:

```dotenv
FACE_DATABASE_PATH=D:/face-data
```

### Windows path or OneDrive problems

- Very long paths can exceed the Windows limit. Keep the project and database
  in a short path such as `C:\face-service`.
- OneDrive may lock files while syncing. Pause syncing during registration, or
  move `FACE_DATABASE_PATH` outside OneDrive.

### Python 3.13 or 3.9 is rejected

The pinned MediaPipe and ONNX Runtime versions do not provide wheels for those
versions. Install Python 3.12 and recreate the virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python scripts\install.py
```

### Linux: no window appears

The camera preview needs a desktop session. Check:

```bash
echo "$DISPLAY"
echo "$WAYLAND_DISPLAY"
```

If both are empty, run the application from a desktop terminal, not over SSH.
For a virtual display, `xvfb-run python main.py --check` can test the GUI build,
but a real camera preview still needs a display.

### macOS: camera permission

Grant camera access to the application that runs Python (Terminal, iTerm2 or
your IDE) in **System Settings → Privacy & Security → Camera**, then restart
that application.

---

## Known limitations

- **No liveness detection.** A printed photo or a screen showing a face can
  fool the system. Do not use this project for security-critical access control
  without adding anti-spoofing.
- **One face at a time.** Registration rejects frames with zero or multiple
  faces; recognition expects exactly one face in the captured frame.
- **Thresholds are generic.** Lighting, age, glasses, masks, extreme poses and
  camera quality affect accuracy. Tune the threshold for your data.
- **Small registration set.** One photo per pose is enough for a demonstration
  but not for high accuracy. More photos per person would help.
- **No encryption.** Photos and embeddings are stored as ordinary files. Protect
  the `faces/` folder and backups.
- **Interactive desktop workflow.** There is no headless API or web interface.
- **OpenCV detector only by default.** RetinaFace is not installed because it
  requires TensorFlow. MTCNN, SSD, Dlib and other DeepFace detectors are also
  not installed.
- **Pose estimation is approximate.** It is good enough for guidance, not for
  medical or biometric measurement.
- **The low-light fallback changes alignment.** When the OpenCV detector fails,
  the MediaPipe crop is resized without the same alignment as the OpenCV
  pipeline. It worked in testing, but recognition may be less accurate in very
  dim conditions. Register people in good lighting when possible.
- **Platform coverage.** A clean install was verified on Windows 11 and on WSL
  Ubuntu 24.04. Linux runtime checks, the camera preview and macOS were not
  executed. Python 3.10/3.11 were not executed either.
- **No automated test suite or CI.** The checks in this audit were run manually.

---

## Future improvements

- Add liveness/anti-spoofing detection.
- Capture several photos per pose and choose the sharpest one.
- Add face-quality checks (blur, brightness, face size).
- Store embeddings in an encrypted database instead of a plain `.npz` file.
- Add a simple GUI (Tkinter or PySide) instead of the terminal menu.
- Add an attendance log with timestamps.
- Add a REST API for integration with other systems.
- Support more DeepFace models and detectors as optional extras.
- Add unit tests and GitHub Actions for Windows, Linux and macOS.
- Package the application with PyInstaller for users who do not want Python.

---

## Dependency notes

### Why ONNX instead of TensorFlow?

DeepFace's TensorFlow backend requires `tensorflow` and `tf-keras` (roughly
2 GB installed) and pins older Keras/protobuf combinations. The ONNX backend
uses `onnxruntime` (about 15 MB) and produces equivalent ArcFace embeddings. On
the same four test photos, the distances from both backends agreed to about
`1e-7`. The application therefore defaults to:

```dotenv
DEEPFACE_BACKEND_ENGINE=onnx
```

Advanced users can install TensorFlow and `tf-keras` themselves and set
`DEEPFACE_BACKEND_ENGINE=tensorflow`, but that configuration is not part of the
tested installation.

### Why are DeepFace and MediaPipe installed with `--no-deps`?

- **DeepFace** declares `opencv-contrib-python-headless`. **MediaPipe** declares
  `opencv-contrib-python`. Installing both puts two different `cv2` builds in
  the same environment, and whichever was installed last wins. That causes
  confusing `cv2.imshow` failures. This project installs one GUI-capable build
  (`opencv-contrib-python`) and skips the headless package.
- **MediaPipe 0.10.21** declares `jax`, `jaxlib`, `sounddevice` and
  `sentencepiece`. Its Face Mesh solution does not import them. Current JAX
  releases also require NumPy 2, which conflicts with MediaPipe's NumPy `<2`
  requirement. Skipping them keeps the environment small and consistent.
- The real runtime dependencies are listed explicitly in `requirements.txt`.

### Why these pins?

| Package | Pin | Reason |
| --- | --- | --- |
| `numpy` | `1.26.4` | Last NumPy 1.x line; required by MediaPipe 0.10.21 and compatible with OpenCV 4.11. |
| `pandas` | `2.2.3` | pandas 3.x requires NumPy 2 and breaks the NumPy 1.26 environment. |
| `protobuf` | `4.25.9` | Compatible with MediaPipe 0.10.21 and DeepFace 0.0.103. |
| `opencv-contrib-python` | `4.11.0.86` | Provides `cv2`, GUI support and the bundled Haar cascade used by the OpenCV detector. |
| `onnxruntime` | `1.23.2` | Has wheels for Python 3.10–3.13 on Windows/Linux and for Intel/Apple-silicon macOS. |
| `mediapipe` | `0.10.21` | Stable Face Mesh API with wheels for Python 3.9–3.12. |
| `deepface` | `0.0.103` | Provides the ArcFace model and ONNX backend. |
| `matplotlib` | `3.10.8` | Required by MediaPipe drawing utilities; still supports Python 3.10. |
| `werkzeug` | `3.1.9` | Imported by DeepFace 0.0.103 but missing from its declared dependencies. |

### Why is `retina-face` not installed?

`retina-face` imports TensorFlow at runtime. The default detector is OpenCV,
which is bundled with OpenCV and needs no extra model download. If you want
RetinaFace, install TensorFlow and `retina-face` yourself and set
`FACE_DETECTOR_BACKEND=retinaface`; that configuration is not tested here.

---

## License and attribution

- The original project repository did not include a licence file. Add an
  explicit licence before redistributing the application.
- **DeepFace** – <https://github.com/serengil/deepface> (MIT).
- **ArcFace model weights** – distributed by DeepFace; check the model
  repository's terms before commercial use.
- **MediaPipe** – <https://github.com/google-ai-edge/mediapipe> (Apache 2.0).
  The bundled `assets/canonical_face_model.obj` comes from MediaPipe; see
  `assets/NOTICE.md`.
- **OpenCV** – <https://opencv.org> (Apache 2.0).
- **ONNX Runtime** – <https://onnxruntime.ai> (MIT).

---

## Quick reference

```bash
# Install
python scripts/install.py

# Verify (downloads the ArcFace model on the first run)
python main.py --check
python main.py --check --camera

# Run
python main.py

# Clear the embedding cache
python main.py --clear-cache
```
