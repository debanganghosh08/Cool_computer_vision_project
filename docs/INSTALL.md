# 📦 Install

Three commands if everything goes well, and this page for when it does not.

---

## ✅ What you need

| | |
|---|---|
| **Python** | 3.10, 3.11 or 3.12. 3.13 works on recent MediaPipe — see below |
| **OS** | Windows, macOS or Linux |
| **GPU** | none |
| **Disk** | about 400 MB (MediaPipe and OpenCV are large wheels) + 7.8 MB for the model |
| **ffmpeg** | optional. Only needed to keep the sound on an exported video |

Check your Python first:

```bash
python --version
```

---

## 🚀 The normal path

```bash
git clone https://github.com/debanganghosh08/Cool_computer_vision_project.git
cd Cool_computer_vision_project
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or let the script do it:

```powershell
.\scripts\setup.ps1        # Windows
```
```bash
bash scripts/setup.sh      # macOS / Linux
```

Then:

```bash
python -m handportal --input 0 --preview
```

Hands up, palms to the camera, pull them apart. 🕳️

---

## 🧠 Always use a virtual environment

Not a style preference. MediaPipe pins a specific NumPy range, and installing it into your
system Python is a reliable way to break something else you installed last month. The `.venv`
folder is already in `.gitignore`. Delete it and start again any time; nothing else is affected.

**You know it is active** when your prompt starts with `(.venv)`. If it does not, `pip install`
went somewhere you did not mean.

---

## 🧰 The model

The first run downloads `hand_landmarker.task` (7.8 MB) from Google's public model host into
`models/`. It happens once, and after that the project is fully offline.

If your network blocks it, download it yourself:

```
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

Save it as `models/hand_landmarker.task`, or point at it anywhere:

```bash
python -m handportal -i 0 --preview --model /path/to/hand_landmarker.task
```

You can also fetch it ahead of time:

```bash
python scripts/get_model.py
```

---

## 🐍 Python 3.13

MediaPipe did not ship 3.13 wheels for a long time, which is why `requirements.txt` pins a
version tested on 3.12. If you are on 3.13 and `pip install mediapipe==0.10.21` fails with
"no matching distribution", you have two options.

**Let pip pick a newer MediaPipe** (newer releases do publish 3.13 wheels):

```bash
pip install "mediapipe>=0.10.30" opencv-python numpy
```

Everything in this project uses the MediaPipe **Tasks** API (`HandLandmarker`), which is the
current, supported interface, so a newer MediaPipe is the safe direction to move in. The older
`mp.solutions.hands` interface is not used anywhere here.

**Or install a second Python** alongside your current one and point the venv at it:

```bash
py -3.12 -m venv .venv        # Windows, if you have 3.12 installed
```

---

## 🔊 ffmpeg (optional)

OpenCV writes picture only. Without ffmpeg on your PATH, an exported video is silent — the
render still succeeds, you just lose the audio. With ffmpeg present, the original soundtrack is
copied back automatically.

```bash
ffmpeg -version
```

- **Windows:** `winget install Gyan.FFmpeg`, or download a build and add it to PATH
- **macOS:** `brew install ffmpeg`
- **Linux:** `sudo apt install ffmpeg`

Skip the step entirely with `--no-audio`.

---

## 🧪 Check the install

```bash
python tests/test_geometry.py
```

35 tests, about a second, and they need neither a camera nor the model. If those pass, the maths
is fine and anything still wrong is a camera, a codec, or a path.

Then the real thing:

```bash
python -m handportal --input 0 --preview
```

---

## 🧹 Uninstall

Delete the folder. There is nothing else: no global install, no service, no registry entry, no
config in your home directory.
