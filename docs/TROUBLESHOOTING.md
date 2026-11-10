# 🔧 Troubleshooting

Every error message this project can produce, and what to do about it.

---

## 🕳️ The portal never appears

**Both hands must be in frame.** The portal is defined by two palms; one hand gives you nothing.

Every run prints the number that answers this:

```
[handportal] 504 frames in 56.4s (8.9 fps) | both hands seen in 84% of frames
```

Below about 25% you also get a tip printed. If that number is low:

- **Get both hands fully in shot.** Fingers cut off by the frame edge are often not detected.
- **Palms toward the camera.** The back of the hand tracks fine; a hand edge-on does not.
- **Light the room.** MediaPipe is good in poor light, not magic.
- **Slow down.** Motion blur destroys tracking. This is also why the portal fades on fast swipes
  — that is deliberate, not a bug.
- **Try `--detect-confidence 0.3`** to accept weaker detections.

Check what is actually being tracked by running the `skeleton` world, which draws the landmarks:

```bash
python -m handportal -i 0 --preview --world skeleton
```

If no bones are drawn, it is tracking, not the portal.

---

## 🫨 The portal jitters

```bash
python -m handportal -i clip.mp4 --min-cutoff 0.6 --beta 0.01 -o out.mp4
```

Lower `--min-cutoff` = heavier smoothing when your hands are still.

---

## 🐌 The portal lags behind my hands

The other direction:

```bash
python -m handportal -i clip.mp4 --min-cutoff 1.8 --beta 0.06 -o out.mp4
```

Higher `--beta` = the filter backs off faster when you move.

---

## 📏 The portal is too fat / too thin

```bash
--thickness-min 0.03 --thickness-max 0.12     # slimmer, more of a slit
--thickness-min 0.08 --thickness-max 0.28     # chunkier
```

And if it does not open enough for the hand distance you are actually using, move the range:

```bash
--spread-closed 0.15 --spread-wide 0.60
```

---

## 💥 The portal keeps blowing up to full frame

You have `--spread-full` set. Turn it off (`--spread-full 0`, the default) and use `--reveal-at`
for the final reveal instead. See the "wrong answer #4" section in
[HOW_IT_WORKS.md](HOW_IT_WORKS.md) for why.

---

## 🔇 My exported video has no sound

OpenCV writes picture only; the sound is copied back with ffmpeg afterwards. No ffmpeg on PATH,
no sound. Install it ([INSTALL.md](INSTALL.md)) and render again, or accept it with `--no-audio`.

Webcam captures have no audio to copy in the first place.

---

## ❌ `ModuleNotFoundError: No module named 'mediapipe'`

The virtual environment is not active, or the install went into a different Python.

```bash
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

Your prompt should read `(.venv)`. Confirm with `python -c "import mediapipe; print(mediapipe.__version__)"`.

---

## ❌ `ERROR: Could not find a version that satisfies the requirement mediapipe==0.10.21`

Your Python is too new (usually 3.13) for that pinned version.

```bash
pip install "mediapipe>=0.10.30" opencv-python numpy
```

This project uses the MediaPipe Tasks API, which newer versions keep. Details in
[INSTALL.md](INSTALL.md).

---

## ❌ `Could not open '0'. Is another app using the camera?`

Close Zoom, Teams, Meet, OBS, the Camera app. Only one program gets the webcam at a time.

Multiple cameras? Try `--input 1`, `--input 2`.

---

## ❌ `Could not download the hand landmark model`

Your network blocked `storage.googleapis.com`. Download it in a browser and save it as
`models/hand_landmarker.task`. The URL is printed in the error, and is also in
[INSTALL.md](INSTALL.md).

---

## ❌ `Could not open <path> for writing`

Usually the extension. Use `.mp4`. The folder is created for you, so a missing directory is not
the cause.

---

## 🖼️ The preview window is frozen or will not close

Click the window first, then press `q` or `Esc` — OpenCV only reads keys when its window has
focus. `Ctrl+C` in the terminal also works and still finalises the output file.

On a headless machine there is no window to show, so use `--output` without `--preview`.

---

## 🐢 It is slow

Measured on a CPU-only laptop at 720x1280: about 11 fps, roughly 3x realtime. A 17-second clip
takes about a minute. That is expected.

To go faster:

- **Cheap worlds:** `thermal` (2 ms), `pixel` (2 ms), `glitch` (8 ms) against `comic` (49 ms)
- **Drop `--lightning`**
- **`--glow 0`**
- **Smaller input.** Cost scales with pixel count; 720p renders roughly twice as fast as 1080p
- **`--max-frames 120`** while you are still dialling in the look

MediaPipe is about 55 ms of every frame and there is no way around it on CPU.

---

## 🎨 It works but looks cheap

Three things, in order of impact:

1. **Thinner.** `--thickness-max 0.12`. A big coloured box reads as a sticker; a slit reads as a
   cut through the world.
2. **Match the rim to the world.** `--world storm:cyan --color cyan`. A magenta rim on a green
   world looks like two effects stacked.
3. **One reveal, late.** `--reveal-at` once, near the end. Repeated full reveals stop being a
   reveal.

---

## 🧪 Nothing above helped

```bash
python tests/test_geometry.py
```

If those 20 pass, the geometry, smoothing and masking are all fine and the problem is input,
output or environment. If any fail, open an issue with the failure and your
`python --version` and `pip freeze`.
