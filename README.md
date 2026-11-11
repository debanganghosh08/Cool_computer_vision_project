# 🕳️✋ HandPortal

**Open a portal between your two hands. Whatever is inside comes from another world.** 🔥

You have seen the edit. Someone holds their hands up, pulls them apart, and a glowing slab
opens across the frame — and through that slab they are somewhere else. Everyone assumes it
is an app, a template, or an AI video generator. ❄️

It is not. It is about 1,300 lines of Python, one hand tracking model, and OpenCV.

This repo is that effect, open source, running on your laptop, offline, free. 😂

```bash
python -m handportal --input 0 --preview
```

That is it. Webcam on, hands up, portal open.

---

## 👀 What it actually does

| Step | What happens |
|---|---|
| 1️⃣ | **MediaPipe** finds 21 landmarks on each of your hands, every frame |
| 2️⃣ | Each hand gives two corners: its **index fingertip** and its **thumb tip** |
| 3️⃣ | A **One Euro filter** smooths all four corners — no jitter when still, no lag when fast |
| 4️⃣ | A **box is built between your hands**, cornered on those four fingertips: a sliver when they touch, wide open when you pull apart |
| 5️⃣ | That rectangle becomes an **alpha mask** |
| 6️⃣ | A **world** is rendered — a restyled version of your own frame |
| 7️⃣ | The two are **composited** through the mask, with a glowing rim and crawling bolts ⚡ |

No green screen. No second camera. No cloud. No account. No API key. 🚫☁️

---

## 🌍 The worlds

The thing nobody expects: **you do not need a second video.** Most worlds are generated
from the frame you already shot, so one phone, one take, and the other side of the portal
builds itself.

| World | What it looks like |
|---|---|
| `storm` ⚡ | Duotone sky world with lightning. The cinematic one. **Default.** |
| `matrix` 💚 | Falling green code, your silhouette ghosted behind it |
| `thermal` 🔥 | Heat camera |
| `edges` 🩻 | Glowing line-art x-ray |
| `blueprint` 📐 | Engineering drawing on navy, with a scrolling grid |
| `comic` 💥 | Flat colour, hard ink lines, halftone dots |
| `pixel` 🎮 | 8-bit console |
| `glitch` 📺 | Broken signal — channel shift, tearing, scanlines |
| `skeleton` 🦴 | The hand tracking itself, drawn: bones, joints, lit fingertips |
| `neon` 🌈 | You, repainted in loud pastel neon: candy gradient, drifting wash, glowing mint outlines |
| `video:clip.mp4` 🎬 | **Your own footage** inside the portal |
| `image:poster.png` 🖼️ | Any still image |

Some take an argument:

```bash
--world storm:cyan      # magenta, cyan, amber, violet, green, red
--world pixel:48        # chunkier pixels
--world matrix:70       # more columns
--world neon:ice        # cooler neon: navy, blue, cyan, mint
--world comic:9         # bigger halftone dots
```

---

## ⚡ Quick start

```bash
git clone https://github.com/debanganghosh08/Cool_computer_vision_project.git
cd Cool_computer_vision_project

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

The hand landmark model (~7.8 MB) downloads itself the first time you run it. 📦

**Live on your webcam:**

```bash
python -m handportal --input 0 --preview
```

**On a video you shot:**

```bash
python -m handportal --input myclip.mp4 --world storm --lightning --output out/portal.mp4
```

**The full cinematic version, portal swallowing the frame at 14.3 seconds:**

```bash
python -m handportal -i myclip.mp4 -w storm:magenta --lightning --reveal-at 14.3 -o out/reel.mp4
```

Full setup, including what to do when something breaks: **[docs/INSTALL.md](docs/INSTALL.md)**

---

## 🎬 For editors: export the mask, not just the video

If you finish in Premiere, CapCut, Resolve or After Effects, you do not want a baked video —
you want the **matte**. Pass `--matte` and you get a second file: the portal in pure black and
white, frame-synced to your clip. Drop it in as a track matte and put anything you like behind
it. 🧊

```bash
python -m handportal -i myclip.mp4 -w storm -o out/portal.mp4 --matte out/matte.mp4
```

Now the portal is a *tool*, not a filter. 🛠️

---

## 🎛️ Every knob

```
--world, -w         what is inside the portal (see the table above)
--color, -c         edge colour: magenta cyan lime amber white violet red, or #ff33cc
--lightning         crawling bolts along the edge
--feather           soft edge, in pixels (default 7)
--glow / --glow-width / --rim    the edge treatment
--mirror            flip left-right, for selfie view

--anchor            fingertips (default): a box between your hands, cornered on index + thumb tips
                    palms: the older band through both palm centres
--thickness-min     (palms) how thin the band is when your hands are together  (default 0.04)
--thickness-max     how thick when they are wide                       (default 0.17)
--spread-closed     hand distance that counts as "closed"              (default 0.22)
--spread-wide       hand distance that counts as "open"                (default 0.80)

--reveal-at SEC     the second the portal swallows the whole frame
--reveal-over SEC   how long that takes (default 0.6)

--min-cutoff        lower = smoother when still
--beta              higher = less lag when moving fast

--matte PATH        also write the mask as a black and white video
--max-frames N      stop after N frames, for a quick test
--preview           live window (press q to quit)
```

`python -m handportal --help` prints the lot. 📖

---

## 🧠 Three things that are harder than they look

Covered properly in **[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md)**, short version here.

**1. The wrist is the wrong anchor.** 🤚
It is the obvious landmark and it is useless: it sits at the edge of the hand and swings a long
way whenever the forearm rolls, so the portal slides around. The mean of the wrist and the four
knuckles sits in the middle of the palm and barely moves when the hand rotates.

**2. Handedness lies, screen position does not.** 🔄
MediaPipe tells you which hand is left and which is right. That label flips the moment the image
is mirrored — and a portal that swaps ends mid-shot looks broken. HandPortal sorts the two hands
by their x position on screen instead. Boring, and it never fails.

**3. A moving average is the wrong smoother.** 📉
Average over a window and the jitter goes, but so does your timing — the portal lags behind your
hands and the illusion dies. The One Euro filter changes its own cutoff with the speed of the
signal: heavy smoothing when you are still, almost none when you move. Two lines of maths, and it
is the single biggest difference between "homemade" and "clean".

And one that only shows up when you test on real footage: **driving the full reveal off how wide
your hands are does not work.** A spread wide enough to trigger it is just what "palms up to
camera" looks like, so it fires constantly, mid-gesture. The reveal is a timed beat instead
(`--reveal-at`), which is also what an editor actually wants. ⏱️

---

## 📊 Speed

Measured on this machine, CPU only, 720x1280 video, no GPU:

| Stage | ms / frame |
|---|---|
| MediaPipe hand tracking | 54 |
| World render (`storm`) | 15 |
| Composite + glow + bolts | 21 |
| **Total** | **≈ 90 ms, about 11 fps** |

A 17-second clip renders in about a minute. 🏃

It started at 2.9 fps. The entire 4x came from one change: the compositor was written in float32
numpy, and full-frame float expressions cost about 27 ms each — there were six of them per frame.
Rewritten with 8-bit OpenCV ops, the same maths costs 5 ms. Wide blurs are also done at quarter
resolution, since a blur that wide has no fine detail to lose anyway. 🧊

Preview on a webcam runs comfortably; if your machine struggles, use a cheap world
(`thermal`, `pixel`, `glitch`) and drop `--lightning`.

---

## 🧪 Tests

```bash
python tests/test_geometry.py        # no pytest needed
python -m pytest -q                  # if you have it
```

35 tests, all of the geometry, smoothing, masking and colour parsing, no camera and no model
required. They run in under a second. ✅

---

## 📁 Layout

```
handportal/
  tracker.py      MediaPipe -> two stable palm points
  smoothing.py    One Euro filter
  portal.py       two points -> a rotated rectangle
  compositor.py   mask, glow, rim, bolts, blend
  worlds.py       everything you can see inside the portal
  video.py        reading, writing, putting the audio back
  cli.py          the command line
tests/            geometry and smoothing tests
docs/             install, how it works, troubleshooting
```

---

## 🔧 Add your own world

Nine lines:

```python
# handportal/worlds.py
class SepiaWorld(World):
    name = "sepia"

    def render(self, ctx: WorldContext) -> np.ndarray:
        gray = cv2.cvtColor(ctx.frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_PINK)

BUILTIN["sepia"] = SepiaWorld
```

```bash
python -m handportal -i 0 --preview --world sepia
```

`ctx` gives you the frame, the frame number, the time in seconds, the size, and the hand
landmarks if you want to draw on them. PRs with good-looking worlds are very welcome. 🎨

---

## ❓ FAQ

**Does it need a GPU?** No. Everything here runs on the CPU. 💻

**Does it work on a Mac / Linux?** Yes. It is Python, OpenCV and MediaPipe, nothing
platform-specific. Webcam index may differ.

**Does anything get uploaded?** No. There is no network call in this project except the one-time
model download from Google's public model host. Your video never leaves your machine. 🔒

**Can I use it on client work?** Yes, MIT licensed. Do what you like.

**Why not just use an app?** Use the app if you want the effect once. Use this if you want to
change it — your own world, your own colour, your own timing, an exported matte, batch runs over
a folder, and no watermark. 😎

**My portal does not appear.** Both hands have to be in frame. The run prints what percentage of
frames saw both — if that number is low, that is your answer. See
[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

---

## 📜 Licence

MIT. Build something cool with it. ⭐

If you make something good with this, star the repo — it genuinely helps.
