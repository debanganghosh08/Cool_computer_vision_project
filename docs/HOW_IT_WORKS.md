# 🧠 How it works

The effect looks like compositing magic. It is four small problems, and three of them have a
wrong obvious answer that costs you a day if you take it.

---

## The pipeline

```
frame ──> HandLandmarker ──> 2 palm points ──> One Euro filter ──> rotated rectangle
                                                                          │
                                                                          ▼
                                               world.render(frame) ──> alpha mask ──> composite
```

One file each:

| File | Job |
|---|---|
| `tracker.py` | MediaPipe in, two stable palm points out |
| `smoothing.py` | One Euro filter |
| `portal.py` | two points in, one rectangle out |
| `worlds.py` | what is inside the portal |
| `compositor.py` | the mask, the glow, the blend |

---

## 1. Finding the hands

MediaPipe's `HandLandmarker` returns 21 landmarks per hand, normalised to 0..1, which we
multiply by the frame size to get pixels. We use the **VIDEO** running mode, which carries
tracking state between frames and is both faster and steadier than running detection fresh on
every frame. It needs a monotonic timestamp in milliseconds, which is why the loop passes
`index * 1000 / fps` rather than a wall clock.

### 🤚 Wrong answer #1: anchoring to the wrist

Landmark 0 is the wrist and it is the obvious anchor. It is also the worst one. It sits at the
very edge of the hand, so when your forearm rolls — which it does constantly, because that is
how people gesture — it swings through a large arc while the palm barely moves. A portal pinned
to it slides and wobbles.

The fix is three lines:

```python
PALM_POINTS = (0, 5, 9, 13, 17)        # wrist + the knuckle of each finger
palm = pts[list(PALM_POINTS)].mean(axis=0)
```

That point sits in the middle of the palm and is nearly invariant to hand rotation.

### 🔄 Wrong answer #2: trusting handedness

MediaPipe labels each hand "Left" or "Right". Use that to decide which end of the portal is
which and it will break, because the label is relative to the real hand, not the picture — so it
flips the moment the image is mirrored, which selfie cameras do by default. A portal whose ends
swap for a frame snaps through 90 degrees.

```python
found.sort(key=lambda hand: hand.palm[0])   # sort by x on screen
```

Screen position cannot flip. Boring beats clever.

---

## 2. Smoothing

Raw landmarks jitter by a few pixels per frame even on a completely still hand. Untreated, the
portal edge shimmers and the whole thing looks cheap.

### 📉 Wrong answer #3: a moving average

Average the last N frames and the jitter goes — along with your timing. The portal now trails
your hands by N/2 frames, and because the viewer is watching your hands *and* the portal in the
same picture, that lag is the single most visible way to give the trick away.

The **One Euro filter** (Casiez, Roussel & Vogel, CHI 2012) is a low-pass filter whose cutoff
frequency rises with the speed of the signal:

```python
cutoff = min_cutoff + beta * abs(speed_estimate)
```

Still hand → low speed → low cutoff → heavy smoothing → no jitter.
Fast hand → high speed → high cutoff → light smoothing → no lag.

Two knobs, both exposed on the command line:

- `--min-cutoff` lower = smoother when still, slightly laggier
- `--beta` higher = less lag when moving fast

### Angles need care

Filtering an angle directly explodes at the ±π wrap: the value jumps by 2π, the filter sees an
enormous "movement", and the portal spins a full turn. `OneEuroAngle` filters `cos` and `sin`
separately and recombines with `atan2`, so the wrap is simply not representable.

There is also a hard limit on rotation speed (`max_spin_dps`, 420°/s). If MediaPipe briefly
mixes up which hand is which, that clamp turns a violent snap into a short lean.

---

## 3. The rectangle

Given two smoothed palm points:

```
axis            = right − left
angle           = atan2(axis.y, axis.x)
half_length     = |axis| / 2 × (1 + overhang)
spread          = |axis| / frame_width
openness        = smoothstep(spread_closed, spread_wide, spread)
thickness       = thickness_min + (thickness_max − thickness_min) × openness
```

`overhang` pushes the ends slightly past the palms so the seam hides behind the hand itself.

`smoothstep` rather than a straight line, so the band eases in and out instead of changing at a
constant rate — the difference between a mechanism and a move.

### 📐 Wrong answer #5: pinning the box to the palms

The first version ran a band through both palm centres and let it reach 10% past them. It
looked fine in a still and wrong in a real clip: the band sat ON the hands, covered them, and
poked out past the fingertips. The effect people expect is a box BETWEEN the hands, the
"director's frame" you make with your index fingers and thumbs.

So the default anchor is now the fingertips. Each hand gives two corners, its index tip
(landmark 8) and its thumb tip (landmark 4). The box is left index → right index → right thumb →
left thumb. When your fingertips touch it is a sliver; as they part it opens, and its corners
are your fingertips. Three things it needed, all found on real footage:

- **Faster smoothing.** Fingertips move far faster than palms. The palm settings
  (min_cutoff 1.2, beta 0.02) left a corner 13 px behind a moving fingertip on average and 25 px
  at worst. Grid-tested on real clips, 1.0 / 0.2 gets that to about 2 px (under 5 px at p95),
  and still shakes less than the raw landmarks do.
- **Top and bottom decided per frame, after smoothing.** The four labelled tips are smoothed
  first, then each hand's upper tip (against the normal of the left-to-right axis) becomes a top
  corner. The box can never fold into a bow-tie, however the hands rotate.
- **No guessing at the edge of the frame.** A hand that is half or more outside the frame still
  gets landmarks, but the off-screen ones are guesses: one index tip jumped 170 px in a single
  frame. Such a hand no longer counts. If a hand leaves fast, the whole last good box fades over
  a few frames instead of half of it freezing while the other half follows the other hand (that
  made a wedge). A slow hand that drops out for a frame still gets the usual hold. When the box
  comes back it fades in, and its corner filters start fresh.

Measured on a 15 s clip after all of this: every corner within 10 px of the real fingertip,
1.6 px on average. `--anchor palms` keeps the old band.

### ⏱️ Wrong answer #4: triggering the full reveal on hand spread

This one only shows up on real footage. The natural ending is the portal swallowing the frame,
and the natural trigger looks like "hands very wide". So: when spread goes past ~0.9 of the
frame width, blow the rectangle up to cover everything.

Measured on an actual clip, an ordinary "palms up toward the camera" pose sits at **spread 0.85**,
and a relaxed arms-apart gesture reaches **0.88**. The reveal fired every few seconds, in the
middle of normal talking.

So the reveal is a **timed beat** by default:

```bash
--reveal-at 14.3 --reveal-over 0.6
```

Deterministic, repeatable, and editable to the frame — which is what you want anyway, because
the reveal is supposed to land on a word or a beat, not on however wide your arms happened to
go. The spread trigger is still there (`--spread-full`) for anyone who wants it, switched off by
default.

### When a hand disappears

Hands leave frame and MediaPipe drops them for a frame or two constantly. The portal keeps the
last known position for `--hold` frames (default 6), then fades over 4 more.

With `--anchor palms`, unless the hand left **through the edge**. Swing your arms out wide and both hands leave the
frame on purpose; a portal that vanishes there looks like a glitch. So a hand last seen within
15% of a border is pinned just outside that border for up to 2 seconds, and the band keeps
stretching off-screen toward it. A hand lost in the middle of the frame (motion blur, a bad
frame) is not pinned, so a tracking hiccup can never freeze the portal in place.

The default fingertip box does not do this. Its corners ARE the fingertips; once a hand has
left the frame there is nothing to pin a corner to, and pinning anyway bent the box into a
wedge. So it fades out, and comes back the moment the hands are back in shot.

The subtle bug here, and it was in this code: the miss has to be counted against what was
**actually detected**, before the stand-in position is filled in. Count it afterwards and the
counter never advances, the hold window never expires, and the portal hangs on screen forever
after you have walked away. There is a test for it.

---

## 4. Compositing

```python
alpha = fillConvexPoly(corners) ──> blur by `feather`
out   = base × (1 − alpha) + world × alpha
```

Then the edge treatment, in this order: the outward **glow** (blur the hard mask, subtract the
hard mask, colourise the halo), the **bolts**, and finally the **rim** stroked straight onto the
output with `polylines`.

### 🚀 The performance story

The first working version ran at **2.9 fps** and the compositor was 60% of it. The guess was
that the wide Gaussian blurs were to blame. They were not — profiling said otherwise:

```
numpy  base*(1-m) + world*m      27 ms
numpy  out += colour*halo*k      30 ms
cv2    equivalent 8-bit blend     5 ms
```

Full-frame float32 expressions in numpy are simply expensive at 720x1280, and the compositor had
six of them per frame. Rewriting the same maths in 8-bit OpenCV calls — `cv2.multiply`,
`cv2.add`, `cv2.convertScaleAbs` — took compositing from **204 ms to 20 ms** and the whole
pipeline from 2.9 to **11 fps**. 8 bits is plenty of precision for a mask.

Two smaller wins on the same theme:

- `soft_blur()` does wide blurs at quarter resolution. A blur that wide has no fine detail left
  in it by definition, so shrinking first is free in quality and about 16x faster.
- The bolt clip mask used a blur radius that scaled with portal thickness. During the full
  reveal that became a 500-pixel kernel on a full frame. It is a fixed small radius now — the
  bolts are drawn on the edge, so it never needed to scale.

The remaining 54 ms is MediaPipe, which is where the time should be going.

---

## 5. Worlds

A world answers one question: *what should this frame look like on the other side?*

```python
class World:
    def render(self, ctx: WorldContext) -> np.ndarray: ...
```

`ctx` carries the frame, the frame index, the time in seconds, the size, and the hand landmarks.

The important design decision: **most worlds restyle the frame you already have** rather than
playing a second video. That is what makes this a one-take effect on one phone. `storm` does not
paint flat colour into the portal — flat colour reads as a sticker — it crushes the frame to a
two-colour duotone and lights it from a storm overhead, so you still see the person and the room
through the window, recoloured. That is what makes a viewer believe it is somewhere else.

`video:` and `image:` are there for when you do have a second layer.

---

## 📚 Reference

- MediaPipe Hand Landmarker — https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
- Casiez, Roussel & Vogel, *1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in
  Interactive Systems*, CHI 2012 — https://gery.casiez.net/1euro/
