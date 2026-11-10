# 🎬 Recipes

Copy, paste, change one thing.

---

## The one to run first

```bash
python -m handportal --input 0 --preview
```

Webcam, live, default world. Hands up, palms to camera, pull apart.

---

## Pick a world, live

```bash
python -m handportal -i 0 --preview -w matrix
python -m handportal -i 0 --preview -w thermal
python -m handportal -i 0 --preview -w skeleton -c cyan
python -m handportal -i 0 --preview -w comic -c amber
```

`skeleton` is also the debug view — it draws exactly what the tracker sees.

---

## Render a clip

```bash
python -m handportal -i myclip.mp4 -w storm --lightning -o out/portal.mp4
```

---

## The full cinematic version

```bash
python -m handportal \
  -i myclip.mp4 \
  -w storm:magenta \
  --lightning \
  --color magenta \
  --thickness-max 0.15 \
  --reveal-at 14.3 \
  --reveal-over 0.7 \
  -o out/reel.mp4
```

The portal breathes with the hands for the whole clip, then swallows the frame at 14.3 s.

---

## Your own footage inside the portal

```bash
python -m handportal -i me.mp4 -w video:city_drone.mp4 -o out/portal.mp4
python -m handportal -i me.mp4 -w image:poster.png   -o out/portal.mp4
```

The video loops if it is shorter than the clip.

---

## Export a matte for your editor

```bash
python -m handportal -i me.mp4 -w storm -o out/portal.mp4 --matte out/matte.mp4
```

`matte.mp4` is the portal in black and white, frame-synced. Use it as a track matte in Premiere,
CapCut, Resolve or After Effects and put whatever you want behind it.

---

## Dial the look in quickly

```bash
python -m handportal -i me.mp4 -w storm -o out/test.mp4 --max-frames 90
```

90 frames renders in about 10 seconds. Get the look right, then drop `--max-frames`.

---

## Colour matching

```bash
python -m handportal -i me.mp4 -w storm:cyan  -c cyan    -o out/cyan.mp4
python -m handportal -i me.mp4 -w storm:green -c lime    -o out/green.mp4
python -m handportal -i me.mp4 -w storm:amber -c "#ffaa22" -o out/amber.mp4
```

Match the rim to the world. A magenta rim on a green world reads as two effects stacked.

---

## Batch a folder

**Windows (PowerShell):**

```powershell
Get-ChildItem clips\*.mp4 | ForEach-Object {
    python -m handportal -i $_.FullName -w storm --lightning -o "out\$($_.BaseName)_portal.mp4"
}
```

**macOS / Linux:**

```bash
for f in clips/*.mp4; do
  python -m handportal -i "$f" -w storm --lightning -o "out/$(basename "${f%.*}")_portal.mp4"
done
```

---

## Slimmer, sharper, no glow

```bash
python -m handportal -i me.mp4 -w edges --thickness-max 0.10 --feather 2 --glow 0 --rim 2 -o out/clean.mp4
```

---

## Fastest possible settings

```bash
python -m handportal -i me.mp4 -w thermal --glow 0 --feather 3 -o out/fast.mp4
```

Roughly twice the speed of the default, because `thermal` costs 2 ms a frame against 15 for
`storm`, and the glow pass is skipped.
