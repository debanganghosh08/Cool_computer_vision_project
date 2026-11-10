"""What you see inside the portal.

A world is anything that can answer "give me a full-size frame for time t".
Most of them restyle the camera frame itself, which is the whole point: you
shoot once, on one phone, and the other world is generated, not filmed.

Add your own: write a class with a render(ctx) method and register it in BUILTIN.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .compositor import soft_blur


@dataclass
class WorldContext:
    """Everything a world is allowed to know about the current frame."""

    frame: np.ndarray  # the untouched camera frame, BGR
    index: int  # frame number, from 0
    time: float  # seconds since the start
    size: tuple[int, int]  # (height, width)
    hands: list[np.ndarray]  # 0-2 arrays of (21, 2) landmarks in pixels


class World:
    name = "base"

    def render(self, ctx: WorldContext) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------
# Worlds built out of the camera frame
# --------------------------------------------------------------------------


class ThermalWorld(World):
    """Heat-camera look. Cheap, instantly readable, always lands."""

    name = "thermal"

    def render(self, ctx: WorldContext) -> np.ndarray:
        gray = cv2.cvtColor(ctx.frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)


class EdgesWorld(World):
    """Line-art x-ray. Canny edges glowing on near-black."""

    name = "edges"

    def render(self, ctx: WorldContext) -> np.ndarray:
        gray = cv2.cvtColor(ctx.frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 55, 140)
        glow = soft_blur(cv2.dilate(edges, np.ones((2, 2), np.uint8)), 3.0)
        line = cv2.convertScaleAbs(edges, alpha=0.9)
        return cv2.merge([
            line,
            cv2.add(edges, cv2.convertScaleAbs(glow, alpha=0.55)),
            cv2.add(line, cv2.convertScaleAbs(glow, alpha=0.25)),
        ])


class BlueprintWorld(World):
    """Engineering drawing: cyan lines and a measured grid on navy."""

    name = "blueprint"

    def render(self, ctx: WorldContext) -> np.ndarray:
        h, w = ctx.size
        out = np.full((h, w, 3), (90, 40, 12), dtype=np.uint8)  # deep navy, BGR
        step = max(24, h // 26)
        offset = int(ctx.time * 14) % step
        for x in range(-step + offset, w, step):
            cv2.line(out, (x, 0), (x, h), (120, 80, 30), 1, cv2.LINE_AA)
        for y in range(-step + offset, h, step):
            cv2.line(out, (0, y), (w, y), (120, 80, 30), 1, cv2.LINE_AA)
        gray = cv2.cvtColor(ctx.frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 45, 120)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
        out[edges > 0] = (255, 235, 150)
        return out


class ComicWorld(World):
    """Printed-comic look: flat colour, hard ink lines, halftone dots."""

    name = "comic"

    def __init__(self, dot: int = 6) -> None:
        self.dot = max(3, dot)
        self._screen: np.ndarray | None = None
        # Posterising with (x // 42) * 42 + 21 is a full-frame integer divide.
        # The same answer comes out of a 256-entry lookup table far quicker.
        self._poster = ((np.arange(256, dtype=np.uint8) // 42) * 42 + 21).astype(np.uint8)

    def _halftone(self, size: tuple[int, int]) -> np.ndarray:
        if self._screen is not None and self._screen.shape == size:
            return self._screen
        h, w = size
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        period = float(self.dot)
        wave = np.sin(xx * math.pi / period) * np.sin(yy * math.pi / period)
        shade = 1.0 - (wave > 0.35).astype(np.float32) * 0.22
        self._screen = cv2.cvtColor((shade * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        return self._screen

    def render(self, ctx: WorldContext) -> np.ndarray:
        # Bilateral filtering is the expensive part and its whole job is to throw
        # detail away, so run it at half size and scale the result back up.
        h, w = ctx.size
        small = cv2.resize(ctx.frame, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        flat = cv2.resize(
            cv2.bilateralFilter(small, 7, 120, 120), (w, h), interpolation=cv2.INTER_LINEAR
        )
        gray = cv2.cvtColor(ctx.frame, cv2.COLOR_BGR2GRAY)
        ink = cv2.adaptiveThreshold(
            cv2.medianBlur(gray, 5), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 6
        )
        out = cv2.multiply(cv2.LUT(flat, self._poster), self._halftone(ctx.size), scale=1 / 255.0)
        out[ink == 0] = 18
        return out


class PixelWorld(World):
    """Game-console look: big pixels, few colours."""

    name = "pixel"

    def __init__(self, blocks: int = 72) -> None:
        self.blocks = max(16, blocks)

    def render(self, ctx: WorldContext) -> np.ndarray:
        h, w = ctx.size
        small_w = self.blocks
        small_h = max(1, int(h * small_w / w))
        small = cv2.resize(ctx.frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
        small = (small // 36) * 36
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


class GlitchWorld(World):
    """Broken-signal look: channel shift, slice tearing, scanlines."""

    name = "glitch"

    def render(self, ctx: WorldContext) -> np.ndarray:
        h, w = ctx.size
        out = ctx.frame.copy()
        shift = int(6 + 5 * math.sin(ctx.time * 7.3))
        out[:, :, 2] = np.roll(out[:, :, 2], shift, axis=1)
        out[:, :, 0] = np.roll(out[:, :, 0], -shift, axis=1)
        rng = np.random.default_rng(ctx.index)
        for _ in range(4):
            y = int(rng.integers(0, max(1, h - 20)))
            band = int(rng.integers(6, 26))
            out[y : y + band] = np.roll(out[y : y + band], int(rng.integers(-40, 40)), axis=1)
        out[::3] = (out[::3] * 0.72).astype(np.uint8)
        return out


class MatrixWorld(World):
    """Falling code. Columns of characters, brightest at the head."""

    name = "matrix"

    GLYPHS = "01<>[]{}/|=+*#$%&ABCDEFGHJKLMNPQRSTUVWXYZ"

    def __init__(self, cols: int = 42, speed: float = 16.0) -> None:
        self.cols = max(10, cols)
        self.speed = speed
        self._state: dict | None = None

    def _init_state(self, size: tuple[int, int]) -> None:
        h, w = size
        rng = np.random.default_rng(7)
        cell = w / self.cols
        rows = int(h / (cell * 1.12)) + 2
        self._state = {
            "size": size,
            "cell": cell,
            "rows": rows,
            "head": rng.uniform(-rows, 0, self.cols),
            "rate": rng.uniform(0.55, 1.6, self.cols),
            "tail": rng.integers(6, max(7, rows), self.cols),
            "chars": rng.integers(0, len(self.GLYPHS), (rows + 4, self.cols)),
            "rng": rng,
        }

    def render(self, ctx: WorldContext) -> np.ndarray:
        if self._state is None or self._state["size"] != ctx.size:
            self._init_state(ctx.size)
        st = self._state
        assert st is not None
        h, w = ctx.size
        cell = st["cell"]
        rows = st["rows"]
        head = st["head"]
        rate = st["rate"]
        tail = st["tail"]
        chars = st["chars"]
        rng = st["rng"]

        out = np.zeros((h, w, 3), dtype=np.uint8)
        # A ghost of the real frame keeps the body readable behind the code.
        gray = cv2.cvtColor(ctx.frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        out[:, :, 1] = np.clip(gray * 0.30, 0, 255).astype(np.uint8)

        scale = cell / 28.0
        for c in range(self.cols):
            x = int(c * cell + cell * 0.12)
            top = head[c]
            length = int(tail[c])
            for k in range(length):
                r = int(top - k)
                if r < 0 or r >= rows:
                    continue
                y = int(r * cell * 1.12 + cell)
                fade = 1.0 - k / max(length, 1)
                if k == 0:
                    color = (210, 255, 210)
                else:
                    g = int(60 + 175 * fade)
                    color = (int(20 * fade), g, int(45 * fade))
                glyph = self.GLYPHS[int(chars[r % chars.shape[0], c])]
                cv2.putText(
                    out, glyph, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA
                )
            head[c] += rate[c] * self.speed / 30.0
            if head[c] - tail[c] > rows:
                head[c] = -float(rng.integers(0, rows // 2 + 1))
                tail[c] = int(rng.integers(6, max(7, rows)))
                rate[c] = float(rng.uniform(0.55, 1.6))
        if ctx.index % 3 == 0:
            idx = rng.integers(0, chars.size, max(1, chars.size // 40))
            chars.flat[idx] = rng.integers(0, len(self.GLYPHS), idx.size)
        return out


class StormWorld(World):
    """Same room, different sky.

    This is the one that sells the trick, so it is worth saying what it does and
    does not do. It does NOT paint a flat colour into the portal - a flat colour
    reads as a sticker, not a window. It takes the real frame, crushes it to a
    two-colour duotone, and lights it from a storm overhead. You still see the
    person and the room through the portal, recoloured, which is exactly what
    makes a viewer believe they are looking into somewhere else.
    """

    name = "storm"

    # (shadow colour, highlight colour), BGR
    PALETTE = {
        "magenta": ((92, 6, 72), (240, 150, 255)),
        "cyan": ((78, 46, 6), (255, 240, 160)),
        "amber": ((8, 44, 92), (150, 225, 255)),
        "violet": ((110, 14, 52), (255, 175, 215)),
        "green": ((14, 62, 14), (170, 255, 190)),
        "red": ((20, 10, 80), (160, 160, 255)),
    }

    def __init__(self, hue: str = "magenta") -> None:
        self.hue = hue
        self._noise: np.ndarray | None = None
        self._lut: np.ndarray | None = None
        self._sky: np.ndarray | None = None
        self._sky_size: tuple[int, int] | None = None
        # equalizeHist flattens a face into a white blob under hard light. CLAHE
        # lifts local contrast instead and keeps the features readable.
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _get_lut(self) -> np.ndarray:
        """256-entry duotone lookup. Built once, then it is a single cv2.LUT call."""
        if self._lut is not None:
            return self._lut
        lo_t, hi_t = self.PALETTE.get(self.hue, self.PALETTE["magenta"])
        lo = np.array(lo_t, dtype=np.float32)
        hi = np.array(hi_t, dtype=np.float32)
        t = (np.linspace(0.0, 1.0, 256, dtype=np.float32) ** 0.85)[:, None]
        table = lo[None, :] * (1.0 - t) + hi[None, :] * t
        self._lut = np.clip(table, 0, 255).astype(np.uint8).reshape(256, 1, 3)
        return self._lut

    def _get_sky(self, size: tuple[int, int]) -> np.ndarray:
        """A vertical wash, cached, so we are not rebuilding a gradient every frame."""
        if self._sky is not None and self._sky_size == size:
            return self._sky
        h, w = size
        lo_t, hi_t = self.PALETTE.get(self.hue, self.PALETTE["magenta"])
        lo = np.array(lo_t, dtype=np.float32)
        hi = np.array(hi_t, dtype=np.float32)
        ramp = (np.linspace(1.0, 0.0, h, dtype=np.float32) ** 1.5)[:, None, None]
        column = lo[None, None, :] * (1 - ramp) + hi[None, None, :] * ramp
        self._sky = np.repeat(column, w, axis=1).astype(np.uint8)
        self._sky_size = size
        return self._sky

    def _get_noise(self, size: tuple[int, int]) -> np.ndarray:
        want = (max(2, size[0] // 8), max(2, size[1] // 8))
        if self._noise is not None and self._noise.shape[:2] == want:
            return self._noise
        rng = np.random.default_rng(3)
        self._noise = cv2.GaussianBlur(rng.random(want).astype(np.float32), (0, 0), 6)
        return self._noise

    def render(self, ctx: WorldContext) -> np.ndarray:
        h, w = ctx.size
        gray = self._clahe.apply(cv2.cvtColor(ctx.frame, cv2.COLOR_BGR2GRAY))
        duotone = cv2.LUT(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), self._get_lut())

        # Light the duotone with the sky, strongest where the frame is dark, so the
        # background glows and the subject keeps its shape. Every step here stays
        # in 8-bit OpenCV rather than float numpy - see compositor.portal_mask_u8
        # for why that is worth caring about.
        dark = cv2.bitwise_not(gray)
        dark3 = cv2.merge([dark, dark, dark])
        out = cv2.add(
            cv2.convertScaleAbs(duotone, alpha=0.88),
            cv2.multiply(self._get_sky(ctx.size), dark3, scale=0.38 / 255.0),
        )

        noise = self._get_noise(ctx.size)
        dx = int(ctx.time * 22) % max(noise.shape[1], 1)
        clouds = cv2.resize(np.roll(noise, dx, axis=1), (w, h), interpolation=cv2.INTER_LINEAR)
        clouds8 = cv2.cvtColor((clouds * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        out = cv2.addWeighted(out, 1.0, clouds8, 0.22, -28.0)

        rng = np.random.default_rng(ctx.index // 4)
        if rng.random() < 0.45:
            x = float(rng.uniform(w * 0.1, w * 0.9))
            y = 0.0
            pts = [[int(x), 0]]
            while y < h:
                y += rng.uniform(h * 0.07, h * 0.16)
                x += rng.normal(0, w * 0.05)
                pts.append([int(x), int(min(y, h))])
            poly = np.array(pts, dtype=np.int32)
            bolt = np.zeros((h, w), dtype=np.uint8)
            cv2.polylines(bolt, [poly], False, 255, 3, cv2.LINE_AA)
            bolt = soft_blur(bolt, 7.0)
            cv2.polylines(bolt, [poly], False, 255, 1, cv2.LINE_AA)
            out = cv2.addWeighted(out, 1.0, cv2.merge([bolt, bolt, bolt]), 0.75, 0.0)
        return out


class SkeletonWorld(World):
    """The tracking itself, drawn: bones, joints, lit fingertips, on a dark frame."""

    name = "skeleton"

    BONES = (
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17),
    )
    TIPS = (4, 8, 12, 16, 20)

    def __init__(self, color: tuple[int, int, int] = (255, 230, 90)) -> None:
        self.color = color

    def render(self, ctx: WorldContext) -> np.ndarray:
        out = cv2.convertScaleAbs(ctx.frame, alpha=0.22)
        layer = np.zeros_like(out)
        for pts in ctx.hands:
            p = pts.astype(np.int32)
            for a, b in self.BONES:
                cv2.line(layer, tuple(p[a]), tuple(p[b]), self.color, 2, cv2.LINE_AA)
            for i, (x, y) in enumerate(p):
                r = 6 if i in self.TIPS else 3
                cv2.circle(layer, (int(x), int(y)), r, (255, 255, 255), -1, cv2.LINE_AA)
        glow = cv2.convertScaleAbs(soft_blur(layer, 9.0), alpha=0.9)
        return cv2.add(cv2.add(out, glow), layer)


# --------------------------------------------------------------------------
# A world loaded from a file
# --------------------------------------------------------------------------


class MediaWorld(World):
    """Your own video or still image inside the portal. Video loops."""

    name = "media"

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"World media not found: {self.path}")
        self._still: np.ndarray | None = None
        self._cap = None
        still = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
        if still is not None:
            self._still = still
            return
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise ValueError(f"Could not open {self.path} as an image or a video.")
        self._cap = cap

    def render(self, ctx: WorldContext) -> np.ndarray:
        if self._still is not None:
            return self._still
        assert self._cap is not None
        ok, frame = self._cap.read()
        if not ok:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
        if not ok:
            return np.zeros((*ctx.size, 3), dtype=np.uint8)
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

BUILTIN = {
    "storm": StormWorld,
    "matrix": MatrixWorld,
    "thermal": ThermalWorld,
    "edges": EdgesWorld,
    "blueprint": BlueprintWorld,
    "comic": ComicWorld,
    "pixel": PixelWorld,
    "glitch": GlitchWorld,
    "skeleton": SkeletonWorld,
}


def build_world(spec: str) -> World:
    """'matrix', 'storm:cyan', 'video:clip.mp4', 'image:poster.png'."""
    spec = spec.strip()
    if spec.startswith(("video:", "image:", "media:")):
        return MediaWorld(spec.split(":", 1)[1])
    name, _, arg = spec.partition(":")
    name = name.lower()
    if name not in BUILTIN:
        raise ValueError(
            f"Unknown world {spec!r}. Built in: {', '.join(sorted(BUILTIN))}. "
            f"Or use video:<path> / image:<path>."
        )
    cls = BUILTIN[name]
    if not arg:
        return cls()
    if cls is StormWorld:
        return StormWorld(hue=arg)
    if cls is MatrixWorld:
        return MatrixWorld(cols=int(arg))
    if cls is PixelWorld:
        return PixelWorld(blocks=int(arg))
    if cls is ComicWorld:
        return ComicWorld(dot=int(arg))
    return cls()
