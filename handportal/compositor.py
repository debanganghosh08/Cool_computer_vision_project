"""Build the portal mask and blend the two worlds through it."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .portal import Portal

# Colours are (B, G, R) because that is what OpenCV uses.
PRESETS: dict[str, tuple[int, int, int]] = {
    "magenta": (200, 40, 255),
    "cyan": (255, 220, 60),
    "lime": (90, 255, 120),
    "amber": (40, 170, 255),
    "white": (255, 255, 255),
    "violet": (255, 70, 150),
    "red": (60, 60, 255),
}


def parse_color(value: str) -> tuple[int, int, int]:
    """Accept a preset name, '#rrggbb', or 'r,g,b'. Returns BGR."""
    v = value.strip().lower()
    if v in PRESETS:
        return PRESETS[v]
    if v.startswith("#"):
        v = v[1:]
    if len(v) == 6 and all(c in "0123456789abcdef" for c in v):
        r, g, b = int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)
        return (b, g, r)
    if "," in v:
        parts = [int(p) for p in v.split(",")]
        if len(parts) == 3:
            r, g, b = parts
            return (b, g, r)
    raise ValueError(
        f"Could not read the colour {value!r}. "
        f"Use one of {', '.join(sorted(PRESETS))}, or #ff33cc, or 255,51,204."
    )


@dataclass
class EdgeStyle:
    color: tuple[int, int, int] = PRESETS["magenta"]
    feather: int = 7  # soft edge on the mask, in pixels
    glow: float = 1.0  # 0 turns the glow off
    glow_width: int = 26  # how far the glow bleeds outward, in pixels
    rim: int = 3  # hard bright line right on the edge, in pixels, 0 for none
    lightning: bool = False
    lightning_amount: float = 1.0


def soft_blur(img: np.ndarray, sigma: float, max_sigma_full_res: float = 4.0) -> np.ndarray:
    """A wide, soft blur, done cheaply.

    A Gaussian is O(sigma) per pixel, so a wide glow blurred at full resolution
    costs more than everything else in the frame put together - on a 720x1280
    frame a sigma of 14 was taking over 100 ms on its own. A wide blur has no
    fine detail left in it by definition, so shrink first, blur small, grow back.
    The result is visually identical and roughly 16x faster.
    """
    if sigma <= 0.05:
        return img.copy()
    if sigma <= max_sigma_full_res:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    h, w = img.shape[:2]
    scale = min(4, max(2, int(sigma / max_sigma_full_res)))
    small = cv2.resize(img, (max(2, w // scale), max(2, h // scale)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / scale)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def portal_mask_u8(portal: Portal, size: tuple[int, int], feather: int = 7) -> np.ndarray:
    """The portal as an 8-bit mask, 0..255.

    Everything downstream stays in 8 bits on purpose. The obvious way to write a
    compositor is in float32, and it is also roughly five times slower: a single
    full-frame float expression on a 720x1280 frame costs about 27 ms in numpy
    against 5 ms for the same blend through OpenCV in 8 bits, and there are half
    a dozen of them per frame. 8 bits is plenty for a mask.
    """
    h, w = size
    mask = np.zeros((h, w), dtype=np.uint8)
    if not portal.present or portal.half_length <= 0.5 or portal.half_thickness <= 0.5:
        return mask
    cv2.fillPoly(mask, [np.round(portal.corners()).astype(np.int32)], 255, lineType=cv2.LINE_AA)
    if feather > 0:
        mask = soft_blur(mask, feather * 0.8)
    if portal.opacity < 1.0:
        mask = cv2.convertScaleAbs(mask, alpha=float(portal.opacity))
    return mask


def portal_mask(portal: Portal, size: tuple[int, int], feather: int = 7) -> np.ndarray:
    """The same mask as float32 in 0..1, for callers that want it that way."""
    return portal_mask_u8(portal, size, feather).astype(np.float32) / 255.0


_SOLID_CACHE: dict[tuple[int, int, tuple[int, int, int]], np.ndarray] = {}


def _solid(size: tuple[int, int], color: tuple[int, int, int]) -> np.ndarray:
    key = (size[0], size[1], color)
    got = _SOLID_CACHE.get(key)
    if got is None:
        got = np.full((size[0], size[1], 3), color, dtype=np.uint8)
        if len(_SOLID_CACHE) > 8:
            _SOLID_CACHE.clear()
        _SOLID_CACHE[key] = got
    return got


def _lightning_on_edges(q, color, rng, amount, layer):
    """Bolts along the top (q0 to q1) and bottom (q3 to q2) edges of a four-corner portal."""
    width = float(np.linalg.norm((q[0] + q[1]) / 2 - (q[3] + q[2]) / 2))
    if width < 12 or float(np.linalg.norm(q[1] - q[0])) < 40:
        return layer
    for _ in range(max(1, int(round(3 * amount)))):
        a, b = (q[0], q[1]) if rng.random() < 0.5 else (q[3], q[2])
        d = b - a
        n = np.array([-d[1], d[0]], dtype=np.float32) / max(float(np.linalg.norm(d)), 1.0)
        start = rng.uniform(0.0, 0.6)
        length = rng.uniform(0.2, 0.5)
        pts = [a + d * min(1.0, start + length * i / 14) + n * rng.normal(0.0, width * 0.07) for i in range(15)]
        poly = np.round(np.array(pts)).astype(np.int32)
        cv2.polylines(layer, [poly], False, color, thickness=2, lineType=cv2.LINE_AA)
        cv2.polylines(layer, [poly], False, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    return cv2.GaussianBlur(layer, (0, 0), 3.0)


def _lightning_overlay(
    portal: Portal,
    size: tuple[int, int],
    color: tuple[int, int, int],
    rng: np.random.Generator,
    amount: float,
) -> np.ndarray:
    """Jagged bolts crawling along the two long edges of the portal."""
    h, w = size
    layer = np.zeros((h, w, 3), dtype=np.uint8)
    if portal.quad is not None:
        return _lightning_on_edges(portal.corners(), color, rng, amount, layer)
    if portal.half_length < 20 or portal.half_thickness < 6:
        return layer
    ca, sa = math.cos(portal.angle), math.sin(portal.angle)
    ux, uy = ca, sa
    vx, vy = -sa, ca
    cx, cy = portal.center
    bolts = max(1, int(round(3 * amount)))
    for _ in range(bolts):
        side = 1.0 if rng.random() < 0.5 else -1.0
        start = rng.uniform(-0.95, 0.2)
        length = rng.uniform(0.25, 0.8)
        steps = 14
        pts: list[list[int]] = []
        for i in range(steps + 1):
            t = start + length * (i / steps)
            if t > 1.0:
                break
            along = t * portal.half_length
            jitter = rng.normal(0.0, portal.half_thickness * 0.14)
            across = side * portal.half_thickness + jitter
            pts.append([int(cx + ux * along + vx * across), int(cy + uy * along + vy * across)])
        if len(pts) < 2:
            continue
        poly = np.array(pts, dtype=np.int32)
        cv2.polylines(layer, [poly], False, color, thickness=2, lineType=cv2.LINE_AA)
        cv2.polylines(layer, [poly], False, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    return cv2.GaussianBlur(layer, (0, 0), 3.0)


def compose(
    base: np.ndarray,
    world: np.ndarray,
    portal: Portal,
    style: EdgeStyle,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend world into base through the portal.

    Returns (composited BGR frame, float mask 0..1) so the caller can also write
    the mask out as an alpha matte for an external editor.
    """
    h, w = base.shape[:2]
    mask = portal_mask_u8(portal, (h, w), style.feather)

    if world.shape[:2] != (h, w):
        world = cv2.resize(world, (w, h), interpolation=cv2.INTER_LINEAR)

    if not portal.present:
        return base.copy(), mask

    alpha = cv2.merge([mask, mask, mask])
    inv = cv2.bitwise_not(alpha)
    out = cv2.add(
        cv2.multiply(base, inv, scale=1 / 255.0),
        cv2.multiply(world, alpha, scale=1 / 255.0),
    )

    if style.glow > 0 or style.rim > 0 or style.lightning:
        hard = portal_mask_u8(portal, (h, w), feather=0)

        if style.glow > 0 and style.glow_width > 0:
            spread = soft_blur(hard, style.glow_width / 3.0)
            halo = cv2.subtract(spread, hard)  # saturating, so no clipping needed
            halo3 = cv2.merge([halo, halo, halo])
            out = cv2.add(
                out,
                cv2.multiply(halo3, _solid((h, w), style.color), scale=style.glow * 1.35 / 255.0),
            )

        if style.lightning:
            gen = rng if rng is not None else np.random.default_rng()
            bolts = _lightning_overlay(portal, (h, w), style.color, gen, style.lightning_amount)
            # Clip the bolts to a band hugging the portal edge. Without this they
            # trail off into the middle of the shot and read as random scribbles.
            # Fixed width on purpose: the bolts are drawn on the edge, so this
            # never needs to scale with the portal, and scaling it with portal
            # thickness is what made this the slowest line in the program.
            near = soft_blur(hard, 9.0)
            near3 = cv2.merge([near, near, near])
            out = cv2.add(out, cv2.multiply(bolts, near3, scale=2.2 * portal.opacity / 255.0))

        if style.rim > 0:
            # Stroke the four sides directly. Building a rim by eroding a mask and
            # blending costs two more full-frame passes to light a few thousand
            # pixels; drawing the outline is exact and essentially free.
            poly = portal.corners().astype(np.int32)
            cv2.polylines(out, [poly], True, style.color, style.rim, cv2.LINE_AA)

    return out, mask
