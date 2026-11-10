"""Turn two palm positions into a portal rectangle.

The portal is a rectangle, not an axis-aligned box:

    * its long axis is the line between the two palms, so it tilts as the hands tilt
    * its length follows the distance between the palms, so it opens and closes
    * its thickness grows as the hands spread, so hands together give a thin slit
      and hands thrown wide swallow the whole frame

Everything is computed in pixels and every value is smoothed before use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .smoothing import OneEuro, OneEuroAngle, OneEuroPoint
from .tracker import Frame


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """Classic smoothstep. Returns 0 below edge0, 1 above edge1, an S-curve between."""
    if edge1 <= edge0:
        return 0.0 if x < edge0 else 1.0
    t = (x - edge0) / (edge1 - edge0)
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


@dataclass
class PortalConfig:
    """Every number you can turn. Defaults are tuned for a 9:16 phone video."""

    # How far past each palm the rectangle reaches, as a fraction of palm distance.
    # A little overhang hides the seam behind the hand itself.
    overhang: float = 0.10

    # Thickness, as a fraction of frame height, at the closed and the wide ends.
    # Keep these small. The effect reads as a slit cut through the world; once the
    # band is taller than about a quarter of the frame it stops looking like a cut
    # and starts looking like a coloured box pasted on top.
    thickness_min: float = 0.04
    thickness_max: float = 0.17

    # Palm distance, as a fraction of frame width, that maps to those two ends.
    spread_closed: float = 0.22
    spread_wide: float = 0.80

    # The final reveal, where the portal swallows the whole frame.
    #
    # It is OFF by default, and that is a deliberate choice learned the hard way:
    # a hand spread wide enough to trigger it (0.85 of the frame width) is just
    # what "palms up to camera" looks like, so driving the reveal off spread alone
    # fires it constantly, in the middle of ordinary gestures.
    #
    # Two ways to switch it on:
    #   reveal_at   seconds into the clip. Deterministic, repeatable, editable.
    #   spread_full hand distance that triggers it. Set it above spread_wide.
    reveal_at: float | None = None
    reveal_over: float = 0.6
    spread_full: float = 0.0  # 0 = never trigger on spread
    thickness_full: float = 1.60

    # The portal fades in across this palm-distance range (fraction of frame width).
    appear_at: float = 0.09
    appear_over: float = 0.07

    # Frames to keep the last portal alive after a hand is lost, and the fade out.
    hold_frames: int = 6
    fade_frames: int = 4

    # One Euro settings. Raise beta if the portal lags behind fast hands;
    # lower min_cutoff if it trembles when the hands are still.
    min_cutoff: float = 1.2
    beta: float = 0.02

    # Clamp how fast the portal may rotate, in degrees per second. MediaPipe
    # sometimes swaps which hand is which for a frame or two; without this the
    # portal snaps through 90 degrees and back.
    max_spin_dps: float = 420.0


@dataclass
class Portal:
    """The portal for a single frame."""

    present: bool
    center: tuple[float, float] = (0.0, 0.0)
    angle: float = 0.0  # radians, 0 = horizontal
    half_length: float = 0.0  # along the hand-to-hand axis
    half_thickness: float = 0.0  # across it
    opacity: float = 0.0  # 0..1
    openness: float = 0.0  # 0 = slit, 1 = fully spread

    def corners(self) -> np.ndarray:
        """Four corners, clockwise, as float32 (4, 2)."""
        ca, sa = math.cos(self.angle), math.sin(self.angle)
        ux, uy = ca, sa  # along the axis
        vx, vy = -sa, ca  # across it
        cx, cy = self.center
        hl, ht = self.half_length, self.half_thickness
        return np.array(
            [
                [cx - ux * hl - vx * ht, cy - uy * hl - vy * ht],
                [cx + ux * hl - vx * ht, cy + uy * hl - vy * ht],
                [cx + ux * hl + vx * ht, cy + uy * hl + vy * ht],
                [cx - ux * hl + vx * ht, cy - uy * hl + vy * ht],
            ],
            dtype=np.float32,
        )


@dataclass
class PortalTracker:
    """Stateful: feed it one tracker Frame per video frame, get one Portal back."""

    width: int
    height: int
    config: PortalConfig = field(default_factory=PortalConfig)

    def __post_init__(self) -> None:
        c = self.config
        self._f_center = OneEuroPoint(c.min_cutoff, c.beta)
        self._f_angle = OneEuroAngle(c.min_cutoff, c.beta)
        self._f_length = OneEuro(c.min_cutoff, c.beta)
        self._last_left: tuple[float, float] | None = None
        self._last_right: tuple[float, float] | None = None
        self._missing = 0
        self._last_portal = Portal(present=False)
        self._last_angle: float | None = None

    def reset(self) -> None:
        self.__post_init__()

    def _time_burst(self, t: float | None) -> float:
        c = self.config
        if c.reveal_at is None or t is None:
            return 0.0
        return smoothstep(c.reveal_at, c.reveal_at + max(c.reveal_over, 1e-3), t)

    def update(self, frame: Frame, dt: float, t: float | None = None) -> Portal:
        time_burst = self._time_burst(t)
        left = frame.left.palm if frame.left else None
        right = frame.right.palm if frame.right else None

        # Count the miss against what was actually DETECTED, before any stand-in
        # is filled in. Counting it afterwards means the counter never advances,
        # the hold window never expires, and the portal hangs on screen forever
        # after the hands have gone.
        if left is None or right is None:
            self._missing += 1
        else:
            self._missing = 0

        # A hand that blinks out for a few frames keeps its last position, so the
        # portal does not pop. Past hold_frames it fades away instead - unless a
        # timed reveal is running, which must survive the hands leaving the shot.
        stand_in = self._missing <= self.config.hold_frames or time_burst > 0.0
        if stand_in:
            if left is None:
                left = self._last_left
            if right is None:
                right = self._last_right

        if left is None or right is None:
            return self._fade_out()

        self._last_left, self._last_right = left, right

        cx, cy = (left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0
        dx, dy = right[0] - left[0], right[1] - left[1]
        raw_dist = math.hypot(dx, dy)
        raw_angle = math.atan2(dy, dx)

        cx, cy = self._f_center((cx, cy), dt)
        angle = self._f_angle(raw_angle, dt)
        dist = self._f_length(raw_dist, dt)

        angle = self._limit_spin(angle, dt)

        spread_n = dist / max(self.width, 1)
        # Two stages. Most of the time the band just breathes between thickness_min
        # and thickness_max. Past spread_full it blows open into the final reveal.
        openness = smoothstep(self.config.spread_closed, self.config.spread_wide, spread_n)
        burst = time_burst
        if self.config.spread_full > self.config.spread_wide:
            burst = max(burst, smoothstep(self.config.spread_wide, self.config.spread_full, spread_n))

        half_length = dist * 0.5 * (1.0 + self.config.overhang)
        thickness = self.config.thickness_min + (
            self.config.thickness_max - self.config.thickness_min
        ) * openness
        thickness += (self.config.thickness_full - self.config.thickness_max) * burst
        half_thickness = thickness * self.height * 0.5
        # At the very end the band must also be long enough to clear the frame,
        # even if the hands never quite reach the edges.
        if burst > 0.0:
            reach = math.hypot(self.width, self.height) * 0.75
            half_length = half_length * (1.0 - burst) + max(half_length, reach) * burst

        opacity = smoothstep(
            self.config.appear_at,
            self.config.appear_at + self.config.appear_over,
            spread_n,
        )
        opacity = max(opacity, burst)

        self._last_portal = Portal(
            present=opacity > 0.001,
            center=(cx, cy),
            angle=angle,
            half_length=half_length,
            half_thickness=half_thickness,
            opacity=opacity,
            openness=openness,
        )
        return self._last_portal

    # -- internals ---------------------------------------------------------
    def _limit_spin(self, angle: float, dt: float) -> float:
        if self._last_angle is None:
            self._last_angle = angle
            return angle
        limit = math.radians(self.config.max_spin_dps) * max(dt, 1e-3)
        delta = math.atan2(math.sin(angle - self._last_angle), math.cos(angle - self._last_angle))
        if abs(delta) > limit:
            delta = math.copysign(limit, delta)
        self._last_angle = self._last_angle + delta
        return self._last_angle

    def _fade_out(self) -> Portal:
        over = self._missing - self.config.hold_frames
        if over <= 0:
            return self._last_portal
        k = 1.0 - over / max(self.config.fade_frames, 1)
        if k <= 0.0:
            self._last_left = self._last_right = None
            self._last_portal = Portal(present=False)
            return self._last_portal
        p = self._last_portal
        self._last_portal = Portal(
            present=True,
            center=p.center,
            angle=p.angle,
            half_length=p.half_length,
            half_thickness=p.half_thickness,
            opacity=p.opacity * k,
            openness=p.openness,
        )
        return self._last_portal
