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
from dataclasses import dataclass, field, replace

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

    # Palms anchor only. A hand that leaves THROUGH THE EDGE of the frame (arms
    # swung out wide) is pinned just outside that edge, so the band keeps stretching off-screen
    # toward it instead of vanishing. Only a hand last seen within edge_margin
    # of the border counts; a hand lost mid-frame (motion blur, a bad frame)
    # still holds and fades as above. edge_hold_s = 0 turns it off.
    edge_margin: float = 0.15  # fraction of width / height
    edge_hold_s: float = 2.0

    # What the portal is pinned to.
    #   "fingertips" (default): a four-cornered box BETWEEN the hands. Its corners
    #       are each hand's index tip and thumb tip, so it starts exactly at the
    #       fingertips, is a sliver when they touch, and opens as they part. This
    #       is the "director's frame" look.
    #   "palms": the older look, a rotated band through both palm centres that
    #       runs past them by `overhang` and covers the hands.
    anchor: str = "fingertips"
    # Fingertips move much faster than palms. Measured on real footage, the palm
    # settings (1.2 / 0.02) left a corner 13 px behind a moving fingertip on
    # average (25 px at p95); 1.0 / 0.2 gets that to 2 px (5 px p95) while still
    # shaking less than the raw landmarks do.
    tip_min_cutoff: float = 1.0
    tip_beta: float = 0.2
    # A hand lost while its fingertips were moving faster than this (fraction of
    # the frame width per second) has LEFT the shot, it did not blink. Holding its
    # last corners while the other hand keeps moving bent the box into a wedge on
    # real footage (an arm swung out at 85 px a frame), so the whole last box fades
    # instead. Slow hands that drop out for a frame still get hold_frames.
    fast_exit: float = 0.6
    # Frames to fade the portal in when it comes back after being gone.
    appear_frames: int = 4
    # A hand with at least this share of its 21 landmarks outside the frame is
    # leaving, and its fingertips are guesses: on real footage MediaPipe moved an
    # off-screen index tip 170 px in one frame (16 of 21 points outside). In the
    # fingertip anchor such a hand counts as gone, and the box fades from the last
    # frame where both hands were really in shot.
    max_outside: float = 0.5

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
    # Fingertip anchor: the four corners [left top, right top, right bottom,
    # left bottom]. When set it IS the portal's shape; the fields above still
    # describe the hands (centre, angle, spread) for opacity and the reveal.
    quad: np.ndarray | None = None

    def corners(self) -> np.ndarray:
        """Four corners, clockwise, as float32 (4, 2)."""
        if self.quad is not None:
            return np.asarray(self.quad, dtype=np.float32)
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
        self._pins: dict[str, tuple[float, float] | None] = {"left": None, "right": None}
        self._pin_age = {"left": 0.0, "right": 0.0}
        # Fingertip anchor: one filter per LABELLED tip (L index, L thumb, R index,
        # R thumb), smoothed before deciding which is top or bottom, so a swap can
        # never smear a corner across the box.
        self._f_tips = [OneEuroPoint(c.tip_min_cutoff, c.tip_beta) for _ in range(4)]
        self._tips: dict[str, np.ndarray | None] = {"left": None, "right": None}
        self._tips_palm: dict[str, tuple[float, float] | None] = {"left": None, "right": None}
        self._tip_speed = {"left": 0.0, "right": 0.0}  # frame widths per second, last seen
        self._hold = c.hold_frames
        self._ramp = 0.0

    def reset(self) -> None:
        self.__post_init__()

    def _outside_edge(self, pt: tuple[float, float]) -> tuple[float, float] | None:
        """Where a hand that just left through a border should be pinned, or None."""
        x, y = pt
        mx, my = self.config.edge_margin * self.width, self.config.edge_margin * self.height
        out_x, out_y = 0.04 * self.width, 0.04 * self.height
        if x < mx:
            return (-out_x, y)
        if x > self.width - mx:
            return (self.width + out_x, y)
        if y < my:
            return (x, -out_y)
        if y > self.height - my:
            return (x, self.height + out_y)
        return None

    def _edge_pin(self, side: str, seen, last, dt: float):
        """The palm to use for one side: seen, pinned outside an edge, or None."""
        if seen is not None or self.config.edge_hold_s <= 0.0:
            self._pins[side], self._pin_age[side] = None, 0.0
            return seen
        if self._pins[side] is None:
            if last is None:
                return None
            self._pins[side] = self._outside_edge(last)
            self._pin_age[side] = 0.0
            if self._pins[side] is None:
                return None
        self._pin_age[side] += dt
        if self._pin_age[side] > self.config.edge_hold_s:
            return None
        return self._pins[side]

    def _time_burst(self, t: float | None) -> float:
        c = self.config
        if c.reveal_at is None or t is None:
            return 0.0
        return smoothstep(c.reveal_at, c.reveal_at + max(c.reveal_over, 1e-3), t)

    def update(self, frame: Frame, dt: float, t: float | None = None) -> Portal:
        time_burst = self._time_burst(t)
        self._last_dt = dt
        if self.config.anchor == "fingertips":
            frame = Frame(left=self._usable(frame.left, "left"), right=self._usable(frame.right, "right"))
        left = frame.left.palm if frame.left else None
        right = frame.right.palm if frame.right else None
        # Edge hold is for the palms anchor. With fingertip corners, a hand that has
        # left the frame has no fingertips left to pin a corner to: pinning it bent
        # the box into a wedge (measured on real footage), so the box fades instead
        # and comes back the moment the hands do.
        if self.config.anchor == "palms":
            left = self._edge_pin("left", left, self._last_left, dt)
            right = self._edge_pin("right", right, self._last_right, dt)

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
        self._hold = self.config.hold_frames
        if self.config.anchor == "fingertips" and self._missing > 0:
            for side, seen in (("left", left), ("right", right)):
                if seen is None and self._tip_speed[side] > self.config.fast_exit:
                    self._hold = 0
        stand_in = self._missing <= self._hold or time_burst > 0.0
        if stand_in:
            if left is None:
                left = self._last_left
            if right is None:
                right = self._last_right

        if left is None or right is None:
            return self._fade_out()

        self._last_left, self._last_right = left, right
        tips_l = self._tips_for("left", frame.left, left)
        tips_r = self._tips_for("right", frame.right, right)

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
        self._ramp = min(1.0, self._ramp + 1.0 / max(self.config.appear_frames, 1))
        opacity = max(opacity * self._ramp, burst)

        quad = None
        if self.config.anchor == "fingertips" and tips_l is not None and tips_r is not None:
            quad = self._fingertip_quad(tips_l, tips_r, dt)
            if burst > 0.0:
                quad = self._grow_to_frame(quad, burst)

        self._last_portal = Portal(
            present=opacity > 0.001,
            center=(cx, cy),
            angle=angle,
            half_length=half_length,
            half_thickness=half_thickness,
            opacity=opacity,
            openness=openness,
            quad=quad,
        )
        return self._last_portal

    # -- fingertip anchor --------------------------------------------------
    def _usable(self, hand, side: str):
        """None for a hand that is mostly outside the frame; it is leaving, so no hold either."""
        if hand is None:
            return None
        pts = np.asarray(hand.points, dtype=np.float32)
        outside = (pts[:, 0] < 0) | (pts[:, 0] > self.width - 1) | (pts[:, 1] < 0) | (pts[:, 1] > self.height - 1)
        if float(outside.mean()) >= self.config.max_outside:
            self._tip_speed[side] = float("inf")
            return None
        return hand

    def _tips_for(self, side: str, hand, palm_used) -> np.ndarray | None:
        """(index tip, thumb tip) for one side: seen now, or carried along with a held or pinned palm."""
        if hand is not None:
            tips = np.asarray(hand.points, dtype=np.float32)[[8, 4]].copy()
            prev = self._tips[side]
            if prev is not None and self._tips_palm[side] is not None:
                step = float(np.max(np.linalg.norm(tips - prev, axis=1)))
                self._tip_speed[side] = step / max(self._last_dt, 1e-3) / max(self.width, 1)
            self._tips[side] = tips
            self._tips_palm[side] = hand.palm
            return self._tips[side]
        last, last_palm = self._tips[side], self._tips_palm[side]
        if last is None or last_palm is None or palm_used is None:
            return None
        # a held or edge-pinned hand moves its fingertips by the same offset as its palm
        shift = np.array([palm_used[0] - last_palm[0], palm_used[1] - last_palm[1]], dtype=np.float32)
        return last + shift

    def _fingertip_quad(self, tips_l: np.ndarray, tips_r: np.ndarray, dt: float) -> np.ndarray:
        raw = [tips_l[0], tips_l[1], tips_r[0], tips_r[1]]
        sm = [np.array(f((float(p[0]), float(p[1])), dt), dtype=np.float32) for f, p in zip(self._f_tips, raw)]
        li, lt, ri, rt = sm
        # "top" is the tip further against the normal of the left-to-right axis, so
        # the box never folds into a bow-tie, whatever the rotation of the hands
        ax = (ri + rt) / 2.0 - (li + lt) / 2.0
        n = float(np.hypot(ax[0], ax[1]))
        ux, uy = (float(ax[0]) / n, float(ax[1]) / n) if n > 1e-3 else (1.0, 0.0)
        vx, vy = -uy, ux

        def up(p):
            return float(p[0]) * vx + float(p[1]) * vy

        l_top, l_bot = (li, lt) if up(li) <= up(lt) else (lt, li)
        r_top, r_bot = (ri, rt) if up(ri) <= up(rt) else (rt, ri)
        return np.stack([l_top, r_top, r_bot, l_bot]).astype(np.float32)

    def _grow_to_frame(self, quad: np.ndarray, burst: float) -> np.ndarray:
        """Final reveal: each corner travels out to a box far bigger than the frame."""
        c = quad.mean(axis=0)
        top = quad[1] - quad[0]
        ang = math.atan2(float(top[1]), float(top[0])) if float(np.hypot(top[0], top[1])) > 1e-3 else 0.0
        big = Portal(
            present=True,
            center=(float(c[0]), float(c[1])),
            angle=ang,
            half_length=math.hypot(self.width, self.height) * 0.75,
            half_thickness=self.config.thickness_full * self.height * 0.5,
        ).corners()
        return (quad * (1.0 - burst) + big * burst).astype(np.float32)

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
        over = self._missing - self._hold
        if over <= 0:
            return self._last_portal
        k = 1.0 - over / max(self.config.fade_frames, 1)
        if k <= 0.0:
            self._ramp = 0.0
            self._tips = {"left": None, "right": None}
            self._tips_palm = {"left": None, "right": None}
            self._tip_speed = {"left": 0.0, "right": 0.0}
            for f in self._f_tips:
                f.reset()  # otherwise the corners start from where the fingertips were before
            self._last_left = self._last_right = None
            self._last_portal = Portal(present=False)
            return self._last_portal
        p = self._last_portal
        self._last_portal = replace(p, present=True, opacity=p.opacity * k)
        return self._last_portal
