"""One Euro filter.

Raw landmark coordinates jitter by a few pixels every frame even when a hand is
perfectly still. Averaging over a window kills the jitter but adds lag, and lag
is very visible when a portal edge is supposed to be stuck to a palm.

The One Euro filter (Casiez, Roussel, Vogel, CHI 2012) solves both at once: it is
a low-pass filter whose cutoff frequency rises with the speed of the signal.
Slow movement -> heavy smoothing -> no jitter. Fast movement -> light smoothing
-> no lag. Two knobs:

    min_cutoff  lower = smoother when still (and slightly laggier)
    beta        higher = less lag when moving fast
"""

from __future__ import annotations

import math


class _LowPass:
    __slots__ = ("_value", "_has_value")

    def __init__(self) -> None:
        self._value = 0.0
        self._has_value = False

    def __call__(self, x: float, alpha: float) -> float:
        if not self._has_value:
            self._value = x
            self._has_value = True
        else:
            self._value = alpha * x + (1.0 - alpha) * self._value
        return self._value

    @property
    def value(self) -> float:
        return self._value

    @property
    def ready(self) -> bool:
        return self._has_value

    def reset(self) -> None:
        self._value = 0.0
        self._has_value = False


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuro:
    """Smooth one scalar signal."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.015, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x = _LowPass()
        self._dx = _LowPass()
        self._last: float | None = None

    def reset(self) -> None:
        self._x.reset()
        self._dx.reset()
        self._last = None

    def __call__(self, x: float, dt: float) -> float:
        if dt <= 0.0:
            dt = 1.0 / 30.0
        prev = self._last if self._last is not None else x
        dx = (x - prev) / dt
        self._last = x
        edx = self._dx(dx, _alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self._x(x, _alpha(cutoff, dt))


class OneEuroPoint:
    """Smooth a 2D point with one filter per axis."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.015) -> None:
        self._fx = OneEuro(min_cutoff, beta)
        self._fy = OneEuro(min_cutoff, beta)

    def reset(self) -> None:
        self._fx.reset()
        self._fy.reset()

    def __call__(self, point: tuple[float, float], dt: float) -> tuple[float, float]:
        return self._fx(point[0], dt), self._fy(point[1], dt)


class OneEuroAngle:
    """Smooth an angle in radians without the +pi / -pi jump.

    Filtering an angle directly makes the portal spin a full turn whenever the
    value wraps, so the angle is carried as a unit vector and filtered there.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.015) -> None:
        self._fc = OneEuro(min_cutoff, beta)
        self._fs = OneEuro(min_cutoff, beta)

    def reset(self) -> None:
        self._fc.reset()
        self._fs.reset()

    def __call__(self, angle: float, dt: float) -> float:
        c = self._fc(math.cos(angle), dt)
        s = self._fs(math.sin(angle), dt)
        if c == 0.0 and s == 0.0:
            return angle
        return math.atan2(s, c)
