"""Tests for the parts that do not need a camera, a model or a GPU.

    python -m pytest -q          (if you have pytest)
    python tests/test_geometry.py   (if you do not)

Nothing here imports mediapipe, so these run on any machine in under a second.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from handportal.compositor import EdgeStyle, parse_color, portal_mask  # noqa: E402
from handportal.portal import Portal, PortalConfig, PortalTracker, smoothstep  # noqa: E402
from handportal.smoothing import OneEuro, OneEuroAngle  # noqa: E402
from handportal.tracker import Frame, Hand  # noqa: E402

W, H = 720, 1280


def fake_hand(x: float, y: float) -> Hand:
    pts = np.tile(np.array([[x, y]], dtype=np.float32), (21, 1))
    pts[12] = [x, y - 90]  # so span is non-zero
    return Hand(points=pts, palm=(x, y), score=1.0)


def frame_at(lx: float, rx: float, y: float = 640.0) -> Frame:
    return Frame(left=fake_hand(lx, y), right=fake_hand(rx, y))


def settle(tracker: PortalTracker, lx: float, rx: float, n: int = 40, t0: float = 0.0) -> Portal:
    """Feed the same pose for n frames so the smoothing converges."""
    portal = Portal(present=False)
    for i in range(n):
        portal = tracker.update(frame_at(lx, rx), 1 / 30, t=t0 + i / 30)
    return portal


# -- smoothstep -------------------------------------------------------------


def test_smoothstep_clamps_and_rises():
    assert smoothstep(0.0, 1.0, -5.0) == 0.0
    assert smoothstep(0.0, 1.0, 5.0) == 1.0
    assert smoothstep(0.0, 1.0, 0.5) == 0.5
    assert smoothstep(0.0, 1.0, 0.25) < 0.25  # S-curve, slow at the start


def test_smoothstep_degenerate_range():
    assert smoothstep(1.0, 1.0, 0.9) == 0.0
    assert smoothstep(1.0, 1.0, 1.1) == 1.0


# -- one euro ---------------------------------------------------------------


def test_one_euro_converges_to_a_constant():
    f = OneEuro()
    for _ in range(200):
        out = f(10.0, 1 / 30)
    assert abs(out - 10.0) < 1e-3


def test_one_euro_kills_jitter():
    """Noise around a constant must come out far smaller than it went in."""
    rng = np.random.default_rng(0)
    f = OneEuro(min_cutoff=0.8, beta=0.01)
    raw, smooth = [], []
    for _ in range(300):
        x = 100.0 + rng.normal(0, 3.0)
        raw.append(x)
        smooth.append(f(x, 1 / 30))
    assert np.std(smooth[100:]) < np.std(raw[100:]) / 3


def test_one_euro_angle_does_not_jump_at_pi():
    """The whole reason angles are filtered as a vector."""
    f = OneEuroAngle()
    for _ in range(60):
        f(math.pi - 0.01, 1 / 30)
    out = f(-math.pi + 0.01, 1 / 30)
    # The true move is 0.02 rad across the wrap, not 6.26 rad the other way.
    assert abs(abs(out) - math.pi) < 0.2


# -- portal geometry --------------------------------------------------------


def test_no_hands_means_no_portal():
    t = PortalTracker(W, H)
    for _ in range(20):
        p = t.update(Frame(None, None), 1 / 30)
    assert not p.present


def test_hands_together_give_a_thin_portal():
    t = PortalTracker(W, H)
    p = settle(t, 340, 380)
    assert p.half_thickness * 2 < H * 0.12


def test_wider_hands_give_a_thicker_and_longer_portal():
    narrow = settle(PortalTracker(W, H), 300, 420)
    wide = settle(PortalTracker(W, H), 90, 630)
    assert wide.half_length > narrow.half_length
    assert wide.half_thickness > narrow.half_thickness


def test_portal_never_bursts_on_spread_by_default():
    """A wide 'palms to camera' pose must stay a band, not swallow the frame.

    This is the regression that mattered: with an automatic spread trigger the
    reveal fired during ordinary gestures.
    """
    p = settle(PortalTracker(W, H), 20, 700)
    assert p.half_thickness * 2 < H * 0.30


def test_spread_burst_fires_when_explicitly_enabled():
    cfg = PortalConfig(spread_full=0.92)
    p = settle(PortalTracker(W, H, cfg), 10, 710)
    assert p.half_thickness * 2 > H


def test_timed_reveal_opens_the_whole_frame():
    cfg = PortalConfig(reveal_at=1.0, reveal_over=0.3)
    t = PortalTracker(W, H, cfg)
    before = settle(t, 300, 420, n=20, t0=0.0)
    after = settle(t, 300, 420, n=30, t0=1.5)
    assert before.half_thickness * 2 < H * 0.3
    assert after.half_thickness * 2 > H
    assert after.half_length * 2 > W


def test_portal_angle_follows_the_hand_line():
    t = PortalTracker(W, H)
    for i in range(60):
        t.update(Frame(left=fake_hand(200, 800), right=fake_hand(520, 480)), 1 / 30, t=i / 30)
    p = t.update(Frame(left=fake_hand(200, 800), right=fake_hand(520, 480)), 1 / 30, t=2.0)
    expected = math.atan2(480 - 800, 520 - 200)
    assert abs(p.angle - expected) < 0.12


def test_corners_are_a_rectangle():
    p = Portal(present=True, center=(360, 640), angle=0.4, half_length=200, half_thickness=60, opacity=1.0)
    c = p.corners()
    side_a = np.linalg.norm(c[1] - c[0])
    side_b = np.linalg.norm(c[2] - c[1])
    assert abs(side_a - 400) < 1e-3
    assert abs(side_b - 120) < 1e-3
    # opposite sides equal, and the diagonals match => it really is a rectangle
    assert abs(np.linalg.norm(c[3] - c[2]) - side_a) < 1e-3
    assert abs(np.linalg.norm(c[2] - c[0]) - np.linalg.norm(c[3] - c[1])) < 1e-3


def test_hold_then_fade_when_a_hand_is_lost():
    cfg = PortalConfig(hold_frames=5, fade_frames=4)
    t = PortalTracker(W, H, cfg)
    settle(t, 200, 520)
    held = t.update(Frame(left=fake_hand(200, 640), right=None), 1 / 30)
    assert held.present and held.opacity > 0.5
    for _ in range(30):
        gone = t.update(Frame(None, None), 1 / 30)
    assert not gone.present


def tip_hand(palm_x, palm_y, index, thumb):
    """A hand whose index tip and thumb tip are at given points."""
    h = fake_hand(palm_x, palm_y)
    h.points[8] = index
    h.points[4] = thumb
    return h


def tip_frame(lx, rx, y=640.0):
    # director's frame: index tips up and inward, thumb tips below them
    left = tip_hand(lx, y, (lx + 60, y - 80), (lx + 70, y + 20))
    right = tip_hand(rx, y, (rx - 60, y - 70), (rx - 70, y + 25))
    return Frame(left=left, right=right)


def test_fingertip_box_starts_at_the_fingertips():
    t = PortalTracker(W, H, PortalConfig())
    for i in range(40):
        p = t.update(tip_frame(150, 570), 1 / 30, t=i / 30)
    q = p.corners()
    want = np.array([[210, 560], [510, 570], [500, 665], [220, 660]], dtype=np.float32)
    assert np.abs(q - want).max() < 1.0, q


def test_fingertip_box_is_a_sliver_when_tips_touch():
    t = PortalTracker(W, H, PortalConfig())
    f = Frame(left=tip_hand(300, 640, (355, 600), (355, 630)), right=tip_hand(420, 640, (365, 600), (365, 630)))
    for i in range(40):
        p = t.update(f, 1 / 30, t=i / 30)
    q = p.corners()
    assert np.linalg.norm(q[1] - q[0]) < 12 and np.linalg.norm(q[2] - q[3]) < 12


def test_fingertip_box_never_folds_into_a_bow_tie():
    # thumbs ABOVE index tips on one side must still give top-then-bottom corners
    t = PortalTracker(W, H, PortalConfig())
    f = Frame(left=tip_hand(150, 640, (220, 680), (210, 580)), right=tip_hand(570, 640, (500, 580), (510, 680)))
    for i in range(30):
        p = t.update(f, 1 / 30, t=i / 30)
    q = p.corners()
    assert q[0][1] < q[3][1] and q[1][1] < q[2][1]


def test_fingertip_corner_keeps_up_with_a_moving_finger():
    t = PortalTracker(W, H, PortalConfig())
    for i in range(40):
        p = t.update(tip_frame(150 + i * 4, 570), 1 / 30, t=i / 30)  # left hand slides 4 px a frame
    assert abs(p.corners()[0][0] - (150 + 39 * 4 + 60)) < 6


def test_fingertip_reveal_covers_the_whole_frame():
    t = PortalTracker(W, H, PortalConfig(reveal_at=1.0, reveal_over=0.5))
    for i in range(60):
        p = t.update(tip_frame(150, 570), 1 / 30, t=i / 30)
    m = portal_mask(p, (H, W), feather=0)
    assert m.min() > 0.99


def test_palms_anchor_still_available():
    t = PortalTracker(W, H, PortalConfig(anchor="palms"))
    for i in range(40):
        p = t.update(tip_frame(150, 570), 1 / 30, t=i / 30)
    assert p.quad is None and p.half_length > 210


def test_fast_exit_fades_the_whole_box_instead_of_freezing_half():
    t = PortalTracker(W, H, PortalConfig(hold_frames=6, fade_frames=4))
    for i in range(30):
        last = t.update(tip_frame(150, 570), 1 / 30, t=i / 30)
    t.update(tip_frame(150, 640), 1 / 30, t=30 / 30)  # right hand swings out: 70 px in one frame
    before = t.update(tip_frame(150, 700), 1 / 30, t=31 / 30)
    gone = t.update(Frame(left=tip_hand(120, 640, (180, 560), (190, 660)), right=None), 1 / 30, t=32 / 30)
    assert gone.opacity < before.opacity  # fading at once, no hold
    assert np.allclose(gone.corners(), before.corners())  # the last whole box, not a wedge


def test_hand_mostly_outside_the_frame_is_not_a_corner():
    t = PortalTracker(W, H, PortalConfig(hold_frames=6, fade_frames=4))
    for i in range(30):
        before = t.update(tip_frame(150, 570), 1 / 30, t=i / 30)
    leaving = tip_hand(700, 640, (690, 900), (650, 700))  # tips are guesses
    leaving.points[:12, 0] = W + 50  # 12 of 21 landmarks past the right edge
    p = t.update(Frame(left=tip_hand(150, 640, (210, 560), (220, 660)), right=leaving), 1 / 30, t=1.0)
    assert p.opacity < before.opacity  # fading
    assert np.allclose(p.corners(), before.corners())  # the guessed tip never became a corner


def test_corners_start_fresh_after_the_box_was_gone():
    t = PortalTracker(W, H, PortalConfig(hold_frames=2, fade_frames=2))
    for i in range(30):
        t.update(tip_frame(150, 570), 1 / 30, t=i / 30)
    for i in range(10):
        t.update(Frame(None, None), 1 / 30, t=(30 + i) / 30)
    p = t.update(tip_frame(100, 520, y=900), 1 / 30, t=41 / 30)  # hands come back somewhere else
    assert abs(p.corners()[0][0] - 160) < 1 and abs(p.corners()[0][1] - 820) < 1


def test_slow_hand_that_blinks_out_is_still_held():
    t = PortalTracker(W, H, PortalConfig(hold_frames=6, fade_frames=4))
    for i in range(30):
        t.update(tip_frame(150, 570), 1 / 30, t=i / 30)
    held = t.update(Frame(left=tip_hand(150, 640, (210, 560), (220, 660)), right=None), 1 / 30, t=1.0)
    assert held.opacity > 0.99


def test_box_fades_in_when_it_comes_back():
    t = PortalTracker(W, H, PortalConfig())
    first = t.update(tip_frame(150, 570), 1 / 30, t=0.0)
    assert 0.0 < first.opacity < 0.5
    for i in range(1, 6):
        p = t.update(tip_frame(150, 570), 1 / 30, t=i / 30)
    assert p.opacity > 0.99


def test_fingertip_box_fades_when_a_hand_leaves_the_frame():
    t = PortalTracker(W, H, PortalConfig(hold_frames=5, fade_frames=4))
    for i in range(40):
        t.update(tip_frame(150, 680), 1 / 30, t=i / 30)
    for _ in range(30):
        p = t.update(Frame(left=fake_hand(150, 640), right=None), 1 / 30)
    assert not p.present


def test_hand_leaving_through_the_edge_keeps_the_band():
    # palms anchor, arms swung out wide: the right hand was last seen near the right border
    t = PortalTracker(W, H, PortalConfig(hold_frames=5, fade_frames=4, anchor="palms"))
    settle(t, 200, 680)
    for _ in range(30):  # one second, far past hold_frames
        p = t.update(Frame(left=fake_hand(200, 640), right=None), 1 / 30)
    assert p.present and p.opacity > 0.5
    # the band now reaches past the right edge of the frame
    assert p.center[0] + p.half_length > W


def test_edge_pin_expires():
    t = PortalTracker(W, H, PortalConfig(hold_frames=5, fade_frames=4, edge_hold_s=0.5, anchor="palms"))
    settle(t, 200, 680)
    for _ in range(60):
        p = t.update(Frame(left=fake_hand(200, 640), right=None), 1 / 30)
    assert not p.present


def test_edge_pin_releases_when_the_hand_comes_back():
    t = PortalTracker(W, H, PortalConfig(anchor="palms"))
    settle(t, 200, 680)
    for _ in range(10):
        t.update(Frame(left=fake_hand(200, 640), right=None), 1 / 30)
    p = settle(t, 200, 520, n=60)
    assert abs((p.center[0] + p.half_length / 1.1) - 520) < 40  # follows the real hand again


# -- mask and colour --------------------------------------------------------


def test_mask_is_empty_without_a_portal():
    m = portal_mask(Portal(present=False), (H, W))
    assert m.max() == 0.0


def test_mask_covers_roughly_the_rectangle_area():
    p = Portal(present=True, center=(360, 640), angle=0.0, half_length=150, half_thickness=50, opacity=1.0)
    m = portal_mask(p, (H, W), feather=0)
    assert abs(m.sum() - 300 * 100) / (300 * 100) < 0.02
    assert m.max() <= 1.0 and m.min() >= 0.0


def test_mask_opacity_scales_the_mask():
    p = Portal(present=True, center=(360, 640), angle=0.0, half_length=150, half_thickness=50, opacity=0.5)
    m = portal_mask(p, (H, W), feather=0)
    # The mask is stored as 8-bit, so 0.5 lands on 128/255, not exactly 0.5.
    assert abs(m.max() - 0.5) < 1.5 / 255


def test_parse_color_forms_all_agree():
    assert parse_color("#ff33cc") == (204, 51, 255)  # BGR
    assert parse_color("255,51,204") == (204, 51, 255)
    assert parse_color("magenta") == parse_color("MAGENTA")


def test_parse_color_rejects_nonsense():
    for bad in ("nope", "#12", "1,2", ""):
        try:
            parse_color(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} should not have parsed")


def test_edge_style_defaults_are_sane():
    s = EdgeStyle()
    assert s.feather >= 0 and s.glow_width > 0


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  pass  {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"  FAIL  {name}: {exc}")
    print(f"\n{'all green' if not failures else str(failures) + ' failed'}")
    raise SystemExit(1 if failures else 0)
