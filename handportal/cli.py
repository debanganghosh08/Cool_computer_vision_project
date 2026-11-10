"""Command line for HandPortal.

    python -m handportal --input clip.mp4 --world storm --output out/portal.mp4
    python -m handportal --input 0 --world matrix --preview
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .compositor import EdgeStyle, PRESETS, compose, parse_color
from .portal import PortalConfig, PortalTracker
from .tracker import HandTracker
from .video import Source, make_writer, mux_audio
from .worlds import WorldContext, build_world

BANNER = r"""
  _  _              _ ___         _        _
 | || |__ _ _ _  __| | _ \___ _ _| |_ __ _| |
 | __ / _` | ' \/ _` |  _/ _ \ '_|  _/ _` | |
 |_||_\__,_|_||_\__,_|_| \___/_|  \__\__,_|_|
 open a portal between your two hands
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="handportal",
        description="Open a portal between your two hands and show another world inside it.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m handportal --input 0 --preview\n"
            "  python -m handportal --input clip.mp4 --world storm:magenta --lightning -o out/reel.mp4\n"
            "  python -m handportal --input clip.mp4 --world video:other.mp4 -o out/reel.mp4 --matte out/matte.mp4\n"
        ),
    )
    p.add_argument("--input", "-i", default="0", help="video file, or a camera index like 0 (default: 0)")
    p.add_argument("--output", "-o", help="where to write the mp4 (omit with --preview)")
    p.add_argument(
        "--world",
        "-w",
        default="storm",
        help=(
            "what is inside the portal: storm, matrix, thermal, edges, blueprint, comic, "
            "pixel, glitch, skeleton, or video:<path> / image:<path>. "
            "Some take an argument, like storm:cyan or pixel:48. (default: storm)"
        ),
    )
    p.add_argument("--preview", action="store_true", help="show a live window while it runs (q to quit)")
    p.add_argument("--matte", help="also write the portal mask as a black and white mp4, for an editor")
    p.add_argument("--mirror", action="store_true", help="flip the picture left to right (selfie view)")

    look = p.add_argument_group("look")
    look.add_argument(
        "--color", "-c", default="magenta",
        help=f"edge colour: {', '.join(sorted(PRESETS))}, or #ff33cc, or 255,51,204",
    )
    look.add_argument("--feather", type=int, default=7, help="soft edge in pixels (default: 7)")
    look.add_argument("--glow", type=float, default=1.0, help="edge glow strength, 0 turns it off")
    look.add_argument("--glow-width", type=int, default=26, help="how far the glow bleeds, in pixels")
    look.add_argument("--rim", type=int, default=3, help="bright line on the edge, in pixels, 0 for none")
    look.add_argument("--lightning", action="store_true", help="crawl bolts along the portal edge")

    shape = p.add_argument_group("portal shape")
    shape.add_argument("--thickness-min", type=float, default=0.04, help="thickness when hands are together, as a fraction of height")
    shape.add_argument("--thickness-max", type=float, default=0.17, help="thickness when hands are wide, as a fraction of height")
    shape.add_argument("--thickness-full", type=float, default=1.60, help="thickness at the final reveal (above 1.0 covers the frame)")
    shape.add_argument("--spread-closed", type=float, default=0.22, help="hand distance that counts as closed, as a fraction of width")
    shape.add_argument("--spread-wide", type=float, default=0.80, help="hand distance that counts as fully open")
    shape.add_argument("--spread-full", type=float, default=0.0, help="hand distance that triggers the full-frame reveal (0 = never, see docs)")
    shape.add_argument("--overhang", type=float, default=0.10, help="how far past each palm the portal reaches")
    shape.add_argument("--hold", type=int, default=6, help="frames to keep the portal alive after a hand is lost")

    reveal = p.add_argument_group("final reveal")
    reveal.add_argument("--reveal-at", type=float, help="seconds into the clip when the portal swallows the whole frame")
    reveal.add_argument("--reveal-over", type=float, default=0.6, help="how long that takes, in seconds (default: 0.6)")

    smooth = p.add_argument_group("smoothing")
    smooth.add_argument("--min-cutoff", type=float, default=1.2, help="lower = smoother when still")
    smooth.add_argument("--beta", type=float, default=0.02, help="higher = less lag when moving fast")

    other = p.add_argument_group("other")
    other.add_argument("--model", help="path to hand_landmarker.task (downloaded on first run if absent)")
    other.add_argument("--detect-confidence", type=float, default=0.5)
    other.add_argument("--max-frames", type=int, default=0, help="stop after N frames, for a quick test")
    other.add_argument("--no-audio", action="store_true", help="do not copy the original sound onto the export")
    other.add_argument("--quiet", "-q", action="store_true")
    other.add_argument("--version", action="version", version=f"handportal {__version__}")
    return p


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.output and not args.preview:
        print("Nothing to do: pass --output to render a file, or --preview to watch it live.", file=sys.stderr)
        return 2

    if not args.quiet:
        print(BANNER, file=sys.stderr)

    try:
        style = EdgeStyle(
            color=parse_color(args.color),
            feather=args.feather,
            glow=args.glow,
            glow_width=args.glow_width,
            rim=args.rim,
            lightning=args.lightning,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        world = build_world(args.world)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        source = Source.open(args.input)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    config = PortalConfig(
        overhang=args.overhang,
        thickness_min=args.thickness_min,
        thickness_max=args.thickness_max,
        thickness_full=args.thickness_full,
        spread_closed=args.spread_closed,
        spread_wide=args.spread_wide,
        spread_full=args.spread_full,
        reveal_at=args.reveal_at,
        reveal_over=args.reveal_over,
        hold_frames=args.hold,
        min_cutoff=args.min_cutoff,
        beta=args.beta,
    )

    out_path = Path(args.output) if args.output else None
    tmp_path = out_path.with_name(out_path.stem + "_silent" + out_path.suffix) if out_path else None
    matte_path = Path(args.matte) if args.matte else None

    writer = None
    matte_writer = None
    rng = np.random.default_rng(1234)
    started = time.time()
    index = 0
    seen_hands = 0

    if not args.quiet:
        kind = "camera" if source.is_camera else "file"
        print(
            f"[handportal] {kind} {args.input} | {source.width}x{source.height} @ {source.fps:g} fps"
            + (f" | {source.frames} frames" if source.frames else ""),
            file=sys.stderr,
        )
        print(f"[handportal] world: {args.world}", file=sys.stderr)

    try:
        tracker = HandTracker(
            args.model,
            video_mode=True,
            min_detection_confidence=args.detect_confidence,
            quiet=args.quiet,
        )
    except (SystemExit, RuntimeError) as exc:
        source.close()
        print(f"error: {exc}", file=sys.stderr)
        return 1

    portal_tracker = PortalTracker(width=source.width, height=source.height, config=config)
    dt = 1.0 / source.fps

    try:
        while True:
            frame = source.read()
            if frame is None:
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)

            timestamp_ms = int(index * 1000.0 / source.fps)
            detected = tracker.detect(frame, timestamp_ms)
            if detected.both:
                seen_hands += 1
            portal = portal_tracker.update(detected, dt, t=index / source.fps)

            ctx = WorldContext(
                frame=frame,
                index=index,
                time=index / source.fps,
                size=(frame.shape[0], frame.shape[1]),
                hands=[h.points for h in detected.hands],
            )
            world_frame = world.render(ctx)
            out_frame, mask = compose(frame, world_frame, portal, style, rng)

            if out_path is not None:
                if writer is None:
                    writer = make_writer(tmp_path, source.fps, (frame.shape[1], frame.shape[0]))
                writer.write(out_frame)
            if matte_path is not None:
                if matte_writer is None:
                    matte_writer = make_writer(
                        matte_path, source.fps, (frame.shape[1], frame.shape[0]), gray=True
                    )
                matte_writer.write(mask)

            if args.preview:
                cv2.imshow("HandPortal - press q to quit", out_frame)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break

            index += 1
            if args.max_frames and index >= args.max_frames:
                break
            if not args.quiet and source.frames and index % 30 == 0:
                pct = 100.0 * index / source.frames
                print(f"\r[handportal] {index}/{source.frames} frames ({pct:5.1f}%)", end="", file=sys.stderr)
    except KeyboardInterrupt:
        print("\n[handportal] stopped", file=sys.stderr)
    finally:
        if not args.quiet and source.frames:
            print(file=sys.stderr)
        source.close()
        tracker.close()
        world.close()
        if writer is not None:
            writer.release()
        if matte_writer is not None:
            matte_writer.release()
        if args.preview:
            cv2.destroyAllWindows()

    if index == 0:
        print("error: no frames were read.", file=sys.stderr)
        return 1

    if out_path is not None and tmp_path is not None:
        muxed = False
        if not args.no_audio and not source.is_camera and source.path is not None:
            muxed = mux_audio(tmp_path, source.path, out_path)
        if muxed:
            tmp_path.unlink(missing_ok=True)
        else:
            out_path.unlink(missing_ok=True)
            tmp_path.replace(out_path)

    elapsed = time.time() - started
    if not args.quiet:
        hands_pct = 100.0 * seen_hands / max(index, 1)
        print(
            f"[handportal] {index} frames in {elapsed:.1f}s "
            f"({index / max(elapsed, 1e-6):.1f} fps) | both hands seen in {hands_pct:.0f}% of frames",
            file=sys.stderr,
        )
        if hands_pct < 25:
            print(
                "[handportal] tip: both hands were rarely visible. Keep them inside the frame, "
                "palms toward the camera, and make sure the room is lit.",
                file=sys.stderr,
            )
        if out_path is not None:
            print(f"[handportal] wrote {out_path}", file=sys.stderr)
        if matte_path is not None:
            print(f"[handportal] wrote {matte_path}", file=sys.stderr)
    return 0


def main() -> None:
    raise SystemExit(run())
